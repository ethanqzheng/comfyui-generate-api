"""
ComfyUI执行器 - 与ComfyUI后端通信并执行工作流
"""

# Ensure package root is in sys.path BEFORE any other imports
import sys
from pathlib import Path

# Get the package root directory (parent of 'core' directory)
_package_root = Path(__file__).resolve().parent.parent
_package_root_str = str(_package_root)
if _package_root_str not in sys.path:
    sys.path.insert(0, _package_root_str)

import asyncio
import json
import uuid
import logging
import base64
from typing import Dict, Any, Optional, Tuple, Set
import httpx
from PIL import Image
import io
import websockets
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

from api.types.generate import GenerateRequest
from api.types.workflow import Workflow
from core.workflow_analyzer import WorkflowAnalyzer, WorkflowGraph

# Import utils modules using importlib for more reliable loading
import importlib.util
import types

def _import_utils_module(module_name, package_root):
    """Dynamically import a module from utils package"""
    module_path = package_root / "utils" / f"{module_name}.py"
    if not module_path.exists():
        raise ImportError(f"Cannot find module {module_name} at {module_path}")
    
    # Ensure utils package exists in sys.modules
    if "utils" not in sys.modules:
        utils_pkg = types.ModuleType("utils")
        utils_pkg.__path__ = [str(package_root / "utils")]
        sys.modules["utils"] = utils_pkg
    
    spec = importlib.util.spec_from_file_location(f"utils.{module_name}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create spec for utils.{module_name}")
    
    module = importlib.util.module_from_spec(spec)
    # Add to sys.modules to make it importable
    sys.modules[f"utils.{module_name}"] = module
    spec.loader.exec_module(module)
    return module

# Try normal import first
try:
    from utils.progress import (
        log_job_prompt,
        log_parameter_update,
        log_workflow_submitted,
    )
    from utils.progress_tracker import PromptProgressTracker, ProgressSnapshot
except ImportError:
    # Fallback: use importlib to dynamically load
    _progress_module = _import_utils_module("progress", _package_root)
    _tracker_module = _import_utils_module("progress_tracker", _package_root)
    
    log_job_prompt = _progress_module.log_job_prompt
    log_parameter_update = _progress_module.log_parameter_update
    log_workflow_submitted = _progress_module.log_workflow_submitted
    PromptProgressTracker = _tracker_module.PromptProgressTracker
    ProgressSnapshot = _tracker_module.ProgressSnapshot

logger = logging.getLogger(__name__)


class ComfyExecutor:
    """ComfyUI执行器"""

    def __init__(self, host: str = "localhost", port: int = 8188, timeout: int = 300):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.base_url = f"http://{host}:{port}"
        self.ws_url = f"ws://{host}:{port}/ws"

        # ComfyUI input目录（用于图生图）
        # 从 core/executor.py 开始：parents[0]=core, parents[1]=comfyui-generate-api, 
        # parents[2]=custom_nodes, parents[3]=ComfyUI root
        comfy_root = Path(__file__).resolve().parents[3]  # core/executor.py -> ComfyUI root
        self.comfyui_input_dir = comfy_root / "input"
        self.comfyui_input_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("ComfyUI input directory: %s", self.comfyui_input_dir)

        # HTTP客户端
        self.http_client: Optional[httpx.AsyncClient] = None
        self.client_id = str(uuid.uuid4())

        # 工作流分析器
        self.workflow_analyzer = WorkflowAnalyzer()

        # 进度跟踪
        self.progress_tracker = PromptProgressTracker()
        self._ws_task: Optional[asyncio.Task] = None
        self._ws_connection: Optional[websockets.WebSocketClientProtocol] = None
        self._ws_should_stop = False

        # 连接状态
        self.connected = False

    async def initialize(self):
        """初始化执行器"""
        timeout = httpx.Timeout(timeout=self.timeout)
        self.http_client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
        )

        if await self.check_connection():
            self.connected = True
            logger.info("ComfyUI executor initialized successfully")
            await self._ensure_progress_listener()
        else:
            raise ConnectionError(f"Unable to connect to ComfyUI server: {self.base_url}")

    async def cleanup(self):
        """清理资源"""
        await self._stop_progress_listener()
        if self.http_client:
            await self.http_client.aclose()
            self.http_client = None

        self.connected = False
        logger.info("ComfyUI executor cleaned up")

    async def check_connection(self) -> bool:
        """检查ComfyUI连接状态"""
        try:
            if not self.http_client:
                return False

            response = await self.http_client.get("/system_stats")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Connection check failed: {e}")
            return False

    async def submit_prompt(self, request: GenerateRequest, workflow: Workflow) -> Dict[str, Any]:
        """执行工作流并立即返回ComfyUI的prompt_id"""
        if not self.http_client:
            raise RuntimeError("ComfyExecutor is not initialized")

        if request.input_image:
            request.input_image = await self._process_input_image(request.input_image)

        provided_params = self._get_provided_params(request)
        workflow_data, applied_params = self._prepare_workflow_data(request, workflow, provided_params)

        payload = {
            "prompt": workflow_data,
            "client_id": self.client_id,
            "extra_data": {"workflow": workflow.name},
        }

        # 记录 payload 关键信息（避免输出整个 workflow_data）
        logger.debug(
            "Submitting prompt to ComfyUI: workflow=%s, payload: %s", 
            workflow.name, json.dumps(payload, indent=2, ensure_ascii=False),
        )
        response = await self.http_client.post("/prompt", json=payload)

        if response.status_code == 400:
            try:
                error_data = response.json()
                logger.error(
                    "ComfyUI server returns 400 error: %s",
                    json.dumps(error_data, indent=2, ensure_ascii=False),
                )
            except Exception:
                logger.error("ComfyUI server returns 400 error: %s", response.text)

        response.raise_for_status()

        result = response.json()
        prompt_id = result.get("prompt_id")
        log_workflow_submitted(prompt_id)

        param_values = {k: getattr(request, k) for k in provided_params}
        log_job_prompt(prompt_id, request.prompt, workflow.name, param_values)

        return {
            "prompt_id": prompt_id,
            "client_id": self.client_id,
            "applied_params": applied_params,
            "workflow_data": workflow_data,
        }

    async def forward_request(self, method: str, path: str, **kwargs) -> httpx.Response:
        """通用请求转发到ComfyUI"""
        if not self.http_client:
            raise RuntimeError("ComfyExecutor is not initialized")

        if not path.startswith("/"):
            path = f"/{path}"

        return await self.http_client.request(method, path, **kwargs)

    async def fetch_history(self, prompt_id: str) -> Dict[str, Any]:
        """获取ComfyUI生成历史"""
        if not self.http_client:
            raise RuntimeError("ComfyExecutor is not initialized")

        response = await self.http_client.get(f"/history/{prompt_id}")
        response.raise_for_status()
        return response.json()

    async def fetch_queue(self) -> Dict[str, Any]:
        """获取ComfyUI队列状态（包括正在执行和等待的任务）"""
        if not self.http_client:
            raise RuntimeError("ComfyExecutor is not initialized")

        try:
            response = await self.http_client.get("/queue")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code if e.response else 0
            logger.error("Failed to fetch queue: HTTP %d - %s", status_code, e)
            raise
        except Exception as e:
            logger.error("Failed to fetch queue: %s", e, exc_info=True)
            raise

    async def check_prompt_in_queue(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        """检查prompt_id是否在队列中（正在执行或等待执行）
        
        返回:
            - 如果在队列中，返回包含状态信息的字典: {"status": "running" | "pending", "position": int}
            - 如果不在队列中，返回 None
        """
        try:
            queue_data = await self.fetch_queue()
            running_count = len(queue_data.get("queue_running", []))
            pending_count = len(queue_data.get("queue_pending", []))
            logger.info("Queue data received: queue_running=%d items, queue_pending=%d items", 
                       running_count, pending_count)
            
            def extract_prompt_id_from_item(item: Any) -> Optional[str]:
                """从队列项中提取 prompt_id，支持多种格式
                
                ComfyUI 队列格式通常是: [prompt_id, extra_data_dict]
                其中 prompt_id 是字符串（UUID 格式：36 字符，4 个连字符）
                """
                if isinstance(item, str):
                    # 直接是 prompt_id 字符串（优先匹配 UUID 格式：36 字符，4 个连字符）
                    if len(item) == 36 and item.count('-') == 4:
                        return item
                    # 如果不是 UUID 格式，仍然返回（兼容非标准格式）
                    return item
                
                if isinstance(item, dict):
                    # 格式: {prompt_id: ...} 或 {job_id: ...} 或 {id: ...}
                    for key in ("prompt_id", "job_id", "id"):
                        if key in item:
                            return item[key]
                    return None
                
                if isinstance(item, list):
                    # 遍历列表中的所有元素，查找 prompt_id
                    for elem in item:
                        if isinstance(elem, str) and len(elem) == 36 and elem.count('-') == 4:
                            return elem
                        if isinstance(elem, dict):
                            for key in ("prompt_id", "job_id", "id"):
                                if key in elem:
                                    return elem[key]
                
                return None
            
            # 检查正在执行的任务
            for item in queue_data.get("queue_running", []):
                if extract_prompt_id_from_item(item) == prompt_id:
                    logger.info("Found prompt_id %s in running queue", prompt_id)
                    return {"status": "running", "position": 0}
            
            # 检查等待执行的任务
            for idx, item in enumerate(queue_data.get("queue_pending", [])):
                if extract_prompt_id_from_item(item) == prompt_id:
                    logger.info("Found prompt_id %s in pending queue at position %d", prompt_id, idx + 1)
                    return {"status": "pending", "position": idx + 1}
            
            return None
        except httpx.HTTPStatusError as e:
            logger.error("Failed to check queue status (HTTP error): %s", e)
            # 如果队列 API 不可用，返回 None 而不是抛出异常
            # 这样调用者可以继续处理（比如返回 404）
            return None
        except Exception as e:
            logger.error("Failed to check queue status: %s", e, exc_info=True)
            return None

    async def get_progress(self, prompt_id: str) -> Optional[ProgressSnapshot]:
        """获取prompt的最新进度"""
        if not prompt_id:
            return None
        return await self.progress_tracker.get_progress(prompt_id)

    async def _ensure_progress_listener(self):
        """启动WebSocket监听，用于拉取实时进度"""
        if self._ws_task and not self._ws_task.done():
            return
        self._ws_should_stop = False
        self._ws_task = asyncio.create_task(self._progress_ws_loop())

    async def _stop_progress_listener(self):
        """停止WebSocket监听"""
        self._ws_should_stop = True
        if self._ws_connection:
            try:
                await self._ws_connection.close()
            except Exception:
                pass
            self._ws_connection = None

        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
            self._ws_task = None

    async def _progress_ws_loop(self):
        """持续监听ComfyUI WebSocket获取进度"""
        if not self.client_id:
            logger.warning("WebSocket listener skipped: client_id missing")
            return

        url = f"{self.ws_url}?clientId={self.client_id}"
        feature_flags_msg = json.dumps({"type": "feature_flags", "data": {"supports_preview_metadata": False}})
        backoff = 1

        while not self._ws_should_stop:
            try:
                async with websockets.connect(url, ping_interval=30, ping_timeout=30) as ws:
                    self._ws_connection = ws
                    backoff = 1
                    try:
                        await ws.send(feature_flags_msg)
                    except Exception as exc:
                        logger.debug("Failed to send feature flags: %s", exc)

                    async for message in ws:
                        if self._ws_should_stop:
                            break
                        if isinstance(message, bytes):
                            continue
                        await self._handle_ws_message(message)
            except asyncio.CancelledError:
                break
            except (ConnectionClosedError, ConnectionClosedOK):
                if self._ws_should_stop:
                    break
            except Exception as exc:
                if self._ws_should_stop:
                    break
                logger.warning("WebSocket连接异常: %s", exc)
            finally:
                self._ws_connection = None

            if self._ws_should_stop:
                break

            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)

    async def _handle_ws_message(self, raw_message: str):
        """处理来自ComfyUI的WebSocket消息"""
        try:
            message = json.loads(raw_message)
        except json.JSONDecodeError:
            logger.debug("Invalid WebSocket payload: %s", raw_message)
            return

        msg_type = message.get("type")
        data = message.get("data") or {}

        if msg_type == "progress":
            await self.progress_tracker.update_scalar(
                data.get("prompt_id"),
                data.get("node"),
                data.get("value"),
                data.get("max"),
            )
        elif msg_type == "progress_state":
            await self.progress_tracker.update_nodes(
                data.get("prompt_id"),
                data.get("nodes"),
            )
        elif msg_type == "executing":
            prompt_id = data.get("prompt_id")
            node_id = data.get("node")
            if node_id is None:
                await self.progress_tracker.mark_completed(prompt_id)
            else:
                await self.progress_tracker.set_current_node(prompt_id, node_id)
        elif msg_type in ("execution_error", "execution_interrupted"):
            await self.progress_tracker.mark_failed(data.get("prompt_id"))
        elif msg_type == "execution_start":
            await self.progress_tracker.set_current_node(data.get("prompt_id"), None)
    async def _process_input_image(self, input_image: str) -> str:
        """处理输入图像，支持base64或文件名"""
        if not input_image:
            raise ValueError("input_image is empty")

        if input_image.startswith("data:image"):
            header, base64_data = input_image.split(",", 1)
            image_data = base64.b64decode(base64_data)
            filename = f"upload_{uuid.uuid4().hex}.png"

            if not self.comfyui_input_dir:
                raise ValueError("ComfyUI input directory is not configured")

            file_path = self.comfyui_input_dir / filename
            image = Image.open(io.BytesIO(image_data))
            image.save(file_path, format="PNG")

            logger.info("Base64 image saved: %s to %s", filename, file_path)
            
            # Verify file exists and is readable
            if not file_path.exists():
                raise FileNotFoundError(f"Failed to save image to {file_path}")
            
            # Return just the filename, not the full path
            # ComfyUI LoadImage node expects just the filename
            return filename

        if input_image.startswith("/") or input_image.startswith("http"):
            raise ValueError("URL or absolute path is not supported, please use base64 or upload file")

        # For existing filenames, verify they exist in the input directory
        if self.comfyui_input_dir:
            file_path = self.comfyui_input_dir / input_image
            if not file_path.exists():
                logger.warning("Input image file does not exist: %s", file_path)
                raise FileNotFoundError(f"Input image file not found: {input_image} in {self.comfyui_input_dir}")

        return input_image

    def _prepare_workflow_data(
        self,
        request: GenerateRequest,
        workflow: Workflow,
        provided_params: Set[str],
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """准备工作流数据 - 使用智能分析器"""
        workflow_data = json.loads(json.dumps(workflow.raw_data))

        graph = self.workflow_analyzer.analyze_workflow(workflow_data)

        applied_params = self._update_parameters_with_graph(
            workflow_data,
            graph,
            request,
            provided_params,
        )

        if request.custom_params:
            self._apply_custom_params(workflow_data, request.custom_params)

        return workflow_data, applied_params

    def _update_parameters_with_graph(
        self,
        workflow_data: Dict[str, Any],
        graph: WorkflowGraph,
        request: GenerateRequest,
        provided_params: set,
    ) -> Dict[str, Dict[str, Any]]:
        """使用图分析结果更新参数"""
        applied: Dict[str, Dict[str, Any]] = {}


        positive_node, negative_node = graph.get_prompt_nodes()

        if positive_node:
            workflow_data[positive_node.id]["inputs"]["text"] = request.prompt
            log_parameter_update(positive_node.id, "CLIPTextEncode", "prompt", request.prompt[:30] + "...")
            applied.setdefault(positive_node.id, {})["text"] = request.prompt

        if negative_node:
            negative_prompt = request.negative_prompt or ""
            workflow_data[negative_node.id]["inputs"]["text"] = negative_prompt
            log_parameter_update(negative_node.id, "CLIPTextEncode", "negative", negative_prompt[:30] + "...")
            applied.setdefault(negative_node.id, {})["text"] = negative_prompt

        image_input_node = graph.get_image_input_node()
        if image_input_node and "input_image" in provided_params and request.input_image:
            workflow_data[image_input_node.id]["inputs"]["image"] = request.input_image
            log_parameter_update(image_input_node.id, "LoadImage", "image", request.input_image)
            applied.setdefault(image_input_node.id, {})["image"] = request.input_image

        for param_name, param_info in graph.input_parameters.items():
            if "node_id" not in param_info or "input_name" not in param_info:
                continue

            node_id = param_info["node_id"]
            input_name = param_info["input_name"]

            if param_name not in provided_params:
                continue

            request_value = getattr(request, param_name, None)
            if request_value is None:
                continue

            if param_name == "seed" and request_value < 0:
                request_value = int.from_bytes(uuid.uuid4().bytes[:4], "big")
                log_parameter_update(node_id, workflow_data[node_id]["class_type"], "seed", f"{request_value} (随机)")
            else:
                log_parameter_update(node_id, workflow_data[node_id]["class_type"], input_name, request_value)

            workflow_data[node_id]["inputs"][input_name] = request_value
            applied.setdefault(node_id, {})[input_name] = request_value

        return applied


    def _get_provided_params(self, request: GenerateRequest) -> set:
        """获取用户实际提供的参数（排除默认值）"""
        provided_params = set()
        optional_params = [
            "width",
            "height",
            "steps",
            "cfg_scale",
            "seed",
            "sampler_name",
            "batch_size",
            "input_image",
            "denoise",
        ]

        for param in optional_params:
            if hasattr(request, param):
                current_value = getattr(request, param)
                if current_value is not None:
                    provided_params.add(param)

        return provided_params

    def _apply_custom_params(self, workflow_data: Dict[str, Any], custom_params: Dict[str, Any]):
        """应用自定义参数"""
        for param_name, param_value in custom_params.items():
            for node_id, node_data in workflow_data.items():
                if isinstance(node_data, dict) and "inputs" in node_data:
                    if param_name in node_data["inputs"]:
                        node_data["inputs"][param_name] = param_value
