"""
Attach minimal HTTP APIs to ComfyUI's built-in PromptServer (aiohttp).

Registered endpoints:
  - POST   /api/generate
  - GET    /api/generate/{prompt_id}
  - POST   /api/upload_files
  - GET    /api/workflows
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure package root is in sys.path before importing other modules
_package_root = Path(__file__).resolve().parent
if str(_package_root) not in sys.path:
    sys.path.insert(0, str(_package_root))

import asyncio
import logging
import shutil
from typing import Any, Dict, Optional, Tuple

from aiohttp import web
import server

from api.types.generate import (
    GenerateRequest,
    GenerateResponse,
    GenerateStatusResponse,
)
from api.types.workflow import (
    WorkflowInfo,
    WorkflowListResponse,
)
from core.workflow_loader import WorkflowLoader
from core.executor import ComfyExecutor

logger = logging.getLogger(__name__)

COMPONENTS_KEY = "comfyui_generate_api_components"
_components_lock = asyncio.Lock()


async def _ensure_components(app: web.Application) -> Tuple[WorkflowLoader, ComfyExecutor]:
    """Lazily create and initialize WorkflowLoader and ComfyExecutor."""
    if COMPONENTS_KEY in app:
        data = app[COMPONENTS_KEY]
        return data["workflow_loader"], data["comfy_executor"]

    async with _components_lock:
        if COMPONENTS_KEY in app:
            data = app[COMPONENTS_KEY]
            return data["workflow_loader"], data["comfy_executor"]

        # 计算 workflows 目录：从 server_setup.py 开始，parents[1] 是包根目录
        package_root = Path(__file__).resolve().parent
        workflows_dir = package_root / "workflows"
        workflows_dir.mkdir(parents=True, exist_ok=True)
        
        # 尝试从 PromptServer 获取 host 和 port
        try:
            import server
            server_instance = server.PromptServer.instance
            host = getattr(server_instance, "address", "127.0.0.1")
            port = getattr(server_instance, "port", 9300)
            # 0.0.0.0 无法作为客户端地址使用，回落到 localhost
            if host in {"0.0.0.0", "::"}:
                host = "127.0.0.1"
        except Exception:
            host = "127.0.0.1"
            port = 9300
        
        logger.info(
            "[comfyui-generate-api] initializing components, workflows_dir=%s, comfyui=%s:%s",
            workflows_dir,
            host,
            port,
        )

        loader = WorkflowLoader(workflows_dir)
        await loader.initialize()

        executor = ComfyExecutor(
            host=host,
            port=port,
        )
        await executor.initialize()

        app[COMPONENTS_KEY] = {
            "workflow_loader": loader,
            "comfy_executor": executor,
        }

        return loader, executor


async def generate_handler(request: web.Request) -> web.Response:
    """POST /api/generate"""
    try:
        payload = await request.json()
    except Exception:
        return web.json_response(
            {"detail": "invalid JSON body"}, status=400
        )

    try:
        generate_req = GenerateRequest(**payload)
    except Exception as exc:
        return web.json_response(
            {"detail": f"invalid request: {exc}"}, status=400
        )

    workflow_loader, comfy_executor = await _ensure_components(request.app)

    workflow = workflow_loader.get_workflow(generate_req.workflow)
    if not workflow:
        return web.json_response(
            {"detail": f"workflow: '{generate_req.workflow}' file not found"},
            status=404,
        )

    if not comfy_executor.connected:
        return web.json_response(
            {"detail": "ComfyUI Server connection error"},
            status=503,
        )

    try:
        result = await comfy_executor.submit_prompt(generate_req, workflow)
    except Exception as exc:
        logger.error("submit_prompt failed: %s", exc, exc_info=True)
        return web.json_response(
            {"detail": "failed to submit prompt"}, status=502
        )

    if not result.get("prompt_id"):
        return web.json_response(
            {"detail": "ComfyUI did not return prompt_id"},
            status=502,
        )

    resp = GenerateResponse(
        prompt_id=result["prompt_id"],
        workflow=workflow.name,
        status="submitted",
        message="workflow submitted to ComfyUI",
        client_id=result.get("client_id"),
    )
    return web.json_response(resp.model_dump())


async def _build_queue_status_response(
    comfy_executor: ComfyExecutor,
    prompt_id: str,
    queue_status: Dict[str, Any],
) -> GenerateStatusResponse:
    status_str = queue_status["status"]
    position = queue_status.get("position", 0)
    progress_value: Optional[float] = None

    if status_str == "running":
        message = "generating"
        progress_snapshot = await comfy_executor.get_progress(prompt_id)
        if progress_snapshot:
            progress_value = progress_snapshot.percent
    elif status_str == "pending":
        message = f"queued (position: {position})"
    else:
        message = "processing"

    return GenerateStatusResponse(
        prompt_id=prompt_id,
        status=status_str,
        completed=False,
        message=message,
        files=[],
        error=None,
        elapsed_time=None,
        progress=progress_value,
    )


async def generate_status_handler(request: web.Request) -> web.Response:
    """GET /api/generate/{prompt_id}"""
    prompt_id = request.match_info.get("prompt_id", "")
    if not prompt_id:
        return web.json_response({"detail": "prompt_id required"}, status=400)

    _, comfy_executor = await _ensure_components(request.app)

    if not comfy_executor.connected:
        return web.json_response(
            {"detail": "ComfyUI Server connection error"},
            status=503,
        )

    import httpx

    try:
        history = await comfy_executor.fetch_history(prompt_id)
    except httpx.HTTPStatusError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            queue_status = await comfy_executor.check_prompt_in_queue(prompt_id)
            if queue_status:
                resp = await _build_queue_status_response(
                    comfy_executor, prompt_id, queue_status
                )
                return web.json_response(resp.model_dump())
            return web.json_response(
                {"detail": "prompt_id not found"}, status=404
            )
        logger.error("fetch history failed: %s", exc)
        return web.json_response(
            {"detail": "failed to fetch prompt history"},
            status=502,
        )
    except Exception as exc:
        logger.error("fetch history error: %s", exc, exc_info=True)
        return web.json_response(
            {"detail": "failed to fetch prompt history"},
            status=502,
        )

    record = history.get(prompt_id)
    if not record:
        queue_status = await comfy_executor.check_prompt_in_queue(prompt_id)
        if queue_status:
            resp = await _build_queue_status_response(
                comfy_executor, prompt_id, queue_status
            )
            return web.json_response(resp.model_dump())
        return web.json_response(
            {"detail": "prompt_id not found"}, status=404
        )

    from api.helpers import (
        _extract_output_urls,
        _extract_status_message,
        _calculate_elapsed_time,
    )

    status_info = record.get("status") or {}
    status_str = status_info.get("status_str", "unknown")
    completed = bool(status_info.get("completed", False))

    message = _extract_status_message(status_info, status_str)
    files = _extract_output_urls(record)
    elapsed_time = _calculate_elapsed_time(status_info)

    if completed and files:
        file_count = len(files)
        message = f"generate completed, {file_count} file{'s' if file_count > 1 else ''} generated"
    elif completed and not files:
        message = "generate completed, no files generated"
    elif not completed:
        message = message or "generating"

    error: Optional[str] = None
    if status_str in ("error", "failed"):
        error = message
        message = "generation failed"

    resp = GenerateStatusResponse(
        prompt_id=prompt_id,
        status=status_str,
        completed=completed,
        message=message,
        files=files,
        error=error,
        elapsed_time=elapsed_time,
        progress=100.0 if completed else None,
    )
    return web.json_response(resp.model_dump())


async def upload_files_handler(request: web.Request) -> web.Response:
    """POST /api/upload_files"""
    _, comfy_executor = await _ensure_components(request.app)

    reader = await request.multipart()
    field = await reader.next()
    if field is None or field.name != "file":
        return web.json_response(
            {"detail": "file field required"}, status=400
        )

    filename = field.filename
    if not filename:
        return web.json_response(
            {"detail": "filename missing"}, status=400
        )

    from pathlib import Path as _Path

    allowed_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".bmp",
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".webm",
    }
    file_ext = _Path(filename).suffix.lower()

    if file_ext not in allowed_extensions:
        return web.json_response(
            {
                "detail": f"file format invalid, not in {sorted(allowed_extensions)}"
            },
            status=400,
        )

    if not comfy_executor.comfyui_input_dir:
        return web.json_response(
            {"detail": "input directory not configured"},
            status=503,
        )

    import re
    import uuid

    keep_filename = request.query.get("keep_filename", "false").lower() in {
        "1",
        "true",
        "yes",
    }

    if keep_filename:
        unique_filename = re.sub(r"[^\w\-_\.]", "_", filename)
    else:
        unique_filename = f"upload_{uuid.uuid4().hex}{file_ext}"

    file_path = comfy_executor.comfyui_input_dir / unique_filename

    with file_path.open("wb") as f:
        while True:
            chunk = await field.read_chunk()
            if not chunk:
                break
            f.write(chunk)

    logger.info("file uploaded: %s", unique_filename)
    return web.json_response(
        {
            "filename": unique_filename,
            "message": "file upload success",
            "path": str(file_path),
        }
    )


async def list_workflows_handler(request: web.Request) -> web.Response:
    """GET /api/workflows"""
    try:
        workflow_loader, _ = await _ensure_components(request.app)

        workflow_names = workflow_loader.list_workflows()
        workflows_info = []

        for name in workflow_names:
            info = workflow_loader.get_workflow_info(name)
            if info:
                workflows_info.append(WorkflowInfo(**info))

        resp = WorkflowListResponse(
            workflows=workflows_info,
            total=len(workflows_info),
        )
        return web.json_response(resp.model_dump())
    except Exception as exc:
        logger.error("list_workflows_handler failed: %s", exc, exc_info=True)
        return web.json_response(
            {"detail": f"failed to list workflows: {exc}"}, status=500
        )


def setup_server() -> None:
    """Register routes on ComfyUI's PromptServer."""
    try:
        server_instance = server.PromptServer.instance
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("[comfyui-generate-api] failed to get PromptServer instance: %s", exc)
        return

    app: web.Application = server_instance.app

    # Register API routes
    app.router.add_post("/api/generate", generate_handler)
    app.router.add_get("/api/generate/{prompt_id}", generate_status_handler)
    app.router.add_post("/api/upload_files", upload_files_handler)
    app.router.add_get("/api/workflows", list_workflows_handler)

    logger.info(
        "[comfyui-generate-api] routes registered: /api/generate, /api/workflows, /api/upload_files"
    )


