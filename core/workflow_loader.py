"""
Workflow JSON文件动态加载器
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import aiofiles
import asyncio
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading

from api.types.workflow import Workflow, WorkflowNode
from core.workflow_analyzer import WorkflowAnalyzer

logger = logging.getLogger(__name__)


class WorkflowFileHandler(FileSystemEventHandler):
    """Workflow文件变化监控处理器"""
    
    def __init__(self, loader: 'WorkflowLoader'):
        self.loader = loader
    
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('.json'):
            self._schedule_async_task(Path(event.src_path))
    
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.json'):
            self._schedule_async_task(Path(event.src_path))
    
    def on_deleted(self, event):
        if not event.is_directory and event.src_path.endswith('.json'):
            workflow_name = Path(event.src_path).stem
            self.loader.workflows.pop(workflow_name, None)
            logger.info(f"unloaded workflow: {workflow_name}")
    
    def _schedule_async_task(self, file_path: Path):
        """在主事件循环中调度异步任务"""
        try:
            # 获取主事件循环
            loop = self.loader._loop
            if loop and not loop.is_closed():
                # 使用call_soon_threadsafe在主循环中调度任务
                asyncio.run_coroutine_threadsafe(
                    self.loader._load_single_workflow(file_path), 
                    loop
                )
        except Exception as e:
            logger.error(f" failed to schedule async task: {e}")


class WorkflowLoader:
    """Workflow动态加载器"""
    
    def __init__(self, workflows_dir: Path):
        self.workflows_dir = Path(workflows_dir)
        self.workflows: Dict[str, Workflow] = {}
        self.observer: Optional[Observer] = None
        self._lock = asyncio.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.analyzer = WorkflowAnalyzer()
    
    async def initialize(self):
        """初始化加载器"""
        # 保存当前事件循环引用
        self._loop = asyncio.get_running_loop()
        
        # 确保工作流目录存在
        self.workflows_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载所有现有的workflow
        await self.load_all_workflows()
        
        # 启动文件监控
        self.start_file_watcher()
        
        logger.info(f"workflow loader initialized, loaded {len(self.workflows)} workflows")
    
    async def load_all_workflows(self):
        """加载所有workflow文件"""
        if not self.workflows_dir.exists():
            logger.warning(f"workflow directory not exist: {self.workflows_dir}")
            return
        
        json_files = list(self.workflows_dir.glob("*.json"))
        if not json_files:
            logger.warning(f"no found workflow files in {self.workflows_dir}")
            return
        
        tasks = [self._load_single_workflow(file_path) for file_path in json_files]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _load_single_workflow(self, file_path: Path) -> bool:
        """加载单个workflow文件"""
        async with self._lock:
            try:
                async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    workflow_data = json.loads(content)
                
                # 解析workflow
                workflow = self._parse_workflow(workflow_data, file_path.stem)
                self.workflows[workflow.name] = workflow
                
                logger.info(f"loaded workflow: {workflow.name} ({len(workflow.nodes)} nodes)")
                return True
                
            except Exception as e:
                logger.error(f"failed to load workflow {file_path}: {e}")
                return False
    
    def _parse_workflow(self, workflow_data: Dict[str, Any], name: str) -> Workflow:
        """解析workflow数据 - 使用智能分析器"""
        
        # 使用智能分析器分析工作流
        graph = self.analyzer.analyze_workflow(workflow_data)
        
        # 直接使用统一的模型，包含智能分析结果
        nodes = []
        for i, (node_id, analyzer_node) in enumerate(graph.nodes.items()):
            workflow_node = WorkflowNode(
                id=analyzer_node.id,
                class_type=analyzer_node.class_type,
                inputs=analyzer_node.inputs,
                meta=analyzer_node.meta or {},
                # 智能分析结果
                role=analyzer_node.role.value if analyzer_node.role else None,
                title=analyzer_node.title,
                execution_order=graph.execution_order.index(node_id) if node_id in graph.execution_order else None
            )
            nodes.append(workflow_node)
        
        return Workflow(
            name=name,
            nodes=nodes,
            workflow_type=graph.workflow_type,
            input_params=graph.input_parameters,
            raw_data=workflow_data
        )
    
    def start_file_watcher(self):
        """启动文件监控"""
        if self.observer:
            return
        
        self.observer = Observer()
        handler = WorkflowFileHandler(self)
        self.observer.schedule(handler, str(self.workflows_dir), recursive=False)
        self.observer.start()
        logger.info(f"workflow file watcher started: {self.workflows_dir}")
    
    def stop_file_watcher(self):
        """停止文件监控"""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            logger.info("workflow file watcher stopped")
        
    def get_workflow(self, name: str) -> Optional[Workflow]:
        """获取指定的workflow"""
        return self.workflows.get(name)
    
    def list_workflows(self) -> List[str]:
        """获取所有workflow名称"""
        return list(self.workflows.keys())
    
    def get_workflow_info(self, name: str) -> Optional[Dict[str, Any]]:
        """获取workflow详细信息"""
        workflow = self.workflows.get(name)
        if not workflow:
            return None
        
        return {
            'name': workflow.name,
            'type': workflow.workflow_type,
            'nodes_count': len(workflow.nodes),
            'input_params': workflow.input_params,
            'description': f"A {workflow.workflow_type} workflow with {len(workflow.nodes)} nodes"
        }
    
    async def reload_workflow(self, name: str) -> bool:
        """重新加载指定的workflow"""
        file_path = self.workflows_dir / f"{name}.json"
        if file_path.exists():
            return await self._load_single_workflow(file_path)
        return False
    
    def __del__(self):
        """析构函数"""
        if self.observer:
            self.stop_file_watcher()
