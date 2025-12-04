"""
简单的进度条和日志工具
"""

import logging
import time
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def log_job_prompt(prompt_id: str, prompt: str, workflow: str, params: Dict[str, Any]):
    """记录任务开始"""
    short_prompt = prompt[:30] + "..." if len(prompt) > 50 else prompt
    
    param_str = ""
    if params:
        param_parts = []
        for key, value in params.items():
            if key == 'cfg_scale':
                param_parts.append(f"cfg={value}")
            else:
                param_parts.append(f"{key}={value}")
        if param_parts:
            param_str = f"[{', '.join(param_parts)}]"

    # 使用标准日志输出，这样会显示文件名和行号
    identifier = prompt_id
    metadata = []
    if workflow:
        metadata.append(f"workflow={workflow}")
    if identifier:
        metadata.append(f"prompt_id={identifier}")
    metadata_str = f" [{', '.join(metadata)}]" if metadata else ""
    
    base_message = f"{metadata_str}, Prompt: {short_prompt}"
    if param_str:
        base_message = f"{base_message}, Params: {param_str}"
    
    logger.info(base_message)

def log_parameter_update(node_id: str, param_type: str, param_name: str, value: Any):
    """记录参数更新（调试级别）"""
    logger.debug(f"Node {node_id}: {param_name}={value}")

def log_workflow_submitted(prompt_id: str):
    """记录工作流提交"""
    logger.debug(f"Workflow submitted: {prompt_id}")

def log_image_copied(filename: str):
    """记录图像复制"""
    logger.debug(f"Image copied: {filename}")
