"""
API数据类型定义模块
"""

from .generate import (
    ImageFormat,
    GenerateRequest,
    GenerateResponse,
    GenerateStatusResponse,
)

from .workflow import (
    NodeRole,
    WorkflowNode,
    Workflow,
    WorkflowInfo,
    WorkflowListResponse,
)

__all__ = [
    # Generate types
    "ImageFormat", 
    "GenerateRequest",
    "GenerateResponse",
    "GenerateStatusResponse",
    
    # Workflow types
    "NodeRole",
    "WorkflowNode",
    "Workflow",
    "WorkflowInfo",
    "WorkflowListResponse",
]
