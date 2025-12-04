"""
Workflow数据模型
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from enum import Enum


class NodeRole(str, Enum):
    """节点角色枚举"""
    MODEL_LOADER = "model_loader"
    POSITIVE_PROMPT = "positive_prompt"
    NEGATIVE_PROMPT = "negative_prompt"
    LATENT_SOURCE = "latent_source"
    IMAGE_INPUT = "image_input"
    SAMPLER = "sampler"
    DECODER = "decoder"
    OUTPUT = "output"
    PREPROCESSOR = "preprocessor"
    POSTPROCESSOR = "postprocessor"
    CONTROL = "control"
    UNKNOWN = "unknown"


class WorkflowNode(BaseModel):
    """Workflow节点模型"""
    id: str = Field(description="节点ID")
    class_type: str = Field(description="节点类型")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="节点输入")
    meta: Dict[str, Any] = Field(default_factory=dict, description="节点元数据")
    
    # 智能分析器扩展字段
    role: Optional[NodeRole] = Field(default=None, description="节点角色")
    title: Optional[str] = Field(default=None, description="节点标题")
    execution_order: Optional[int] = Field(default=None, description="执行顺序")


class Workflow(BaseModel):
    """Workflow模型"""
    name: str = Field(description="工作流名称")
    nodes: List[WorkflowNode] = Field(description="节点列表")
    workflow_type: str = Field(description="工作流类型", default="custom")
    input_params: Dict[str, Any] = Field(default_factory=dict, description="可配置参数")
    raw_data: Dict[str, Any] = Field(description="原始JSON数据")
    description: Optional[str] = Field(default=None, description="工作流描述")
    
    class Config:
        json_encoders = {
            # 自定义JSON编码器
        }


class WorkflowInfo(BaseModel):
    """Workflow信息模型"""
    name: str = Field(description="工作流名称")
    type: str = Field(description="工作流类型")
    nodes_count: int = Field(description="节点数量")
    input_params: Dict[str, Any] = Field(description="输入参数定义")
    description: str = Field(description="工作流描述")


class WorkflowListResponse(BaseModel):
    """Workflow列表响应模型"""
    workflows: List[WorkflowInfo] = Field(description="工作流列表")
    total: int = Field(description="总数量")


class WorkflowUploadRequest(BaseModel):
    """Workflow上传请求模型"""
    name: str = Field(description="工作流名称", min_length=1, max_length=100)
    workflow_data: Dict[str, Any] = Field(description="工作流JSON数据")
    description: Optional[str] = Field(default=None, description="工作流描述")
    overwrite: bool = Field(default=False, description="是否覆盖已存在的工作流")
