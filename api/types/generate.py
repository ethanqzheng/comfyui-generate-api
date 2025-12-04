"""
图像生成相关数据模型
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, validator
from enum import Enum


class ImageFormat(str, Enum):
    """图像格式枚举"""
    PNG = "png"
    JPEG = "jpeg"
    WEBP = "webp"


class GenerateRequest(BaseModel):
    """图像生成请求模型"""
    prompt: str = Field(description="positive prompt", min_length=1, max_length=2000)
    negative_prompt: Optional[str] = Field(
        default="",
        description="negative prompt",
        max_length=2000,
    )
    workflow: str = Field(description="workflow file name")

    input_image: Optional[str] = Field(
        default=None,
        description="input image filename (need to upload to ComfyUI input directory) or base64 encoded image data",
    )
    denoise: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="denoise, 0.0=fully preserve original image, 1.0=fully redraw",
    )

    width: Optional[int] = Field(default=None, ge=64, le=2048, description="image width")
    height: Optional[int] = Field(default=None, ge=64, le=2048, description="image height")

    steps: Optional[int] = Field(default=None, ge=1, le=100, description="steps")
    cfg_scale: Optional[float] = Field(default=None, ge=1.0, le=20.0, description="CFG Scale")
    seed: Optional[int] = Field(default=-1, description="random seed, -1 for random")
    sampler_name: Optional[str] = Field(default=None, description="sampler name")

    batch_size: Optional[int] = Field(default=None, ge=1, le=4, description="batch size")
    output_format: Optional[ImageFormat] = Field(default=ImageFormat.PNG, description="output format")

    custom_params: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="custom parameters, will override the corresponding parameters in workflow",
    )

    @validator("width", "height")
    def validate_dimensions(cls, v):
        """验证图像尺寸必须是8的倍数"""
        if v % 8 != 0:
            raise ValueError("width and height must be multiples of 8")
        return v


class GenerateResponse(BaseModel):
    """图像生成响应模型"""
    prompt_id: str = Field(description="ComfyUI prompt id")
    workflow: str = Field(description="workflow file name")
    status: str = Field(default="submitted", description="generation status")
    message: str = Field(description="status message")
    client_id: Optional[str] = Field(default=None, description="ComfyUI client id")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="metadata for frontend follow-ups")


class GenerateStatusResponse(BaseModel):
    """生成状态查询响应"""
    prompt_id: str = Field(description="ComfyUI prompt id")
    status: str = Field(default="unknown", description="当前状态字符串")
    completed: bool = Field(default=False, description="任务是否完成")
    message: str = Field(default="", description="状态消息")
    files: List[str] = Field(default_factory=list, description="生成的输出文件URL列表")
    error: Optional[str] = Field(default=None, description="错误信息（如果有）")
    elapsed_time: Optional[float] = Field(default=None, description="执行耗时（秒）")
    progress: Optional[float] = Field(default=None, description="生成进度（0-100）")
