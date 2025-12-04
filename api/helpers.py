"""
图像生成相关的通用工具函数（不依赖 FastAPI）。

保留给 aiohttp 路由使用的逻辑：
  - _extract_output_urls
  - _extract_status_message
  - _calculate_elapsed_time
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _extract_output_urls(record: Dict[str, Any]) -> List[str]:
    """从record中提取输出文件的URL列表，使用ComfyUI的/view路由"""
    from urllib.parse import quote
    
    files: List[str] = []
    raw_outputs = record.get("outputs")
    
    # 处理ComfyUI原始格式（字典格式）
    if isinstance(raw_outputs, dict):
        for node_outputs in raw_outputs.values():
            if not isinstance(node_outputs, dict):
                continue
            for value in node_outputs.values():
                if not isinstance(value, list):
                    continue
                for item in value:
                    if not isinstance(item, dict):
                        continue
                    filename = item.get("filename")
                    if not filename:
                        continue
                    subfolder = (item.get("subfolder") or "").strip("/")
                    
                    # 使用ComfyUI的/view路由，格式: /view?filename=xxx&type=output[&subfolder=xxx]
                    params = [f"filename={quote(filename)}", "type=output"]
                    if subfolder:
                        params.append(f"subfolder={quote(subfolder)}")
                    files.append(f"/view?{'&'.join(params)}")
    
    # 处理已处理的数组格式
    elif isinstance(raw_outputs, list):
        for output in raw_outputs:
            if isinstance(output, dict):
                url = output.get("url")
                if url:
                    files.append(url)
                elif output.get("filename"):
                    # 如果有filename但没有url，构建/view URL
                    filename = output["filename"]
                    subfolder = output.get("subfolder", "").strip("/")
                    params = [f"filename={quote(filename)}", "type=output"]
                    if subfolder:
                        params.append(f"subfolder={quote(subfolder)}")
                    files.append(f"/view?{'&'.join(params)}")
                elif output.get("relative_path"):
                    # 兼容旧格式：relative_path可能是 "subfolder/filename" 或 "filename"
                    relative_path = output["relative_path"]
                    path_parts = relative_path.split("/")
                    if len(path_parts) > 1:
                        subfolder = "/".join(path_parts[:-1])
                        filename = path_parts[-1]
                    else:
                        subfolder = ""
                        filename = path_parts[0]
                    params = [f"filename={quote(filename)}", "type=output"]
                    if subfolder:
                        params.append(f"subfolder={quote(subfolder)}")
                    files.append(f"/view?{'&'.join(params)}")

    return files


def _extract_status_message(status_info: Dict[str, Any], status_str: str) -> str:
    """从状态信息中提取消息"""
    messages = status_info.get("messages")
    if not isinstance(messages, list) or not messages:
        return status_str
    
    last_message = messages[-1]
    if not isinstance(last_message, (list, tuple)) or len(last_message) < 2:
        return status_str
    
    message_type = last_message[0]
    message_data = last_message[1] if isinstance(last_message[1], dict) else {}
    detail_msg = message_data.get("detail") or message_data.get("prompt_id")
    return f"{message_type}: {detail_msg}" if detail_msg else str(message_type)


def _calculate_elapsed_time(status_info: Dict[str, Any]) -> Optional[float]:
    """从状态信息中计算执行耗时（秒）"""
    messages = status_info.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return None

    start_time: Optional[int] = None
    end_time: Optional[int] = None

    for message in messages:
        if not isinstance(message, (list, tuple)) or len(message) < 2:
            continue

        message_type = message[0]
        message_data = message[1] if isinstance(message[1], dict) else {}
        timestamp = message_data.get("timestamp")

        if not isinstance(timestamp, (int, float)):
            continue

        timestamp = int(timestamp)
        if message_type == "execution_start" and start_time is None:
            start_time = timestamp
        elif message_type in ("execution_success", "execution_error") and end_time is None:
            end_time = timestamp

    if start_time and end_time:
        # 时间戳是毫秒，转换为秒
        return round((end_time - start_time) / 1000.0, 3)

    return None
