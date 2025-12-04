"""
用于跟踪ComfyUI任务进度的工具
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_percent(value: Optional[float], max_value: Optional[float]) -> float:
    if value is None or max_value in (None, 0):
        return 0.0
    if max_value == 0:
        return 0.0
    ratio = _clamp(value / max_value, 0.0, 1.0)
    return round(ratio * 100.0, 2)


@dataclass
class ProgressSnapshot:
    """单个prompt的进度快照"""

    prompt_id: str
    percent: float
    value: float
    max_value: float
    state: str = "running"
    node_id: Optional[str] = None
    display_node_id: Optional[str] = None
    updated_at: float = field(default_factory=lambda: time.time())
    source: str = "scalar"


class PromptProgressTracker:
    """维护prompt进度信息"""

    def __init__(self, stale_after: float = 600.0):
        self._data: Dict[str, ProgressSnapshot] = {}
        self._lock = asyncio.Lock()
        self._stale_after = stale_after

    async def update_scalar(self, prompt_id: Optional[str], node_id: Optional[str], value: Any, max_value: Any):
        """处理progress事件（单节点）"""
        if not prompt_id:
            return

        value_f = _safe_float(value)
        max_f = _safe_float(max_value)
        percent = _to_percent(value_f, max_f) if value_f is not None and max_f is not None else 0.0

        snapshot = ProgressSnapshot(
            prompt_id=prompt_id,
            percent=percent,
            value=value_f or 0.0,
            max_value=max_f or 1.0,
            node_id=node_id,
            display_node_id=node_id,
            source="scalar",
        )
        await self._store_snapshot(snapshot)

    async def update_nodes(self, prompt_id: Optional[str], nodes: Any):
        """处理progress_state事件（多节点状态）"""
        if not prompt_id or not isinstance(nodes, dict):
            return

        ratios = []
        running_node = None
        for node_data in nodes.values():
            if not isinstance(node_data, dict):
                continue
            node_value = _safe_float(node_data.get("value"))
            node_max = _safe_float(node_data.get("max"))
            if node_value is not None and node_max not in (None, 0):
                ratios.append(_clamp(node_value / node_max, 0.0, 1.0))
            node_state = node_data.get("state")
            if node_state == "running":
                running_node = node_data

        if not ratios and running_node is None:
            return

        percent = round(sum(ratios) / len(ratios) * 100.0, 2) if ratios else 0.0
        node_id = None
        display_node_id = None
        if running_node:
            node_id = running_node.get("node_id") or running_node.get("real_node_id")
            display_node_id = (
                running_node.get("display_node_id")
                or running_node.get("real_node_id")
                or running_node.get("node_id")
            )
            value = _safe_float(running_node.get("value")) or 0.0
            max_value = _safe_float(running_node.get("max")) or 1.0
        else:
            value = percent
            max_value = 100.0

        snapshot = ProgressSnapshot(
            prompt_id=prompt_id,
            percent=percent,
            value=value,
            max_value=max_value,
            node_id=node_id,
            display_node_id=display_node_id or node_id,
            source="nodes",
        )
        await self._store_snapshot(snapshot)

    async def mark_completed(self, prompt_id: Optional[str]):
        """任务完成时清理进度"""
        if not prompt_id:
            return
        async with self._lock:
            self._data.pop(prompt_id, None)

    async def mark_failed(self, prompt_id: Optional[str]):
        """任务失败时清理进度"""
        await self.mark_completed(prompt_id)

    async def set_current_node(self, prompt_id: Optional[str], node_id: Optional[str]):
        """记录当前执行节点（无进度值时使用）"""
        if not prompt_id:
            return
        async with self._lock:
            snapshot = self._data.get(prompt_id)
            now = time.time()
            if snapshot:
                snapshot.node_id = node_id
                snapshot.display_node_id = node_id or snapshot.display_node_id
                snapshot.updated_at = now
            else:
                self._data[prompt_id] = ProgressSnapshot(
                    prompt_id=prompt_id,
                    percent=0.0,
                    value=0.0,
                    max_value=1.0,
                    node_id=node_id,
                    display_node_id=node_id,
                    source="executing",
                    updated_at=now,
                )

    async def get_progress(self, prompt_id: str) -> Optional[ProgressSnapshot]:
        """获取prompt进度"""
        async with self._lock:
            self._purge_stale_locked()
            snapshot = self._data.get(prompt_id)
            if not snapshot:
                return None
            return replace(snapshot)

    async def _store_snapshot(self, snapshot: ProgressSnapshot):
        async with self._lock:
            self._purge_stale_locked()
            snapshot.updated_at = time.time()
            self._data[snapshot.prompt_id] = snapshot

    def _purge_stale_locked(self):
        """清理过期数据（需在锁内调用）"""
        now = time.time()
        stale_keys = [
            prompt_id
            for prompt_id, snapshot in self._data.items()
            if (now - snapshot.updated_at) > self._stale_after
        ]
        for prompt_id in stale_keys:
            self._data.pop(prompt_id, None)

