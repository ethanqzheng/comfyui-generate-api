"""
comfyui-generate-api custom node entry for the embedded API server.

This variant does NOT start a separate FastAPI/uvicorn server.
Instead, it attaches a few HTTP routes directly to ComfyUI's PromptServer
using aiohttp, similar to ComfyUI-disty-Flow.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ensure local package is importable
PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from server_setup import setup_server

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# register HTTP routes on ComfyUI's built-in PromptServer
setup_server()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

