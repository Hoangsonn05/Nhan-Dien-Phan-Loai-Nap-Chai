"""Bottle Cap Vision - software-only Python/OpenCV foundation."""

from .config import AppConfig, DEFAULT_COLORS, load_config, save_config
from .models import CountEvent, Detection, FramePacket, FrameResult, RunSummary, Track

__all__ = [
    "AppConfig",
    "DEFAULT_COLORS",
    "load_config",
    "save_config",
    "Detection",
    "Track",
    "CountEvent",
    "FramePacket",
    "FrameResult",
    "RunSummary",
]
__version__ = "1.3.0"
