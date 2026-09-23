"""Bottle Cap Vision - software-only Python/OpenCV foundation."""

from .color_stabilizer import ColorStabilizer
from .config import AppConfig, DEFAULT_COLORS, load_config, save_config
from .models import CountEvent, Detection, FramePacket, FrameResult, RunSummary, Track

__all__ = [
    "AppConfig",
    "DEFAULT_COLORS",
    "load_config",
    "save_config",
    "ColorStabilizer",
    "Detection",
    "Track",
    "CountEvent",
    "FramePacket",
    "FrameResult",
    "RunSummary",
]
__version__ = "1.3.0"
