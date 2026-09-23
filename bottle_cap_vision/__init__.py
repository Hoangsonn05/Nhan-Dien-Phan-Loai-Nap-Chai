"""Bottle Cap Vision - software-only Python/OpenCV foundation."""

from .color_stabilizer import ColorStabilizer
from .config import AppConfig, DEFAULT_COLORS, DetectorConfig, load_config, save_config
from .detector import DetectorProtocol, HSVDetector, create_detector
from .models import CountEvent, Detection, FramePacket, FrameResult, RunSummary, Track

__all__ = [
    "AppConfig",
    "DEFAULT_COLORS",
    "DetectorConfig",
    "load_config",
    "save_config",
    "ColorStabilizer",
    "DetectorProtocol",
    "HSVDetector",
    "create_detector",
    "Detection",
    "Track",
    "CountEvent",
    "FramePacket",
    "FrameResult",
    "RunSummary",
]
__version__ = "1.4.0"
