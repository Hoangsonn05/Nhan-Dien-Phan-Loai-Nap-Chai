"""Bottle Cap Vision - software-only Python/OpenCV foundation."""

from .config import AppConfig, DEFAULT_COLORS, load_config, save_config
from .models import CountEvent, Detection, Track

__all__ = [
    "AppConfig",
    "DEFAULT_COLORS",
    "load_config",
    "save_config",
    "Detection",
    "Track",
    "CountEvent",
]
__version__ = "1.2.0"
