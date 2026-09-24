"""Practical realtime camera composition for the learned pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from .config import AppConfig
from .detector import DetectorProtocol, create_detector
from .models import RunSummary
from .pipeline import BottleCapPipeline
from .practical_demo import PracticalDemoError, load_learned_config
from .runtime import run
from .sinks import DisplaySink
from .sources import CameraSource


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.learned-p9-full.json"


def run_camera(
    *,
    camera_index: int = 0,
    config_path: str | Path = DEFAULT_CONFIG,
    max_frames: int | None = None,
    detector_factory: Callable[[AppConfig], DetectorProtocol] | None = None,
) -> RunSummary:
    """Load one learned detector and run stateful sequential camera inference."""
    config, _resolved_config, checkpoint, _metadata = load_learned_config(config_path)
    factory = detector_factory or create_detector
    print(f"Loading full learned model: {checkpoint}")
    try:
        detector = factory(config)
    except Exception as exc:
        raise PracticalDemoError(
            f"Full learned model failed compatibility/load validation ({checkpoint}): {exc}"
        ) from exc

    pipeline = BottleCapPipeline(config, detector=detector)
    return run(
        CameraSource(camera_index),
        pipeline,
        [DisplaySink(config, window_name="BottleCapVision - Practical Camera Demo")],
        max_frames=max_frames,
    )
