"""Practical video-file composition for the full learned pipeline."""
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
from .sources import VideoSource


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.learned-p9-full.json"


def _validated_video_path(video_path: str | Path) -> Path:
    path = Path(video_path).resolve()
    if not path.exists():
        raise PracticalDemoError(f"Video file does not exist: {path}")
    if not path.is_file():
        raise PracticalDemoError(f"Video input is not a file: {path}")
    return path


def run_video(
    video_path: str | Path,
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    max_frames: int | None = None,
    detector_factory: Callable[[AppConfig], DetectorProtocol] | None = None,
) -> RunSummary:
    """Load one learned detector and process a video as one stateful stream."""
    resolved_video = _validated_video_path(video_path)
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
    display = DisplaySink(
        config,
        window_name="BottleCapVision - Practical Video Demo",
        synchronize_to_source_fps=True,
    )
    summary = run(
        VideoSource(resolved_video),
        pipeline,
        [display],
        max_frames=max_frames,
    )
    if display.source_fps is None:
        print("Source FPS: N/A (playback follows processing speed)")
    else:
        print(f"Source FPS: {display.source_fps:.2f}")
    return summary
