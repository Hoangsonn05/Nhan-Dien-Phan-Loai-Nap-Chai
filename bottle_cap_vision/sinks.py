from __future__ import annotations

import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
from time import perf_counter
from typing import TextIO

import cv2

from .config import AppConfig
from .models import FramePacket, FrameResult, RunSummary, RuntimeMetrics, SourceMetadata
from .visualization import annotate_frame


class ImageSink:
    """Write an annotated frame to one image path."""

    def __init__(self, path: str | Path, config: AppConfig):
        self.path = Path(path)
        self.config = config
        self.last_result: FrameResult | None = None

    def open(self, metadata: SourceMetadata) -> None:
        pass

    def write(self, packet: FramePacket, result: FrameResult, metrics: RuntimeMetrics) -> bool:
        annotated = annotate_frame(packet.frame, result, self.config, metrics.processing_fps)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(self.path), annotated):
            raise RuntimeError(f"Cannot write output image: {self.path}")
        self.last_result = result
        return True

    def close(self, summary: RunSummary | None = None) -> None:
        pass


class VideoSink:
    """Write annotated frames to an OpenCV video writer."""

    def __init__(self, path: str | Path, config: AppConfig):
        self.path = Path(path)
        self.config = config
        self._metadata = SourceMetadata()
        self._writer = None

    def open(self, metadata: SourceMetadata) -> None:
        self._metadata = metadata
        fps = metadata.fps
        width = metadata.width
        height = metadata.height
        if fps is None or not math.isfinite(fps) or fps <= 0:
            raise RuntimeError("Cannot write output video without valid source FPS metadata")
        if width is None or height is None or width <= 0 or height <= 0:
            raise RuntimeError("Cannot write output video without valid source dimensions")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(self.path), fourcc, fps, (width, height))
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f"Cannot write output video: {self.path}")
        self._writer = writer

    def write(self, packet: FramePacket, result: FrameResult, metrics: RuntimeMetrics) -> bool:
        annotated = annotate_frame(packet.frame, result, self.config, metrics.processing_fps)
        if self._writer is None:
            raise RuntimeError("VideoSink must be opened before writing")
        height, width = annotated.shape[:2]
        if (width, height) != (self._metadata.width, self._metadata.height):
            raise RuntimeError(
                "Output frame dimensions differ from source metadata: "
                f"frame={width}x{height}, metadata={self._metadata.width}x{self._metadata.height}"
            )
        self._writer.write(annotated)
        return True

    def close(self, summary: RunSummary | None = None) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


class DisplaySink:
    """Display annotated frames and translate q/ESC into a stop request."""

    def __init__(
        self,
        config: AppConfig,
        window_name: str = "Bottle Cap Vision - Python/OpenCV",
        *,
        synchronize_to_source_fps: bool = False,
    ):
        self.config = config
        self.window_name = window_name
        self.synchronize_to_source_fps = synchronize_to_source_fps
        self.source_fps: float | None = None
        self._window_created = False
        self._frame_period: float | None = None
        self._cycle_started_at: float | None = None

    def open(self, metadata: SourceMetadata) -> None:
        fps = metadata.fps
        self.source_fps = fps if fps is not None and math.isfinite(fps) and fps > 0 else None
        if self.synchronize_to_source_fps and self.source_fps is not None:
            self._frame_period = 1.0 / self.source_fps
            self._cycle_started_at = perf_counter()
        else:
            self._frame_period = None
            self._cycle_started_at = None

    def write(self, packet: FramePacket, result: FrameResult, metrics: RuntimeMetrics) -> bool:
        annotated = annotate_frame(packet.frame, result, self.config, metrics.processing_fps)
        cv2.imshow(self.window_name, annotated)
        self._window_created = True
        delay_ms = 1
        if self._frame_period is not None and self._cycle_started_at is not None:
            remaining = self._frame_period - (perf_counter() - self._cycle_started_at)
            if remaining > 0:
                # OpenCV keeps its event loop responsive while waiting.  The
                # epsilon prevents exact millisecond budgets such as 30 ms
                # from becoming 31 ms due to floating-point representation.
                delay_ms = max(1, math.ceil(remaining * 1000.0 - 1e-9))
        key = cv2.waitKey(delay_ms) & 0xFF
        if self._frame_period is not None:
            self._cycle_started_at = perf_counter()
        return key not in (27, ord("q"), ord("Q"))

    def close(self, summary: RunSummary | None = None) -> None:
        if self._window_created:
            cv2.destroyWindow(self.window_name)
            self._window_created = False
        self._frame_period = None
        self._cycle_started_at = None


class CsvSink:
    """Stream per-frame result records to CSV without core-owned state."""

    FIELDNAMES = ["frame", "detections", "tracks", "new_events", "total"]

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._handle: TextIO | None = None
        self._writer = None

    def open(self, metadata: SourceMetadata) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", newline="", encoding="utf-8-sig")
        self._writer = csv.DictWriter(self._handle, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()

    def write(self, packet: FramePacket, result: FrameResult, metrics: RuntimeMetrics) -> bool:
        if self._writer is None:
            raise RuntimeError("CsvSink must be opened before writing")
        self._writer.writerow({
            "frame": result.frame_index,
            "detections": len(result.detections),
            "tracks": len(result.tracks),
            "new_events": json.dumps(
                [{"track_id": event.track_id, "color": event.color} for event in result.events],
                ensure_ascii=False,
            ),
            "total": result.total_count,
        })
        return True

    def close(self, summary: RunSummary | None = None) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
            self._writer = None


class JsonSink:
    """Write the final count summary and events as JSON."""

    def __init__(self, path: str | Path, *, source: str = ""):
        self.path = Path(path)
        self.source = source

    def open(self, metadata: SourceMetadata) -> None:
        pass

    def write(self, packet: FramePacket, result: FrameResult, metrics: RuntimeMetrics) -> bool:
        return True

    def close(self, summary: RunSummary | None = None) -> None:
        if summary is None:
            return
        payload = {
            "source": self.source,
            "frames_processed": summary.frames_processed,
            "total_count": summary.total_count,
            "count_by_color": summary.count_by_color,
            "count_events": [asdict(event) for event in summary.count_events],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
