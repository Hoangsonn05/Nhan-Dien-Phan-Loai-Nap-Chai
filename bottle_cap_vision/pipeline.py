from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Optional

import cv2

from .config import AppConfig
from .counter import LineCounter
from .detector import ColorCapDetector
from .tracker import NearestNeighborTracker
from .visualization import draw_line, draw_panel, draw_tracks


class BottleCapPipeline:
    """Coordinates detection, tracking, counting and output generation.

    P0 keeps detector/tracker/counter behavior unchanged. This class is the
    single runtime entry point for both image and video processing.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.detector = ColorCapDetector(config)
        tc = config.tracking
        self.tracker = NearestNeighborTracker(
            max_distance=tc.max_distance,
            max_missing_frames=tc.max_missing_frames,
            min_confirmed_frames=tc.min_confirmed_frames,
            max_history=tc.max_history,
        ) if tc.enabled else None
        cc = config.counting
        self.counter = LineCounter(
            orientation=cc.orientation,
            line_position=cc.line_position,
            direction=cc.direction,
            deadband=cc.deadband,
        ) if cc.enabled else None
        self.frame_records: list[dict] = []

    def process_frame(self, frame, frame_index: int = 0):
        detections = self.detector.detect(frame)
        tracks = self.tracker.update(detections) if self.tracker else []
        events = self.counter.update(tracks, frame_index, frame.shape) if self.counter else []
        return detections, tracks, events

    def annotate_frame(self, frame, tracks, fps: float = 0.0) -> None:
        """Draw runtime overlays in one canonical place."""
        draw_line(frame, self.config)
        if tracks:
            draw_tracks(frame, tracks)
        if self.counter:
            draw_panel(frame, self.counter, fps)

    def run_image(self, source: str | Path, output: str | Path) -> dict:
        """Process one image and save its annotated result."""
        source = Path(source)
        output = Path(output)
        frame = cv2.imread(str(source))
        if frame is None:
            raise RuntimeError(f"Cannot read image: {source}")

        detections, tracks, events = self.process_frame(frame, 0)
        self.annotate_frame(frame, tracks, 0.0)
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), frame):
            raise RuntimeError(f"Cannot write output image: {output}")

        return {
            "source": str(source),
            "output": str(output),
            "detections": detections,
            "tracks": tracks,
            "events": events,
        }

    def run_video(self, source: str | Path, output: Optional[str | Path] = None,
                  csv_output: Optional[str | Path] = None, display: bool = True,
                  max_frames: int | None = None) -> dict:
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {source}")
        writer = None
        fps_samples = []
        frame_index = 0
        try:
            fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if output:
                output = Path(output)
                output.parent.mkdir(parents=True, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output), fourcc, fps_in, (width, height))
            while True:
                if max_frames is not None and frame_index >= max_frames:
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                start = time.perf_counter()
                detections, tracks, events = self.process_frame(frame, frame_index)
                elapsed = time.perf_counter() - start
                if elapsed > 0:
                    fps_samples.append(1.0 / elapsed)

                fps_now = (sum(fps_samples[-30:]) / min(len(fps_samples), 30)) if fps_samples else 0.0
                self.annotate_frame(frame, tracks, fps_now)

                self.frame_records.append({
                    "frame": frame_index,
                    "detections": len(detections),
                    "tracks": len(tracks),
                    "new_events": [{"track_id": e.track_id, "color": e.color} for e in events],
                    "total": self.counter.total if self.counter else None,
                })

                if writer:
                    writer.write(frame)
                if display:
                    cv2.imshow("Bottle Cap Vision - Python/OpenCV", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord('q')):
                        break
                frame_index += 1
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()

        avg_fps = sum(fps_samples) / len(fps_samples) if fps_samples else 0.0
        result = {
            "source": str(source),
            "frames_processed": frame_index,
            "average_processing_fps": avg_fps,
            "total_count": self.counter.total if self.counter else 0,
            "count_by_color": dict(self.counter.by_color) if self.counter else {},
        }
        if csv_output:
            self.export_csv(csv_output)
        return result

    def export_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["frame", "detections", "tracks", "new_events", "total"])
            writer.writeheader()
            for row in self.frame_records:
                writer.writerow({
                    **row,
                    "new_events": json.dumps(row["new_events"], ensure_ascii=False),
                })

    def export_result_json(self, path: str | Path, source: str = "") -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source": source,
            "total_count": self.counter.total if self.counter else 0,
            "count_by_color": dict(self.counter.by_color) if self.counter else {},
            "count_events": [e.__dict__ for e in self.counter.events] if self.counter else [],
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
