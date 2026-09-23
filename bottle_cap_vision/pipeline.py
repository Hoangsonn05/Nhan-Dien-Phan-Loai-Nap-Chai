from __future__ import annotations

from .color_stabilizer import ColorStabilizer
from .config import AppConfig
from .counter import LineCounter
from .detector import ColorCapDetector
from .models import FrameResult
from .tracker import NearestNeighborTracker


class BottleCapPipeline:
    """Stateful in-memory vision core for detection, tracking and counting."""

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
        sc = config.color_stabilization
        self.color_stabilizer = ColorStabilizer(
            enabled=sc.enabled,
            window_size=sc.window_size,
            min_observations=sc.min_observations,
        )
        cc = config.counting
        self.counter = LineCounter(
            orientation=cc.orientation,
            line_position=cc.line_position,
            direction=cc.direction,
            deadband=cc.deadband,
        ) if cc.enabled else None

    def process_frame(self, frame, frame_index: int = 0) -> FrameResult:
        detections = self.detector.detect(frame)
        tracks = self.tracker.update(detections) if self.tracker else []
        tracks = self.color_stabilizer.update(tracks)
        events = self.counter.update(tracks, frame_index, frame.shape) if self.counter else []
        return FrameResult(
            frame_index=frame_index,
            detections=detections,
            tracks=tracks,
            events=events,
            total_count=self.counter.total if self.counter else 0,
            count_by_color=dict(self.counter.by_color) if self.counter else {},
        )
