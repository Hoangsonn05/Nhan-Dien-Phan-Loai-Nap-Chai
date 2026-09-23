"""Shared runtime contracts for the BottleCapVision core.

This module owns data exchanged between detector, tracker and counter so later
algorithm changes do not force cross-module type rewrites.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Detection:
    """One detector output in full-frame coordinates."""

    color: str
    centroid: tuple[int, int]
    bbox: tuple[int, int, int, int]
    area: float
    circularity: float
    solidity: float
    color_ratio: float


@dataclass
class Track:
    """Mutable state for one tracked object.

    ``observed_color`` is the latest fresh detector classification, while
    ``color`` is the public output color resolved by the core stabilizer.
    """

    track_id: int
    color: str
    centroid: tuple[int, int]
    bbox: tuple[int, int, int, int]
    history: deque[tuple[int, int]] = field(default_factory=deque)
    age: int = 1
    missing: int = 0
    confirmed: bool = False
    counted: bool = False
    observed_color: str | None = None

    def __post_init__(self) -> None:
        if self.observed_color is None:
            self.observed_color = self.color


@dataclass(frozen=True)
class CountEvent:
    """Immutable event emitted when a confirmed track is counted."""

    track_id: int
    color: str
    frame_index: int


@dataclass(frozen=True)
class FramePacket:
    """One in-memory frame supplied by a runtime source."""

    frame: np.ndarray
    frame_index: int


@dataclass(frozen=True)
class SourceMetadata:
    """Optional media properties exposed by a source to output sinks."""

    fps: float | None = None
    width: int | None = None
    height: int | None = None


@dataclass(frozen=True)
class FrameResult:
    """Structured business result produced by the core for one frame."""

    frame_index: int
    detections: list[Detection]
    tracks: list[Track]
    events: list[CountEvent]
    total_count: int
    count_by_color: dict[str, int]


@dataclass(frozen=True)
class RuntimeMetrics:
    """Presentation-only metrics calculated outside the vision core."""

    processing_fps: float = 0.0


@dataclass(frozen=True)
class RunSummary:
    """Final runtime summary independent of source and output paths."""

    frames_processed: int
    average_processing_fps: float
    total_count: int
    count_by_color: dict[str, int]
    count_events: list[CountEvent]
