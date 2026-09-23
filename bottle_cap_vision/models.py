"""Shared runtime contracts for the BottleCapVision core.

This module owns data exchanged between detector, tracker and counter so later
algorithm changes do not force cross-module type rewrites.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


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
    """Mutable state for one tracked object."""

    track_id: int
    color: str
    centroid: tuple[int, int]
    bbox: tuple[int, int, int, int]
    history: deque[tuple[int, int]] = field(default_factory=deque)
    age: int = 1
    missing: int = 0
    confirmed: bool = False
    counted: bool = False


@dataclass(frozen=True)
class CountEvent:
    """Immutable event emitted when a confirmed track is counted."""

    track_id: int
    color: str
    frame_index: int
