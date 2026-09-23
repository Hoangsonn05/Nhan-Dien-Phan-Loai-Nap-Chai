from __future__ import annotations

from collections import Counter as PyCounter
from dataclasses import dataclass, field
from typing import Dict, Iterable

from .models import CountEvent, Track


@dataclass
class LineCounter:
    orientation: str = "horizontal"
    line_position: float = 0.60
    direction: str = "positive"
    deadband: float = 5.0
    total: int = 0
    by_color: PyCounter = field(default_factory=PyCounter)
    events: list[CountEvent] = field(default_factory=list)
    _track_last_stable_side: dict[int, str] = field(default_factory=dict, init=False, repr=False)

    _NEGATIVE_SIDE = "negative"
    _NEAR_SIDE = "near"
    _POSITIVE_SIDE = "positive"

    def _signed_position(self, point: tuple[int, int], width: int, height: int) -> float:
        if self.orientation == "vertical":
            return point[0] - width * self.line_position
        return point[1] - height * self.line_position

    def _classify_side(self, signed_position: float) -> str:
        if signed_position < -self.deadband:
            return self._NEGATIVE_SIDE
        if signed_position > self.deadband:
            return self._POSITIVE_SIDE
        return self._NEAR_SIDE

    @staticmethod
    def _crossing_direction(previous_side: str, current_side: str) -> str | None:
        if previous_side == LineCounter._NEGATIVE_SIDE and current_side == LineCounter._POSITIVE_SIDE:
            return "positive"
        if previous_side == LineCounter._POSITIVE_SIDE and current_side == LineCounter._NEGATIVE_SIDE:
            return "negative"
        return None

    def update(self, tracks: Iterable[Track], frame_index: int, frame_shape: tuple[int, int]) -> list[CountEvent]:
        h, w = frame_shape[:2]
        new_events: list[CountEvent] = []
        for tr in tracks:
            if tr.counted or not tr.confirmed or len(tr.history) < 1:
                continue
            current_position = self._signed_position(tr.history[-1], w, h)
            current_side = self._classify_side(current_position)

            if current_side == self._NEAR_SIDE:
                continue

            previous_side = self._track_last_stable_side.get(tr.track_id)
            if previous_side is None:
                self._track_last_stable_side[tr.track_id] = current_side
                continue

            if previous_side == current_side:
                continue

            crossing_direction = self._crossing_direction(previous_side, current_side)
            self._track_last_stable_side[tr.track_id] = current_side

            if crossing_direction is None:
                continue
            if self.direction != "both" and crossing_direction != self.direction:
                continue

            tr.counted = True
            event = CountEvent(tr.track_id, tr.color, frame_index)
            self.events.append(event)
            self.total += 1
            self.by_color[tr.color] += 1
            new_events.append(event)
            self._track_last_stable_side.pop(tr.track_id, None)
        return new_events

    def snapshot(self) -> Dict[str, int]:
        result = {k: int(self.by_color.get(k, 0)) for k in sorted(self.by_color)}
        result["TOTAL"] = int(self.total)
        return result
