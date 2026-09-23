from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Iterable

from .models import Track


@dataclass
class _ColorState:
    history: deque[str]
    stable_color: str


class ColorStabilizer:
    """Resolve bounded per-track observations into a stable output color.

    The first fresh observation is immediately usable as a provisional color.
    A different color replaces it only when that color is the unique window
    winner and has at least ``min_observations`` votes. Ties therefore retain
    the current stable color.
    """

    def __init__(self, *, enabled: bool = True, window_size: int = 5, min_observations: int = 3):
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be a bool")
        if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 1:
            raise ValueError("window_size must be a positive integer")
        if isinstance(min_observations, bool) or not isinstance(min_observations, int) or min_observations < 1:
            raise ValueError("min_observations must be a positive integer")
        if min_observations > window_size:
            raise ValueError("min_observations must not exceed window_size")
        self.enabled = enabled
        self.window_size = window_size
        self.min_observations = min_observations
        self._states: dict[int, _ColorState] = {}

    def history_for(self, track_id: int) -> tuple[str, ...]:
        state = self._states.get(track_id)
        return tuple(state.history) if state else ()

    def temporal_confidence_for(self, track_id: int) -> float | None:
        """Return stable-color vote ratio, not detector/model probability."""
        state = self._states.get(track_id)
        if not state or not state.history:
            return None
        return state.history.count(state.stable_color) / len(state.history)

    def _record(self, track: Track) -> _ColorState:
        observed_color = track.observed_color
        if observed_color is None:
            raise ValueError(f"Fresh track {track.track_id} has no observed color")

        state = self._states.get(track.track_id)
        if state is None:
            state = _ColorState(
                history=deque(maxlen=self.window_size),
                stable_color=observed_color,
            )
            self._states[track.track_id] = state
        state.history.append(observed_color)

        votes = Counter(state.history)
        highest = max(votes.values())
        winners = [color for color, count in votes.items() if count == highest]
        if len(winners) == 1:
            winner = winners[0]
            if winner == state.stable_color or highest >= self.min_observations:
                state.stable_color = winner
        return state

    def update(self, tracks: Iterable[Track]) -> list[Track]:
        current_tracks = list(tracks)
        active_ids = {track.track_id for track in current_tracks}
        for track_id in set(self._states) - active_ids:
            del self._states[track_id]

        for track in current_tracks:
            if not self.enabled:
                if track.missing == 0 and track.observed_color is not None:
                    track.color = track.observed_color
                continue

            state = self._states.get(track.track_id)
            if track.missing == 0:
                state = self._record(track)
            if state is not None:
                track.color = state.stable_color
        return current_tracks
