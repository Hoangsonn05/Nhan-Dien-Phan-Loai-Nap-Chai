from __future__ import annotations

from collections import deque
from typing import Dict, List
import math

from .models import Detection, Track


class NearestNeighborTracker:
    """Simple tracker intended for fixed-camera conveyor videos."""

    def __init__(self, max_distance: float = 80.0, max_missing_frames: int = 12,
                 min_confirmed_frames: int = 2, max_history: int = 20):
        self.max_distance = max_distance
        self.max_missing_frames = max_missing_frames
        self.min_confirmed_frames = min_confirmed_frames
        self.max_history = max_history
        self._next_id = 1
        self.tracks: Dict[int, Track] = {}

    @staticmethod
    def _distance(a: tuple[int, int], b: tuple[int, int]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _update_track(self, track: Track, det: Detection) -> None:
        track.color = det.color
        track.centroid = det.centroid
        track.bbox = det.bbox
        track.history.append(det.centroid)
        while len(track.history) > self.max_history:
            track.history.popleft()
        track.age += 1
        track.missing = 0
        if track.age >= self.min_confirmed_frames:
            track.confirmed = True

    def update(self, detections: List[Detection]) -> List[Track]:
        unmatched_tracks = set(self.tracks.keys())
        unmatched_dets = set(range(len(detections)))
        candidates = []
        for tid, tr in self.tracks.items():
            for di, det in enumerate(detections):
                candidates.append((self._distance(tr.centroid, det.centroid), tid, di))
        for dist, tid, di in sorted(candidates):
            if dist > self.max_distance or tid not in unmatched_tracks or di not in unmatched_dets:
                continue
            self._update_track(self.tracks[tid], detections[di])
            unmatched_tracks.remove(tid)
            unmatched_dets.remove(di)

        for tid in list(unmatched_tracks):
            self.tracks[tid].missing += 1
            if self.tracks[tid].missing > self.max_missing_frames:
                del self.tracks[tid]

        for di in unmatched_dets:
            det = detections[di]
            history = deque([det.centroid], maxlen=self.max_history)
            tr = Track(
                track_id=self._next_id,
                color=det.color,
                centroid=det.centroid,
                bbox=det.bbox,
                history=history,
                age=1,
                missing=0,
                confirmed=(self.min_confirmed_frames <= 1),
            )
            self.tracks[self._next_id] = tr
            self._next_id += 1

        return list(self.tracks.values())
