from __future__ import annotations

from collections import deque

import numpy as np

from bottle_cap_vision.color_stabilizer import ColorStabilizer
from bottle_cap_vision.config import AppConfig
from bottle_cap_vision.models import Detection, Track
from bottle_cap_vision.pipeline import BottleCapPipeline
from bottle_cap_vision.tracker import NearestNeighborTracker


def _track(track_id: int, color: str, *, missing: int = 0) -> Track:
    return Track(
        track_id=track_id,
        color=color,
        observed_color=color,
        centroid=(10, 10),
        bbox=(5, 5, 10, 10),
        history=deque([(10, 10)]),
        missing=missing,
        confirmed=True,
    )


def _observe(stabilizer: ColorStabilizer, track: Track, color: str) -> str:
    track.observed_color = color
    track.missing = 0
    return stabilizer.update([track])[0].color


def _detection(color: str, y: int, *, x: int = 10) -> Detection:
    return Detection(color, (x, y), (x - 5, y - 5, 10, 10), 300, 0.8, 0.9, 0.8)


def test_single_and_two_frame_noise_do_not_switch_stable_color():
    stabilizer = ColorStabilizer(window_size=5, min_observations=3)
    track = _track(1, "Red")

    single_noise = [_observe(stabilizer, track, color) for color in ["Red", "Red", "Pink", "Red", "Red"]]
    two_frame_noise = [_observe(stabilizer, track, color) for color in ["Red", "Pink", "Pink", "Red", "Red"]]

    assert single_noise == ["Red"] * 5
    assert two_frame_noise == ["Red"] * 5


def test_temporal_confidence_is_vote_ratio_not_detector_probability():
    stabilizer = ColorStabilizer(window_size=5, min_observations=3)
    track = _track(1, "Red")

    for color in ["Red", "Red", "Pink", "Red", "Red"]:
        _observe(stabilizer, track, color)

    assert stabilizer.temporal_confidence_for(1) == 0.8


def test_sustained_real_color_change_eventually_switches():
    stabilizer = ColorStabilizer(window_size=5, min_observations=3)
    track = _track(1, "Red")

    output = [_observe(stabilizer, track, color) for color in ["Red", "Red", "Red", "Pink", "Pink", "Pink"]]

    assert output == ["Red", "Red", "Red", "Red", "Red", "Pink"]


def test_voting_is_generic_and_tie_keeps_current_stable_color():
    stabilizer = ColorStabilizer(window_size=4, min_observations=2)
    track = _track(1, "Purple")

    output = [_observe(stabilizer, track, color) for color in ["Purple", "Lilac", "Purple", "Lilac"]]

    assert output == ["Purple"] * 4


def test_new_track_uses_first_observation_as_provisional_color():
    stabilizer = ColorStabilizer(window_size=5, min_observations=3)

    result = stabilizer.update([_track(7, "Orange")])

    assert result[0].color == "Orange"
    assert stabilizer.history_for(7) == ("Orange",)


def test_missing_frames_do_not_add_votes_but_keep_history():
    tracker = NearestNeighborTracker(max_distance=25, max_missing_frames=2, min_confirmed_frames=1)
    stabilizer = ColorStabilizer(window_size=5, min_observations=3)

    stabilizer.update(tracker.update([_detection("Red", 10)]))
    stabilizer.update(tracker.update([]))
    stabilizer.update(tracker.update([]))
    tracks = stabilizer.update(tracker.update([_detection("Pink", 10)]))

    assert stabilizer.history_for(1) == ("Red", "Pink")
    assert tracks[0].track_id == 1
    assert tracks[0].color == "Red"


def test_history_is_bounded_to_window_size():
    stabilizer = ColorStabilizer(window_size=3, min_observations=2)
    track = _track(1, "A")

    for color in ["A", "B", "C", "D"]:
        _observe(stabilizer, track, color)

    assert stabilizer.history_for(1) == ("B", "C", "D")


def test_tracks_have_independent_color_state():
    stabilizer = ColorStabilizer(window_size=5, min_observations=3)
    red = _track(1, "Red")
    blue = _track(2, "Blue")

    stabilizer.update([red, blue])
    for color in ["Pink", "Pink"]:
        red.observed_color = color
        blue.observed_color = "Blue"
        stabilizer.update([red, blue])

    assert red.color == "Red"
    assert blue.color == "Blue"
    assert stabilizer.history_for(1) == ("Red", "Pink", "Pink")
    assert stabilizer.history_for(2) == ("Blue", "Blue", "Blue")


def test_state_is_removed_when_tracker_no_longer_returns_track():
    tracker = NearestNeighborTracker(max_distance=25, max_missing_frames=0, min_confirmed_frames=1)
    stabilizer = ColorStabilizer(window_size=5, min_observations=3)
    stabilizer.update(tracker.update([_detection("Red", 10)]))

    stabilizer.update(tracker.update([]))

    assert stabilizer.history_for(1) == ()


def test_tracker_id_remains_stable_when_observed_color_flickers():
    tracker = NearestNeighborTracker(max_distance=25, min_confirmed_frames=1)
    stabilizer = ColorStabilizer(window_size=5, min_observations=3)
    ids = []
    colors = []

    for color in ["Red", "Pink", "Red"]:
        tracks = stabilizer.update(tracker.update([_detection(color, 10)]))
        ids.append(tracks[0].track_id)
        colors.append(tracks[0].color)

    assert ids == [1, 1, 1]
    assert colors == ["Red", "Red", "Red"]


def test_crossing_event_uses_stable_color_and_counts_once():
    cfg = AppConfig.default()
    cfg.tracking.min_confirmed_frames = 1
    cfg.tracking.max_distance = 30
    cfg.counting.line_position = 0.5
    cfg.counting.deadband = 5
    pipeline = BottleCapPipeline(cfg)
    observations = iter([
        _detection("Red", 35),
        _detection("Red", 40),
        _detection("Red", 45),
        _detection("Pink", 65),
        _detection("Red", 70),
    ])
    pipeline.detector.detect = lambda frame: [next(observations)]
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    track_ids = []
    stable_colors = []
    events = []
    for index in range(5):
        result = pipeline.process_frame(frame, index)
        track_ids.append(result.tracks[0].track_id)
        stable_colors.append(result.tracks[0].color)
        events.extend(result.events)

    assert track_ids == [1, 1, 1, 1, 1]
    assert stable_colors == ["Red"] * 5
    assert len(events) == 1
    assert events[0].color == "Red"
    assert pipeline.counter.total == 1
