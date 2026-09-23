import numpy as np

from bottle_cap_vision.config import AppConfig
from bottle_cap_vision.counter import LineCounter
from bottle_cap_vision.detector import ColorCapDetector, Detection
from bottle_cap_vision.tracker import NearestNeighborTracker


def _make_detection(y: int, color: str = "Red", x: int = 10) -> Detection:
    return Detection(color, (x, y), (0, 0, 1, 1), 300, 0.8, 0.9, 0.8)


def _make_track_sequence(
    sequence: list[int],
    *,
    orientation: str = "horizontal",
    direction: str = "positive",
    deadband: float = 5,
    line_position: float = 0.5,
    frame_shape: tuple[int, int, int] = (100, 100, 3),
    color: str = "Red",
    x: int = 10,
) -> tuple[LineCounter, list[tuple[int, list[int], list[tuple[int, bool]], list[int]]]]:
    tracker = NearestNeighborTracker(max_distance=100, min_confirmed_frames=1)
    counter = LineCounter(
        line_position=line_position,
        orientation=orientation,
        direction=direction,
        deadband=deadband,
    )
    events_by_frame = []
    for frame_index, value in enumerate(sequence):
        if orientation == "vertical":
            detection = _make_detection(x=value, y=10, color=color)
        else:
            detection = _make_detection(y=value, color=color, x=x)
        tracks = tracker.update([detection])
        events = counter.update(tracks, frame_index, frame_shape)
        events_by_frame.append(
            (
                value,
                [event.track_id for event in events],
                [(track.track_id, track.counted) for track in tracks],
                [event.frame_index for event in events],
            )
        )
    return counter, events_by_frame


def test_detector_empty_frame():
    detector = ColorCapDetector(AppConfig.default())
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    assert detector.detect(frame) == []


def test_tracker_assigns_stable_id_for_same_color_nearby():
    tracker = NearestNeighborTracker(max_distance=50, min_confirmed_frames=1)
    d1 = Detection("Blue", (50, 50), (40, 40, 20, 20), 300, 0.8, 0.9, 0.8)
    d2 = Detection("Blue", (55, 52), (45, 42, 20, 20), 300, 0.8, 0.9, 0.8)
    t1 = tracker.update([d1])[0]
    t2 = tracker.update([d2])[0]
    assert t1.track_id == t2.track_id == 1


def test_tracker_keeps_identity_through_color_flicker():
    tracker = NearestNeighborTracker(max_distance=50, min_confirmed_frames=1)
    detections = [
        Detection("Red", (100, 100), (90, 90, 20, 20), 300, 0.8, 0.9, 0.8),
        Detection("Pink", (103, 102), (93, 92, 20, 20), 300, 0.8, 0.9, 0.8),
        Detection("Red", (106, 104), (96, 94, 20, 20), 300, 0.8, 0.9, 0.8),
    ]

    ids = [tracker.update([det])[0].track_id for det in detections]

    assert ids == [1, 1, 1]


def test_tracker_keeps_distinct_tracks_for_two_objects_same_color():
    tracker = NearestNeighborTracker(max_distance=50, min_confirmed_frames=1)
    frame1 = [
        Detection("Red", (50, 50), (40, 40, 20, 20), 300, 0.8, 0.9, 0.8),
        Detection("Red", (150, 150), (140, 140, 20, 20), 300, 0.8, 0.9, 0.8),
    ]
    frame2 = [
        Detection("Red", (55, 52), (45, 42, 20, 20), 300, 0.8, 0.9, 0.8),
        Detection("Red", (155, 152), (145, 142, 20, 20), 300, 0.8, 0.9, 0.8),
    ]

    ids1 = sorted(track.track_id for track in tracker.update(frame1))
    ids2 = sorted(track.track_id for track in tracker.update(frame2))

    assert ids1 == [1, 2]
    assert ids2 == [1, 2]


def test_tracker_keeps_distinct_tracks_for_two_objects_different_colors():
    tracker = NearestNeighborTracker(max_distance=50, min_confirmed_frames=1)
    frame1 = [
        Detection("Red", (50, 50), (40, 40, 20, 20), 300, 0.8, 0.9, 0.8),
        Detection("Blue", (150, 150), (140, 140, 20, 20), 300, 0.8, 0.9, 0.8),
    ]
    frame2 = [
        Detection("Blue", (155, 152), (145, 142, 20, 20), 300, 0.8, 0.9, 0.8),
        Detection("Red", (55, 52), (45, 42, 20, 20), 300, 0.8, 0.9, 0.8),
    ]

    ids1 = sorted(track.track_id for track in tracker.update(frame1))
    ids2 = sorted(track.track_id for track in tracker.update(frame2))

    assert ids1 == [1, 2]
    assert ids2 == [1, 2]


def test_tracker_preserves_id_through_detection_order_changes():
    tracker = NearestNeighborTracker(max_distance=50, min_confirmed_frames=1)
    frame1 = [
        Detection("Red", (50, 50), (40, 40, 20, 20), 300, 0.8, 0.9, 0.8),
        Detection("Blue", (150, 150), (140, 140, 20, 20), 300, 0.8, 0.9, 0.8),
    ]
    frame2 = [
        Detection("Blue", (155, 152), (145, 142, 20, 20), 300, 0.8, 0.9, 0.8),
        Detection("Red", (55, 52), (45, 42, 20, 20), 300, 0.8, 0.9, 0.8),
    ]

    tracks1 = {track.color: track.track_id for track in tracker.update(frame1)}
    tracks2 = {track.color: track.track_id for track in tracker.update(frame2)}

    assert tracks1 == {"Red": 1, "Blue": 2}
    assert tracks2 == {"Red": 1, "Blue": 2}


def test_tracker_respects_max_distance_and_missing_frame_timeout():
    tracker = NearestNeighborTracker(max_distance=25, max_missing_frames=1, min_confirmed_frames=1)
    first = Detection("Red", (50, 50), (40, 40, 20, 20), 300, 0.8, 0.9, 0.8)
    far = Detection("Pink", (120, 120), (110, 110, 20, 20), 300, 0.8, 0.9, 0.8)

    tracks1 = tracker.update([first])
    tracks2 = tracker.update([far])
    assert {track.track_id for track in tracks1} == {1}
    assert sorted(track.track_id for track in tracks2) == [1, 2]

    tracker.update([])
    assert 1 not in tracker.tracks
    assert 2 in tracker.tracks

    tracker.update([])
    assert tracker.tracks == {}


def test_tracker_uses_one_to_one_assignment():
    tracker = NearestNeighborTracker(max_distance=100, min_confirmed_frames=1)
    frame1 = [
        Detection("Red", (50, 50), (40, 40, 20, 20), 300, 0.8, 0.9, 0.8),
        Detection("Blue", (150, 150), (140, 140, 20, 20), 300, 0.8, 0.9, 0.8),
    ]
    frame2 = [
        Detection("Pink", (52, 52), (42, 42, 20, 20), 300, 0.8, 0.9, 0.8),
        Detection("Pink", (148, 148), (138, 138, 20, 20), 300, 0.8, 0.9, 0.8),
    ]

    tracker.update(frame1)
    tracks = tracker.update(frame2)

    assert sorted(track.track_id for track in tracks) == [1, 2]
    assert {track.centroid for track in tracks} == {(52, 52), (148, 148)}


def test_counter_counts_exactly_once_after_crossing():
    counter, events_by_frame = _make_track_sequence([35, 65, 70], deadband=1)

    assert events_by_frame[0][1] == []
    assert events_by_frame[1][1] == [1]
    assert events_by_frame[2][1] == []
    assert counter.total == 1
    assert counter.by_color["Red"] == 1


def test_counter_counts_gradual_positive_crossing():
    counter, events_by_frame = _make_track_sequence([35, 44, 48, 52, 56, 65], deadband=5)

    assert [events for _, events, _, _ in events_by_frame] == [[], [], [], [], [1], []]
    assert counter.total == 1
    assert len(counter.events) == 1
    assert counter.events[0].track_id == 1


def test_counter_counts_gradual_negative_crossing():
    counter, events_by_frame = _make_track_sequence([65, 56, 52, 48, 44, 35], direction="negative", deadband=5)

    assert [events for _, events, _, _ in events_by_frame] == [[], [], [], [], [1], []]
    assert counter.total == 1


def test_counter_ignores_deadband_jitter():
    counter, events_by_frame = _make_track_sequence([48, 51, 49, 52, 50], deadband=5)

    assert [events for _, events, _, _ in events_by_frame] == [[], [], [], [], []]
    assert counter.total == 0


def test_counter_ignores_approach_and_return_without_crossing():
    counter, events_by_frame = _make_track_sequence([35, 44, 48, 44, 35], deadband=5)

    assert [events for _, events, _, _ in events_by_frame] == [[], [], [], [], []]
    assert counter.total == 0


def test_counter_does_not_count_when_starting_inside_deadband():
    counter, events_by_frame = _make_track_sequence([50, 52, 60], deadband=5)

    assert [events for _, events, _, _ in events_by_frame] == [[], [], []]
    assert counter.total == 0


def test_counter_respects_direction_filter_positive():
    positive_counter, positive_events = _make_track_sequence([35, 65], direction="positive", deadband=5)
    negative_counter, negative_events = _make_track_sequence([65, 35], direction="positive", deadband=5)

    assert [events for _, events, _, _ in positive_events] == [[], [1]]
    assert positive_counter.total == 1
    assert [events for _, events, _, _ in negative_events] == [[], []]
    assert negative_counter.total == 0


def test_counter_respects_direction_filter_negative():
    negative_counter, negative_events = _make_track_sequence([65, 35], direction="negative", deadband=5)
    positive_counter, positive_events = _make_track_sequence([35, 65], direction="negative", deadband=5)

    assert [events for _, events, _, _ in negative_events] == [[], [1]]
    assert negative_counter.total == 1
    assert [events for _, events, _, _ in positive_events] == [[], []]
    assert positive_counter.total == 0


def test_counter_accepts_both_directions():
    first_counter, first_events = _make_track_sequence([35, 65], direction="both", deadband=5)
    second_counter, second_events = _make_track_sequence([65, 35], direction="both", deadband=5)

    assert [events for _, events, _, _ in first_events] == [[], [1]]
    assert [events for _, events, _, _ in second_events] == [[], [1]]
    assert first_counter.total == 1
    assert second_counter.total == 1


def test_counter_counts_only_once_per_track():
    counter, events_by_frame = _make_track_sequence([35, 65, 35, 65], deadband=5)

    assert [events for _, events, _, _ in events_by_frame] == [[], [1], [], []]
    assert counter.total == 1


def test_counter_counts_independent_tracks_separately():
    tracker = NearestNeighborTracker(max_distance=100, min_confirmed_frames=1)
    counter = LineCounter(line_position=0.5, orientation="horizontal", direction="positive", deadband=5)

    first = tracker.update([_make_detection(35, color="Red", x=10), _make_detection(35, color="Blue", x=80)])
    assert counter.update(first, 0, (100, 100, 3)) == []

    second = tracker.update([_make_detection(65, color="Red", x=10), _make_detection(65, color="Blue", x=80)])
    events = counter.update(second, 1, (100, 100, 3))

    assert sorted(event.track_id for event in events) == [1, 2]
    assert counter.total == 2
    assert counter.by_color["Red"] == 1
    assert counter.by_color["Blue"] == 1


def test_counter_uses_horizontal_axis_for_horizontal_lines():
    counter, events_by_frame = _make_track_sequence([35, 65], orientation="horizontal", direction="positive", deadband=5, frame_shape=(100, 200, 3), x=90)

    assert [events for _, events, _, _ in events_by_frame] == [[], [1]]
    assert counter.total == 1


def test_counter_uses_vertical_axis_for_vertical_lines():
    counter, events_by_frame = _make_track_sequence(
        [35, 65],
        orientation="vertical",
        direction="positive",
        deadband=5,
        line_position=0.25,
        frame_shape=(100, 200, 3),
    )

    assert [events for _, events, _, _ in events_by_frame] == [[], [1]]
    assert counter.total == 1


def test_shared_runtime_contracts_are_canonical():
    from bottle_cap_vision.models import CountEvent as ModelCountEvent
    from bottle_cap_vision.models import Detection as ModelDetection
    from bottle_cap_vision.models import Track as ModelTrack
    from bottle_cap_vision.counter import CountEvent as CounterCountEvent
    from bottle_cap_vision.detector import Detection as DetectorDetection
    from bottle_cap_vision.tracker import Track as TrackerTrack

    assert DetectorDetection is ModelDetection
    assert TrackerTrack is ModelTrack
    assert CounterCountEvent is ModelCountEvent


def test_pipeline_processes_an_in_memory_frame():
    from bottle_cap_vision.models import FrameResult
    from bottle_cap_vision.pipeline import BottleCapPipeline

    result = BottleCapPipeline(AppConfig.default()).process_frame(
        np.zeros((80, 120, 3), dtype=np.uint8),
        frame_index=7,
    )

    assert isinstance(result, FrameResult)
    assert result.frame_index == 7
    assert result.detections == []
    assert result.tracks == []
    assert result.events == []
