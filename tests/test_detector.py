from __future__ import annotations

import cv2
import numpy as np
import pytest

from bottle_cap_vision import detector as detector_module
from bottle_cap_vision.config import AppConfig, ColorConfig, HSVRange
from bottle_cap_vision.detector import ColorCapDetector
from bottle_cap_vision.models import Detection
from bottle_cap_vision.tracker import NearestNeighborTracker


GREEN = HSVRange((55, 200, 200), (65, 255, 255))


def _detection(
    color: str,
    centroid: tuple[int, int],
    bbox: tuple[int, int, int, int],
    *,
    area: float = 100.0,
    color_ratio: float = 0.8,
) -> Detection:
    return Detection(color, centroid, bbox, area, 0.8, 0.9, color_ratio)


def _synthetic_detector(*colors: str) -> ColorCapDetector:
    cfg = AppConfig.default()
    cfg.colors = [ColorConfig(name, [GREEN]) for name in colors]
    cfg.detection.min_area = 10
    cfg.detection.max_area = 1000
    cfg.detection.min_circularity = 0.0
    cfg.detection.min_solidity = 0.0
    cfg.detection.morphology_kernel = 3
    cfg.detection.open_iterations = 0
    cfg.detection.close_iterations = 0
    cfg.detection.min_color_ratio = 0.0
    return ColorCapDetector(cfg)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ((10, 10, 20, 20), (10, 10, 20, 20), 1.0),
        ((0, 0, 10, 10), (20, 20, 10, 10), 0.0),
        ((0, 0, 20, 20), (10, 0, 20, 20), 1 / 3),
        ((0, 0, 10, 10), (10, 0, 10, 10), 0.0),
    ],
)
def test_bbox_iou(left, right, expected):
    assert detector_module.bbox_iou(left, right) == pytest.approx(expected)


def test_same_object_same_color_is_deduplicated():
    candidates = [
        _detection("Green", (30, 30), (10, 10, 40, 40), color_ratio=0.7),
        _detection("Green", (31, 30), (11, 10, 40, 40), color_ratio=0.9),
    ]

    result = ColorCapDetector._deduplicate(candidates)

    assert result == [candidates[1]]


def test_same_object_different_colors_uses_existing_evidence():
    candidates = [
        _detection("Red", (30, 30), (10, 10, 40, 40), area=1200, color_ratio=0.9),
        _detection("Pink", (43, 30), (23, 10, 40, 40), area=1100, color_ratio=0.8),
    ]

    result = ColorCapDetector._deduplicate(candidates)

    assert result == [candidates[0]]


def test_nearby_distinct_objects_are_not_merged_by_centroid_distance():
    candidates = [
        _detection("Blue", (15, 15), (10, 10, 10, 10)),
        _detection("Green", (25, 15), (20, 10, 10, 10)),
    ]

    assert len(ColorCapDetector._deduplicate(candidates)) == 2


def test_low_overlap_candidates_remain_distinct():
    candidates = [
        _detection("Purple", (10, 10), (0, 0, 20, 20)),
        _detection("Lilac", (25, 10), (15, 0, 20, 20)),
    ]

    assert len(ColorCapDetector._deduplicate(candidates)) == 2


def test_strong_containment_is_deduplicated():
    candidates = [
        _detection("Purple", (20, 20), (0, 0, 40, 40), area=1200),
        _detection("Lilac", (20, 20), (15, 15, 10, 10), area=100),
    ]

    assert ColorCapDetector._deduplicate(candidates) == [candidates[0]]


def test_equal_evidence_tie_is_deterministic_for_generic_colors():
    alpha = _detection("Alpha", (20, 20), (10, 10, 20, 20))
    zeta = _detection("Zeta", (20, 20), (10, 10, 20, 20))

    forward = ColorCapDetector._deduplicate([zeta, alpha])
    reverse = ColorCapDetector._deduplicate([alpha, zeta])

    assert forward == reverse == [alpha]


def test_empty_candidate_list_returns_empty():
    assert ColorCapDetector._deduplicate([]) == []


def test_real_detector_keeps_two_nearby_physical_objects():
    detector = _synthetic_detector("Green")
    frame = np.zeros((50, 60, 3), dtype=np.uint8)
    green_bgr = tuple(int(value) for value in cv2.cvtColor(
        np.uint8([[[60, 255, 255]]]), cv2.COLOR_HSV2BGR
    )[0, 0])
    cv2.circle(frame, (20, 25), 4, green_bgr, -1)
    cv2.circle(frame, (30, 25), 4, green_bgr, -1)

    detections = detector.detect(frame)

    assert sorted(det.centroid for det in detections) == [(20, 25), (30, 25)]


def test_real_detector_collapses_overlapping_generic_color_masks():
    detector = _synthetic_detector("Zeta", "Alpha")
    frame = np.zeros((50, 60, 3), dtype=np.uint8)
    green_bgr = tuple(int(value) for value in cv2.cvtColor(
        np.uint8([[[60, 255, 255]]]), cv2.COLOR_HSV2BGR
    )[0, 0])
    cv2.circle(frame, (25, 25), 8, green_bgr, -1)

    detections = detector.detect(frame)

    assert len(detections) == 1
    assert detections[0].color == "Alpha"


def test_area_filters_still_reject_small_and_large_candidates():
    detector = _synthetic_detector("Green")
    detector.config.detection.min_area = 50
    detector.config.detection.max_area = 200
    frame = np.zeros((60, 100, 3), dtype=np.uint8)
    green_bgr = tuple(int(value) for value in cv2.cvtColor(
        np.uint8([[[60, 255, 255]]]), cv2.COLOR_HSV2BGR
    )[0, 0])
    cv2.circle(frame, (15, 30), 3, green_bgr, -1)
    cv2.circle(frame, (45, 30), 6, green_bgr, -1)
    cv2.circle(frame, (80, 30), 10, green_bgr, -1)

    detections = detector.detect(frame)

    assert [det.centroid for det in detections] == [(45, 30)]


def test_roi_detection_coordinates_remain_full_frame():
    detector = _synthetic_detector("Green")
    detector.config.roi.enabled = True
    detector.config.roi.x1 = 20
    detector.config.roi.y1 = 15
    detector.config.roi.x2 = 50
    detector.config.roi.y2 = 45
    frame = np.zeros((60, 80, 3), dtype=np.uint8)
    green_bgr = tuple(int(value) for value in cv2.cvtColor(
        np.uint8([[[60, 255, 255]]]), cv2.COLOR_HSV2BGR
    )[0, 0])
    cv2.circle(frame, (35, 30), 6, green_bgr, -1)

    detections = detector.detect(frame)

    assert len(detections) == 1
    assert detections[0].centroid == (35, 30)
    assert detections[0].bbox == (29, 24, 13, 13)


def test_nearby_detector_outputs_create_two_tracker_tracks():
    candidates = [
        _detection("Green", (15, 15), (10, 10, 10, 10)),
        _detection("Green", (25, 15), (20, 10, 10, 10)),
    ]
    detections = ColorCapDetector._deduplicate(candidates)
    tracker = NearestNeighborTracker(max_distance=50, min_confirmed_frames=1)

    tracks = tracker.update(detections)

    assert len(tracks) == 2
