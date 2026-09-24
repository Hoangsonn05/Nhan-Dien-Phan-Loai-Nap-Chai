from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import bottle_cap_vision.practical_demo as practical_demo
from bottle_cap_vision.config import load_config
from bottle_cap_vision.models import Detection
from bottle_cap_vision.practical_demo import (
    DemoReport,
    DisplayMode,
    PracticalDemoError,
    ProcessedImage,
    build_comparison_board,
    build_display_board,
    build_footer_lines,
    cycle_display_mode,
    discover_images,
    process_images,
    show_results,
)
from bottle_cap_vision.visualization import annotate_detections
from scripts import demo_images


ROOT = Path(__file__).resolve().parents[1]


def _image(path: Path, value: int = 0) -> bytes:
    frame = np.full((40, 60, 3), value, np.uint8)
    assert cv2.imwrite(str(path), frame)
    return path.read_bytes()


def _config(tmp_path: Path, *, checkpoint_exists: bool = True) -> Path:
    checkpoint = tmp_path / "best.pt"
    metadata = tmp_path / "checkpoint_metadata.json"
    if checkpoint_exists:
        checkpoint.write_bytes(b"checkpoint")
    metadata.write_text("{}", encoding="utf-8")
    raw = json.loads((ROOT / "configs" / "config.learned-p9-full.json").read_text(encoding="utf-8"))
    raw["detector"]["checkpoint"] = str(checkpoint)
    raw["detector"]["metadata"] = str(metadata)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


class FakeDetector:
    def __init__(self, results: list[list[Detection]]) -> None:
        self.results = iter(results)
        self.calls = 0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self.calls += 1
        assert frame.shape == (40, 60, 3)
        return next(self.results)


def test_discovery_filters_hidden_and_unsupported_files_in_deterministic_order(tmp_path):
    _image(tmp_path / "B.PNG")
    _image(tmp_path / "a.jpg")
    _image(tmp_path / ".hidden.jpeg")
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")
    (tmp_path / "folder.png").mkdir()

    assert [path.name for path in discover_images(tmp_path)] == ["a.jpg", "B.PNG"]


def test_discovery_accepts_every_documented_image_extension(tmp_path):
    names = ["a.jpg", "b.jpeg", "c.png", "d.bmp", "e.webp", "f.tif", "g.tiff"]
    for name in names:
        (tmp_path / name).write_bytes(b"discovery does not decode")

    assert [path.name for path in discover_images(tmp_path)] == names


def test_recursive_discovery_skips_hidden_directories(tmp_path):
    visible = tmp_path / "nested"
    hidden = tmp_path / ".cache"
    visible.mkdir()
    hidden.mkdir()
    _image(visible / "cap.webp")
    _image(hidden / "ignored.png")

    assert [path.relative_to(tmp_path).as_posix() for path in discover_images(tmp_path, recursive=True)] == [
        "nested/cap.webp"
    ]


@pytest.mark.parametrize("create", [False, True])
def test_missing_or_empty_input_folder_fails_clearly(tmp_path, create):
    folder = tmp_path / "images"
    if create:
        folder.mkdir()
    with pytest.raises(PracticalDemoError, match="does not exist|Khong tim thay"):
        discover_images(folder)


def test_processing_loads_detector_once_and_preserves_independent_detection_contract(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    originals = {
        "one.png": _image(input_dir / "one.png", 20),
        "two.jpg": _image(input_dir / "two.jpg", 40),
    }
    config = _config(tmp_path)
    detector = FakeDetector([
        [Detection("Red", (15, 16), (10, 11, 10, 10), class_id=7, confidence=0.923)],
        [],
    ])
    factory_calls = []

    report = process_images(
        input_dir,
        tmp_path / "output",
        config,
        detector_factory=lambda cfg: factory_calls.append(cfg) or detector,
    )

    assert len(factory_calls) == 1
    assert detector.calls == 2
    assert not hasattr(practical_demo, "BottleCapPipeline")
    assert report.summary["images_successfully_processed"] == 2
    assert report.summary["failed_images"] == 0
    assert report.summary["total_detections"] == 1
    assert report.summary["detections_by_color"]["Red"] == 1
    assert all((report.output_dir / "annotated" / name).is_file() for name in originals)
    assert {name: (input_dir / name).read_bytes() for name in originals} == originals

    predictions = json.loads((report.output_dir / "predictions.json").read_text(encoding="utf-8"))
    first = predictions["images"][0]["detections"][0]
    assert first == {
        "class_id": 7,
        "color": "Red",
        "confidence": 0.923,
        "bbox": [10, 11, 10, 10],
        "centroid": [15, 16],
    }
    assert predictions["images"][1]["detections"] == []
    with (report.output_dir / "predictions.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["image"] == "one.png"
    assert rows[0]["class_id"] == "7"


def test_corrupt_image_is_recorded_and_remaining_image_is_processed(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "bad.jpg").write_bytes(b"not an image")
    _image(input_dir / "good.png")
    detector = FakeDetector([[]])

    report = process_images(
        input_dir,
        tmp_path / "output",
        _config(tmp_path),
        detector_factory=lambda _cfg: detector,
    )

    assert detector.calls == 1
    assert report.summary["images_successfully_processed"] == 1
    assert report.summary["failed_images"] == 1
    records = report.predictions["images"]
    assert [(item["file"], item["status"]) for item in records] == [
        ("bad.jpg", "ERROR"),
        ("good.png", "OK"),
    ]


def test_recursive_duplicate_basenames_get_stable_collision_safe_outputs(tmp_path):
    input_dir = tmp_path / "input"
    (input_dir / "a").mkdir(parents=True)
    (input_dir / "b").mkdir()
    _image(input_dir / "a" / "cap.png")
    _image(input_dir / "b" / "cap.png")
    detector = FakeDetector([[], []])

    report = process_images(
        input_dir,
        tmp_path / "output",
        _config(tmp_path),
        recursive=True,
        detector_factory=lambda _cfg: detector,
    )

    names = [item.annotated_path.name for item in report.processed_images]
    assert len(names) == len(set(names)) == 2
    assert all(name.startswith("cap__") and name.endswith(".png") for name in names)


def test_annotation_is_frame_local_uses_filled_label_background_and_does_not_mutate_input(monkeypatch):
    source = np.zeros((100, 160, 3), np.uint8)
    before = source.copy()
    detection = Detection("Blue", (50, 50), (30, 30, 40, 40), class_id=0, confidence=0.8)
    rectangle_thicknesses = []
    original_rectangle = cv2.rectangle

    def recording_rectangle(*args, **kwargs):
        thickness = kwargs.get("thickness", args[4] if len(args) > 4 else None)
        rectangle_thicknesses.append(thickness)
        return original_rectangle(*args, **kwargs)

    monkeypatch.setattr(cv2, "rectangle", recording_rectangle)

    detected = annotate_detections(source, [detection])
    empty = annotate_detections(source, [])

    assert np.array_equal(source, before)
    assert not np.array_equal(detected, source)
    assert np.array_equal(empty, source)
    assert -1 in rectangle_thicknesses
    assert any(thickness is not None and thickness >= 2 for thickness in rectangle_thicknesses)


def test_annotation_uses_stable_distinct_class_colors():
    source = np.zeros((120, 240, 3), np.uint8)
    detections = [
        Detection("Blue", (40, 70), (20, 60, 40, 40), class_id=0, confidence=0.77),
        Detection("Red", (150, 70), (130, 60, 40, 40), class_id=7, confidence=0.88),
    ]

    first = annotate_detections(source, detections)
    second = annotate_detections(source, detections)

    assert np.array_equal(first, second)
    assert not np.array_equal(first[60, 20], first[60, 130])


def test_comparison_board_preserves_input_and_stays_within_display_bounds():
    original = np.zeros((1200, 600, 3), np.uint8)
    annotated = original.copy()
    board = build_comparison_board(original, annotated, filename="cap.jpg", index=3, total=24)

    assert board.shape[1] <= 1800
    assert board.shape[0] <= 950
    assert original.shape == (1200, 600, 3)


@pytest.mark.parametrize("mode", list(DisplayMode))
def test_each_display_mode_renders_deterministically_with_summary_outside_image(mode):
    original = np.full((320, 640, 3), 50, np.uint8)
    annotated = np.full((320, 640, 3), 100, np.uint8)
    kwargs = {
        "filename": "caps.jpg",
        "index": 1,
        "total": 2,
        "detection_count": 3,
        "counts_by_color": (("Blue", 2), ("Red", 1)),
        "mode": mode,
    }

    first = build_display_board(original, annotated, **kwargs)
    second = build_display_board(original, annotated, **kwargs)

    assert np.array_equal(first, second)
    assert first.shape[1] == 1800
    assert first.shape[0] <= 950
    assert np.array_equal(original, np.full((320, 640, 3), 50, np.uint8))
    assert np.array_equal(annotated, np.full((320, 640, 3), 100, np.uint8))
    if mode is DisplayMode.RESULT_ONLY:
        rendered_result = first[practical_demo.HEADER_HEIGHT:practical_demo.HEADER_HEIGHT + 400, 500:1300]
        assert np.all(rendered_result == 100)


def test_mode_cycle_and_footer_information_are_complete():
    assert cycle_display_mode(DisplayMode.SIDE_BY_SIDE) is DisplayMode.RESULT_ONLY
    assert cycle_display_mode(DisplayMode.RESULT_ONLY) is DisplayMode.ORIGINAL_ONLY
    assert cycle_display_mode(DisplayMode.ORIGINAL_ONLY) is DisplayMode.SIDE_BY_SIDE

    lines = build_footer_lines(
        filename="real/caps.jpg",
        index=2,
        total=5,
        detection_count=3,
        counts_by_color=(("Blue", 2), ("Red", 1)),
        mode=DisplayMode.RESULT_ONLY,
    )
    text = "\n".join(lines)
    assert "Image 2 / 5 | real/caps.jpg" in text
    assert "Detected caps: 3 | Blue: 2 | Red: 1" in text
    assert "M/Tab Mode" in text
    assert "Mode: RESULT_ONLY" in text

    wrapped = build_footer_lines(
        filename="caps.jpg",
        index=1,
        total=1,
        detection_count=10,
        counts_by_color=tuple((f"LongColorName{index}", 1) for index in range(10)),
    )
    assert len(wrapped) > 3
    assert wrapped[-1].startswith("A/Left Prev")


def test_no_gui_cli_never_opens_opencv_window(monkeypatch, tmp_path):
    report = DemoReport({}, {"images_successfully_processed": 0, "failed_images": 0, "total_detections": 0, "detections_by_color": {}}, [], tmp_path)
    monkeypatch.setattr(demo_images, "process_images", lambda *args, **kwargs: report)
    monkeypatch.setattr(cv2, "namedWindow", lambda *args: pytest.fail("namedWindow called"))
    monkeypatch.setattr(cv2, "imshow", lambda *args: pytest.fail("imshow called"))
    monkeypatch.setattr(cv2, "waitKey", lambda *args: pytest.fail("waitKey called"))
    monkeypatch.setattr(cv2, "waitKeyEx", lambda *args: pytest.fail("waitKeyEx called"))

    assert demo_images.main(["--no-gui"]) == 0


def test_gui_navigation_uses_one_window_and_blocking_keys(monkeypatch, tmp_path):
    source = tmp_path / "source.png"
    annotated = tmp_path / "annotated.png"
    _image(source)
    _image(annotated)
    images = [ProcessedImage(source, annotated, "source.png")]
    keys = iter([ord("d"), ord("a"), ord("q")])
    calls = {"named": 0, "show": 0, "destroy": 0}
    monkeypatch.setattr(cv2, "namedWindow", lambda *args: calls.__setitem__("named", calls["named"] + 1))
    monkeypatch.setattr(cv2, "imshow", lambda *args: calls.__setitem__("show", calls["show"] + 1))
    monkeypatch.setattr(cv2, "waitKeyEx", lambda delay: next(keys) if delay == 0 else pytest.fail("non-blocking wait"))
    monkeypatch.setattr(cv2, "destroyWindow", lambda *args: calls.__setitem__("destroy", calls["destroy"] + 1))

    show_results(images)

    assert calls == {"named": 1, "show": 3, "destroy": 1}


def test_gui_mode_keys_cycle_without_opening_additional_windows(monkeypatch, tmp_path):
    source = tmp_path / "source.png"
    annotated = tmp_path / "annotated.png"
    _image(source)
    _image(annotated)
    images = [ProcessedImage(source, annotated, "source.png", 2, (("Blue", 2),))]
    keys = iter([ord("m"), 9, ord("M"), ord("q")])
    modes = []
    original_builder = practical_demo.build_display_board

    def recording_builder(*args, **kwargs):
        modes.append(kwargs["mode"])
        return original_builder(*args, **kwargs)

    monkeypatch.setattr(practical_demo, "build_display_board", recording_builder)
    monkeypatch.setattr(cv2, "namedWindow", lambda *args: None)
    monkeypatch.setattr(cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(cv2, "waitKeyEx", lambda _delay: next(keys))
    monkeypatch.setattr(cv2, "destroyWindow", lambda *args: None)

    show_results(images)

    assert modes == [
        DisplayMode.SIDE_BY_SIDE,
        DisplayMode.RESULT_ONLY,
        DisplayMode.ORIGINAL_ONLY,
        DisplayMode.SIDE_BY_SIDE,
    ]


def test_default_full_config_uses_full_p9_checkpoint_and_cpu():
    config = load_config(ROOT / "configs" / "config.learned-p9-full.json")

    assert config.detector.backend == "learned"
    assert "p9_full_001" in config.detector.checkpoint
    assert "p9_smoke_001" not in config.detector.checkpoint
    assert "p9_full_001" in config.detector.metadata
    assert config.detector.device == "cpu"
    assert config.detector.confidence_threshold == 0.25


def test_missing_full_checkpoint_fails_before_detector_creation(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _image(input_dir / "cap.png")
    factory_called = False

    def factory(_cfg):
        nonlocal factory_called
        factory_called = True
        return FakeDetector([[]])

    with pytest.raises(PracticalDemoError, match="checkpoint does not exist"):
        process_images(
            input_dir,
            tmp_path / "output",
            _config(tmp_path, checkpoint_exists=False),
            detector_factory=factory,
        )
    assert factory_called is False
