from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from bottle_cap_vision import camera_demo
from bottle_cap_vision.config import AppConfig
from bottle_cap_vision.models import RunSummary
from bottle_cap_vision.pipeline import BottleCapPipeline
from bottle_cap_vision.runtime import run
from bottle_cap_vision.sinks import DisplaySink
from bottle_cap_vision.sources import CameraSource


def _frames(count: int) -> list[np.ndarray]:
    return [np.full((80, 120, 3), index, dtype=np.uint8) for index in range(count)]


class FakeCapture:
    def __init__(
        self,
        frames=(),
        *,
        opened: bool = True,
        width: float = 120,
        height: float = 80,
        fps: float = 25.0,
    ):
        self.frames = list(frames)
        self.opened = opened
        self.width = width
        self.height = height
        self.fps = fps
        self.released = False
        self.reads = 0

    def isOpened(self):
        return self.opened

    def get(self, key):
        return {
            cv2.CAP_PROP_FRAME_WIDTH: self.width,
            cv2.CAP_PROP_FRAME_HEIGHT: self.height,
            cv2.CAP_PROP_FPS: self.fps,
        }[key]

    def read(self):
        self.reads += 1
        return (True, self.frames.pop(0)) if self.frames else (False, None)

    def release(self):
        self.released = True


class EmptyDetector:
    def __init__(self):
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return []


def test_camera_source_opens_requested_index_and_yields_sequential_frames(monkeypatch):
    frames = _frames(2)
    capture = FakeCapture(frames, fps=29.97)
    requested = []
    monkeypatch.setattr(cv2, "VideoCapture", lambda index: (requested.append(index) or capture))
    source = CameraSource(3)

    source.open()
    packets = iter(source)
    first = next(packets)
    second = next(packets)
    with pytest.raises(RuntimeError, match="Camera read failed at frame index 2"):
        next(packets)
    source.close()

    assert requested == [3]
    assert source.metadata.width == 120
    assert source.metadata.height == 80
    assert source.metadata.fps == pytest.approx(29.97)
    assert [first.frame_index, second.frame_index] == [0, 1]
    assert first.frame is frames[0]
    assert second.frame is frames[1]
    assert capture.released


@pytest.mark.parametrize("fps", [0.0, -1.0, float("nan"), float("inf")])
def test_camera_source_does_not_invent_fps(monkeypatch, fps):
    capture = FakeCapture(fps=fps)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _index: capture)
    source = CameraSource()

    source.open()
    source.close()

    assert source.metadata.fps is None
    assert capture.released


@pytest.mark.parametrize(
    ("capture", "message"),
    [
        (FakeCapture(opened=False), "Cannot open camera index 7"),
        (FakeCapture(width=0), "invalid dimensions"),
        (FakeCapture(height=float("nan")), "invalid dimensions"),
    ],
)
def test_camera_source_open_failures_are_clear_and_release(monkeypatch, capture, message):
    monkeypatch.setattr(cv2, "VideoCapture", lambda _index: capture)

    with pytest.raises(RuntimeError, match=message):
        CameraSource(7).open()

    assert capture.released


@pytest.mark.parametrize(
    ("read_result", "message"),
    [
        ((False, None), "Camera read failed"),
        ((True, None), "empty frame"),
        ((True, np.empty((0, 0, 3), dtype=np.uint8)), "empty frame"),
    ],
)
def test_camera_source_rejects_failed_or_empty_reads(monkeypatch, read_result, message):
    capture = FakeCapture()
    capture.read = lambda: read_result
    monkeypatch.setattr(cv2, "VideoCapture", lambda _index: capture)
    source = CameraSource()
    source.open()

    with pytest.raises(RuntimeError, match=message):
        next(iter(source))
    source.close()

    assert capture.released


def test_camera_capture_released_after_max_frames(monkeypatch):
    capture = FakeCapture(_frames(3))
    monkeypatch.setattr(cv2, "VideoCapture", lambda _index: capture)
    detector = EmptyDetector()
    cfg = AppConfig.default()

    summary = run(
        CameraSource(),
        BottleCapPipeline(cfg, detector=detector),
        [],
        max_frames=2,
    )

    assert summary.frames_processed == detector.calls == 2
    assert capture.reads == 2
    assert capture.released


@pytest.mark.parametrize("exit_key", [ord("q"), 27])
def test_display_exit_releases_camera_and_closes_window(monkeypatch, exit_key):
    capture = FakeCapture(_frames(3))
    calls = {"destroy": 0}
    monkeypatch.setattr(cv2, "VideoCapture", lambda _index: capture)
    monkeypatch.setattr(cv2, "imshow", lambda *_args: None)
    monkeypatch.setattr(cv2, "waitKey", lambda _delay: exit_key)
    monkeypatch.setattr(
        cv2,
        "destroyWindow",
        lambda _name: calls.__setitem__("destroy", calls["destroy"] + 1),
    )
    cfg = AppConfig.default()

    summary = run(
        CameraSource(),
        BottleCapPipeline(cfg, detector=EmptyDetector()),
        [DisplaySink(cfg)],
    )

    assert summary.frames_processed == 1
    assert capture.released
    assert calls["destroy"] == 1


def test_processing_exception_releases_camera(monkeypatch):
    capture = FakeCapture(_frames(1))
    monkeypatch.setattr(cv2, "VideoCapture", lambda _index: capture)

    class FailingCore:
        def process_frame(self, frame, frame_index=0):
            raise RuntimeError("inference failed")

    with pytest.raises(RuntimeError, match="inference failed"):
        run(CameraSource(), FailingCore(), [])

    assert capture.released


def test_camera_composition_loads_detector_once_and_detects_once_per_frame(monkeypatch):
    capture = FakeCapture(_frames(3))
    monkeypatch.setattr(cv2, "VideoCapture", lambda _index: capture)
    monkeypatch.setattr(cv2, "imshow", lambda *_args: None)
    monkeypatch.setattr(cv2, "waitKey", lambda _delay: -1)
    monkeypatch.setattr(cv2, "destroyWindow", lambda _name: None)
    cfg = AppConfig.default()
    cfg.detector.backend = "learned"
    checkpoint = Path("full.pt")
    monkeypatch.setattr(
        camera_demo,
        "load_learned_config",
        lambda _path: (cfg, Path("config.json"), checkpoint, Path("metadata.json")),
    )
    lifecycle = {"loads": 0, "calls": 0}

    class Detector:
        def detect(self, frame):
            lifecycle["calls"] += 1
            return []

    def factory(_config):
        lifecycle["loads"] += 1
        return Detector()

    summary = camera_demo.run_camera(max_frames=3, detector_factory=factory)

    assert summary.frames_processed == 3
    assert lifecycle == {"loads": 1, "calls": 3}
    assert capture.released


def test_camera_detector_initialization_failure_is_clear_and_does_not_open_camera(monkeypatch):
    cfg = AppConfig.default()
    cfg.detector.backend = "learned"
    checkpoint = Path("full.pt")
    monkeypatch.setattr(
        camera_demo,
        "load_learned_config",
        lambda _path: (cfg, Path("config.json"), checkpoint, Path("metadata.json")),
    )
    monkeypatch.setattr(
        cv2,
        "VideoCapture",
        lambda _index: pytest.fail("camera opened before detector initialization completed"),
    )

    def failed_factory(_config):
        raise ValueError("metadata mismatch")

    with pytest.raises(RuntimeError, match="compatibility/load validation.*metadata mismatch"):
        camera_demo.run_camera(detector_factory=failed_factory)


def test_camera_cli_defaults_instructions_and_summary(monkeypatch, capsys):
    from scripts import demo_camera

    captured_args = {}
    summary = RunSummary(
        frames_processed=2,
        processing_seconds=0.5,
        average_processing_fps=4.0,
        total_count=1,
        count_by_color={"Red": 1},
        count_events=[],
    )

    def fake_run_camera(**kwargs):
        captured_args.update(kwargs)
        return summary

    monkeypatch.setattr(demo_camera, "run_camera", fake_run_camera)

    assert demo_camera.main(["--max-frames", "2"]) == 0
    output = capsys.readouterr().out
    assert captured_args == {
        "camera_index": 0,
        "config_path": str(camera_demo.DEFAULT_CONFIG),
        "max_frames": 2,
    }
    assert "Camera realtime started" in output
    assert "Q / Esc: Exit" in output
    assert "frames_processed: 2" in output
    assert "average_processing_fps: 4.00" in output
    assert "total_count: 1" in output
    assert "count_by_color: {'Red': 1}" in output


def test_camera_cli_rejects_non_positive_max_frames():
    from scripts import demo_camera

    with pytest.raises(SystemExit):
        demo_camera.parse_args(["--max-frames", "0"])
