from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from bottle_cap_vision import sinks, video_demo
from bottle_cap_vision.config import AppConfig
from bottle_cap_vision.models import FramePacket, FrameResult, RunSummary, RuntimeMetrics, SourceMetadata
from bottle_cap_vision.practical_demo import PracticalDemoError
from bottle_cap_vision.sinks import DisplaySink


def _frame(value: int = 0) -> np.ndarray:
    return np.full((80, 120, 3), value, dtype=np.uint8)


def _result(index: int = 0) -> FrameResult:
    return FrameResult(index, [], [], [], 0, {})


class FakeVideoSource:
    instances = []

    def __init__(self, path):
        self.path = Path(path)
        self.metadata = SourceMetadata(fps=20.0, width=120, height=80)
        self.closed = False
        self.frames = [_frame(index) for index in range(4)]
        type(self).instances.append(self)

    def open(self):
        pass

    def __iter__(self):
        for index, frame in enumerate(self.frames):
            yield FramePacket(frame, index)

    def close(self):
        self.closed = True


def _patch_learned_config(monkeypatch, tmp_path):
    cfg = AppConfig.default()
    cfg.detector.backend = "learned"
    checkpoint = tmp_path / "full.pt"
    checkpoint.touch()
    monkeypatch.setattr(
        video_demo,
        "load_learned_config",
        lambda _path: (cfg, tmp_path / "config.json", checkpoint, tmp_path / "metadata.json"),
    )
    return cfg


def test_video_composition_creates_detector_and_pipeline_once_and_processes_each_frame(
    monkeypatch, tmp_path, capsys
):
    video_path = tmp_path / "input.mp4"
    video_path.touch()
    _patch_learned_config(monkeypatch, tmp_path)
    FakeVideoSource.instances.clear()
    monkeypatch.setattr(video_demo, "VideoSource", FakeVideoSource)
    monkeypatch.setattr(cv2, "imshow", lambda *_args: None)
    monkeypatch.setattr(cv2, "waitKey", lambda _delay: -1)
    monkeypatch.setattr(cv2, "destroyWindow", lambda _name: None)
    lifecycle = {"loads": 0, "pipelines": 0, "detects": 0}

    class Detector:
        def detect(self, frame):
            lifecycle["detects"] += 1
            return []

    def factory(_config):
        lifecycle["loads"] += 1
        return Detector()

    real_pipeline = video_demo.BottleCapPipeline

    def pipeline_factory(config, detector):
        lifecycle["pipelines"] += 1
        return real_pipeline(config, detector)

    monkeypatch.setattr(video_demo, "BottleCapPipeline", pipeline_factory)

    summary = video_demo.run_video(video_path, max_frames=3, detector_factory=factory)

    assert summary.frames_processed == 3
    assert lifecycle == {"loads": 1, "pipelines": 1, "detects": 3}
    assert FakeVideoSource.instances[0].closed
    assert "Source FPS: 20.00" in capsys.readouterr().out


def test_realtime_display_waits_remaining_fast_frame_budget(monkeypatch):
    times = iter([0.0, 0.020, 0.050])
    waits = []
    monkeypatch.setattr(sinks, "perf_counter", lambda: next(times))
    monkeypatch.setattr(sinks, "annotate_frame", lambda frame, *_args: frame)
    monkeypatch.setattr(cv2, "imshow", lambda *_args: None)
    monkeypatch.setattr(cv2, "waitKey", lambda delay: (waits.append(delay) or -1))

    sink = DisplaySink(AppConfig.default(), synchronize_to_source_fps=True)
    sink.open(SourceMetadata(fps=20.0, width=120, height=80))
    assert sink.write(FramePacket(_frame(), 0), _result(), RuntimeMetrics(50.0))

    assert waits == [30]


def test_realtime_display_does_not_add_sync_delay_when_processing_is_slow(monkeypatch):
    times = iter([0.0, 0.070, 0.071])
    waits = []
    monkeypatch.setattr(sinks, "perf_counter", lambda: next(times))
    monkeypatch.setattr(sinks, "annotate_frame", lambda frame, *_args: frame)
    monkeypatch.setattr(cv2, "imshow", lambda *_args: None)
    monkeypatch.setattr(cv2, "waitKey", lambda delay: (waits.append(delay) or -1))

    sink = DisplaySink(AppConfig.default(), synchronize_to_source_fps=True)
    sink.open(SourceMetadata(fps=20.0, width=120, height=80))
    assert sink.write(FramePacket(_frame(), 0), _result(), RuntimeMetrics(14.0))

    assert waits == [1]


@pytest.mark.parametrize("fps", [None, 0.0, -1.0, float("nan"), float("inf")])
def test_realtime_display_does_not_invent_missing_or_invalid_fps(monkeypatch, fps):
    waits = []
    monkeypatch.setattr(sinks, "annotate_frame", lambda frame, *_args: frame)
    monkeypatch.setattr(cv2, "imshow", lambda *_args: None)
    monkeypatch.setattr(cv2, "waitKey", lambda delay: (waits.append(delay) or -1))

    sink = DisplaySink(AppConfig.default(), synchronize_to_source_fps=True)
    sink.open(SourceMetadata(fps=fps, width=120, height=80))
    assert sink.write(FramePacket(_frame(), 0), _result(), RuntimeMetrics(10.0))

    assert sink.source_fps is None
    assert waits == [1]


def test_video_composition_reports_missing_source_timing_without_faking_fps(
    monkeypatch, tmp_path, capsys
):
    video_path = tmp_path / "input.mp4"
    video_path.touch()
    _patch_learned_config(monkeypatch, tmp_path)

    class NoFpsVideoSource(FakeVideoSource):
        def __init__(self, path):
            super().__init__(path)
            self.metadata = SourceMetadata(width=120, height=80)

    monkeypatch.setattr(video_demo, "VideoSource", NoFpsVideoSource)
    monkeypatch.setattr(cv2, "imshow", lambda *_args: None)
    monkeypatch.setattr(cv2, "waitKey", lambda _delay: -1)
    monkeypatch.setattr(cv2, "destroyWindow", lambda _name: None)

    summary = video_demo.run_video(
        video_path,
        max_frames=1,
        detector_factory=lambda _cfg: type(
            "Detector", (), {"detect": lambda self, frame: []}
        )(),
    )

    assert summary.frames_processed == 1
    assert "Source FPS: N/A (playback follows processing speed)" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("key", "expected_frames"),
    [(ord("q"), 1), (ord("Q"), 1), (27, 1), (-1, 4)],
)
def test_video_user_stop_or_eof_closes_source_and_window(
    monkeypatch, tmp_path, key, expected_frames
):
    video_path = tmp_path / "input.mp4"
    video_path.touch()
    _patch_learned_config(monkeypatch, tmp_path)
    FakeVideoSource.instances.clear()
    monkeypatch.setattr(video_demo, "VideoSource", FakeVideoSource)
    calls = {"destroy": 0}
    monkeypatch.setattr(cv2, "imshow", lambda *_args: None)
    monkeypatch.setattr(cv2, "waitKey", lambda _delay: key)
    monkeypatch.setattr(
        cv2,
        "destroyWindow",
        lambda _name: calls.__setitem__("destroy", calls["destroy"] + 1),
    )

    summary = video_demo.run_video(video_path, detector_factory=lambda _cfg: type(
        "Detector", (), {"detect": lambda self, frame: []}
    )())

    assert summary.frames_processed == expected_frames
    assert FakeVideoSource.instances[0].closed
    assert calls["destroy"] == 1


def test_video_processing_failure_is_not_swallowed_and_still_closes_source(monkeypatch, tmp_path):
    video_path = tmp_path / "input.mp4"
    video_path.touch()
    _patch_learned_config(monkeypatch, tmp_path)
    FakeVideoSource.instances.clear()
    monkeypatch.setattr(video_demo, "VideoSource", FakeVideoSource)
    monkeypatch.setattr(cv2, "imshow", lambda *_args: None)
    monkeypatch.setattr(cv2, "waitKey", lambda _delay: -1)

    class FailingDetector:
        def detect(self, frame):
            raise RuntimeError("inference failed")

    with pytest.raises(RuntimeError, match="inference failed"):
        video_demo.run_video(video_path, detector_factory=lambda _cfg: FailingDetector())

    assert FakeVideoSource.instances[0].closed


@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_video_input_must_be_an_existing_file(tmp_path, kind):
    path = tmp_path / "input.mp4"
    if kind == "directory":
        path.mkdir()
    with pytest.raises(PracticalDemoError, match="does not exist|not a file"):
        video_demo.run_video(path)


def test_video_cli_defaults_instructions_and_summary(monkeypatch, capsys, tmp_path):
    from scripts import demo_video

    video_path = tmp_path / "input.mp4"
    summary = RunSummary(2, 0.5, 4.0, 1, {"Red": 1}, [])
    captured = {}

    def fake_run_video(path, **kwargs):
        captured["path"] = path
        captured.update(kwargs)
        return summary

    monkeypatch.setattr(demo_video, "run_video", fake_run_video)

    assert demo_video.main([str(video_path), "--max-frames", "2"]) == 0
    output = capsys.readouterr().out
    assert captured == {
        "path": str(video_path),
        "config_path": str(video_demo.DEFAULT_CONFIG),
        "max_frames": 2,
    }
    assert "Video practical started" in output
    assert "Q / Esc: Exit" in output
    assert "frames_processed: 2" in output
    assert "average_processing_fps: 4.00" in output
    assert "total_count: 1" in output
    assert "count_by_color: {'Red': 1}" in output


def test_video_cli_rejects_non_positive_max_frames():
    from scripts import demo_video

    with pytest.raises(SystemExit):
        demo_video.parse_args(["input.mp4", "--max-frames", "0"])
