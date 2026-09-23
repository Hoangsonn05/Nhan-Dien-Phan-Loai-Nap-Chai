from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

from bottle_cap_vision.config import AppConfig
from bottle_cap_vision.detector import create_detector
from bottle_cap_vision.inference import LearnedDetector
from bottle_cap_vision.models import Detection, SourceMetadata
from bottle_cap_vision.pipeline import BottleCapPipeline
from bottle_cap_vision.runtime import run
from bottle_cap_vision.sinks import CsvSink, JsonSink, VideoSink
from bottle_cap_vision.sources import VideoSource


def _make_frames(count: int, *, width: int = 160, height: int = 120) -> list[np.ndarray]:
    frames = []
    for index in range(count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.circle(frame, (30 + index * 7, 25 + index * 6), 10, (0, 0, 255), -1)
        frames.append(frame)
    return frames


def _write_video(path: Path, frames: list[np.ndarray], *, fps: float = 12.0) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    assert writer.isOpened(), "local mp4v encoder is required for the video integration test"
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def _decode_video(path: Path) -> tuple[list[np.ndarray], tuple[int, int], float]:
    capture = cv2.VideoCapture(str(path))
    assert capture.isOpened()
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            assert frame is not None and frame.size > 0
            frames.append(frame)
    finally:
        capture.release()
    return frames, (width, height), fps


@pytest.fixture
def encoded_video(tmp_path: Path) -> Path:
    path = tmp_path / "input.mp4"
    _write_video(path, _make_frames(6))
    return path


@pytest.mark.parametrize("max_frames", [1, 4])
def test_hsv_encoded_video_outputs_are_frame_synchronized(
    encoded_video: Path,
    tmp_path: Path,
    max_frames: int,
):
    source = VideoSource(encoded_video)
    source.open()
    assert source.metadata == SourceMetadata(fps=12.0, width=160, height=120)
    assert [packet.frame_index for packet in source] == list(range(6))
    source.close()

    cfg = AppConfig.default()
    output_video = tmp_path / f"hsv-{max_frames}.mp4"
    csv_path = tmp_path / f"hsv-{max_frames}.csv"
    json_path = tmp_path / f"hsv-{max_frames}.json"
    summary = run(
        VideoSource(encoded_video),
        BottleCapPipeline(cfg, detector=create_detector(cfg)),
        [
            VideoSink(output_video, cfg),
            CsvSink(csv_path),
            JsonSink(json_path, source=str(encoded_video)),
        ],
        max_frames=max_frames,
    )

    decoded, dimensions, output_fps = _decode_video(output_video)
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert summary.frames_processed == max_frames
    assert summary.processing_seconds >= 0
    if summary.processing_seconds > 0:
        assert summary.average_processing_fps == pytest.approx(
            max_frames / summary.processing_seconds
        )
    assert len(decoded) == max_frames
    assert dimensions == (160, 120)
    assert output_fps == pytest.approx(12.0)
    assert [int(row["frame"]) for row in rows] == list(range(max_frames))
    assert payload["frames_processed"] == max_frames
    assert int(rows[-1]["total"]) == payload["total_count"] == summary.total_count


def test_stop_request_keeps_partial_video_csv_json_and_summary_in_sync(
    encoded_video: Path,
    tmp_path: Path,
):
    class StopSink:
        def __init__(self):
            self.writes = 0
            self.closed = False

        def open(self, metadata):
            pass

        def write(self, packet, result, metrics):
            self.writes += 1
            return self.writes < 2

        def close(self, summary=None):
            self.closed = True

    cfg = AppConfig.default()
    stop_sink = StopSink()
    output_video = tmp_path / "stopped.mp4"
    csv_path = tmp_path / "stopped.csv"
    json_path = tmp_path / "stopped.json"
    summary = run(
        VideoSource(encoded_video),
        BottleCapPipeline(cfg),
        [stop_sink, VideoSink(output_video, cfg), CsvSink(csv_path), JsonSink(json_path)],
    )

    decoded, _, _ = _decode_video(output_video)
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert stop_sink.closed
    assert stop_sink.writes == summary.frames_processed == len(decoded) == len(rows) == 2
    assert payload["frames_processed"] == 2


def test_stateful_pipeline_crosses_once_and_runs_are_isolated(
    encoded_video: Path,
    tmp_path: Path,
):
    class SequenceDetector:
        def __init__(self, positions, colors):
            self.positions = iter(positions)
            self.colors = iter(colors)

        def detect(self, frame):
            y = next(self.positions)
            color = next(self.colors)
            return [Detection(color, (50, y), (45, y - 5, 10, 10), confidence=0.9)]

    class SnapshotSink:
        def __init__(self):
            self.snapshots = []

        def open(self, metadata):
            pass

        def write(self, packet, result, metrics):
            self.snapshots.append(
                ([track.track_id for track in result.tracks],
                 [track.color for track in result.tracks],
                 [event.track_id for event in result.events])
            )
            return True

        def close(self, summary=None):
            pass

    cfg = AppConfig.default()
    cfg.tracking.min_confirmed_frames = 1
    cfg.tracking.max_distance = 100
    cfg.color_stabilization.window_size = 3
    cfg.color_stabilization.min_observations = 2
    cfg.counting.line_position = 0.5
    cfg.counting.deadband = 5
    positions = [45, 54, 58, 62, 66, 75]
    colors = ["Red", "Red", "Pink", "Red", "Red", "Red"]
    snapshot = SnapshotSink()
    csv_path = tmp_path / "crossing.csv"
    json_path = tmp_path / "crossing.json"
    first_summary = run(
        VideoSource(encoded_video),
        BottleCapPipeline(cfg, detector=SequenceDetector(positions, colors)),
        [snapshot, CsvSink(csv_path), JsonSink(json_path)],
    )

    assert [ids for ids, _, _ in snapshot.snapshots] == [[1]] * 6
    assert [stable for _, stable, _ in snapshot.snapshots] == [["Red"]] * 6
    assert [events for _, _, events in snapshot.snapshots] == [[], [], [], [], [1], []]
    assert first_summary.total_count == 1
    assert [(event.track_id, event.color, event.frame_index) for event in first_summary.count_events] == [
        (1, "Red", 4)
    ]

    second_summary = run(
        VideoSource(encoded_video),
        BottleCapPipeline(cfg, detector=SequenceDetector(positions, colors)),
        [],
    )
    assert second_summary.total_count == 1
    assert second_summary.count_events[0].track_id == 1


def test_learned_checkpoint_runs_sequential_frames_through_same_runtime(
    encoded_video: Path,
    tmp_path: Path,
):
    root = Path(__file__).resolve().parents[1]
    checkpoint_dir = root / "outputs" / "experiments" / "p9_smoke_001" / "checkpoints"
    cfg = AppConfig.default()
    cfg.detector.backend = "learned"
    cfg.detector.checkpoint = str(checkpoint_dir / "best.pt")
    cfg.detector.metadata = str(checkpoint_dir / "checkpoint_metadata.json")
    cfg.detector.device = "cpu"
    detector = create_detector(cfg)
    assert isinstance(detector, LearnedDetector)

    class CountingDetector:
        def __init__(self, delegate):
            self.delegate = delegate
            self.calls = 0

        def detect(self, frame):
            self.calls += 1
            return self.delegate.detect(frame)

    counting_detector = CountingDetector(detector)
    output_video = tmp_path / "learned.mp4"
    csv_path = tmp_path / "learned.csv"
    json_path = tmp_path / "learned.json"
    summary = run(
        VideoSource(encoded_video),
        BottleCapPipeline(cfg, detector=counting_detector),
        [VideoSink(output_video, cfg), CsvSink(csv_path), JsonSink(json_path)],
        max_frames=3,
    )

    decoded, _, _ = _decode_video(output_video)
    assert counting_detector.calls == summary.frames_processed == len(decoded) == 3
    assert json.loads(json_path.read_text(encoding="utf-8"))["frames_processed"] == 3


def test_learned_detector_is_constructed_once_and_called_once_per_frame(
    encoded_video: Path,
    monkeypatch,
):
    lifecycle = {"loads": 0, "calls": 0}

    class SpyLearnedDetector:
        def __init__(self, **kwargs):
            lifecycle["loads"] += 1

        def detect(self, frame):
            lifecycle["calls"] += 1
            return []

    monkeypatch.setattr("bottle_cap_vision.inference.LearnedDetector", SpyLearnedDetector)
    cfg = AppConfig.default()
    cfg.detector.backend = "learned"
    cfg.detector.checkpoint = "checkpoint.pt"
    cfg.detector.metadata = "checkpoint.json"
    detector = create_detector(cfg)

    summary = run(
        VideoSource(encoded_video),
        BottleCapPipeline(cfg, detector=detector),
        [],
        max_frames=4,
    )

    assert summary.frames_processed == 4
    assert lifecycle == {"loads": 1, "calls": 4}


def test_failure_cleanup_preserves_original_processing_error(encoded_video: Path):
    class TrackedSource(VideoSource):
        def __init__(self, path):
            super().__init__(path)
            self.close_calls = 0

        def close(self):
            self.close_calls += 1
            super().close()

    class FailingSink:
        def __init__(self):
            self.closed = False

        def open(self, metadata):
            pass

        def write(self, packet, result, metrics):
            raise ValueError("frame write failed")

        def close(self, summary=None):
            self.closed = True
            raise RuntimeError("sink cleanup failed")

    source = TrackedSource(encoded_video)
    sink = FailingSink()
    with pytest.raises(ValueError, match="frame write failed") as raised:
        run(source, BottleCapPipeline(AppConfig.default()), [sink])

    assert source.close_calls == 1
    assert sink.closed
    assert any("sink cleanup failed" in note for note in getattr(raised.value, "__notes__", []))


def test_video_sink_rejects_missing_fps_and_unopenable_writer(monkeypatch, tmp_path: Path):
    sink = VideoSink(tmp_path / "missing-fps.mp4", AppConfig.default())
    with pytest.raises(RuntimeError, match="FPS"):
        sink.open(SourceMetadata(width=160, height=120))

    class FailedWriter:
        def __init__(self):
            self.released = False

        def isOpened(self):
            return False

        def release(self):
            self.released = True

    failed = FailedWriter()
    monkeypatch.setattr(cv2, "VideoWriter", lambda *args: failed)
    sink = VideoSink(tmp_path / "cannot-open.mp4", AppConfig.default())
    with pytest.raises(RuntimeError, match="Cannot write output video"):
        sink.open(SourceMetadata(fps=12.0, width=160, height=120))
    assert failed.released


def test_no_display_cli_uses_the_same_video_runtime_without_gui_calls(
    encoded_video: Path,
    tmp_path: Path,
    monkeypatch,
):
    import main as cli

    monkeypatch.setattr(cv2, "imshow", lambda *args: pytest.fail("imshow called"))
    monkeypatch.setattr(cv2, "waitKey", lambda *args: pytest.fail("waitKey called"))
    monkeypatch.setattr(cv2, "destroyWindow", lambda *args: pytest.fail("destroyWindow called"))
    output_video = tmp_path / "cli.mp4"
    csv_path = tmp_path / "cli.csv"
    json_path = tmp_path / "cli.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--source",
            str(encoded_video),
            "--config",
            "configs/config.json",
            "--output",
            str(output_video),
            "--csv",
            str(csv_path),
            "--json",
            str(json_path),
            "--no-display",
            "--max-frames",
            "2",
        ],
    )

    assert cli.main() == 0
    assert len(_decode_video(output_video)[0]) == 2
    assert json.loads(json_path.read_text(encoding="utf-8"))["frames_processed"] == 2
