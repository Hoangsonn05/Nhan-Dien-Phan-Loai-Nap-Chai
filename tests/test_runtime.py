from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

from bottle_cap_vision.config import AppConfig
from bottle_cap_vision.models import Detection, FramePacket, SourceMetadata
from bottle_cap_vision.pipeline import BottleCapPipeline
from bottle_cap_vision.runtime import run
from bottle_cap_vision.sinks import CsvSink, DisplaySink, ImageSink, JsonSink, VideoSink
from bottle_cap_vision.sources import ImageSource, VideoSource


class FakeSource:
    def __init__(self, frames):
        self.frames = frames
        self.opened = False
        self.closed = False
        self.metadata = SourceMetadata(fps=25.0, width=120, height=80)

    def open(self):
        self.opened = True

    def __iter__(self):
        for index, frame in enumerate(self.frames):
            yield FramePacket(frame=frame, frame_index=index)

    def close(self):
        self.closed = True


class MemorySink:
    def __init__(self, *, stop_after=None, fail=False):
        self.stop_after = stop_after
        self.fail = fail
        self.items = []
        self.opened = False
        self.closed = False
        self.summary = None

    def open(self, metadata):
        self.opened = True

    def write(self, packet, result, metrics):
        self.items.append((packet, result, metrics))
        if self.fail:
            raise RuntimeError("sink failed")
        return self.stop_after is None or len(self.items) < self.stop_after

    def close(self, summary=None):
        self.closed = True
        self.summary = summary


def _blank_frames(count):
    return [np.zeros((80, 120, 3), dtype=np.uint8) for _ in range(count)]


def test_image_source_yields_one_indexed_frame_and_rejects_invalid_path(tmp_path):
    path = tmp_path / "input.png"
    assert cv2.imwrite(str(path), _blank_frames(1)[0])

    source = ImageSource(path)
    source.open()
    packets = list(source)
    source.close()

    assert len(packets) == 1
    assert packets[0].frame_index == 0
    assert packets[0].frame.shape == (80, 120, 3)
    with pytest.raises(RuntimeError, match="Cannot read image"):
        ImageSource(tmp_path / "missing.png").open()


def test_video_source_yields_sequential_frames_and_releases(monkeypatch):
    frames = _blank_frames(2)

    class FakeCapture:
        def __init__(self, path):
            self.remaining = list(frames)
            self.released = False

        def isOpened(self):
            return True

        def get(self, key):
            return {
                cv2.CAP_PROP_FPS: 25.0,
                cv2.CAP_PROP_FRAME_WIDTH: 120,
                cv2.CAP_PROP_FRAME_HEIGHT: 80,
                cv2.CAP_PROP_FRAME_COUNT: 2,
            }[key]

        def read(self):
            return (True, self.remaining.pop(0)) if self.remaining else (False, None)

        def release(self):
            self.released = True

    capture = FakeCapture("unused")
    monkeypatch.setattr(cv2, "VideoCapture", lambda path: capture)
    source = VideoSource("movie.mp4")

    source.open()
    packets = list(source)
    source.close()

    assert [packet.frame_index for packet in packets] == [0, 1]
    assert source.metadata.fps == 25.0
    assert capture.released is True


def test_fake_source_core_memory_sinks_and_max_frames_are_independent():
    source = FakeSource(_blank_frames(3))
    sink_a = MemorySink()
    sink_b = MemorySink()

    summary = run(
        source,
        BottleCapPipeline(AppConfig.default()),
        [sink_a, sink_b],
        max_frames=2,
    )

    assert summary.frames_processed == 2
    assert [item[1].frame_index for item in sink_a.items] == [0, 1]
    assert [item[1].frame_index for item in sink_b.items] == [0, 1]
    assert source.closed and sink_a.closed and sink_b.closed


@pytest.mark.parametrize("max_frames", [0, -1, False])
def test_runner_rejects_non_positive_max_frames(max_frames):
    with pytest.raises(ValueError, match="positive integer"):
        run(
            FakeSource(_blank_frames(1)),
            BottleCapPipeline(AppConfig.default()),
            [MemorySink()],
            max_frames=max_frames,
        )


def test_processing_fps_is_frames_divided_by_total_core_time(monkeypatch):
    import bottle_cap_vision.runtime as runtime

    times = iter([0.0, 1.0, 1.0, 4.0])
    monkeypatch.setattr(runtime, "perf_counter", lambda: next(times))
    sink = MemorySink()

    summary = runtime.run(
        FakeSource(_blank_frames(2)),
        BottleCapPipeline(AppConfig.default()),
        [sink],
    )

    assert summary.processing_seconds == 4.0
    assert summary.average_processing_fps == 0.5
    assert [item[2].processing_fps for item in sink.items] == [1.0, 0.5]


def test_video_source_reports_midstream_decode_failure_and_runner_releases(monkeypatch):
    frame = _blank_frames(1)[0]

    class TruncatedCapture:
        def __init__(self):
            self.reads = 0
            self.released = False

        def isOpened(self):
            return True

        def get(self, key):
            return {
                cv2.CAP_PROP_FPS: 25.0,
                cv2.CAP_PROP_FRAME_WIDTH: 120,
                cv2.CAP_PROP_FRAME_HEIGHT: 80,
                cv2.CAP_PROP_FRAME_COUNT: 3,
            }[key]

        def read(self):
            self.reads += 1
            return (True, frame) if self.reads == 1 else (False, None)

        def release(self):
            self.released = True

    capture = TruncatedCapture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda path: capture)

    with pytest.raises(RuntimeError, match="decoded=1, expected=3"):
        run(VideoSource("truncated.mp4"), BottleCapPipeline(AppConfig.default()), [])
    assert capture.released


def test_runner_closes_resources_on_stop_and_exception():
    stopped_source = FakeSource(_blank_frames(3))
    stopped_sink = MemorySink(stop_after=1)
    summary = run(stopped_source, BottleCapPipeline(AppConfig.default()), [stopped_sink])
    assert summary.frames_processed == 1
    assert stopped_source.closed and stopped_sink.closed

    failed_source = FakeSource(_blank_frames(1))
    failed_sink = MemorySink(fail=True)
    with pytest.raises(RuntimeError, match="sink failed"):
        run(failed_source, BottleCapPipeline(AppConfig.default()), [failed_sink])
    assert failed_source.closed and failed_sink.closed
    assert failed_sink.summary is None


def test_image_csv_and_json_outputs_are_sink_owned(tmp_path):
    source_path = tmp_path / "input.png"
    image_path = tmp_path / "nested" / "annotated.png"
    csv_path = tmp_path / "nested" / "frames.csv"
    json_path = tmp_path / "nested" / "result.json"
    assert cv2.imwrite(str(source_path), _blank_frames(1)[0])
    cfg = AppConfig.default()

    summary = run(
        ImageSource(source_path),
        BottleCapPipeline(cfg),
        [ImageSink(image_path, cfg), CsvSink(csv_path), JsonSink(json_path, source=str(source_path))],
    )

    assert image_path.exists()
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [{"frame": "0", "detections": "0", "tracks": "0", "new_events": "[]", "total": "0"}]
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload == {
        "source": str(source_path),
        "frames_processed": 1,
        "total_count": 0,
        "count_by_color": {},
        "count_events": [],
    }
    assert summary.frames_processed == 1


def test_no_display_path_never_calls_gui(monkeypatch):
    monkeypatch.setattr(cv2, "imshow", lambda *args: pytest.fail("imshow called"))
    monkeypatch.setattr(cv2, "waitKey", lambda *args: pytest.fail("waitKey called"))

    run(FakeSource(_blank_frames(1)), BottleCapPipeline(AppConfig.default()), [MemorySink()])


def test_video_sink_writes_frames_and_releases_writer(monkeypatch, tmp_path):
    class FakeWriter:
        def __init__(self):
            self.frames = []
            self.released = False

        def isOpened(self):
            return True

        def write(self, frame):
            self.frames.append(frame)

        def release(self):
            self.released = True

    writer = FakeWriter()
    monkeypatch.setattr(cv2, "VideoWriter_fourcc", lambda *args: 123)
    monkeypatch.setattr(cv2, "VideoWriter", lambda *args: writer)

    summary = run(
        FakeSource(_blank_frames(2)),
        BottleCapPipeline(AppConfig.default()),
        [VideoSink(tmp_path / "output.mp4", AppConfig.default())],
    )

    assert summary.frames_processed == 2
    assert len(writer.frames) == 2
    assert writer.released is True


def test_display_sink_requests_stop_and_closes_window(monkeypatch):
    calls = {"imshow": 0, "waitKey": 0, "destroyWindow": 0}
    monkeypatch.setattr(cv2, "imshow", lambda *args: calls.__setitem__("imshow", calls["imshow"] + 1))
    monkeypatch.setattr(cv2, "waitKey", lambda *args: (calls.__setitem__("waitKey", calls["waitKey"] + 1) or ord("q")))
    monkeypatch.setattr(cv2, "destroyWindow", lambda *args: calls.__setitem__("destroyWindow", calls["destroyWindow"] + 1))

    summary = run(
        FakeSource(_blank_frames(3)),
        BottleCapPipeline(AppConfig.default()),
        [DisplaySink(AppConfig.default()), MemorySink()],
    )

    assert summary.frames_processed == 1
    assert calls == {"imshow": 1, "waitKey": 1, "destroyWindow": 1}


def test_pipeline_module_has_no_runtime_io_calls():
    source = (Path(__file__).parents[1] / "bottle_cap_vision" / "pipeline.py").read_text(encoding="utf-8")
    forbidden = (
        "cv2.imread", "cv2.VideoCapture", "cv2.VideoWriter", "cv2.imwrite",
        "cv2.imshow", "cv2.waitKey", "csv.DictWriter", "json.dumps",
        ".write_text(", ".mkdir(",
    )
    assert not [name for name in forbidden if name in source]


def test_image_cli_composes_source_core_and_output_sinks(monkeypatch, tmp_path):
    import main as cli

    source = tmp_path / "input.png"
    output = tmp_path / "output" / "annotated.png"
    csv_output = tmp_path / "output" / "frames.csv"
    json_output = tmp_path / "output" / "result.json"
    assert cv2.imwrite(str(source), _blank_frames(1)[0])
    monkeypatch.setattr(sys, "argv", [
        "main.py",
        "--source", str(source),
        "--config", "configs/config.json",
        "--output", str(output),
        "--csv", str(csv_output),
        "--json", str(json_output),
        "--no-display",
        "--max-frames", "1",
    ])

    assert cli.main() == 0
    assert output.exists()
    assert csv_output.exists()
    assert json_output.exists()


@pytest.mark.parametrize("value", ["0", "-1"])
def test_cli_rejects_non_positive_max_frames(monkeypatch, value):
    import main as cli

    monkeypatch.setattr(sys, "argv", ["main.py", "--max-frames", value])
    with pytest.raises(SystemExit) as raised:
        cli.parse_args()
    assert raised.value.code == 2


def test_fake_source_core_stabilizer_and_memory_sink_preserve_stable_color():
    class ColorSnapshotSink(MemorySink):
        def __init__(self):
            super().__init__()
            self.colors = []

        def write(self, packet, result, metrics):
            self.colors.append(result.tracks[0].color)
            return super().write(packet, result, metrics)

    cfg = AppConfig.default()
    cfg.tracking.min_confirmed_frames = 1
    pipeline = BottleCapPipeline(cfg)
    observations = iter(["Red", "Red", "Pink", "Red", "Red"])
    pipeline.detector.detect = lambda frame: [
        Detection(next(observations), (10, 10), (5, 5, 10, 10), 300, 0.8, 0.9, 0.8)
    ]
    source = FakeSource(_blank_frames(5))
    sink = ColorSnapshotSink()

    summary = run(source, pipeline, [sink])

    assert summary.frames_processed == 5
    assert sink.colors == ["Red"] * 5
    assert source.closed and sink.closed
