from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from bottle_cap_vision.config import AppConfig
from bottle_cap_vision.inference import PredictionAdapter
from bottle_cap_vision.models import Detection, FramePacket, SourceMetadata
from scripts import diagnose_camera_fp as diagnose


MAPPING = {0: "red"}


def _result(frame: np.ndarray):
    return SimpleNamespace(
        orig_shape=frame.shape[:2],
        boxes=SimpleNamespace(
            xyxy=np.asarray([[1.25, 2.5, 8.75, 9.5]], dtype=np.float32),
            cls=np.asarray([0.0], dtype=np.float32),
            conf=np.asarray([0.75], dtype=np.float32),
        ),
    )


class FakeModel:
    def predict(self, *, source, **_kwargs):
        return [_result(source)]


class FakeDetector:
    def __init__(self, checkpoint: Path):
        self.metadata = SimpleNamespace(image_size=640, checkpoint=checkpoint)
        self.device = "cpu"
        self.confidence_threshold = 0.25
        self._model = FakeModel()
        self._adapter = PredictionAdapter(MAPPING, {0: "Red"}, 0.25)

    def detect(self, frame):
        return self._adapter.to_detections(self._model.predict(source=frame)[0], frame.shape)


class FakeSource:
    instances = []

    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.metadata = SourceMetadata(fps=29.97, width=12, height=10)
        self.frame = np.arange(10 * 12 * 3, dtype=np.uint8).reshape(10, 12, 3)
        self.opened = False
        self.closed = False
        self.reads = 0
        self.__class__.instances.append(self)

    def open(self):
        self.opened = True

    def __iter__(self):
        while True:
            self.reads += 1
            yield FramePacket(frame=self.frame, frame_index=self.reads - 1)

    def close(self):
        self.closed = True


def _install_loader(monkeypatch, tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text("{}", encoding="utf-8")
    config = AppConfig.default()
    config.detector.backend = "learned"
    detector = FakeDetector(checkpoint)
    monkeypatch.setattr(
        diagnose,
        "load_diagnostic_detector",
        lambda _path: (config, config_path.resolve(), checkpoint.resolve(), metadata_path, detector),
    )
    return config_path, checkpoint, detector


def test_save_raw_frame_is_pixel_equivalent_and_prediction_lossless(tmp_path):
    frame = np.arange(8 * 9 * 3, dtype=np.uint8).reshape(8, 9, 3)
    original = frame.copy()
    detection = Detection(
        color="Red",
        centroid=(4, 5),
        bbox=(1, 2, 6, 5),
        class_id=0,
        confidence=0.756789123,
    )
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")

    frame_path = diagnose.save_capture(
        output_dir=tmp_path,
        frame=frame,
        detections=[detection],
        camera_index=2,
        source_metadata=SourceMetadata(fps=25.0, width=9, height=8),
        config_path=config_path,
        checkpoint=checkpoint,
        checkpoint_hash=diagnose.file_sha256(checkpoint),
    )

    assert np.array_equal(diagnose.read_png(frame_path), original)
    assert np.array_equal(frame, original)
    predictions = json.loads((tmp_path / "predictions.json").read_text())
    assert predictions["predictions"] == [
        {
            "index": 0,
            "class_id": 0,
            "class_name": "Red",
            "confidence": 0.756789123,
            "bbox_xywh": [1, 2, 6, 5],
            "bbox_xyxy": [1, 2, 7, 7],
        }
    ]


def test_capture_s_saves_once_and_cleans_up(monkeypatch, tmp_path):
    FakeSource.instances.clear()
    _install_loader(monkeypatch, tmp_path)
    monkeypatch.setattr(diagnose, "CameraSource", FakeSource)
    monkeypatch.setattr(cv2, "imshow", lambda *_args: None)
    keys = iter([ord("s"), ord("s")])
    monkeypatch.setattr(cv2, "waitKey", lambda _delay: next(keys))
    destroyed = []
    monkeypatch.setattr(cv2, "destroyWindow", destroyed.append)

    result = diagnose.capture_one_frame(
        camera_index=3,
        config_path=tmp_path / "config.json",
        output_dir=tmp_path / "capture",
    )

    source = FakeSource.instances[-1]
    assert result == tmp_path / "capture" / "frame.png"
    assert source.reads == 1
    assert source.closed
    assert destroyed == [diagnose.WINDOW_NAME]
    assert len(list((tmp_path / "capture").glob("*.png"))) == 1


@pytest.mark.parametrize("exit_key", [ord("q"), 27])
def test_capture_quit_does_not_save_and_cleans_up(monkeypatch, tmp_path, exit_key):
    FakeSource.instances.clear()
    _install_loader(monkeypatch, tmp_path)
    monkeypatch.setattr(diagnose, "CameraSource", FakeSource)
    monkeypatch.setattr(cv2, "imshow", lambda *_args: None)
    monkeypatch.setattr(cv2, "waitKey", lambda _delay: exit_key)
    destroyed = []
    monkeypatch.setattr(cv2, "destroyWindow", destroyed.append)

    result = diagnose.capture_one_frame(
        camera_index=0,
        config_path=tmp_path / "config.json",
        output_dir=tmp_path / "capture",
    )

    assert result is None
    assert FakeSource.instances[-1].closed
    assert destroyed == [diagnose.WINDOW_NAME]
    assert not (tmp_path / "capture").exists()


def test_capture_detects_source_frame_mutation(monkeypatch, tmp_path):
    FakeSource.instances.clear()
    config_path, checkpoint, detector = _install_loader(monkeypatch, tmp_path)

    def mutate(frame):
        frame[0, 0, 0] ^= 1
        return []

    detector.detect = mutate
    monkeypatch.setattr(diagnose, "CameraSource", FakeSource)

    with pytest.raises(diagnose.DiagnosticError, match="mutated"):
        diagnose.capture_one_frame(
            camera_index=0,
            config_path=config_path,
            output_dir=tmp_path / "capture",
        )

    assert checkpoint.is_file()
    assert FakeSource.instances[-1].closed


def test_replay_uses_saved_png_without_opening_camera_and_runs_three_boundaries(
    monkeypatch, tmp_path
):
    config_path, checkpoint, detector = _install_loader(monkeypatch, tmp_path)
    frame = np.arange(10 * 12 * 3, dtype=np.uint8).reshape(10, 12, 3)
    live_detection = detector.detect(frame)[0]
    frame_path = diagnose.save_capture(
        output_dir=tmp_path,
        frame=frame,
        detections=[live_detection],
        camera_index=0,
        source_metadata=SourceMetadata(fps=30.0, width=12, height=10),
        config_path=config_path.resolve(),
        checkpoint=checkpoint.resolve(),
        checkpoint_hash=diagnose.file_sha256(checkpoint),
    )
    monkeypatch.setattr(
        diagnose,
        "CameraSource",
        lambda *_args, **_kwargs: pytest.fail("replay must not open camera"),
    )

    summary = diagnose.replay_frame(
        frame_path=frame_path,
        config_path=config_path,
        runs=3,
        fp_index=0,
        actual_object="background card",
    )

    assert summary["origin"] == "RAW_MODEL"
    assert summary["reproduction"] == {
        "raw_ultralytics": "3/3",
        "adapter": "3/3",
        "direct_detector": "3/3",
        "pipeline": True,
    }
    assert summary["frame_mutated"] is False
    assert all(item["raw_equals_adapter"] for item in summary["comparisons"])
    assert all(item["adapter_equals_direct_detector"] for item in summary["comparisons"])
    for name in (
        "raw_predictions.json",
        "adapter_predictions.json",
        "detector_predictions.json",
        "pipeline_predictions.json",
        "replay_summary.json",
    ):
        assert (tmp_path / name).is_file()


def test_replay_rejects_config_or_checkpoint_drift(monkeypatch, tmp_path):
    config_path, checkpoint, detector = _install_loader(monkeypatch, tmp_path)
    frame = np.zeros((10, 12, 3), dtype=np.uint8)
    frame_path = diagnose.save_capture(
        output_dir=tmp_path,
        frame=frame,
        detections=detector.detect(frame),
        camera_index=0,
        source_metadata=SourceMetadata(fps=None, width=12, height=10),
        config_path=config_path,
        checkpoint=checkpoint,
        checkpoint_hash=diagnose.file_sha256(checkpoint),
    )
    checkpoint.write_bytes(b"different checkpoint")

    with pytest.raises(diagnose.DiagnosticError, match="checkpoint_sha256"):
        diagnose.replay_frame(frame_path=frame_path, config_path=config_path)
