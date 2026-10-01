from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from bottle_cap_vision.config import AppConfig, DEFAULT_COLORS
from bottle_cap_vision.dataset import DatasetManifest, build_manifest
from bottle_cap_vision.detector import HSVDetector, create_detector
from bottle_cap_vision.evaluation import evaluate_predictions
from bottle_cap_vision.inference import (
    CheckpointCompatibilityError,
    LearnedDetector,
    PredictionAdapter,
    PredictionContractError,
    detection_to_evaluation_prediction,
    load_model_metadata,
)
from bottle_cap_vision.models import Detection
from bottle_cap_vision.pipeline import BottleCapPipeline
from bottle_cap_vision.tracker import NearestNeighborTracker


CANONICAL_MAPPING = {
    0: "blue",
    1: "brown",
    2: "green",
    3: "lilac",
    4: "orange",
    5: "pink",
    6: "purple",
    7: "red",
    8: "white",
    9: "yellow",
}
RUNTIME_MAPPING = {index: name for index, name in enumerate(DEFAULT_COLORS)}


@dataclass
class Artifact:
    checkpoint: Path
    metadata: Path
    manifest: Path


class FakeInnerModel:
    def __init__(self) -> None:
        self.eval_calls = 0

    def eval(self):
        self.eval_calls += 1
        return self


class FakeModel:
    task = "detect"

    def __init__(self, results=None, names=None) -> None:
        self.names = dict(CANONICAL_MAPPING if names is None else names)
        self.ckpt = {"train_args": {"model": "yolo11n.pt"}}
        self.model = FakeInnerModel()
        self.results = results
        self.predict_calls = 0
        self.predict_requests = []

    def predict(self, *, source, **kwargs):
        self.predict_calls += 1
        self.predict_requests.append((source, kwargs))
        if self.results is not None:
            return self.results
        return [_result(source.shape[:2], [], [], [])]


def _mapping_list(mapping=CANONICAL_MAPPING):
    return [
        {"class_id": class_id, "class_name": class_name}
        for class_id, class_name in mapping.items()
    ]


def _result(orig_shape, xyxy, classes, confidences):
    boxes = SimpleNamespace(
        xyxy=np.asarray(xyxy, dtype=float).reshape((-1, 4)),
        cls=np.asarray(classes, dtype=float),
        conf=np.asarray(confidences, dtype=float),
    )
    return SimpleNamespace(orig_shape=orig_shape, boxes=boxes)


@pytest.fixture
def artifact(tmp_path) -> Artifact:
    dataset = tmp_path / "dataset"
    for index, physical_split in enumerate(("train", "valid", "test")):
        images = dataset / physical_split / "images"
        labels = dataset / physical_split / "labels"
        images.mkdir(parents=True)
        labels.mkdir()
        assert cv2.imwrite(str(images / f"sample_{index}.jpg"), np.full((8, 12, 3), index * 40, np.uint8))
        label_text = "" if index else "".join(
            f"{class_id} 0.5 0.5 0.1 0.1\n" for class_id in CANONICAL_MAPPING
        )
        (labels / f"sample_{index}.txt").write_text(label_text, encoding="utf-8")
    (dataset / "data.yaml").write_text(
        "train: train/images\nval: valid/images\ntest: test/images\n"
        f"nc: {len(CANONICAL_MAPPING)}\n"
        f"names: {list(CANONICAL_MAPPING.values())!r}\n",
        encoding="utf-8",
    )

    experiment = tmp_path / "experiment"
    manifest_path = experiment / "dataset_manifest.json"
    manifest = build_manifest(dataset, manifest_path)
    checkpoints = experiment / "checkpoints"
    checkpoints.mkdir()
    checkpoint = checkpoints / "best.pt"
    checkpoint.write_bytes(b"fake checkpoint")
    training_config = {
        "model": "yolo11n.pt",
        "image_size": 160,
    }
    (experiment / "training_config.json").write_text(json.dumps(training_config), encoding="utf-8")
    backend = {
        "name": "ultralytics",
        "version": "8.4.160",
        "model_identifier": "yolo11n.pt",
    }
    experiment_metadata = {
        "status": "completed",
        "experiment_id": "unit_experiment",
        "backend": backend,
        "dataset": {
            "class_mapping": _mapping_list(),
            "manifest_fingerprint": manifest.fingerprint,
            "manifest_snapshot": "dataset_manifest.json",
        },
    }
    (experiment / "experiment_metadata.json").write_text(
        json.dumps(experiment_metadata), encoding="utf-8"
    )
    checkpoint_metadata = {
        "backend": backend,
        "best_checkpoint": "best.pt",
        "last_checkpoint": "last.pt",
        "class_mapping": _mapping_list(),
        "dataset_fingerprint": manifest.fingerprint,
        "experiment_id": "unit_experiment",
        "experiment_metadata": "../experiment_metadata.json",
        "image_size": 160,
        "training_config": "../training_config.json",
    }
    metadata = checkpoints / "checkpoint_metadata.json"
    metadata.write_text(json.dumps(checkpoint_metadata), encoding="utf-8")
    return Artifact(checkpoint, metadata, manifest_path)


def _learned(artifact: Artifact, model: FakeModel | None = None, **kwargs) -> LearnedDetector:
    selected_model = model or FakeModel()
    return LearnedDetector(
        checkpoint=artifact.checkpoint,
        metadata=artifact.metadata,
        runtime_class_names=DEFAULT_COLORS,
        model_loader=lambda _: selected_model,
        **kwargs,
    )


def test_checkpoint_metadata_matches_canonical_p7_manifest(artifact):
    metadata = load_model_metadata(artifact.checkpoint, artifact.metadata)

    assert metadata.class_mapping == CANONICAL_MAPPING
    assert metadata.image_size == 160
    assert metadata.architecture == "yolo11n.pt"


def test_missing_checkpoint_and_metadata_fail_clearly(artifact, tmp_path):
    with pytest.raises(CheckpointCompatibilityError, match="Checkpoint missing"):
        load_model_metadata(tmp_path / "missing.pt", artifact.metadata)
    with pytest.raises(CheckpointCompatibilityError, match="metadata missing"):
        load_model_metadata(artifact.checkpoint, tmp_path / "missing.json")


@pytest.mark.parametrize(
    "mapping",
    [
        {**CANONICAL_MAPPING, 7: "crimson"},
        {**CANONICAL_MAPPING, 0: "brown", 1: "blue"},
    ],
)
def test_checkpoint_mapping_name_or_order_mismatch_is_rejected(artifact, mapping):
    raw = json.loads(artifact.metadata.read_text(encoding="utf-8"))
    raw["class_mapping"] = _mapping_list(mapping)
    artifact.metadata.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(CheckpointCompatibilityError, match="class mapping"):
        load_model_metadata(artifact.checkpoint, artifact.metadata)


def test_model_embedded_mapping_is_validated(artifact):
    names = {**CANONICAL_MAPPING, 7: "crimson"}

    with pytest.raises(CheckpointCompatibilityError, match="model-embedded class mapping"):
        _learned(artifact, FakeModel(names=names))


def test_model_is_loaded_once_for_multiple_frames(artifact):
    calls = []
    model = FakeModel()
    detector = LearnedDetector(
        checkpoint=artifact.checkpoint,
        metadata=artifact.metadata,
        runtime_class_names=DEFAULT_COLORS,
        model_loader=lambda path: calls.append(path) or model,
    )

    first_frame = np.zeros((20, 30, 3), dtype=np.uint8)
    second_frame = np.ones((20, 30, 3), dtype=np.uint8)
    detector.detect(first_frame)
    detector.detect(second_frame)

    assert calls == [str(artifact.checkpoint.resolve())]
    assert model.predict_calls == 2
    assert model.predict_requests[0][0] is first_frame
    assert model.predict_requests[1][0] is second_frame
    assert all(
        request[1]
        == {
            "imgsz": 160,
            "device": "cpu",
            "conf": 0.25,
            "agnostic_nms": True,
            "verbose": False,
        }
        for request in model.predict_requests
    )
    assert model.model.eval_calls == 1


def test_explicit_cuda_fails_instead_of_silent_cpu_fallback(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(CheckpointCompatibilityError, match="CUDA.*unavailable"):
        LearnedDetector._resolve_device("cuda:0")


def test_hsv_factory_does_not_touch_learned_checkpoint_paths():
    cfg = AppConfig.default()
    cfg.detector.backend = "hsv"
    cfg.detector.checkpoint = "missing.pt"
    cfg.detector.metadata = "missing.json"

    assert isinstance(create_detector(cfg), HSVDetector)


def test_xyxy_is_clipped_rounded_and_converted_on_non_square_frame():
    adapter = PredictionAdapter(CANONICAL_MAPPING, RUNTIME_MAPPING, 0.25)
    result = _result((720, 1280), [[-2.2, 5.2, 1285.0, 719.1]], [7], [0.75])

    detections = adapter.to_detections(result, (720, 1280, 3))

    assert len(detections) == 1
    detection = detections[0]
    assert detection.bbox == (0, 5, 1280, 715)
    assert detection.centroid == (640, 362)
    assert detection.color == "Red"
    assert detection.class_id == 7
    assert detection.confidence == 0.75
    assert detection.area is detection.circularity is detection.solidity is detection.color_ratio is None


def test_backend_must_reverse_letterbox_to_original_frame_before_adapter():
    adapter = PredictionAdapter(CANONICAL_MAPPING, RUNTIME_MAPPING, 0.25)
    resized_result = _result((160, 160), [[10, 10, 20, 20]], [0], [0.8])

    with pytest.raises(PredictionContractError, match="orig_shape"):
        adapter.to_detections(resized_result, (720, 1280, 3))


def test_threshold_filter_and_confidence_propagation():
    adapter = PredictionAdapter(CANONICAL_MAPPING, RUNTIME_MAPPING, 0.5)
    result = _result(
        (100, 200),
        [[0, 0, 10, 10], [20, 30, 40, 50]],
        [0, 7],
        [0.499, 0.8125],
    )

    detections = adapter.to_detections(result, (100, 200, 3))

    assert [(item.class_id, item.confidence) for item in detections] == [(7, 0.8125)]


def test_empty_and_multiple_predictions_are_canonical():
    adapter = PredictionAdapter(CANONICAL_MAPPING, RUNTIME_MAPPING, 0.0)
    assert adapter.to_detections(_result((10, 20), [], [], []), (10, 20, 3)) == []
    detections = adapter.to_detections(
        _result((10, 20), [[1, 1, 4, 5], [10, 2, 19, 9]], [2, 8], [0.7, 0.8]),
        (10, 20, 3),
    )
    assert [item.class_id for item in detections] == [2, 8]


def test_unknown_class_invalid_bbox_and_malformed_prediction_are_errors():
    adapter = PredictionAdapter(CANONICAL_MAPPING, RUNTIME_MAPPING, 0.0)
    with pytest.raises(PredictionContractError, match="unknown model class ID"):
        adapter.to_detections(_result((10, 20), [[1, 1, 4, 5]], [10], [0.7]), (10, 20, 3))
    with pytest.raises(PredictionContractError, match="invalid xyxy bbox"):
        adapter.to_detections(_result((10, 20), [[4, 1, 1, 5]], [0], [0.7]), (10, 20, 3))
    malformed = SimpleNamespace(orig_shape=(10, 20), boxes=SimpleNamespace(xyxy=[[1, 2, 3]], cls=[0], conf=[0.7]))
    with pytest.raises(PredictionContractError, match="shape"):
        adapter.to_detections(malformed, (10, 20, 3))


class SequenceDetector:
    def __init__(self, detections):
        self._detections = iter(detections)

    def detect(self, frame):
        return [next(self._detections)]


def _learned_detection(color: str, y: int, class_id: int) -> Detection:
    return Detection(
        color=color,
        centroid=(50, y),
        bbox=(45, y - 5, 10, 10),
        class_id=class_id,
        confidence=0.9,
    )


def test_pipeline_accepts_interchangeable_detector_contracts():
    cfg = AppConfig.default()
    cfg.counting.enabled = False
    hsv_contract = SequenceDetector([Detection("Red", (10, 10), (5, 5, 10, 10), 80, 0.8, 0.9, 0.8)])
    learned_contract = SequenceDetector([_learned_detection("Red", 10, 7)])
    hsv_pipeline = BottleCapPipeline(cfg, detector=hsv_contract)
    learned_pipeline = BottleCapPipeline(cfg, detector=learned_contract)

    hsv_result = hsv_pipeline.process_frame(np.zeros((30, 30, 3), np.uint8))
    learned_result = learned_pipeline.process_frame(np.zeros((30, 30, 3), np.uint8))

    assert isinstance(hsv_pipeline.tracker, NearestNeighborTracker)
    assert isinstance(learned_pipeline.tracker, NearestNeighborTracker)
    assert hsv_result.tracks[0].color == learned_result.tracks[0].color == "Red"


def test_learned_detection_uses_existing_stabilizer_and_counter_unchanged():
    cfg = AppConfig.default()
    cfg.tracking.min_confirmed_frames = 1
    cfg.tracking.max_distance = 30
    cfg.counting.line_position = 0.5
    cfg.counting.deadband = 5
    detections = [
        _learned_detection("Red", 35, 7),
        _learned_detection("Red", 40, 7),
        _learned_detection("Pink", 45, 5),
        _learned_detection("Red", 65, 7),
        _learned_detection("Red", 70, 7),
    ]
    pipeline = BottleCapPipeline(cfg, detector=SequenceDetector(detections))
    events = []
    for frame_index in range(len(detections)):
        events.extend(pipeline.process_frame(np.zeros((100, 100, 3), np.uint8), frame_index).events)

    assert len(events) == 1
    assert events[0].color == "Red"


def test_detection_converts_to_p8_prediction_without_framework_object():
    detection = Detection(
        color="Red",
        centroid=(40, 30),
        bbox=(20, 10, 40, 40),
        class_id=7,
        confidence=0.875,
    )

    prediction = detection_to_evaluation_prediction(
        detection, sample_id="val:sample.jpg", frame_shape=(100, 200, 3)
    )

    assert prediction.class_id == 7
    assert prediction.score == 0.875
    assert prediction.bbox.x_center == pytest.approx(0.2)
    assert prediction.bbox.y_center == pytest.approx(0.3)
    assert prediction.bbox.width == pytest.approx(0.2)
    assert prediction.bbox.height == pytest.approx(0.4)


def test_production_p9_checkpoint_load_inference_and_p8_contract():
    root = Path(__file__).resolve().parents[1]
    experiment = root / "models" / "p9_full_001"
    checkpoint = experiment / "checkpoints" / "best.pt"
    metadata = experiment / "checkpoints" / "checkpoint_metadata.json"
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    cfg = AppConfig.default()
    cfg.detector.backend = "learned"
    cfg.detector.checkpoint = str(checkpoint)
    cfg.detector.metadata = str(metadata)
    cfg.detector.device = "cpu"
    cfg.detector.confidence_threshold = 0.25
    detector = create_detector(cfg)

    detections = detector.detect(image)

    assert isinstance(detector, LearnedDetector)
    assert detector.metadata.class_mapping == CANONICAL_MAPPING
    assert all(item.class_id in CANONICAL_MAPPING for item in detections)
    assert all(item.confidence is not None and 0 <= item.confidence <= 1 for item in detections)
    assert all(item.bbox[0] >= 0 and item.bbox[1] >= 0 and item.bbox[2] > 0 and item.bbox[3] > 0 for item in detections)

    manifest = DatasetManifest.load(experiment / "dataset_manifest.json")
    sample = next(iter(manifest.split("val")))
    predictions = [
        detection_to_evaluation_prediction(item, sample_id=sample.sample_id, frame_shape=image.shape)
        for item in detections
    ]
    report = evaluate_predictions(manifest, predictions, split="val")
    assert report.total_samples == len(manifest.split("val"))
    assert report.manifest_fingerprint == manifest.fingerprint
