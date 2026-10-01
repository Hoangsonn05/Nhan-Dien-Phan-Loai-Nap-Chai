"""Learned inference adapter for the canonical runtime ``Detection`` contract.

Ultralytics owns BGR preprocessing, resize/letterbox, NMS, and projection of
``boxes.xyxy`` back to the original image.  This module validates that boundary
and converts only final original-frame results; framework objects never leave
the adapter.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from .dataset import BoundingBox, DatasetContractError, DatasetManifest, validate_class_mapping
from .evaluation import EvaluationPrediction
from .models import Detection


class InferenceContractError(RuntimeError):
    """Base error for learned inference contract failures."""


class CheckpointCompatibilityError(InferenceContractError):
    """Raised before inference when a checkpoint or its metadata is incompatible."""


class PredictionContractError(InferenceContractError):
    """Raised when a framework result cannot become a canonical detection."""


@dataclass(frozen=True)
class ModelMetadata:
    backend: str
    backend_version: str
    architecture: str
    experiment_id: str
    dataset_fingerprint: str
    class_mapping: dict[int, str]
    image_size: int
    checkpoint: Path
    metadata_path: Path
    experiment_metadata_path: Path
    training_config_path: Path


def _read_json(path: Path, label: str) -> dict[str, object]:
    if not path.is_file():
        raise CheckpointCompatibilityError(f"{label} missing: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointCompatibilityError(f"Cannot read {label} {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise CheckpointCompatibilityError(f"{label} root must be an object: {path}")
    return raw


def _required_string(raw: Mapping[str, object], key: str, label: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise CheckpointCompatibilityError(f"{label}.{key} must be a non-empty string")
    return value


def _metadata_mapping(raw: object, label: str) -> dict[int, str]:
    if not isinstance(raw, list):
        raise CheckpointCompatibilityError(f"{label} must be a list")
    entries: list[tuple[object, object]] = []
    for item in raw:
        if not isinstance(item, dict) or set(item) != {"class_id", "class_name"}:
            raise CheckpointCompatibilityError(
                f"{label} entries require exactly class_id and class_name"
            )
        entries.append((item["class_id"], item["class_name"]))
    try:
        return validate_class_mapping(entries)
    except DatasetContractError as exc:
        raise CheckpointCompatibilityError(f"Invalid {label}: {exc}") from exc


def _resolve_link(parent: Path, reference: object, label: str) -> Path:
    if not isinstance(reference, str) or not reference:
        raise CheckpointCompatibilityError(f"{label} linkage must be a non-empty string")
    path = Path(reference)
    if path.is_absolute():
        raise CheckpointCompatibilityError(f"{label} linkage must be relative")
    return (parent / path).resolve()


def load_model_metadata(checkpoint: str | Path, metadata: str | Path) -> ModelMetadata:
    """Validate the P9 checkpoint chain against its canonical P7 snapshot."""
    checkpoint_path = Path(checkpoint).resolve()
    metadata_path = Path(metadata).resolve()
    if not checkpoint_path.is_file():
        raise CheckpointCompatibilityError(f"Checkpoint missing: {checkpoint_path}")

    raw = _read_json(metadata_path, "checkpoint metadata")
    backend_raw = raw.get("backend")
    if not isinstance(backend_raw, dict):
        raise CheckpointCompatibilityError("checkpoint metadata.backend must be an object")
    backend = _required_string(backend_raw, "name", "checkpoint metadata.backend")
    backend_version = _required_string(backend_raw, "version", "checkpoint metadata.backend")
    architecture = _required_string(
        backend_raw, "model_identifier", "checkpoint metadata.backend"
    )
    if backend != "ultralytics":
        raise CheckpointCompatibilityError(
            f"Unsupported checkpoint backend {backend!r}; expected 'ultralytics'"
        )

    linked_names = []
    for key in ("best_checkpoint", "last_checkpoint"):
        reference = raw.get(key)
        if isinstance(reference, str) and reference:
            linked_names.append(_resolve_link(metadata_path.parent, reference, key))
    if checkpoint_path not in linked_names:
        raise CheckpointCompatibilityError(
            f"Checkpoint {checkpoint_path} is not linked by {metadata_path}"
        )

    class_mapping = _metadata_mapping(raw.get("class_mapping"), "checkpoint class_mapping")
    fingerprint = _required_string(raw, "dataset_fingerprint", "checkpoint metadata")
    if len(fingerprint) != 64 or any(ch not in "0123456789abcdef" for ch in fingerprint):
        raise CheckpointCompatibilityError(
            "checkpoint metadata.dataset_fingerprint must be a lowercase SHA-256 digest"
        )
    image_size = raw.get("image_size")
    if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size <= 0:
        raise CheckpointCompatibilityError("checkpoint metadata.image_size must be positive")
    experiment_id = _required_string(raw, "experiment_id", "checkpoint metadata")

    experiment_path = _resolve_link(
        metadata_path.parent, raw.get("experiment_metadata"), "experiment_metadata"
    )
    experiment = _read_json(experiment_path, "experiment metadata")
    if experiment.get("status") != "completed":
        raise CheckpointCompatibilityError("P9 experiment metadata is not completed")
    if experiment.get("experiment_id") != experiment_id:
        raise CheckpointCompatibilityError("experiment ID differs between checkpoint and experiment metadata")
    if experiment.get("backend") != backend_raw:
        raise CheckpointCompatibilityError("backend metadata differs between checkpoint and experiment")

    dataset_raw = experiment.get("dataset")
    if not isinstance(dataset_raw, dict):
        raise CheckpointCompatibilityError("experiment metadata.dataset must be an object")
    experiment_mapping = _metadata_mapping(
        dataset_raw.get("class_mapping"), "experiment dataset.class_mapping"
    )
    if experiment_mapping != class_mapping:
        raise CheckpointCompatibilityError("class mapping differs between checkpoint and experiment metadata")
    if dataset_raw.get("manifest_fingerprint") != fingerprint:
        raise CheckpointCompatibilityError("dataset fingerprint differs between checkpoint and experiment metadata")

    manifest_path = _resolve_link(
        experiment_path.parent, dataset_raw.get("manifest_snapshot"), "manifest_snapshot"
    )
    try:
        manifest = DatasetManifest.load(manifest_path)
    except DatasetContractError as exc:
        raise CheckpointCompatibilityError(f"Cannot load canonical P7 manifest snapshot: {exc}") from exc
    if manifest.class_mapping != class_mapping:
        raise CheckpointCompatibilityError("checkpoint class mapping does not exactly match P7 manifest")
    if manifest.fingerprint != fingerprint:
        raise CheckpointCompatibilityError("checkpoint dataset fingerprint does not match P7 manifest")

    training_config_path = _resolve_link(
        metadata_path.parent, raw.get("training_config"), "training_config"
    )
    training_config = _read_json(training_config_path, "training config")
    if training_config.get("model") != architecture:
        raise CheckpointCompatibilityError("training config model differs from checkpoint architecture")
    if training_config.get("image_size") != image_size:
        raise CheckpointCompatibilityError("training config image_size differs from checkpoint metadata")

    return ModelMetadata(
        backend=backend,
        backend_version=backend_version,
        architecture=architecture,
        experiment_id=experiment_id,
        dataset_fingerprint=fingerprint,
        class_mapping=class_mapping,
        image_size=image_size,
        checkpoint=checkpoint_path,
        metadata_path=metadata_path,
        experiment_metadata_path=experiment_path,
        training_config_path=training_config_path,
    )


def _as_numpy(value: object, label: str) -> np.ndarray:
    try:
        converted = value
        if hasattr(converted, "detach"):
            converted = converted.detach()
        if hasattr(converted, "cpu"):
            converted = converted.cpu()
        if hasattr(converted, "numpy"):
            converted = converted.numpy()
        return np.asarray(converted)
    except Exception as exc:
        raise PredictionContractError(f"Cannot read prediction {label}: {exc}") from exc


def _model_mapping(model: object) -> dict[int, str]:
    raw = getattr(model, "names", None)
    if isinstance(raw, Mapping):
        entries = list(raw.items())
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        entries = list(enumerate(raw))
    else:
        raise CheckpointCompatibilityError("loaded model does not expose class names")
    try:
        return validate_class_mapping(entries)
    except DatasetContractError as exc:
        raise CheckpointCompatibilityError(f"Invalid class names embedded in model: {exc}") from exc


class PredictionAdapter:
    """Convert final Ultralytics original-frame ``xyxy`` results to ``Detection``."""

    def __init__(
        self,
        class_mapping: Mapping[int, str],
        runtime_class_names: Mapping[int, str],
        confidence_threshold: float,
    ) -> None:
        self.class_mapping = dict(class_mapping)
        self.runtime_class_names = dict(runtime_class_names)
        self.confidence_threshold = confidence_threshold

    def to_detections(self, result: object, frame_shape: tuple[int, ...]) -> list[Detection]:
        if len(frame_shape) < 2 or frame_shape[0] <= 0 or frame_shape[1] <= 0:
            raise PredictionContractError(f"Invalid frame shape: {frame_shape!r}")
        height, width = int(frame_shape[0]), int(frame_shape[1])
        orig_shape = getattr(result, "orig_shape", None)
        if tuple(orig_shape or ()) != (height, width):
            raise PredictionContractError(
                f"prediction orig_shape {orig_shape!r} does not match frame {(height, width)!r}"
            )
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            raise PredictionContractError("prediction result has no boxes contract")

        xyxy = _as_numpy(getattr(boxes, "xyxy", None), "boxes.xyxy")
        classes = _as_numpy(getattr(boxes, "cls", None), "boxes.cls")
        confidences = _as_numpy(getattr(boxes, "conf", None), "boxes.conf")
        if xyxy.ndim != 2 or xyxy.shape[1:] != (4,):
            raise PredictionContractError(f"boxes.xyxy must have shape (N, 4), got {xyxy.shape}")
        classes = classes.reshape(-1)
        confidences = confidences.reshape(-1)
        if len(xyxy) != len(classes) or len(xyxy) != len(confidences):
            raise PredictionContractError("prediction box/class/confidence lengths differ")
        arrays = ((xyxy, "boxes.xyxy"), (classes, "boxes.cls"), (confidences, "boxes.conf"))
        for array, label in arrays:
            if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_):
                raise PredictionContractError(
                    f"prediction {label} must contain numeric non-boolean values"
                )

        detections: list[Detection] = []
        for index, (raw_box, raw_class, raw_confidence) in enumerate(
            zip(xyxy, classes, confidences)
        ):
            try:
                values = [float(value) for value in raw_box]
                confidence = float(raw_confidence)
                class_value = float(raw_class)
            except (TypeError, ValueError) as exc:
                raise PredictionContractError(
                    f"prediction {index} values are not numeric"
                ) from exc
            if not all(math.isfinite(value) for value in (*values, confidence, class_value)):
                raise PredictionContractError(f"prediction {index} contains non-finite values")
            if not 0.0 <= confidence <= 1.0:
                raise PredictionContractError(
                    f"prediction {index} confidence is outside [0, 1]: {confidence}"
                )
            if not class_value.is_integer():
                raise PredictionContractError(f"prediction {index} class ID is not an integer")
            class_id = int(class_value)
            if class_id not in self.class_mapping:
                raise PredictionContractError(f"unknown model class ID: {class_id}")
            if confidence < self.confidence_threshold:
                continue

            x1, y1, x2, y2 = values
            if x2 <= x1 or y2 <= y1:
                raise PredictionContractError(f"prediction {index} has invalid xyxy bbox: {values}")
            # Partial boxes are clipped, then rounded outward so the canonical
            # integer bbox covers the complete backend prediction.
            x1 = min(max(x1, 0.0), float(width))
            y1 = min(max(y1, 0.0), float(height))
            x2 = min(max(x2, 0.0), float(width))
            y2 = min(max(y2, 0.0), float(height))
            left, top = math.floor(x1), math.floor(y1)
            right, bottom = math.ceil(x2), math.ceil(y2)
            box_width, box_height = right - left, bottom - top
            if box_width <= 0 or box_height <= 0:
                raise PredictionContractError(
                    f"prediction {index} bbox is outside/empty after clipping: {values}"
                )
            bbox = (left, top, box_width, box_height)
            detections.append(
                Detection(
                    color=self.runtime_class_names[class_id],
                    centroid=(left + box_width // 2, top + box_height // 2),
                    bbox=bbox,
                    area=None,
                    circularity=None,
                    solidity=None,
                    color_ratio=None,
                    class_id=class_id,
                    confidence=confidence,
                )
            )
        return detections


class LearnedDetector:
    """Load one validated P9 checkpoint and expose backend-neutral detections."""

    def __init__(
        self,
        *,
        checkpoint: str | Path,
        metadata: str | Path,
        device: str = "cpu",
        confidence_threshold: float = 0.25,
        runtime_class_names: Sequence[str],
        model_loader: Callable[[str], object] | None = None,
    ) -> None:
        if (
            isinstance(confidence_threshold, bool)
            or not isinstance(confidence_threshold, (int, float))
            or not math.isfinite(confidence_threshold)
            or not 0.0 <= confidence_threshold <= 1.0
        ):
            raise CheckpointCompatibilityError("confidence_threshold must be in [0, 1]")
        self.metadata = load_model_metadata(checkpoint, metadata)
        self.device = self._resolve_device(device)
        self.confidence_threshold = float(confidence_threshold)

        runtime_mapping = self._runtime_mapping(runtime_class_names)
        if model_loader is None:
            try:
                import ultralytics
                from ultralytics import YOLO
            except ImportError as exc:
                raise CheckpointCompatibilityError(
                    "Ultralytics is required only when detector.backend='learned'"
                ) from exc
            installed_major = ultralytics.__version__.split(".", 1)[0]
            declared_major = self.metadata.backend_version.split(".", 1)[0]
            if installed_major != declared_major:
                raise CheckpointCompatibilityError(
                    "Ultralytics major version is incompatible with checkpoint metadata: "
                    f"installed={ultralytics.__version__!r}, "
                    f"declared={self.metadata.backend_version!r}"
                )
            model_loader = YOLO
        try:
            self._model = model_loader(str(self.metadata.checkpoint))
        except Exception as exc:
            raise CheckpointCompatibilityError(f"Failed to load checkpoint: {exc}") from exc

        if getattr(self._model, "task", "detect") != "detect":
            raise CheckpointCompatibilityError("checkpoint task must be object detection")
        embedded_mapping = _model_mapping(self._model)
        if embedded_mapping != self.metadata.class_mapping:
            raise CheckpointCompatibilityError(
                "model-embedded class mapping differs from checkpoint metadata"
            )
        checkpoint_data = getattr(self._model, "ckpt", None)
        if isinstance(checkpoint_data, dict):
            train_args = checkpoint_data.get("train_args")
            embedded_model = train_args.get("model") if isinstance(train_args, dict) else None
            if embedded_model is not None and embedded_model != self.metadata.architecture:
                raise CheckpointCompatibilityError(
                    "model-embedded architecture differs from checkpoint metadata"
                )
        inner_model = getattr(self._model, "model", None)
        if inner_model is not None and hasattr(inner_model, "eval"):
            inner_model.eval()
        self._adapter = PredictionAdapter(
            self.metadata.class_mapping,
            runtime_mapping,
            self.confidence_threshold,
        )

    def _runtime_mapping(self, names: Sequence[str]) -> dict[int, str]:
        if len(names) != len(self.metadata.class_mapping):
            raise CheckpointCompatibilityError(
                "runtime color count differs from canonical P7 class mapping"
            )
        mapping: dict[int, str] = {}
        for class_id, canonical_name in self.metadata.class_mapping.items():
            runtime_name = names[class_id]
            if not isinstance(runtime_name, str) or runtime_name.casefold() != canonical_name:
                raise CheckpointCompatibilityError(
                    f"runtime color {runtime_name!r} does not map one-to-one to canonical "
                    f"class {class_id} {canonical_name!r}"
                )
            mapping[class_id] = runtime_name
        return mapping

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "cpu":
            return device
        if device != "auto" and device != "cuda" and not (
            isinstance(device, str) and device.startswith("cuda:") and device[5:].isdigit()
        ):
            raise CheckpointCompatibilityError(f"Invalid inference device: {device!r}")
        try:
            import torch
        except ImportError as exc:
            raise CheckpointCompatibilityError("PyTorch is required for learned inference") from exc
        if device == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        if not torch.cuda.is_available():
            raise CheckpointCompatibilityError(
                f"CUDA device {device!r} requested but CUDA is unavailable"
            )
        index = 0 if device == "cuda" else int(device.split(":", 1)[1])
        if index >= torch.cuda.device_count():
            raise CheckpointCompatibilityError(
                f"CUDA device index {index} is unavailable; found {torch.cuda.device_count()} device(s)"
            )
        return f"cuda:{index}"

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if frame is None or getattr(frame, "size", 0) == 0:
            return []
        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            raise PredictionContractError("learned detector input must be a non-empty NumPy image")
        try:
            try:
                import torch

                inference_context = torch.inference_mode()
            except ImportError:
                inference_context = nullcontext()
            with inference_context:
                results = self._model.predict(
                    source=frame,
                    imgsz=self.metadata.image_size,
                    device=self.device,
                    conf=self.confidence_threshold,
                    agnostic_nms=True,
                    verbose=False,
                )
        except InferenceContractError:
            raise
        except Exception as exc:
            raise InferenceContractError(f"Learned inference failed: {exc}") from exc
        if not isinstance(results, (list, tuple)) or len(results) != 1:
            raise PredictionContractError("model.predict must return exactly one result per input frame")
        return self._adapter.to_detections(results[0], frame.shape)


def detection_to_evaluation_prediction(
    detection: Detection,
    *,
    sample_id: str,
    frame_shape: tuple[int, ...],
) -> EvaluationPrediction:
    """Convert canonical full-frame pixels to the P8 normalized contract."""
    if detection.class_id is None:
        raise PredictionContractError("evaluation requires a canonical detection class_id")
    if len(frame_shape) < 2 or frame_shape[0] <= 0 or frame_shape[1] <= 0:
        raise PredictionContractError(f"Invalid frame shape: {frame_shape!r}")
    height, width = float(frame_shape[0]), float(frame_shape[1])
    x, y, box_width, box_height = detection.bbox
    if x < 0 or y < 0 or box_width <= 0 or box_height <= 0:
        raise PredictionContractError(f"Invalid canonical detection bbox: {detection.bbox!r}")
    if x + box_width > width or y + box_height > height:
        raise PredictionContractError("canonical detection bbox exceeds frame bounds")
    return EvaluationPrediction(
        sample_id=sample_id,
        class_id=detection.class_id,
        bbox=BoundingBox(
            x_center=(x + box_width / 2.0) / width,
            y_center=(y + box_height / 2.0) / height,
            width=box_width / width,
            height=box_height / height,
        ),
        score=detection.confidence,
    )
