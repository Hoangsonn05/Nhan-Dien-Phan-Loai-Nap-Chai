"""Model-agnostic evaluation over canonical :mod:`dataset` manifests.

The evaluator deliberately consumes only ``DatasetManifest`` records and
canonical normalized bounding boxes.  It does not inspect images, raw YOLO
labels, runtime detections, trackers, or counters.

AP/mAP is intentionally deferred until a prediction source exposes a
meaningful ranking score.  The optional prediction ``score`` is preserved as
metadata and validated as finite, but is not used by the metrics in this
module.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable

from .dataset import Annotation, BoundingBox, DatasetContractError, DatasetManifest


EVALUATION_SCHEMA_VERSION = "1.0"
BACKGROUND_LABEL = "BACKGROUND"


class EvaluationContractError(ValueError):
    """Raised when predictions or evaluation options violate the contract."""


@dataclass(frozen=True)
class EvaluationPrediction:
    """One prediction in the same normalized coordinate space as P7.

    ``class_id`` must be a canonical manifest class ID.  ``score`` is optional
    and has no probability or ranking semantics in the P8 metric contract.
    """

    sample_id: str
    class_id: int
    bbox: BoundingBox
    score: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise EvaluationContractError("prediction sample_id must be a non-empty string")
        if isinstance(self.class_id, bool) or not isinstance(self.class_id, int):
            raise EvaluationContractError("prediction class_id must be an integer")
        if not isinstance(self.bbox, BoundingBox):
            raise EvaluationContractError("prediction bbox must be a canonical BoundingBox")
        try:
            BoundingBox.from_dict(self.bbox.to_dict())
        except DatasetContractError as exc:
            raise EvaluationContractError(f"invalid prediction bbox: {exc}") from exc
        if self.score is not None:
            if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
                raise EvaluationContractError("prediction score must be a number when supplied")
            if not math.isfinite(self.score):
                raise EvaluationContractError("prediction score must be finite")

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "class_id": self.class_id,
            "bbox": self.bbox.to_dict(),
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, raw: object) -> "EvaluationPrediction":
        if not isinstance(raw, dict) or set(raw) != {"sample_id", "class_id", "bbox", "score"}:
            raise EvaluationContractError(
                "prediction must contain sample_id, class_id, bbox and score"
            )
        try:
            bbox = BoundingBox.from_dict(raw["bbox"])
        except DatasetContractError as exc:
            raise EvaluationContractError(f"invalid prediction bbox: {exc}") from exc
        return cls(
            sample_id=raw["sample_id"],
            class_id=raw["class_id"],
            bbox=bbox,
            score=raw["score"],
        )


@dataclass(frozen=True)
class MetricCounts:
    tp: int
    fp: int
    fn: int
    precision: float | None
    recall: float | None
    f1: float | None

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


@dataclass(frozen=True)
class PerClassMetrics:
    class_id: int
    class_name: str
    support: int
    tp: int
    fp: int
    fn: int
    precision: float | None
    recall: float | None
    f1: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "support": self.support,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


@dataclass(frozen=True)
class MacroMetrics:
    precision: float | None
    precision_classes: int
    recall: float | None
    recall_classes: int
    f1: float | None
    f1_classes: int

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "precision": self.precision,
            "precision_classes": self.precision_classes,
            "recall": self.recall,
            "recall_classes": self.recall_classes,
            "f1": self.f1,
            "f1_classes": self.f1_classes,
        }


@dataclass(frozen=True)
class ConfusionMatrix:
    """Extended matrix: GT rows, prediction columns, BACKGROUND last."""

    labels: tuple[str, ...]
    values: tuple[tuple[int, ...], ...]

    def to_dict(self) -> dict[str, object]:
        return {"labels": list(self.labels), "values": [list(row) for row in self.values]}


@dataclass(frozen=True)
class EvaluationReport:
    schema_version: str
    manifest_fingerprint: str
    split: str
    iou_threshold: float
    class_mapping: tuple[tuple[int, str], ...]
    total_samples: int
    total_gt_objects: int
    total_predictions: int
    matched_objects: int
    correct_class_matches: int
    per_class: tuple[PerClassMetrics, ...]
    micro: MetricCounts
    macro: MacroMetrics
    confusion_matrix: ConfusionMatrix
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "manifest_fingerprint": self.manifest_fingerprint,
            "split": self.split,
            "iou_threshold": self.iou_threshold,
            "class_mapping": [
                {"class_id": class_id, "class_name": class_name}
                for class_id, class_name in self.class_mapping
            ],
            "total_samples": self.total_samples,
            "total_gt_objects": self.total_gt_objects,
            "total_predictions": self.total_predictions,
            "matched_objects": self.matched_objects,
            "correct_class_matches": self.correct_class_matches,
            "per_class": [metric.to_dict() for metric in self.per_class],
            "micro": self.micro.to_dict(),
            "macro": self.macro.to_dict(),
            "confusion_matrix": self.confusion_matrix.to_dict(),
            "warnings": list(self.warnings),
        }

    def to_json(self) -> str:
        """Return deterministic compact JSON with undefined metrics as null."""

        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json() + "\n", encoding="utf-8")


@dataclass(frozen=True)
class _Match:
    ground_truth_index: int
    prediction_index: int
    iou: float


def bbox_iou(left: BoundingBox, right: BoundingBox) -> float:
    """Compute IoU for two canonical normalized center-format boxes."""

    left_x1, left_y1, left_x2, left_y2 = _box_corners(left)
    right_x1, right_y1, right_x2, right_y2 = _box_corners(right)
    intersection_width = max(0.0, min(left_x2, right_x2) - max(left_x1, right_x1))
    intersection_height = max(0.0, min(left_y2, right_y2) - max(left_y1, right_y1))
    intersection = intersection_width * intersection_height
    union = left.width * left.height + right.width * right.height - intersection
    return intersection / union if union > 0 else 0.0


def evaluate_predictions(
    manifest: DatasetManifest,
    predictions: Iterable[EvaluationPrediction],
    *,
    split: str,
    iou_threshold: float = 0.5,
) -> EvaluationReport:
    """Evaluate predictions for exactly one canonical manifest split.

    Localization correspondence is assigned before class correctness is
    evaluated.  A class-confused match therefore contributes one FN to the GT
    class, one FP to the predicted class, and one off-diagonal matrix entry.

    Precision and recall are ``None`` when their own denominator is zero.  F1
    is ``None`` unless both precision and recall are defined.  Each macro value
    averages only classes where that specific metric is defined, and reports
    the number of included classes.
    """

    threshold = _validate_threshold(iou_threshold)
    try:
        samples = manifest.split(split)
    except KeyError as exc:
        raise EvaluationContractError(f"unknown canonical split: {split!r}") from exc

    split_samples = {sample.sample_id: sample for sample in samples}
    all_sample_ids = {sample.sample_id for sample in manifest.samples}
    grouped_predictions: dict[str, list[EvaluationPrediction]] = defaultdict(list)
    prediction_count = 0
    for prediction in predictions:
        if not isinstance(prediction, EvaluationPrediction):
            raise EvaluationContractError("predictions must contain EvaluationPrediction records")
        if prediction.sample_id not in all_sample_ids:
            raise EvaluationContractError(f"unknown prediction sample_id: {prediction.sample_id!r}")
        if prediction.sample_id not in split_samples:
            raise EvaluationContractError(
                f"prediction sample_id {prediction.sample_id!r} is not in split {split!r}"
            )
        if prediction.class_id not in manifest.class_mapping:
            raise EvaluationContractError(f"unknown prediction class_id: {prediction.class_id!r}")
        grouped_predictions[prediction.sample_id].append(prediction)
        prediction_count += 1

    class_items = tuple(sorted(manifest.class_mapping.items()))
    class_index = {class_id: index for index, (class_id, _) in enumerate(class_items)}
    background_index = len(class_items)
    matrix = [[0 for _ in range(background_index + 1)] for _ in range(background_index + 1)]
    support = {class_id: 0 for class_id, _ in class_items}
    tp = {class_id: 0 for class_id, _ in class_items}
    fp = {class_id: 0 for class_id, _ in class_items}
    fn = {class_id: 0 for class_id, _ in class_items}
    matched_objects = 0

    for sample in sorted(samples, key=lambda item: item.sample_id):
        ground_truth = tuple(sorted(sample.annotations, key=_annotation_key))
        sample_predictions = tuple(sorted(grouped_predictions[sample.sample_id], key=_prediction_key))
        for annotation in ground_truth:
            support[annotation.class_id] += 1
        matches, unmatched_gt, unmatched_predictions = _match_objects(
            ground_truth, sample_predictions, threshold
        )
        matched_objects += len(matches)

        for match in matches:
            annotation = ground_truth[match.ground_truth_index]
            prediction = sample_predictions[match.prediction_index]
            matrix[class_index[annotation.class_id]][class_index[prediction.class_id]] += 1
            if annotation.class_id == prediction.class_id:
                tp[annotation.class_id] += 1
            else:
                fn[annotation.class_id] += 1
                fp[prediction.class_id] += 1

        for index in unmatched_gt:
            annotation = ground_truth[index]
            fn[annotation.class_id] += 1
            matrix[class_index[annotation.class_id]][background_index] += 1
        for index in unmatched_predictions:
            prediction = sample_predictions[index]
            fp[prediction.class_id] += 1
            matrix[background_index][class_index[prediction.class_id]] += 1

    per_class = tuple(
        _per_class_metrics(
            class_id, class_name, support[class_id], tp[class_id], fp[class_id], fn[class_id]
        )
        for class_id, class_name in class_items
    )
    micro = _metric_counts(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    macro = _macro_metrics(per_class)
    warnings = tuple(
        f"Class {class_name!r} has zero ground-truth support; recall is undefined."
        for class_id, class_name in class_items
        if support[class_id] == 0
    )
    labels = tuple(class_name for _, class_name in class_items) + (BACKGROUND_LABEL,)

    return EvaluationReport(
        schema_version=EVALUATION_SCHEMA_VERSION,
        manifest_fingerprint=manifest.fingerprint,
        split=split,
        iou_threshold=threshold,
        class_mapping=class_items,
        total_samples=len(samples),
        total_gt_objects=sum(support.values()),
        total_predictions=prediction_count,
        matched_objects=matched_objects,
        correct_class_matches=sum(tp.values()),
        per_class=per_class,
        micro=micro,
        macro=macro,
        confusion_matrix=ConfusionMatrix(labels, tuple(tuple(row) for row in matrix)),
        warnings=warnings,
    )


def _match_objects(
    ground_truth: tuple[Annotation, ...],
    predictions: tuple[EvaluationPrediction, ...],
    threshold: float,
) -> tuple[tuple[_Match, ...], tuple[int, ...], tuple[int, ...]]:
    candidates: list[tuple[float, int, int]] = []
    for gt_index, annotation in enumerate(ground_truth):
        for prediction_index, prediction in enumerate(predictions):
            overlap = bbox_iou(annotation.bbox, prediction.bbox)
            if overlap >= threshold:
                candidates.append((-overlap, gt_index, prediction_index))
    candidates.sort()

    used_gt: set[int] = set()
    used_predictions: set[int] = set()
    matches: list[_Match] = []
    for negative_overlap, gt_index, prediction_index in candidates:
        if gt_index in used_gt or prediction_index in used_predictions:
            continue
        used_gt.add(gt_index)
        used_predictions.add(prediction_index)
        matches.append(_Match(gt_index, prediction_index, -negative_overlap))

    matches.sort(key=lambda match: (match.ground_truth_index, match.prediction_index))
    unmatched_gt = tuple(index for index in range(len(ground_truth)) if index not in used_gt)
    unmatched_predictions = tuple(
        index for index in range(len(predictions)) if index not in used_predictions
    )
    return tuple(matches), unmatched_gt, unmatched_predictions


def _metric_counts(tp: int, fp: int, fn: int) -> MetricCounts:
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = _harmonic_mean(precision, recall)
    return MetricCounts(tp, fp, fn, precision, recall, f1)


def _per_class_metrics(
    class_id: int, class_name: str, support: int, tp: int, fp: int, fn: int
) -> PerClassMetrics:
    metrics = _metric_counts(tp, fp, fn)
    return PerClassMetrics(
        class_id,
        class_name,
        support,
        metrics.tp,
        metrics.fp,
        metrics.fn,
        metrics.precision,
        metrics.recall,
        metrics.f1,
    )


def _macro_metrics(per_class: tuple[PerClassMetrics, ...]) -> MacroMetrics:
    precision_values = [metric.precision for metric in per_class if metric.precision is not None]
    recall_values = [metric.recall for metric in per_class if metric.recall is not None]
    f1_values = [metric.f1 for metric in per_class if metric.f1 is not None]
    return MacroMetrics(
        _mean(precision_values),
        len(precision_values),
        _mean(recall_values),
        len(recall_values),
        _mean(f1_values),
        len(f1_values),
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _harmonic_mean(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _validate_threshold(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise EvaluationContractError("iou_threshold must be a finite number")
    threshold = float(value)
    if not 0 < threshold <= 1:
        raise EvaluationContractError("iou_threshold must be in (0, 1]")
    return threshold


def _box_corners(bbox: BoundingBox) -> tuple[float, float, float, float]:
    return (
        bbox.x_center - bbox.width / 2,
        bbox.y_center - bbox.height / 2,
        bbox.x_center + bbox.width / 2,
        bbox.y_center + bbox.height / 2,
    )


def _bbox_key(bbox: BoundingBox) -> tuple[float, float, float, float]:
    return bbox.x_center, bbox.y_center, bbox.width, bbox.height


def _annotation_key(annotation: Annotation) -> tuple[float, float, float, float, int, str]:
    return (*_bbox_key(annotation.bbox), annotation.class_id, annotation.class_name)


def _prediction_key(
    prediction: EvaluationPrediction,
) -> tuple[float, float, float, float, int, int, float]:
    score_key = (0, 0.0) if prediction.score is None else (1, float(prediction.score))
    return (*_bbox_key(prediction.bbox), prediction.class_id, *score_key)
