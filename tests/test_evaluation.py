from __future__ import annotations

from dataclasses import replace
import json
import random

import cv2
import numpy as np
import pytest

from bottle_cap_vision.dataset import (
    ANNOTATION_FORMAT,
    SCHEMA_VERSION,
    Annotation,
    BoundingBox,
    DatasetManifest,
    DatasetSample,
    build_manifest,
)
from bottle_cap_vision.evaluation import (
    BACKGROUND_LABEL,
    EvaluationContractError,
    EvaluationPrediction,
    bbox_iou,
    evaluate_predictions,
)


CLASS_MAPPING = {0: "blue", 1: "pink", 2: "red", 3: "unused"}


def _box(x: float, y: float, width: float = 0.2, height: float = 0.2) -> BoundingBox:
    return BoundingBox(x, y, width, height)


def _annotation(class_id: int, bbox: BoundingBox) -> Annotation:
    return Annotation(class_id, CLASS_MAPPING[class_id], bbox)


def _sample(sample_id: str, annotations: tuple[Annotation, ...], *, split: str = "val") -> DatasetSample:
    return DatasetSample(
        sample_id=sample_id,
        image_path=f"valid/images/{sample_id}.jpg",
        annotation_path=f"valid/labels/{sample_id}.txt",
        split=split,
        width=100,
        height=100,
        channels=3,
        content_sha256="a" * 64,
        annotations=annotations,
    )


def _manifest(*samples: DatasetSample) -> DatasetManifest:
    return DatasetManifest(
        schema_version=SCHEMA_VERSION,
        annotation_format=ANNOTATION_FORMAT,
        allow_empty=True,
        class_mapping=CLASS_MAPPING.copy(),
        samples=tuple(samples),
        statistics={},
        fingerprint="f" * 64,
    )


def _prediction(sample_id: str, class_id: int, bbox: BoundingBox, score=None):
    return EvaluationPrediction(sample_id, class_id, bbox, score)


def _class(report, class_id: int):
    return report.per_class[class_id]


def _matrix_cell(report, row: str, column: str) -> int:
    labels = report.confusion_matrix.labels
    return report.confusion_matrix.values[labels.index(row)][labels.index(column)]


def test_perfect_predictions_have_perfect_micro_metrics():
    boxes = (_box(0.2, 0.2), _box(0.5, 0.5), _box(0.8, 0.8))
    manifest = _manifest(_sample("a", tuple(_annotation(i, box) for i, box in enumerate(boxes))))
    predictions = [_prediction("a", i, box) for i, box in enumerate(boxes)]

    report = evaluate_predictions(manifest, predictions, split="val")

    assert report.micro.to_dict() == {
        "tp": 3, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0,
    }
    assert report.matched_objects == report.correct_class_matches == 3


def test_missed_object_is_false_negative_for_its_class():
    manifest = _manifest(_sample("a", (_annotation(2, _box(0.5, 0.5)),)))

    report = evaluate_predictions(manifest, [], split="val")

    assert (_class(report, 2).tp, _class(report, 2).fp, _class(report, 2).fn) == (0, 0, 1)
    assert report.micro.precision is None
    assert report.micro.recall == 0.0
    assert report.micro.f1 is None


def test_false_positive_is_counted_for_predicted_class():
    manifest = _manifest(_sample("negative", ()))

    report = evaluate_predictions(
        manifest, [_prediction("negative", 0, _box(0.5, 0.5))], split="val"
    )

    assert (_class(report, 0).tp, _class(report, 0).fp, _class(report, 0).fn) == (0, 1, 0)
    assert report.micro.precision == 0.0
    assert report.micro.recall is None


def test_wrong_class_with_correct_bbox_is_class_confusion_not_localization_miss():
    bbox = _box(0.5, 0.5)
    manifest = _manifest(_sample("a", (_annotation(2, bbox),)))

    report = evaluate_predictions(manifest, [_prediction("a", 1, bbox)], split="val")

    assert report.matched_objects == 1
    assert report.correct_class_matches == 0
    assert _class(report, 2).fn == 1
    assert _class(report, 1).fp == 1
    assert _matrix_cell(report, "red", "pink") == 1
    assert _matrix_cell(report, "red", BACKGROUND_LABEL) == 0
    assert _matrix_cell(report, BACKGROUND_LABEL, "pink") == 0


def test_below_iou_threshold_is_unmatched_fp_and_fn():
    manifest = _manifest(_sample("a", (_annotation(2, _box(0.3, 0.5)),)))

    report = evaluate_predictions(
        manifest, [_prediction("a", 2, _box(0.7, 0.5))], split="val"
    )

    assert report.matched_objects == 0
    assert (_class(report, 2).tp, _class(report, 2).fp, _class(report, 2).fn) == (0, 1, 1)


def test_one_gt_can_match_only_one_of_two_predictions():
    bbox = _box(0.5, 0.5)
    manifest = _manifest(_sample("a", (_annotation(2, bbox),)))

    report = evaluate_predictions(
        manifest,
        [_prediction("a", 2, bbox), _prediction("a", 2, _box(0.51, 0.5))],
        split="val",
    )

    assert report.matched_objects == 1
    assert (_class(report, 2).tp, _class(report, 2).fp, _class(report, 2).fn) == (1, 1, 0)


def test_one_prediction_cannot_be_reused_for_multiple_gt():
    manifest = _manifest(
        _sample("a", (_annotation(2, _box(0.48, 0.5)), _annotation(2, _box(0.52, 0.5))))
    )

    report = evaluate_predictions(
        manifest, [_prediction("a", 2, _box(0.5, 0.5))], split="val"
    )

    assert report.matched_objects == 1
    assert (_class(report, 2).tp, _class(report, 2).fn) == (1, 1)


def test_prediction_and_ground_truth_order_do_not_change_report():
    annotations = (
        _annotation(2, _box(0.2, 0.2)),
        _annotation(1, _box(0.5, 0.5)),
        _annotation(0, _box(0.8, 0.8)),
    )
    predictions = [
        _prediction("a", 1, annotations[0].bbox),
        _prediction("a", 1, annotations[1].bbox),
        _prediction("a", 0, annotations[2].bbox),
    ]
    first = evaluate_predictions(_manifest(_sample("a", annotations)), predictions, split="val")
    shuffled_predictions = predictions.copy()
    random.Random(123).shuffle(shuffled_predictions)
    second = evaluate_predictions(
        _manifest(_sample("a", tuple(reversed(annotations)))), shuffled_predictions, split="val"
    )

    assert first.to_dict() == second.to_dict()
    assert first.to_json() == second.to_json()


def test_negative_image_with_no_predictions_changes_no_counts():
    report = evaluate_predictions(_manifest(_sample("negative", ())), [], split="val")

    assert report.total_samples == 1
    assert report.total_gt_objects == report.total_predictions == 0
    assert (report.micro.tp, report.micro.fp, report.micro.fn) == (0, 0, 0)


def test_zero_support_has_undefined_recall_and_is_excluded_from_macro_recall():
    bbox = _box(0.5, 0.5)
    manifest = _manifest(_sample("a", (_annotation(2, bbox),)))

    report = evaluate_predictions(
        manifest, [_prediction("a", 2, bbox), _prediction("a", 3, _box(0.8, 0.8))], split="val"
    )

    assert _class(report, 3).support == 0
    assert _class(report, 3).precision == 0.0
    assert _class(report, 3).recall is None
    assert _class(report, 3).f1 is None
    assert report.macro.recall_classes == 1
    assert report.macro.recall == 1.0
    assert any("unused" in warning for warning in report.warnings)


def test_extended_confusion_matrix_records_miss_and_false_positive():
    manifest = _manifest(
        _sample("miss", (_annotation(2, _box(0.5, 0.5)),)),
        _sample("negative", ()),
    )
    predictions = [_prediction("negative", 0, _box(0.5, 0.5))]

    report = evaluate_predictions(manifest, predictions, split="val")

    assert _matrix_cell(report, "red", BACKGROUND_LABEL) == 1
    assert _matrix_cell(report, BACKGROUND_LABEL, "blue") == 1


def test_hand_calculated_micro_and_macro_metrics():
    boxes = (_box(0.2, 0.2), _box(0.5, 0.5), _box(0.8, 0.8))
    manifest = _manifest(
        _sample("a", (_annotation(0, boxes[0]), _annotation(1, boxes[1]), _annotation(2, boxes[2])))
    )
    predictions = [
        _prediction("a", 0, boxes[0]),  # TP blue
        _prediction("a", 2, boxes[1]),  # pink -> red confusion
        _prediction("a", 0, _box(0.5, 0.8)),  # unmatched blue FP
    ]

    report = evaluate_predictions(manifest, predictions, split="val")

    assert (report.micro.tp, report.micro.fp, report.micro.fn) == (1, 2, 2)
    assert report.micro.precision == pytest.approx(1 / 3)
    assert report.micro.recall == pytest.approx(1 / 3)
    assert report.micro.f1 == pytest.approx(1 / 3)
    assert report.macro.precision_classes == 2
    assert report.macro.precision == pytest.approx((0.5 + 0.0) / 2)
    assert report.macro.recall_classes == 3
    assert report.macro.recall == pytest.approx((1.0 + 0.0 + 0.0) / 3)
    assert report.macro.f1_classes == 2
    assert report.macro.f1 == pytest.approx((2 / 3 + 0.0) / 2)


@pytest.mark.parametrize("threshold", [0, -0.1, 1.1, float("nan"), True])
def test_invalid_iou_threshold_is_rejected(threshold):
    with pytest.raises(EvaluationContractError, match="iou_threshold"):
        evaluate_predictions(_manifest(_sample("a", ())), [], split="val", iou_threshold=threshold)


def test_unknown_class_and_unknown_sample_are_rejected():
    manifest = _manifest(_sample("a", ()))

    with pytest.raises(EvaluationContractError, match="unknown prediction class_id"):
        evaluate_predictions(manifest, [_prediction("a", 99, _box(0.5, 0.5))], split="val")
    with pytest.raises(EvaluationContractError, match="unknown prediction sample_id"):
        evaluate_predictions(manifest, [_prediction("missing", 0, _box(0.5, 0.5))], split="val")


def test_prediction_from_another_manifest_split_is_rejected():
    manifest = _manifest(_sample("val", ()), _sample("train", (), split="train"))

    with pytest.raises(EvaluationContractError, match="not in split"):
        evaluate_predictions(manifest, [_prediction("train", 0, _box(0.5, 0.5))], split="val")


@pytest.mark.parametrize(
    "bbox",
    [
        BoundingBox(float("nan"), 0.5, 0.2, 0.2),
        BoundingBox(0.5, 0.5, 0.0, 0.2),
        BoundingBox(0.05, 0.5, 0.2, 0.2),
    ],
)
def test_invalid_prediction_bbox_is_rejected(bbox):
    with pytest.raises(EvaluationContractError, match="bbox"):
        _prediction("a", 0, bbox)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), True])
def test_invalid_optional_score_is_rejected(score):
    with pytest.raises(EvaluationContractError, match="score"):
        _prediction("a", 0, _box(0.5, 0.5), score)


def test_score_is_optional_unbounded_and_does_not_change_metrics():
    bbox = _box(0.5, 0.5)
    manifest = _manifest(_sample("a", (_annotation(2, bbox),)))

    without_score = evaluate_predictions(manifest, [_prediction("a", 2, bbox)], split="val")
    with_score = evaluate_predictions(manifest, [_prediction("a", 2, bbox, 17.5)], split="val")

    assert without_score.micro == with_score.micro
    assert without_score.confusion_matrix == with_score.confusion_matrix


def test_prediction_json_contract_round_trip():
    prediction = _prediction("a", 2, _box(0.5, 0.5), 0.75)

    assert EvaluationPrediction.from_dict(prediction.to_dict()) == prediction


def test_bbox_iou_uses_canonical_center_coordinates():
    assert bbox_iou(_box(0.5, 0.5), _box(0.5, 0.5)) == pytest.approx(1.0)
    assert bbox_iou(_box(0.2, 0.2), _box(0.8, 0.8)) == 0.0


def test_report_propagates_fingerprint_and_serializes_deterministically(tmp_path):
    manifest = _manifest(_sample("a", ()))
    first = evaluate_predictions(manifest, [], split="val")
    second = evaluate_predictions(manifest, [], split="val")
    path = tmp_path / "report.json"
    first.save(path)

    assert first.manifest_fingerprint == "f" * 64
    assert first.to_json() == second.to_json()
    assert json.loads(path.read_text(encoding="utf-8")) == first.to_dict()
    assert "null" in first.to_json()


def test_direct_loaded_p7_manifest_integration(tmp_path):
    root = tmp_path / "dataset"
    for directory in ("train", "valid", "test"):
        (root / directory / "images").mkdir(parents=True)
        (root / directory / "labels").mkdir()
    (root / "data.yaml").write_text(
        "train: ../train/images\nval: ../valid/images\ntest: ../test/images\nnc: 1\nnames: ['blue']\n",
        encoding="utf-8",
    )
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    cv2.imwrite(str(root / "valid" / "images" / "sample.jpg"), image)
    (root / "valid" / "labels" / "sample.txt").write_text(
        "0 0.5 0.5 0.4 0.4\n", encoding="utf-8"
    )
    path = tmp_path / "manifest.json"
    build_manifest(root, path)
    loaded = DatasetManifest.load(path)
    annotation = loaded.split("val")[0].annotations[0]

    report = evaluate_predictions(
        loaded,
        [EvaluationPrediction(loaded.split("val")[0].sample_id, 0, annotation.bbox)],
        split="val",
    )

    assert report.total_samples == report.total_gt_objects == report.total_predictions == 1
    assert report.micro.f1 == 1.0
    assert report.manifest_fingerprint == loaded.fingerprint


def test_greedy_matching_prefers_highest_iou_independent_of_input_order():
    left = _box(0.4, 0.5, 0.3, 0.3)
    right = _box(0.6, 0.5, 0.3, 0.3)
    manifest = _manifest(_sample("a", (_annotation(0, left), _annotation(2, right))))
    predictions = [_prediction("a", 2, right), _prediction("a", 0, _box(0.45, 0.5, 0.3, 0.3))]

    first = evaluate_predictions(manifest, predictions, split="val")
    second = evaluate_predictions(manifest, list(reversed(predictions)), split="val")

    assert first.to_dict() == second.to_dict()
    assert first.correct_class_matches == 2
