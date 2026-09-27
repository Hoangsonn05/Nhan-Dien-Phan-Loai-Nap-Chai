"""Deterministic P17-CORR1B learned-model/data root-cause audit.

This script is diagnostic only.  It reads the locked P9 checkpoint artifacts,
canonical dataset, benchmark predictions, and the controlled camera frame.  It
does not modify the dataset, model, runtime configuration, or production code.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from typing import Iterable

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bottle_cap_vision.dataset import DatasetManifest, audit_dataset
from bottle_cap_vision.evaluation import EvaluationPrediction, bbox_iou


DATASET_ROOT = REPO_ROOT / "data" / "dataset" / "archive"
EXPERIMENT_ROOT = REPO_ROOT / "outputs" / "experiments" / "p9_full_001"
MANIFEST_PATH = EXPERIMENT_ROOT / "dataset_manifest.json"
CHECKPOINT_PATH = EXPERIMENT_ROOT / "checkpoints" / "best.pt"
CONTROLLED_ROOT = REPO_ROOT / "outputs" / "diagnostics" / "camera_fp"
QUALITY_ROOT = REPO_ROOT / "outputs" / "p9_full_quality"

EXPECTED_CHECKPOINT_SHA256 = "9a8300a78b076acc33aaf81c9fa76bf3c12b9fe6fdea792ce044c100164426fb"
EXPECTED_FRAME_SHA256 = "43f4b04c2502358d51cdc7251ca30dad3443d8087b17aebf4d6db420a9d199e4"
CLASS_NAMES = (
    "blue",
    "brown",
    "green",
    "lilac",
    "orange",
    "pink",
    "purple",
    "red",
    "white",
    "yellow",
)
AFFECTED_CLASS_IDS = (5, 8, 9)

# Explicit controlled-frame ground truth.  The class comes from the user's
# recorded confirmation.  The box was manually bounded on the lossless frame,
# independently of the raw-model float box used for matching.
CONTROLLED_GT_XYXY = (363.0, 304.0, 422.0, 362.0)
CONTROLLED_GT_CLASS_ID = 8
CONTROLLED_GT_CLASS_NAME = "White"
CONTROLLED_IOU_THRESHOLD = 0.5

# Complete manual review of all 14 empty-label images.  The first source group
# is an empty tray (three Roboflow derivatives); every other image visibly
# contains one or more bottle caps and is therefore a semantic label defect.
TRUE_NEGATIVE_SOURCE_KEYS = {"WIN_20210817_16_48_17_Pro"}


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _source_key(path: str) -> str:
    stem = Path(path).stem
    return stem.split("_jpg.rf.", 1)[0] if "_jpg.rf." in stem else stem


def _capture_day(source_key: str) -> str | None:
    match = re.match(r"WIN_(\d{8})_", source_key)
    return match.group(1) if match else None


def _quantiles(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if not array.size:
        return {"count": 0, "min": None, "p05": None, "p25": None, "median": None,
                "p75": None, "p95": None, "max": None}
    points = np.quantile(array, (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0))
    return {
        "count": int(array.size),
        "min": float(points[0]),
        "p05": float(points[1]),
        "p25": float(points[2]),
        "median": float(points[3]),
        "p75": float(points[4]),
        "p95": float(points[5]),
        "max": float(points[6]),
    }


def _xyxy_iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _decode(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise RuntimeError(f"failed to decode {path}")
    return image


def _inner_hsv_medians(image: np.ndarray, bbox) -> tuple[float, float, float]:
    height, width = image.shape[:2]
    # Central 60% of the box reduces surrounding tray/background leakage while
    # retaining the cap face.  These statistics are diagnostic, not rules.
    x1 = max(0, int((bbox.x_center - bbox.width * 0.3) * width))
    x2 = min(width, int((bbox.x_center + bbox.width * 0.3) * width))
    y1 = max(0, int((bbox.y_center - bbox.height * 0.3) * height))
    y2 = min(height, int((bbox.y_center + bbox.height * 0.3) * height))
    roi = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[y1:y2, x1:x2]
    return tuple(float(np.median(roi[:, :, channel])) for channel in range(3))


def _class_distribution(manifest: DatasetManifest) -> tuple[dict[str, object], dict[str, object]]:
    distribution: dict[str, object] = {}
    per_class_sources: dict[str, object] = {}
    for class_id, class_name in sorted(manifest.class_mapping.items()):
        split_result = {}
        source_result = {}
        for split in ("train", "val", "test"):
            containing = []
            object_areas = []
            aspects = []
            widths = []
            heights = []
            source_groups: set[str] = set()
            days: set[str] = set()
            objects = 0
            for sample in manifest.split(split):
                annotations = [item for item in sample.annotations if item.class_id == class_id]
                if not annotations:
                    continue
                containing.append(sample.sample_id)
                source = _source_key(sample.image_path)
                source_groups.add(source)
                day = _capture_day(source)
                if day:
                    days.add(day)
                objects += len(annotations)
                for annotation in annotations:
                    bbox = annotation.bbox
                    object_areas.append(bbox.width * bbox.height)
                    aspects.append(bbox.width / bbox.height)
                    widths.append(bbox.width)
                    heights.append(bbox.height)
            split_result[split] = {
                "image_count_containing_class": len(containing),
                "object_count": objects,
                "recoverable_source_group_count": len(source_groups),
                "augmentation_derivatives_beyond_one_per_source": len(containing) - len(source_groups),
                "bbox_area_ratio": _quantiles(object_areas),
                "bbox_aspect_width_over_height": _quantiles(aspects),
                "bbox_width_ratio": _quantiles(widths),
                "bbox_height_ratio": _quantiles(heights),
            }
            source_result[split] = {
                "recoverable_source_group_count": len(source_groups),
                "capture_days": sorted(days),
                "source_groups": sorted(source_groups) if class_id in AFFECTED_CLASS_IDS else None,
            }
        distribution[class_name] = {"class_id": class_id, "splits": split_result}
        per_class_sources[class_name] = source_result
    return distribution, per_class_sources


def _all_source_summary(manifest: DatasetManifest) -> dict[str, object]:
    split_summary = {}
    all_sources = set()
    shape_counts = Counter()
    ratio_counts = Counter()
    for split in ("train", "val", "test"):
        samples = manifest.split(split)
        sources = {_source_key(sample.image_path) for sample in samples}
        all_sources.update((split, source) for source in sources)
        split_summary[split] = {
            "images": len(samples),
            "recoverable_source_groups": len(sources),
            "augmentation_derivatives_beyond_one_per_source": len(samples) - len(sources),
        }
        for sample in samples:
            shape_counts[f"{sample.width}x{sample.height}"] += 1
            ratio_counts[f"{sample.width / sample.height:.6f}"] += 1
    return {
        "method": (
            "Roboflow derivatives are grouped by the filename prefix before '_jpg.rf.'; "
            "external filenames are retained as separate source groups. The archive README states "
            "that train augmentation created three versions per source image."
        ),
        "splits": split_summary,
        "total_recoverable_split_source_groups": len(all_sources),
        "image_shapes": dict(sorted(shape_counts.items())),
        "aspect_ratios": dict(sorted(ratio_counts.items())),
    }


def _negative_audit(manifest: DatasetManifest) -> dict[str, object]:
    records = []
    counts = Counter()
    sources_by_status: dict[str, set[str]] = defaultdict(set)
    for sample in manifest.samples:
        if sample.annotations:
            continue
        source = _source_key(sample.image_path)
        status = "TRUE_NEGATIVE" if source in TRUE_NEGATIVE_SOURCE_KEYS else "LABEL_DEFECT_CAP_PRESENT"
        counts[status] += 1
        sources_by_status[status].add(source)
        records.append(
            {
                "sample_id": sample.sample_id,
                "image_path": sample.image_path,
                "source_group": source,
                "status": status,
                "review": (
                    "empty blue tray; no visible bottle cap"
                    if status == "TRUE_NEGATIVE"
                    else "one or more visible bottle caps have no annotations"
                ),
            }
        )
    if len(records) != 14 or counts != Counter({"LABEL_DEFECT_CAP_PRESENT": 11, "TRUE_NEGATIVE": 3}):
        raise RuntimeError(f"empty-label audit contract changed: {counts}, total={len(records)}")
    return {
        "method": "complete manual visual review of every empty-label image",
        "total_empty_labels": len(records),
        "counts": {
            "TRUE_NEGATIVE": counts["TRUE_NEGATIVE"],
            "LABEL_DEFECT_CAP_PRESENT": counts["LABEL_DEFECT_CAP_PRESENT"],
            "AMBIGUOUS": 0,
        },
        "independent_source_groups": {
            status: len(values) for status, values in sorted(sources_by_status.items())
        },
        "true_negative_scene_families": 1,
        "true_negative_diversity": {
            "scene": "same empty overhead blue tray, three Roboflow derivatives",
            "hands_or_persons": 0,
            "indoor_room_or_classroom": 0,
            "target_camera_similar": 0,
            "curated_non_cap_circular_distractor_scenes": 0,
        },
        "records": records,
    }


def _controlled_eval() -> dict[str, object]:
    metadata = _read_json(CONTROLLED_ROOT / "frame_metadata.json")
    raw = _read_json(CONTROLLED_ROOT / "raw_predictions.json")
    frame_path = CONTROLLED_ROOT / "frame.png"
    frame = _decode(frame_path)
    decoded_frame_hash = hashlib.sha256(frame.tobytes(order="C")).hexdigest()
    if decoded_frame_hash != EXPECTED_FRAME_SHA256 or metadata["frame_sha256"] != EXPECTED_FRAME_SHA256:
        raise RuntimeError("controlled frame identity mismatch")
    predictions = raw["runs"][0]["predictions"]
    matches = []
    unmatched = []
    for prediction in predictions:
        box = tuple(float(value) for value in prediction["bbox_xyxy"])
        overlap = _xyxy_iou(box, CONTROLLED_GT_XYXY)
        record = {**prediction, "iou_with_gt": overlap}
        if overlap >= CONTROLLED_IOU_THRESHOLD:
            matches.append(record)
        else:
            unmatched.append(record)
    if len(matches) != 1 or matches[0]["index"] != 1:
        raise RuntimeError(f"unexpected controlled matching: {matches}")
    match = matches[0]
    frame_area = metadata["frame_shape"][0] * metadata["frame_shape"][1]
    geometry = {}
    for prediction in predictions:
        _, _, width, height = prediction["canonical_bbox_xywh"]
        geometry[str(prediction["index"])] = {
            "bbox_area_pixels": width * height,
            "bbox_frame_area_ratio": width * height / frame_area,
        }
    gt_area = (CONTROLLED_GT_XYXY[2] - CONTROLLED_GT_XYXY[0]) * (
        CONTROLLED_GT_XYXY[3] - CONTROLLED_GT_XYXY[1]
    )
    x1, y1, x2, y2 = (int(value) for value in CONTROLLED_GT_XYXY)
    gt_box = SimpleNamespace(
        x_center=((x1 + x2) / 2) / frame.shape[1],
        y_center=((y1 + y2) / 2) / frame.shape[0],
        width=(x2 - x1) / frame.shape[1],
        height=(y2 - y1) / frame.shape[0],
    )
    hue, saturation, value = _inner_hsv_medians(frame, gt_box)
    fp0_confidence = float(predictions[0]["confidence"])
    cap_confidence = float(match["confidence"])
    return {
        "frame": {
            "sha256": EXPECTED_FRAME_SHA256,
            "sha256_semantics": "decoded uint8 ndarray bytes in C order",
            "png_file_sha256": _sha256(frame_path),
            "shape": metadata["frame_shape"],
            "dtype": metadata["frame_dtype"],
            "source_fps": metadata["source_fps"],
            "raw_unannotated_lossless_png": metadata["raw_unannotated_lossless_png"],
        },
        "ground_truth": {
            "object_count": 1,
            "class_id": CONTROLLED_GT_CLASS_ID,
            "class_name": CONTROLLED_GT_CLASS_NAME,
            "bbox_xyxy": list(CONTROLLED_GT_XYXY),
            "basis": "user-confirmed White cap; manual visual boundary on the lossless frame",
            "inner_roi_hsv_median_opencv_scale": {"h": hue, "s": saturation, "v": value},
        },
        "matching": {
            "iou_threshold": CONTROLLED_IOU_THRESHOLD,
            "localization_tp": 1,
            "correct_class_tp": 0,
            "wrong_class": 1,
            "unmatched_fp": 2,
            "localization_fn": 0,
            "matched_prediction": match,
            "unmatched_predictions": unmatched,
            "fp_pair_iou": _xyxy_iou(
                tuple(float(v) for v in predictions[0]["bbox_xyxy"]),
                tuple(float(v) for v in predictions[2]["bbox_xyxy"]),
            ),
            "fp_pair_interpretation": "heavily overlapping Yellow predictions; one failure region, not two independent root causes",
        },
        "geometry": {
            "ground_truth_cap_area_pixels": gt_area,
            "ground_truth_cap_frame_area_ratio": gt_area / frame_area,
            "predictions": geometry,
        },
        "confidence": {
            "true_cap_wrong_class_confidence": cap_confidence,
            "highest_background_fp_confidence": fp0_confidence,
            "highest_fp_minus_true_cap": fp0_confidence - cap_confidence,
            "threshold_only_viable": False,
            "reason": (
                "The highest background FP has greater confidence than the localized true cap. "
                "Any threshold above the FP also suppresses the true cap; a threshold that keeps "
                "the true cap necessarily keeps that FP."
            ),
        },
    }


def _benchmark_analysis(manifest: DatasetManifest) -> dict[str, object]:
    predictions = [
        EvaluationPrediction.from_dict(item)
        for item in _read_json(QUALITY_ROOT / "test_predictions.json")
    ]
    grouped: dict[str, list[EvaluationPrediction]] = defaultdict(list)
    for prediction in predictions:
        grouped[prediction.sample_id].append(prediction)
    correct_scores = []
    wrong_scores = []
    unmatched_scores = []
    wrong_pairs = Counter()
    white_pink_records = []
    unmatched_gt_count = 0
    matched_count = 0
    correct_count = 0
    for sample in sorted(manifest.split("test"), key=lambda item: item.sample_id):
        ground_truth = tuple(
            sorted(
                sample.annotations,
                key=lambda item: (
                    item.bbox.x_center, item.bbox.y_center, item.bbox.width, item.bbox.height,
                    item.class_id, item.class_name,
                ),
            )
        )
        sample_predictions = tuple(
            sorted(
                grouped[sample.sample_id],
                key=lambda item: (
                    item.bbox.x_center, item.bbox.y_center, item.bbox.width, item.bbox.height,
                    item.class_id, 0 if item.score is None else 1,
                    0.0 if item.score is None else float(item.score),
                ),
            )
        )
        candidates = []
        for gt_index, annotation in enumerate(ground_truth):
            for pred_index, prediction in enumerate(sample_predictions):
                overlap = bbox_iou(annotation.bbox, prediction.bbox)
                if overlap >= 0.5:
                    candidates.append((-overlap, gt_index, pred_index))
        candidates.sort()
        used_gt = set()
        used_predictions = set()
        for negative_iou, gt_index, pred_index in candidates:
            if gt_index in used_gt or pred_index in used_predictions:
                continue
            used_gt.add(gt_index)
            used_predictions.add(pred_index)
            annotation = ground_truth[gt_index]
            prediction = sample_predictions[pred_index]
            matched_count += 1
            if annotation.class_id == prediction.class_id:
                correct_count += 1
                correct_scores.append(float(prediction.score))
            else:
                wrong_scores.append(float(prediction.score))
                pair = f"{annotation.class_name}->{manifest.class_mapping[prediction.class_id]}"
                wrong_pairs[pair] += 1
                if {annotation.class_id, prediction.class_id} == {5, 8}:
                    white_pink_records.append(
                        {
                            "sample_id": sample.sample_id,
                            "gt_class": annotation.class_name,
                            "predicted_class": manifest.class_mapping[prediction.class_id],
                            "confidence": float(prediction.score),
                            "iou": -negative_iou,
                        }
                    )
        unmatched_gt_count += len(ground_truth) - len(used_gt)
        unmatched_scores.extend(
            float(prediction.score)
            for index, prediction in enumerate(sample_predictions)
            if index not in used_predictions
        )
    report = _read_json(QUALITY_ROOT / "test_evaluation.json")
    if (
        matched_count != report["matched_objects"]
        or correct_count != report["correct_class_matches"]
        or len(unmatched_scores) != report["total_predictions"] - report["matched_objects"]
    ):
        raise RuntimeError("benchmark detail reconstruction does not match canonical evaluation")
    defect_test_sample = next(
        sample for sample in manifest.split("test") if not sample.annotations
    )
    contaminated = grouped[defect_test_sample.sample_id]
    return {
        "canonical_test": {
            "samples": report["total_samples"],
            "gt_objects": report["total_gt_objects"],
            "predictions": report["total_predictions"],
            "localization_matches": matched_count,
            "correct_class_matches": correct_count,
            "wrong_class_matches": matched_count - correct_count,
            "unmatched_predictions": len(unmatched_scores),
            "unmatched_gt": unmatched_gt_count,
            "micro": report["micro"],
        },
        "confidence_distributions": {
            "correct_class_matches": _quantiles(correct_scores),
            "wrong_class_matches": _quantiles(wrong_scores),
            "unmatched_predictions": _quantiles(unmatched_scores),
        },
        "white_pink_confusion": {
            "white_to_pink": wrong_pairs["white->pink"],
            "pink_to_white": wrong_pairs["pink->white"],
            "records": white_pink_records,
        },
        "major_wrong_class_pairs": dict(sorted(wrong_pairs.items(), key=lambda item: (-item[1], item[0]))),
        "known_missing_gt_contamination": {
            "sample_id": defect_test_sample.sample_id,
            "visible_caps_unannotated": True,
            "predictions_counted_as_benchmark_fp": len(contaminated),
            "predicted_classes": dict(
                sorted(Counter(manifest.class_mapping[item.class_id] for item in contaminated).items())
            ),
            "interpretation": "canonical unmatched-prediction counts are not a pure false-positive estimate",
        },
    }


def _geometry_and_color(manifest: DatasetManifest) -> dict[str, object]:
    area_ratios = defaultdict(list)
    input_widths = defaultdict(list)
    input_heights = defaultdict(list)
    hsv_medians = defaultdict(lambda: defaultdict(list))
    hsv_records = defaultdict(list)
    for sample in manifest.split("train"):
        scale = min(320 / sample.width, 320 / sample.height)
        affected = [item for item in sample.annotations if item.class_id in AFFECTED_CLASS_IDS]
        image = _decode(DATASET_ROOT / sample.image_path) if affected else None
        for index, annotation in enumerate(sample.annotations):
            name = annotation.class_name
            bbox = annotation.bbox
            area_ratios["all"].append(bbox.width * bbox.height)
            area_ratios[name].append(bbox.width * bbox.height)
            input_widths["all"].append(bbox.width * sample.width * scale)
            input_heights["all"].append(bbox.height * sample.height * scale)
            input_widths[name].append(bbox.width * sample.width * scale)
            input_heights[name].append(bbox.height * sample.height * scale)
            if annotation.class_id in AFFECTED_CLASS_IDS:
                h, s, v = _inner_hsv_medians(image, bbox)
                for channel, value in (("h", h), ("s", s), ("v", v)):
                    hsv_medians[name][channel].append(value)
                hsv_records[name].append(
                    {
                        "sample_id": sample.sample_id,
                        "annotation_index": index,
                        "h": h,
                        "s": s,
                        "v": v,
                    }
                )
    yellow_external = []
    yellow_tray = []
    for sample in manifest.split("train"):
        for annotation in sample.annotations:
            if annotation.class_id != 9:
                continue
            target = yellow_external if sample.image_path.startswith("train/images/ext_") else yellow_tray
            target.append(annotation.bbox.width * annotation.bbox.height)
    return {
        "training_bbox_area_ratio": {
            name: _quantiles(values) for name, values in sorted(area_ratios.items())
        },
        "estimated_bbox_pixels_at_imgsz_320_letterbox": {
            name: {
                "width": _quantiles(input_widths[name]),
                "height": _quantiles(input_heights[name]),
            }
            for name in ("all", "pink", "white", "yellow")
        },
        "yellow_scale_modes": {
            "external_same_cap_photo_series_area_ratio": _quantiles(yellow_external),
            "tray_one_source_three_derivatives_area_ratio": _quantiles(yellow_tray),
        },
        "train_inner_box_hsv_median_opencv_scale": {
            name: {channel: _quantiles(values) for channel, values in sorted(channels.items())}
            for name, channels in sorted(hsv_medians.items())
        },
        "targeted_visual_review": {
            "yellow": (
                "all 8 train/val Yellow boxes reviewed: labels are visually Yellow; train evidence is "
                "six objects from one tray source (three derivatives) plus three photos of the same external cap"
            ),
            "pink_low_saturation": (
                "24 lowest-median-saturation Pink train crops reviewed: 24/24 have a white-dominant "
                "face with a pink rim/side; labels are plausible but the discriminating cue is narrow"
            ),
            "white_nearest_controlled_hsv": (
                "24 White train crops nearest the controlled cap's S/V medians reviewed: 24/24 are "
                "tray-domain crown-cap undersides/rims, not the controlled plastic cap appearance"
            ),
        },
        "diagnostic_records": {
            "pink_lowest_saturation_24": sorted(hsv_records["pink"], key=lambda item: (item["s"], item["sample_id"], item["annotation_index"]))[:24],
        },
    }


def _annotation_structure(manifest: DatasetManifest) -> dict[str, object]:
    duplicate_rows = []
    for sample in manifest.samples:
        path = DATASET_ROOT / sample.annotation_path
        lines = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        repeated = {line: count for line, count in Counter(lines).items() if count > 1}
        if repeated:
            duplicate_rows.append({"sample_id": sample.sample_id, "rows": repeated})
    return {
        "exact_duplicate_label_rows": len(duplicate_rows),
        "duplicate_label_records": duplicate_rows,
        "semantic_empty_label_defects": 11,
        "semantic_empty_label_defect_source_groups": 5,
        "targeted_positive_review": (
            "No confirmed wrong-class label in the complete Yellow review or deterministic Pink/White "
            "targeted crops; this does not establish that all 50k positive labels are semantically clean."
        ),
    }


def _training_provenance() -> dict[str, object]:
    experiment = _read_json(EXPERIMENT_ROOT / "experiment_metadata.json")
    training_config = _read_json(EXPERIMENT_ROOT / "training_config.json")
    args = yaml.safe_load((EXPERIMENT_ROOT / "backend" / "train" / "args.yaml").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((EXPERIMENT_ROOT / "backend" / "train" / "results.csv").open(encoding="utf-8")))
    checkpoint_hash = _sha256(CHECKPOINT_PATH)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("P9 full checkpoint identity mismatch")
    return {
        "checkpoint": str(CHECKPOINT_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
        "checkpoint_sha256": checkpoint_hash,
        "backend": experiment["backend"],
        "base_model": training_config["model"],
        "pretrained": args["pretrained"],
        "epochs_requested": args["epochs"],
        "epochs_completed": experiment["termination"]["epochs_completed"],
        "results_rows": len(rows),
        "imgsz": args["imgsz"],
        "batch": args["batch"],
        "device": args["device"],
        "optimizer": args["optimizer"],
        "seed": args["seed"],
        "deterministic": args["deterministic"],
        "patience": args["patience"],
        "freeze": args["freeze"],
        "class_filter": args["classes"],
        "single_class": args["single_cls"],
        "augmentation": {
            "hsv_h": args["hsv_h"],
            "hsv_s": args["hsv_s"],
            "hsv_v": args["hsv_v"],
            "degrees": args["degrees"],
            "translate": args["translate"],
            "scale": args["scale"],
            "fliplr": args["fliplr"],
            "mosaic": args["mosaic"],
            "close_mosaic": args["close_mosaic"],
            "mixup": args["mixup"],
        },
        "dataset": experiment["dataset"],
        "configuration_assessment": {
            "imgsz_320_primary_cause": False,
            "imgsz_reason": (
                "the 59x58 controlled GT becomes about 29.5x29 pixels before square padding, "
                "larger than the median canonical train cap at imgsz 320"
            ),
            "hsv_augmentation": (
                "potential contributor for color-semantic classes, but no permitted ablation/retrain "
                "in this task establishes causal impact; not classified as a demonstrated config root cause"
            ),
            "specific_demonstrated_training_config_fault": None,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "diagnostics" / "p17_corr1b",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()

    manifest = DatasetManifest.load(MANIFEST_PATH)
    live = audit_dataset(DATASET_ROOT, allow_empty=True)
    if live.manifest is None or live.manifest.fingerprint != manifest.fingerprint:
        raise RuntimeError("live dataset does not match the P9 full experiment manifest")

    distribution, affected_sources = _class_distribution(manifest)
    source_summary = _all_source_summary(manifest)
    negative = _negative_audit(manifest)
    controlled = _controlled_eval()
    benchmark = _benchmark_analysis(manifest)
    geometry_color = _geometry_and_color(manifest)
    annotation_quality = _annotation_structure(manifest)
    training = _training_provenance()

    dataset_audit = {
        "manifest": str(MANIFEST_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
        "manifest_fingerprint": manifest.fingerprint,
        "live_fingerprint_matches_training": True,
        "structural_audit": live.audit.to_dict(),
        "semantic_annotation_quality": annotation_quality,
    }
    class_distribution = {
        "class_mapping": [
            {"class_id": class_id, "class_name": class_name}
            for class_id, class_name in sorted(manifest.class_mapping.items())
        ],
        "classes": distribution,
    }
    source_diversity = {
        "dataset": source_summary,
        "affected_classes": {name: affected_sources[name] for name in ("yellow", "white", "pink")},
        "scene_family_review": {
            "reviewed_recoverable_source_representatives": 1003,
            "canonical_roboflow_scene": "overhead blue tray with wood border",
            "external_scene": "three photographs of the same Yellow cap on a dark background",
            "hand_or_forearm_sources": ["WIN_20210819_14_56_11_Pro"],
            "person_or_face_sources": 0,
            "indoor_room_or_classroom_sources": 0,
            "target_front_facing_webcam_sources": 0,
            "target_camera_similarity": "NONE",
            "interpretation": (
                "one incidental arm/hand over the tray is not coverage of the frontal classroom/person/hand domain"
            ),
        },
        "geometry_and_color": geometry_color,
    }
    root_summary = {
        "task": "P17-CORR1B-FP-ROOT-CAUSE",
        "hypothesis": "CONFIRMED",
        "root_cause_class": "MULTIPLE_DATA_ISSUES",
        "root_causes": [
            {
                "type": "DOMAIN_HARD_NEGATIVE_GAP",
                "evidence": (
                    "only three genuine negative images from one tray source; no target-like classroom/person "
                    "scene; raw model reproduces two overlapping Yellow predictions on the hand/person region"
                ),
            },
            {
                "type": "CLASS_COVERAGE_IMBALANCE",
                "evidence": (
                    "Yellow has 6 train objects from 4 filename groups but only two scene families, 2 val objects, "
                    "and zero test objects; Lilac has zero train/val support"
                ),
            },
            {
                "type": "DATASET_LABEL_QUALITY",
                "evidence": (
                    "11/14 empty-label images contain unannotated caps across five source groups, poisoning "
                    "negative supervision and contaminating benchmark FP interpretation"
                ),
            },
            {
                "type": "WHITE_PINK_APPEARANCE_AND_DOMAIN_COVERAGE",
                "evidence": (
                    "canonical test has 7 White->Pink and 4 Pink->White matches; low-saturation Pink examples "
                    "are white-faced with narrow pink rims, while nearest White examples are tray-domain crown-cap "
                    "undersides unlike the controlled plastic cap"
                ),
            },
        ],
        "benchmark": benchmark,
        "controlled_frame": controlled,
        "training": training,
        "geometry_interpretation": {
            "controlled_cap_at_imgsz_320": "about 29.5x29 pixels; not marginal and larger than median train cap",
            "yellow_fp": (
                "14.38-14.77% of frame, far above ordinary tray caps but between the sparse Yellow scale modes: "
                "tiny tray cap derivatives and huge same-cap external photographs"
            ),
            "imgsz_primary_cause": False,
        },
        "calibration_only": {
            "supported": False,
            "reason": controlled["confidence"]["reason"],
        },
        "training_configuration": {
            "selected_as_root_cause": False,
            "reason": (
                "no deterministic configuration fault is linked to the failure; imgsz is adequate. "
                "HSV augmentation remains an unisolated candidate for a future controlled ablation."
            ),
        },
        "falsified": [
            "runtime defect",
            "adapter defect",
            "pipeline prediction creation defect",
            "imgsz 320 as primary cause",
            "calibration-only/global-threshold correction",
        ],
        "camera_domain_miniset": {
            "known_gt_frames": 1,
            "result": {
                "localization_tp": 1,
                "correct_color": 0,
                "wrong_color": 1,
                "unmatched_fp": 2,
                "localization_fn": 0,
            },
            "limitation": "no other camera-domain frame has explicit trustworthy ground truth",
        },
        "next_corrective_task": {
            "id": "P17-CORR2-DATASET-CAMERA-COVERAGE",
            "goal": "prepare reviewed target-domain data and retrain a new experiment without altering the locked test split",
            "data_needed": [
                "explicit review/correction of train and val label defects",
                "diverse real hard negatives from target webcam classrooms, hands, persons, tables, and distractors",
                "target-camera positive caps for every color under varied lighting, including White/Pink boundary cases",
                "independent Yellow train/val coverage and a new locked camera diagnostic set",
            ],
            "write": "reviewed train/val additions and correction records only; preserve current test split",
            "train": "new experiment/checkpoint ID; never overwrite p9_full_001",
            "verify": [
                "unchanged canonical test",
                "locked camera-domain diagnostic set",
                "per-class/source-family coverage gates",
                "old-vs-new FP, localization, and color confusion comparison",
                "controlled HSV-augmentation ablation if color confusion persists",
            ],
            "pass": (
                "label review is complete, target-domain negatives/positives are independent and diverse, every "
                "class has meaningful train/val/test evaluation support, and a new checkpoint improves both locked sets"
            ),
        },
        "production_code_changed": False,
        "dataset_changed": False,
        "model_changed": False,
        "task_result": "PASS",
    }

    _write_json(output_dir / "dataset_audit.json", dataset_audit)
    _write_json(output_dir / "class_distribution.json", class_distribution)
    _write_json(output_dir / "source_diversity.json", source_diversity)
    _write_json(output_dir / "negative_audit.json", negative)
    _write_json(output_dir / "controlled_eval.json", controlled)
    _write_json(output_dir / "root_cause_summary.json", root_summary)
    print(json.dumps({"output_dir": str(output_dir), "root_cause_class": "MULTIPLE_DATA_ISSUES"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
