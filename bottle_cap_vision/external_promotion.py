"""Manual-review and controlled-promotion boundary for external cap images.

P13A intake deliberately stops at ``PENDING``.  This module owns the later
state transitions and refuses to copy anything into the canonical dataset
until visual review, annotation review, provenance, duplicate, split and
destination checks all pass.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
from typing import Mapping, Sequence

import cv2
import numpy as np

from .dataset import BoundingBox, load_yolo_class_mapping
from .external_intake import (
    ExternalIntakeError,
    pixel_bbox_to_yolo,
    tree_fingerprint,
    validate_canonical_bbox,
)


CANONICAL_CLASS_MAPPING = {
    0: "blue", 1: "brown", 2: "green", 3: "lilac", 4: "orange",
    5: "pink", 6: "purple", 7: "red", 8: "white", 9: "yellow",
}
REVIEW_DECISIONS = frozenset({"APPROVED_FOR_ANNOTATION", "REJECTED", "AMBIGUOUS"})
PROMOTABLE_SPLITS = {"train": "train", "val": "valid"}


class ExternalPromotionError(ValueError):
    """Raised before promotion when any P13B safety gate fails."""


def record_visual_review(
    staging_root: Path,
    staging_id: str,
    *,
    decision: str,
    reviewer: str,
    notes: str,
    source_group: str | None = None,
    source_group_basis: str | None = None,
) -> dict[str, object]:
    """Record the human visual decision without creating ground truth."""

    if decision not in REVIEW_DECISIONS:
        raise ExternalPromotionError(f"unsupported visual review decision: {decision!r}")
    if not reviewer.strip() or not notes.strip():
        raise ExternalPromotionError("reviewer and visual-review notes are required")
    records = _load_queue(staging_root)
    record = _find_record(records, staging_id)
    if source_group is not None:
        if not source_group.strip() or not source_group_basis or not source_group_basis.strip():
            raise ExternalPromotionError("manual source_group requires a documented basis")
        record.setdefault("file_source_group", record.get("source_group"))
        record["source_group"] = source_group
        record["source_group_basis"] = source_group_basis
    record["review_status"] = decision
    record["annotation_status"] = "IMAGE_ONLY" if decision == "APPROVED_FOR_ANNOTATION" else decision
    record["reviewer"] = reviewer
    record["review_notes"] = notes
    _save_queue(staging_root, records)
    return record


def save_proposed_annotation(
    staging_root: Path,
    staging_id: str,
    boxes: Sequence[tuple[int, float, float, float, float]],
) -> Path:
    """Save manually selected pixel boxes as a proposed canonical YOLO label."""

    records = _load_queue(staging_root)
    record = _find_record(records, staging_id)
    if record.get("review_status") != "APPROVED_FOR_ANNOTATION":
        raise ExternalPromotionError("only APPROVED_FOR_ANNOTATION items may receive annotations")
    image_path = _resolve_staging_path(staging_root, record.get("image_path"), "image_path")
    image = _decode_image(image_path)
    height, width = image.shape[:2]
    if not boxes:
        raise ExternalPromotionError("at least one manual bbox is required")
    lines: list[str] = []
    seen: set[tuple[int, float, float, float, float]] = set()
    for raw in boxes:
        if len(raw) != 5:
            raise ExternalPromotionError("manual bbox requires class_id xmin ymin xmax ymax")
        class_id, xmin, ymin, xmax, ymax = raw
        if isinstance(class_id, bool) or not isinstance(class_id, int) or class_id not in CANONICAL_CLASS_MAPPING:
            raise ExternalPromotionError(f"unknown canonical class ID: {class_id!r}")
        try:
            bbox = pixel_bbox_to_yolo(xmin, ymin, xmax, ymax, width, height, clip=False)
        except ExternalIntakeError as exc:
            raise ExternalPromotionError(str(exc)) from exc
        key = (class_id, bbox.x_center, bbox.y_center, bbox.width, bbox.height)
        if key in seen:
            raise ExternalPromotionError("duplicate identical bbox")
        seen.add(key)
        lines.append(_format_yolo(class_id, bbox))
    label_path = staging_root / "review" / "annotations" / f"{staging_id}.txt"
    _atomic_text(label_path, "\n".join(lines) + "\n")
    record["label_path"] = label_path.relative_to(staging_root).as_posix()
    record["annotation_status"] = "PROPOSED"
    record["mapping_status"] = "PROPOSED"
    record["proposed_class_id"] = boxes[0][0] if len({box[0] for box in boxes}) == 1 else None
    record["proposed_class"] = (
        CANONICAL_CLASS_MAPPING[boxes[0][0]] if record["proposed_class_id"] is not None else "multiple"
    )
    _save_queue(staging_root, records)
    return label_path


def validate_annotation(
    image_path: Path,
    label_path: Path,
    class_mapping: Mapping[int, str] = CANONICAL_CLASS_MAPPING,
) -> tuple[tuple[int, BoundingBox], ...]:
    """Strictly validate image/label pairing and canonical YOLO contents."""

    image = _decode_image(image_path)
    if image_path.stem != label_path.stem:
        raise ExternalPromotionError("image and label stems do not pair exactly")
    try:
        lines = label_path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise ExternalPromotionError(f"cannot read annotation: {exc}") from exc
    if not lines or not any(line.strip() for line in lines):
        raise ExternalPromotionError("approved annotation cannot be empty")
    parsed: list[tuple[int, BoundingBox]] = []
    seen: set[tuple[int, float, float, float, float]] = set()
    for line_number, line in enumerate(lines, 1):
        parts = line.split()
        if len(parts) != 5 or not re.fullmatch(r"[0-9]+", parts[0]):
            raise ExternalPromotionError(f"line {line_number}: expected canonical YOLO fields")
        class_id = int(parts[0])
        if class_id not in class_mapping:
            raise ExternalPromotionError(f"line {line_number}: unknown canonical class ID {class_id}")
        try:
            bbox = BoundingBox(*(float(value) for value in parts[1:]))
            validate_canonical_bbox(bbox)
        except (ValueError, TypeError) as exc:
            raise ExternalPromotionError(f"line {line_number}: invalid bbox") from exc
        key = (class_id, bbox.x_center, bbox.y_center, bbox.width, bbox.height)
        if key in seen:
            raise ExternalPromotionError(f"line {line_number}: duplicate identical bbox")
        seen.add(key)
        parsed.append((class_id, bbox))
    if image.size == 0:  # defensive: _decode_image already rejects this
        raise ExternalPromotionError("image decode failed")
    return tuple(parsed)


def generate_annotation_overlay(staging_root: Path, staging_id: str) -> Path:
    """Render the proposed boxes for the mandatory visual overlay review."""

    records = _load_queue(staging_root)
    record = _find_record(records, staging_id)
    if record.get("annotation_status") not in {"PROPOSED", "REVIEWED", "APPROVED"}:
        raise ExternalPromotionError("an overlay requires a proposed annotation")
    image_path = _resolve_staging_path(staging_root, record.get("image_path"), "image_path")
    label_path = _resolve_staging_path(staging_root, record.get("label_path"), "label_path")
    annotations = validate_annotation(image_path, label_path)
    image = _decode_image(image_path).copy()
    height, width = image.shape[:2]
    for class_id, bbox in annotations:
        x1 = int(round((bbox.x_center - bbox.width / 2) * width))
        y1 = int(round((bbox.y_center - bbox.height / 2) * height))
        x2 = int(round((bbox.x_center + bbox.width / 2) * width))
        y2 = int(round((bbox.y_center + bbox.height / 2) * height))
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), max(2, round(min(width, height) / 400)))
        cv2.putText(
            image, f"{class_id} {CANONICAL_CLASS_MAPPING[class_id]}", (x1, max(30, y1 - 12)),
            cv2.FONT_HERSHEY_SIMPLEX, max(0.7, min(width, height) / 1400), (0, 0, 255), 2, cv2.LINE_AA,
        )
    output = staging_root / "review" / "approved_preview" / f"{staging_id}.jpg"
    output.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise ExternalPromotionError("cannot encode annotation overlay")
    temporary = output.with_suffix(".jpg.tmp")
    temporary.write_bytes(encoded.tobytes())
    temporary.replace(output)
    record["overlay_path"] = output.relative_to(staging_root).as_posix()
    record["annotation_status"] = "REVIEWED"
    _save_queue(staging_root, records)
    return output


def approve_annotation(staging_root: Path, staging_id: str, *, reviewer: str, notes: str) -> None:
    """Approve only a valid annotation whose rendered overlay was reviewed."""

    if not reviewer.strip() or not notes.strip():
        raise ExternalPromotionError("overlay reviewer and notes are required")
    records = _load_queue(staging_root)
    record = _find_record(records, staging_id)
    if record.get("review_status") != "APPROVED_FOR_ANNOTATION" or record.get("annotation_status") != "REVIEWED":
        raise ExternalPromotionError("annotation must pass visual review and overlay generation before approval")
    image_path = _resolve_staging_path(staging_root, record.get("image_path"), "image_path")
    label_path = _resolve_staging_path(staging_root, record.get("label_path"), "label_path")
    overlay_path = _resolve_staging_path(staging_root, record.get("overlay_path"), "overlay_path")
    if not overlay_path.is_file():
        raise ExternalPromotionError("annotation overlay is missing")
    validate_annotation(image_path, label_path)
    record["review_status"] = "APPROVED"
    record["annotation_status"] = "APPROVED"
    record["overlay_reviewer"] = reviewer
    record["overlay_review_notes"] = notes
    _save_queue(staging_root, records)


def create_promotion_plan(
    staging_root: Path,
    dataset_root: Path,
    assignments: Mapping[str, str],
    plan_path: Path,
) -> dict[str, object]:
    """Validate approved records and write a deterministic, non-mutating plan."""

    mapping = load_yolo_class_mapping(dataset_root / "data.yaml")[0]
    if mapping != CANONICAL_CLASS_MAPPING:
        raise ExternalPromotionError("canonical class mapping differs from the locked P13B mapping")
    records = _load_queue(staging_root)
    selected = []
    for staging_id, split in sorted(assignments.items()):
        record = _find_record(records, staging_id)
        if split not in PROMOTABLE_SPLITS:
            raise ExternalPromotionError("promotion destination must be train or val; test is forbidden")
        if record.get("review_status") != "APPROVED" or record.get("annotation_status") != "APPROVED":
            raise ExternalPromotionError(f"{staging_id}: only fully APPROVED records may be promoted")
        selected.append((record, split))
    approved_ids = {str(record["staging_id"]) for record in records if record.get("review_status") == "APPROVED"}
    if set(assignments) != approved_ids:
        missing = sorted(approved_ids - set(assignments))
        extra = sorted(set(assignments) - approved_ids)
        raise ExternalPromotionError(f"assignments must cover exactly approved items; missing={missing}, extra={extra}")
    if not selected:
        raise ExternalPromotionError("promotion plan has no approved items")

    group_splits: dict[str, str] = _previous_group_splits(staging_root)
    canonical_hashes = _canonical_image_hashes(dataset_root)
    selected_hashes: set[str] = set()
    items: list[dict[str, object]] = []
    for record, split in selected:
        staging_id = str(record["staging_id"])
        source_id = _required_text(record, "source_id")
        source_group = _required_text(record, "source_group")
        previous_split = group_splits.setdefault(source_group, split)
        if previous_split != split:
            raise ExternalPromotionError(f"source group {source_group!r} would cross train/val")
        image_path = _resolve_staging_path(staging_root, record.get("image_path"), "image_path")
        label_path = _resolve_staging_path(staging_root, record.get("label_path"), "label_path")
        annotations = validate_annotation(image_path, label_path, mapping)
        digest = _sha256(image_path)
        if record.get("canonical_duplicate") is True or digest in canonical_hashes:
            raise ExternalPromotionError(f"{staging_id}: exact duplicate of canonical data")
        if digest in selected_hashes:
            raise ExternalPromotionError(f"{staging_id}: exact duplicate within approved staging data")
        selected_hashes.add(digest)
        source_manifest = _source_manifest(staging_root, source_id)
        license_name = _required_text(source_manifest, "license")
        if license_name.casefold() == "unknown" or record.get("license") != license_name:
            raise ExternalPromotionError(f"{staging_id}: license is unknown or inconsistent")
        attribution = _required_text(source_manifest, "attribution")
        source_page = _required_text(source_manifest, "source_page")
        provenance = record.get("provenance")
        if not isinstance(provenance, dict):
            raise ExternalPromotionError(f"{staging_id}: provenance is missing")
        original_source = _required_text(provenance, "original_source")
        safe_source = re.sub(r"[^a-zA-Z0-9_-]+", "_", source_id).strip("_")
        if not safe_source:
            raise ExternalPromotionError(f"{staging_id}: source_id is not filename-safe")
        destination_name = f"ext_{safe_source}_{staging_id}{image_path.suffix.lower()}"
        split_directory = PROMOTABLE_SPLITS[split]
        destination_image = PurePosixPath(split_directory, "images", destination_name).as_posix()
        destination_label = PurePosixPath(split_directory, "labels", Path(destination_name).stem + ".txt").as_posix()
        if (dataset_root / Path(destination_image)).exists() or (dataset_root / Path(destination_label)).exists():
            raise ExternalPromotionError(f"{staging_id}: destination collision")
        items.append({
            "staging_id": staging_id,
            "source_id": source_id,
            "source_group": source_group,
            "file_source_group": record.get("file_source_group", source_group),
            "source_image_path": image_path.relative_to(staging_root.resolve()).as_posix(),
            "approved_label_path": label_path.relative_to(staging_root.resolve()).as_posix(),
            "canonical_class_ids": sorted({class_id for class_id, _ in annotations}),
            "destination_split": split,
            "destination_image_path": destination_image,
            "destination_label_path": destination_label,
            "sha256": digest,
            "label_sha256": _sha256(label_path),
            "license": license_name,
            "attribution": attribution,
            "source_page": source_page,
            "original_source": original_source,
            "review_status": "APPROVED",
            "annotation_status": "APPROVED",
        })
    seed = json.dumps(items, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    run_id = "p13b-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    plan = {
        "schema_version": "1.0",
        "promotion_run_id": run_id,
        "old_canonical_dataset_fingerprint": tree_fingerprint(dataset_root),
        "canonical_test_fingerprint": tree_fingerprint(dataset_root / "test"),
        "items": items,
    }
    _atomic_json(plan_path, plan)
    return plan


def promote_approved(
    staging_root: Path,
    dataset_root: Path,
    plan_path: Path,
    manifest_path: Path,
    attribution_path: Path,
) -> dict[str, object]:
    """Copy every planned pair, rolling back all new files on any failure."""

    plan = _read_json_object(plan_path)
    items = plan.get("items")
    if not isinstance(items, list) or not items:
        raise ExternalPromotionError("promotion plan items are missing")
    if manifest_path.exists():
        raise ExternalPromotionError("promotion manifest already exists; logs are immutable")
    if tree_fingerprint(dataset_root) != plan.get("old_canonical_dataset_fingerprint"):
        raise ExternalPromotionError("canonical dataset changed after the promotion plan was created")
    test_before = tree_fingerprint(dataset_root / "test")
    if test_before != plan.get("canonical_test_fingerprint"):
        raise ExternalPromotionError("canonical test changed after the promotion plan was created")

    mapping = load_yolo_class_mapping(dataset_root / "data.yaml")[0]
    if mapping != CANONICAL_CLASS_MAPPING:
        raise ExternalPromotionError("canonical class mapping differs from the locked P13B mapping")
    queue = _load_queue(staging_root)
    group_splits = _previous_group_splits(staging_root)
    validated: list[tuple[dict[str, object], Path, Path, Path, Path]] = []
    seen_hashes: set[str] = set()
    for raw in items:
        if not isinstance(raw, dict):
            raise ExternalPromotionError("promotion plan item is invalid")
        staging_id = _required_text(raw, "staging_id")
        record = _find_record(queue, staging_id)
        if record.get("review_status") != "APPROVED" or record.get("annotation_status") != "APPROVED":
            raise ExternalPromotionError(f"{staging_id}: queue approval was revoked after planning")
        source_id = _required_text(raw, "source_id")
        source_group = _required_text(raw, "source_group")
        if record.get("source_id") != source_id or record.get("source_group") != source_group:
            raise ExternalPromotionError(f"{staging_id}: source identity changed after planning")
        split = _required_text(raw, "destination_split")
        if split not in PROMOTABLE_SPLITS:
            raise ExternalPromotionError("promotion destination must be train or val; test is forbidden")
        previous_split = group_splits.setdefault(source_group, split)
        if previous_split != split:
            raise ExternalPromotionError(f"source group {source_group!r} would cross train/val")
        if record.get("image_path") != raw.get("source_image_path") or record.get("label_path") != raw.get("approved_label_path"):
            raise ExternalPromotionError(f"{staging_id}: approved image/label path changed after planning")
        source_image = _resolve_staging_path(staging_root, raw.get("source_image_path"), "source_image_path")
        source_label = _resolve_staging_path(staging_root, raw.get("approved_label_path"), "approved_label_path")
        annotations = validate_annotation(source_image, source_label, mapping)
        class_ids = sorted({class_id for class_id, _ in annotations})
        if raw.get("canonical_class_ids") != class_ids:
            raise ExternalPromotionError(f"{staging_id}: approved class IDs changed after planning")
        image_hash = _sha256(source_image)
        if image_hash != raw.get("sha256") or image_hash in seen_hashes:
            raise ExternalPromotionError(f"{staging_id}: approved image changed or duplicates another plan item")
        seen_hashes.add(image_hash)
        if _sha256(source_label) != raw.get("label_sha256"):
            raise ExternalPromotionError(f"{staging_id}: approved label changed after planning")
        source_manifest = _source_manifest(staging_root, source_id)
        provenance = record.get("provenance")
        if not isinstance(provenance, dict):
            raise ExternalPromotionError(f"{staging_id}: provenance is missing")
        if (
            raw.get("license") != _required_text(source_manifest, "license")
            or raw.get("attribution") != _required_text(source_manifest, "attribution")
            or raw.get("source_page") != _required_text(source_manifest, "source_page")
            or raw.get("original_source") != _required_text(provenance, "original_source")
        ):
            raise ExternalPromotionError(f"{staging_id}: provenance changed after planning")
        safe_source = re.sub(r"[^a-zA-Z0-9_-]+", "_", source_id).strip("_")
        destination_name = f"ext_{safe_source}_{staging_id}{source_image.suffix.lower()}"
        split_directory = PROMOTABLE_SPLITS[split]
        expected_image = PurePosixPath(split_directory, "images", destination_name).as_posix()
        expected_label = PurePosixPath(split_directory, "labels", Path(destination_name).stem + ".txt").as_posix()
        if raw.get("destination_image_path") != expected_image or raw.get("destination_label_path") != expected_label:
            raise ExternalPromotionError(f"{staging_id}: destination naming is not deterministic")
        destination_image = _resolve_dataset_path(dataset_root, expected_image)
        destination_label = _resolve_dataset_path(dataset_root, expected_label)
        if destination_image.exists() or destination_label.exists():
            raise ExternalPromotionError("promotion destination collision")
        validated.append((raw, source_image, source_label, destination_image, destination_label))

    temporary_paths: list[Path] = []
    committed_paths: list[Path] = []
    attribution_before = attribution_path.read_bytes() if attribution_path.exists() else None
    try:
        for raw, source_image, source_label, destination_image, destination_label in validated:
            destination_image.parent.mkdir(parents=True, exist_ok=True)
            destination_label.parent.mkdir(parents=True, exist_ok=True)
            temp_image = destination_image.with_name(destination_image.name + ".p13b.tmp")
            temp_label = destination_label.with_name(destination_label.name + ".p13b.tmp")
            shutil.copy2(source_image, temp_image)
            temporary_paths.append(temp_image)
            shutil.copy2(source_label, temp_label)
            temporary_paths.append(temp_label)
            if _sha256(temp_image) != raw["sha256"]:
                raise ExternalPromotionError("temporary image hash mismatch")
        for _raw, _source_image, _source_label, destination_image, destination_label in validated:
            temp_image = destination_image.with_name(destination_image.name + ".p13b.tmp")
            temp_label = destination_label.with_name(destination_label.name + ".p13b.tmp")
            temp_image.replace(destination_image)
            committed_paths.append(destination_image)
            temp_label.replace(destination_label)
            committed_paths.append(destination_label)
        test_after = tree_fingerprint(dataset_root / "test")
        if test_after != test_before:
            raise ExternalPromotionError("canonical test changed during promotion")
        _write_attribution(attribution_path, str(plan["promotion_run_id"]), items)
        manifest = {
            "schema_version": "1.0",
            "promotion_run_id": plan["promotion_run_id"],
            "promoted_files": items,
            "old_canonical_dataset_fingerprint": plan["old_canonical_dataset_fingerprint"],
            "new_canonical_dataset_fingerprint": tree_fingerprint(dataset_root),
            "canonical_test_fingerprint_before": test_before,
            "canonical_test_fingerprint_after": test_after,
            "reviewer_state": "APPROVED",
        }
        _atomic_json(manifest_path, manifest)
        return manifest
    except Exception:
        for path in reversed(committed_paths):
            path.unlink(missing_ok=True)
        for path in temporary_paths:
            path.unlink(missing_ok=True)
        if attribution_before is None:
            attribution_path.unlink(missing_ok=True)
        else:
            attribution_path.parent.mkdir(parents=True, exist_ok=True)
            attribution_path.write_bytes(attribution_before)
        manifest_path.unlink(missing_ok=True)
        raise


def _load_queue(staging_root: Path) -> list[dict[str, object]]:
    path = staging_root / "review" / "review_queue.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalPromotionError(f"cannot read review queue: {exc}") from exc
    if not isinstance(raw, list) or not all(isinstance(item, dict) for item in raw):
        raise ExternalPromotionError("review queue must be an array of records")
    ids = [item.get("staging_id") for item in raw]
    if any(not isinstance(item, str) or not item for item in ids) or len(ids) != len(set(ids)):
        raise ExternalPromotionError("review queue staging IDs must be unique non-empty strings")
    return raw


def _find_record(records: Sequence[dict[str, object]], staging_id: str) -> dict[str, object]:
    matches = [record for record in records if record.get("staging_id") == staging_id]
    if len(matches) != 1:
        raise ExternalPromotionError(f"review item not found: {staging_id}")
    return matches[0]


def _save_queue(staging_root: Path, records: Sequence[dict[str, object]]) -> None:
    ordered = sorted(records, key=lambda item: (str(item.get("source_id", "")), str(item["staging_id"])))
    _atomic_json(staging_root / "review" / "review_queue.json", ordered)
    fields = [
        "staging_id", "object_id", "source_id", "image_path", "crop_path", "label_path",
        "source_class", "proposed_class", "proposed_class_id", "heuristic_score", "mapping_status",
        "file_source_group", "source_group", "source_group_basis", "canonical_duplicate", "license",
        "review_required", "review_status", "annotation_status", "reviewer", "notes", "overlay_path",
    ]
    path = staging_root / "review" / "review_queue.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in ordered:
            writer.writerow({field: record.get(field) for field in fields})
    temporary.replace(path)


def _source_manifest(staging_root: Path, source_id: str) -> dict[str, object]:
    return _read_json_object(staging_root / "raw" / source_id / "source_manifest.json")


def _previous_group_splits(staging_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted((staging_root / "review").glob("promotion_manifest*.json")):
        raw = _read_json_object(path)
        items = raw.get("promoted_files", [])
        if not isinstance(items, list):
            raise ExternalPromotionError(f"invalid prior promotion manifest: {path}")
        for item in items:
            if not isinstance(item, dict):
                raise ExternalPromotionError(f"invalid prior promotion record: {path}")
            group = _required_text(item, "source_group")
            split = _required_text(item, "destination_split")
            prior = result.setdefault(group, split)
            if prior != split:
                raise ExternalPromotionError(f"prior manifests already leak source group {group!r}")
    return result


def _canonical_image_hashes(dataset_root: Path) -> set[str]:
    supported = {".jpg", ".jpeg", ".png", ".bmp"}
    return {
        _sha256(path)
        for path in dataset_root.rglob("*")
        if path.is_file() and path.suffix.lower() in supported
    }


def _write_attribution(path: Path, run_id: str, items: Sequence[object]) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else "# External data attribution\n"
    if f"## {run_id}\n" in existing:
        raise ExternalPromotionError("attribution already contains this promotion run")
    typed = [item for item in items if isinstance(item, dict)]
    first = typed[0]
    lines = [
        "", f"## {run_id}", "", f"- Source: {first['source_page']}",
        f"- Author/attribution: {first['attribution']}", f"- License: {first['license']}",
        "- Included files:",
    ]
    for item in typed:
        lines.append(f"  - `{item['destination_image_path']}` from {item['original_source']}")
    _atomic_text(path, existing.rstrip() + "\n" + "\n".join(lines) + "\n")


def _decode_image(path: Path) -> np.ndarray:
    try:
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except OSError as exc:
        raise ExternalPromotionError(f"cannot read image {path}: {exc}") from exc
    if image is None:
        raise ExternalPromotionError(f"image decode failed: {path}")
    return image


def _resolve_staging_path(staging_root: Path, raw: object, field: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ExternalPromotionError(f"{field} must be a non-empty staging-relative path")
    return _resolve_relative(staging_root, raw, field)


def _resolve_dataset_path(dataset_root: Path, raw: object) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ExternalPromotionError("destination path must be non-empty")
    path = PurePosixPath(raw)
    if len(path.parts) < 3 or path.parts[0] not in PROMOTABLE_SPLITS.values() or path.parts[1] not in {"images", "labels"}:
        raise ExternalPromotionError("destination must be a canonical train/val image or label path")
    return _resolve_relative(dataset_root, raw, "destination path")


def _resolve_relative(root: Path, raw: str, field: str) -> Path:
    normalized = raw.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or "." in path.parts or path.as_posix() != normalized:
        raise ExternalPromotionError(f"{field} must stay inside its root")
    resolved_root = root.resolve()
    resolved = (resolved_root / Path(*path.parts)).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ExternalPromotionError(f"{field} escapes its root") from exc
    return resolved


def _required_text(raw: Mapping[str, object], field: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ExternalPromotionError(f"required provenance field is missing: {field}")
    return value


def _format_yolo(class_id: int, bbox: BoundingBox) -> str:
    return f"{class_id} {bbox.x_center:.10g} {bbox.y_center:.10g} {bbox.width:.10g} {bbox.height:.10g}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalPromotionError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ExternalPromotionError(f"JSON object required: {path}")
    return raw


def _atomic_json(path: Path, value: object) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)
