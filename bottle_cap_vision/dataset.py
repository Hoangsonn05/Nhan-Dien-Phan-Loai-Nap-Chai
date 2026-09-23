"""Validated, deterministic dataset boundary for the BottleCapVision dataset.

The real dataset is a Roboflow YOLOv5 export with the physical layout
``{train,valid,test}/{images,labels}``.  This module intentionally keeps raw
YOLO parsing on this side of the manifest boundary so evaluation and training
only need the canonical records serialized by :class:`DatasetManifest`.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence

import cv2
import numpy as np


SCHEMA_VERSION = "1.0"
ANNOTATION_FORMAT = "yolo_bbox_normalized"
SUPPORTED_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp"})
SPLIT_DIRECTORIES = {"train": "train", "val": "valid", "test": "test"}
SPLIT_ORDER = {name: index for index, name in enumerate(SPLIT_DIRECTORIES)}


class DatasetContractError(ValueError):
    """Raised when configuration or a serialized manifest breaks the contract."""


class DatasetValidationError(DatasetContractError):
    """Raised when strict audit errors prevent manifest creation."""

    def __init__(self, audit: "DatasetAudit") -> None:
        self.audit = audit
        super().__init__(f"Dataset audit failed with {len(audit.errors)} error(s)")


@dataclass(frozen=True)
class BoundingBox:
    """YOLO center-format box, normalized to the inclusive range [0, 1]."""

    x_center: float
    y_center: float
    width: float
    height: float

    def to_dict(self) -> dict[str, float]:
        return {
            "x_center": self.x_center,
            "y_center": self.y_center,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, raw: object) -> "BoundingBox":
        if not isinstance(raw, dict) or set(raw) != {"x_center", "y_center", "width", "height"}:
            raise DatasetContractError("bbox must contain x_center, y_center, width and height")
        values = [raw[name] for name in ("x_center", "y_center", "width", "height")]
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            raise DatasetContractError("bbox values must be numbers")
        bbox = cls(*(float(value) for value in values))
        _validate_bbox(bbox)
        return bbox


@dataclass(frozen=True)
class Annotation:
    class_id: int
    class_name: str
    bbox: BoundingBox

    def to_dict(self) -> dict[str, object]:
        return {"class_id": self.class_id, "class_name": self.class_name, "bbox": self.bbox.to_dict()}

    @classmethod
    def from_dict(cls, raw: object, class_mapping: Mapping[int, str]) -> "Annotation":
        if not isinstance(raw, dict) or set(raw) != {"class_id", "class_name", "bbox"}:
            raise DatasetContractError("annotation has an invalid schema")
        class_id = raw["class_id"]
        class_name = raw["class_name"]
        if isinstance(class_id, bool) or not isinstance(class_id, int):
            raise DatasetContractError("annotation class_id must be an integer")
        if class_id not in class_mapping or class_mapping[class_id] != class_name:
            raise DatasetContractError(f"annotation class mapping mismatch for class_id {class_id!r}")
        return cls(class_id, str(class_name), BoundingBox.from_dict(raw["bbox"]))


@dataclass(frozen=True)
class DatasetSample:
    sample_id: str
    image_path: str
    annotation_path: str
    split: str
    width: int
    height: int
    channels: int
    content_sha256: str
    annotations: tuple[Annotation, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "image_path": self.image_path,
            "annotation_path": self.annotation_path,
            "split": self.split,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "content_sha256": self.content_sha256,
            "annotations": [annotation.to_dict() for annotation in self.annotations],
        }

    @classmethod
    def from_dict(cls, raw: object, class_mapping: Mapping[int, str]) -> "DatasetSample":
        required = {
            "sample_id", "image_path", "annotation_path", "split", "width", "height",
            "channels", "content_sha256", "annotations",
        }
        if not isinstance(raw, dict) or set(raw) != required:
            raise DatasetContractError("sample has an invalid schema")
        sample_id = raw["sample_id"]
        split = raw["split"]
        if not isinstance(sample_id, str) or not sample_id:
            raise DatasetContractError("sample_id must be a non-empty string")
        if split not in SPLIT_DIRECTORIES:
            raise DatasetContractError(f"unknown canonical split: {split!r}")
        image_path = _validate_relative_path(raw["image_path"], "image_path")
        annotation_path = _validate_relative_path(raw["annotation_path"], "annotation_path")
        dimensions = [raw[name] for name in ("width", "height", "channels")]
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in dimensions):
            raise DatasetContractError("width, height and channels must be positive integers")
        digest = raw["content_sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise DatasetContractError("content_sha256 must be a lowercase SHA-256 hex digest")
        raw_annotations = raw["annotations"]
        if not isinstance(raw_annotations, list):
            raise DatasetContractError("annotations must be a list")
        annotations = tuple(Annotation.from_dict(item, class_mapping) for item in raw_annotations)
        return cls(
            sample_id=sample_id,
            image_path=image_path,
            annotation_path=annotation_path,
            split=split,
            width=raw["width"],
            height=raw["height"],
            channels=raw["channels"],
            content_sha256=digest,
            annotations=annotations,
        )


@dataclass(frozen=True)
class AuditIssue:
    severity: str
    code: str
    message: str
    path: str | None = None
    line: int | None = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"severity": self.severity, "code": self.code, "message": self.message}
        if self.path is not None:
            result["path"] = self.path
        if self.line is not None:
            result["line"] = self.line
        return result


@dataclass(frozen=True)
class DatasetAudit:
    total_images: int
    valid_samples: int
    invalid_samples: int
    total_annotations: int
    split_counts: dict[str, dict[str, int]]
    class_counts: dict[str, dict[str, int]]
    empty_samples: int
    corrupt_images: int
    missing_labels: int
    orphan_labels: int
    duplicate_groups: tuple[tuple[str, ...], ...]
    cross_split_leakage: tuple[tuple[str, ...], ...]
    issues: tuple[AuditIssue, ...]

    @property
    def errors(self) -> tuple[AuditIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "ERROR")

    @property
    def warnings(self) -> tuple[AuditIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "WARNING")

    @property
    def passed(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "total_images": self.total_images,
            "valid_samples": self.valid_samples,
            "invalid_samples": self.invalid_samples,
            "total_annotations": self.total_annotations,
            "split_counts": self.split_counts,
            "class_counts": self.class_counts,
            "empty_samples": self.empty_samples,
            "corrupt_images": self.corrupt_images,
            "missing_labels": self.missing_labels,
            "orphan_labels": self.orphan_labels,
            "duplicate_groups": [list(group) for group in self.duplicate_groups],
            "cross_split_leakage": [list(group) for group in self.cross_split_leakage],
            "errors": [issue.to_dict() for issue in self.errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
        }


@dataclass(frozen=True)
class DatasetManifest:
    schema_version: str
    annotation_format: str
    allow_empty: bool
    class_mapping: dict[int, str]
    samples: tuple[DatasetSample, ...]
    statistics: dict[str, object]
    fingerprint: str

    def split(self, name: str) -> tuple[DatasetSample, ...]:
        if name not in SPLIT_DIRECTORIES:
            raise KeyError(f"Unknown canonical split: {name}")
        return tuple(sample for sample in self.samples if sample.split == name)

    def resolve_image(self, dataset_root: str | Path, sample: DatasetSample) -> Path:
        return _resolve_under_root(Path(dataset_root), sample.image_path)

    def resolve_annotation(self, dataset_root: str | Path, sample: DatasetSample) -> Path:
        return _resolve_under_root(Path(dataset_root), sample.annotation_path)

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "annotation_format": self.annotation_format,
            "allow_empty": self.allow_empty,
            "class_mapping": [
                {"class_id": class_id, "class_name": class_name}
                for class_id, class_name in sorted(self.class_mapping.items())
            ],
            "samples": [sample.to_dict() for sample in self.samples],
            "statistics": self.statistics,
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_canonical_json(self.to_dict()) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "DatasetManifest":
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DatasetContractError(f"Cannot load dataset manifest: {exc}") from exc
        if not isinstance(raw, dict):
            raise DatasetContractError("manifest root must be an object")
        expected = {
            "schema_version", "annotation_format", "allow_empty", "class_mapping",
            "samples", "statistics", "fingerprint",
        }
        if set(raw) != expected:
            raise DatasetContractError("manifest has an invalid root schema")
        if raw["schema_version"] != SCHEMA_VERSION:
            raise DatasetContractError(f"unsupported manifest schema_version: {raw['schema_version']!r}")
        if raw["annotation_format"] != ANNOTATION_FORMAT:
            raise DatasetContractError(f"unsupported annotation_format: {raw['annotation_format']!r}")
        if not isinstance(raw["allow_empty"], bool):
            raise DatasetContractError("allow_empty must be a boolean")
        mapping = _mapping_from_manifest(raw["class_mapping"])
        if not isinstance(raw["samples"], list):
            raise DatasetContractError("samples must be a list")
        samples = tuple(DatasetSample.from_dict(item, mapping) for item in raw["samples"])
        _validate_sample_collection(samples, allow_empty=raw["allow_empty"])
        expected_order = tuple(sorted(samples, key=_sample_sort_key))
        if samples != expected_order:
            raise DatasetContractError("manifest samples are not in deterministic order")
        if not isinstance(raw["statistics"], dict):
            raise DatasetContractError("statistics must be an object")
        expected_statistics = _statistics_for_samples(samples, mapping)
        if raw["statistics"] != expected_statistics:
            raise DatasetContractError("manifest statistics do not match its samples")
        fingerprint = raw["fingerprint"]
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise DatasetContractError("manifest fingerprint must be a SHA-256 digest")
        manifest = cls(
            schema_version=SCHEMA_VERSION,
            annotation_format=ANNOTATION_FORMAT,
            allow_empty=raw["allow_empty"],
            class_mapping=mapping,
            samples=samples,
            statistics=raw["statistics"],
            fingerprint=fingerprint,
        )
        if _manifest_fingerprint(manifest) != fingerprint:
            raise DatasetContractError("manifest fingerprint does not match its semantic content")
        return manifest


@dataclass(frozen=True)
class DatasetBuildResult:
    class_mapping: dict[int, str]
    audit: DatasetAudit
    manifest: DatasetManifest | None


@dataclass
class _AuditState:
    issues: list[AuditIssue] = field(default_factory=list)
    invalid_image_paths: set[str] = field(default_factory=set)
    corrupt_images: int = 0
    missing_labels: int = 0
    orphan_labels: int = 0

    def issue(
        self,
        severity: str,
        code: str,
        message: str,
        *,
        path: str | None = None,
        line: int | None = None,
        invalid_image: str | None = None,
    ) -> None:
        self.issues.append(AuditIssue(severity, code, message, path, line))
        if invalid_image is not None:
            self.invalid_image_paths.add(invalid_image)


def validate_class_mapping(entries: Iterable[tuple[object, object]]) -> dict[int, str]:
    """Validate and return a deterministic ID-to-name mapping."""

    mapping: dict[int, str] = {}
    names: set[str] = set()
    for raw_id, raw_name in entries:
        if isinstance(raw_id, bool) or not isinstance(raw_id, int):
            raise DatasetContractError("class ID must be an integer")
        if raw_id in mapping:
            raise DatasetContractError(f"duplicate class ID: {raw_id}")
        if not isinstance(raw_name, str) or not raw_name.strip() or raw_name != raw_name.strip():
            raise DatasetContractError("class name must be a non-empty, trimmed string")
        if raw_name in names:
            raise DatasetContractError(f"duplicate class name: {raw_name}")
        mapping[raw_id] = raw_name
        names.add(raw_name)
    if not mapping:
        raise DatasetContractError("class mapping must not be empty")
    expected_ids = list(range(len(mapping)))
    if sorted(mapping) != expected_ids:
        raise DatasetContractError("YOLO class IDs must be contiguous from 0 to nc - 1")
    return dict(sorted(mapping.items()))


def load_yolo_class_mapping(path: str | Path) -> tuple[dict[int, str], dict[str, str]]:
    """Read the restricted metadata subset used by the real Roboflow export."""

    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise DatasetContractError(f"Cannot read class mapping metadata {source}: {exc}") from exc
    values: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            raise DatasetContractError(f"Malformed data.yaml line {line_number}")
        key, value = stripped.split(":", 1)
        key = key.strip()
        if key in values:
            raise DatasetContractError(f"Duplicate data.yaml key {key!r}")
        values[key] = value.strip()
    if "nc" not in values or "names" not in values:
        raise DatasetContractError("data.yaml requires both 'nc' and 'names'")
    try:
        nc = int(values["nc"])
    except ValueError as exc:
        raise DatasetContractError("data.yaml 'nc' must be an integer") from exc
    try:
        names = ast.literal_eval(values["names"])
    except (ValueError, SyntaxError) as exc:
        raise DatasetContractError("data.yaml 'names' must be an inline list of strings") from exc
    if not isinstance(names, list) or len(names) != nc:
        raise DatasetContractError("data.yaml 'names' length must equal 'nc'")
    return validate_class_mapping(enumerate(names)), values


def audit_dataset(
    dataset_root: str | Path,
    *,
    allow_empty: bool = True,
    runtime_class_names: Sequence[str] | None = None,
) -> DatasetBuildResult:
    """Audit the real YOLO dataset and return a canonical manifest only on PASS."""

    root = Path(dataset_root).resolve()
    if not root.is_dir():
        raise DatasetContractError(f"Dataset root is not a directory: {root}")
    mapping, metadata = load_yolo_class_mapping(root / "data.yaml")
    state = _AuditState()
    _audit_metadata_paths(root, metadata, state)
    _audit_runtime_mapping(mapping, runtime_class_names, state)

    samples: list[DatasetSample] = []
    split_image_counts = Counter()
    split_annotation_counts = Counter()
    split_empty_counts = Counter()
    split_class_counts: dict[str, Counter[int]] = {split: Counter() for split in SPLIT_DIRECTORIES}
    seen_image_paths: set[str] = set()
    seen_sample_ids: set[str] = set()
    seen_resolved_image_paths: dict[Path, str] = {}

    for split, directory_name in SPLIT_DIRECTORIES.items():
        images_dir = root / directory_name / "images"
        labels_dir = root / directory_name / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            state.issue(
                "ERROR", "missing_split_directory",
                f"Split {split!r} requires {directory_name}/images and {directory_name}/labels",
                path=directory_name,
            )
            continue
        image_files, label_files = _discover_pair_files(root, images_dir, labels_dir, state)
        split_image_counts[split] += len(image_files)
        image_by_stem = _unique_by_stem(root, image_files, "image", state)
        label_by_stem = _unique_by_stem(root, label_files, "label", state)

        for stem in sorted(set(image_by_stem) - set(label_by_stem)):
            image = image_by_stem[stem]
            relative = _relative_posix(root, image)
            state.missing_labels += 1
            state.issue(
                "ERROR", "missing_annotation", "Image has no matching .txt annotation",
                path=relative, invalid_image=relative,
            )
        for stem in sorted(set(label_by_stem) - set(image_by_stem)):
            label = label_by_stem[stem]
            state.orphan_labels += 1
            state.issue(
                "ERROR", "orphan_annotation", "Annotation has no matching image",
                path=_relative_posix(root, label),
            )

        for stem in sorted(set(image_by_stem) & set(label_by_stem)):
            image = image_by_stem[stem]
            label = label_by_stem[stem]
            image_relative = _relative_posix(root, image)
            label_relative = _relative_posix(root, label)
            if image_relative in seen_image_paths:
                state.issue(
                    "ERROR", "duplicate_image_path", "Image path occurs more than once",
                    path=image_relative, invalid_image=image_relative,
                )
                continue
            seen_image_paths.add(image_relative)
            resolved_image = image.resolve()
            if resolved_image in seen_resolved_image_paths:
                previous = seen_resolved_image_paths[resolved_image]
                state.invalid_image_paths.add(previous)
                state.issue(
                    "ERROR", "duplicate_canonical_path",
                    f"Image resolves to the same file as {previous}",
                    path=image_relative, invalid_image=image_relative,
                )
                continue
            seen_resolved_image_paths[resolved_image] = image_relative
            sample_id = f"{split}:{image_relative}"
            if sample_id in seen_sample_ids:
                state.issue(
                    "ERROR", "duplicate_sample_id", "Sample ID occurs more than once",
                    path=image_relative, invalid_image=image_relative,
                )
                continue
            seen_sample_ids.add(sample_id)

            try:
                image_bytes = image.read_bytes()
            except OSError as exc:
                state.corrupt_images += 1
                state.issue(
                    "ERROR", "unreadable_image", str(exc), path=image_relative,
                    invalid_image=image_relative,
                )
                continue
            decoded = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if decoded is None or decoded.size == 0:
                state.corrupt_images += 1
                state.issue(
                    "ERROR", "corrupt_image", "Image bytes cannot be decoded",
                    path=image_relative, invalid_image=image_relative,
                )
                continue
            height, width = decoded.shape[:2]
            channels = 1 if decoded.ndim == 2 else decoded.shape[2]
            del decoded
            if width <= 0 or height <= 0 or channels != 3:
                state.issue(
                    "ERROR", "invalid_image_shape",
                    f"Expected positive dimensions and 3 channels, got {width}x{height}x{channels}",
                    path=image_relative, invalid_image=image_relative,
                )
                continue

            annotations = _parse_yolo_label(label, label_relative, image_relative, mapping, state)
            if annotations is None:
                continue
            if not annotations:
                split_empty_counts[split] += 1
                if not allow_empty:
                    state.issue(
                        "ERROR", "empty_annotation", "Negative/empty samples are disabled",
                        path=label_relative, invalid_image=image_relative,
                    )
                    continue
            split_annotation_counts[split] += len(annotations)
            split_class_counts[split].update(annotation.class_id for annotation in annotations)
            samples.append(
                DatasetSample(
                    sample_id=sample_id,
                    image_path=image_relative,
                    annotation_path=label_relative,
                    split=split,
                    width=width,
                    height=height,
                    channels=channels,
                    content_sha256=hashlib.sha256(image_bytes).hexdigest(),
                    annotations=annotations,
                )
            )

    if not split_image_counts.total():
        state.issue("ERROR", "empty_dataset", "No supported images were discovered")

    samples.sort(key=_sample_sort_key)
    duplicate_groups, leakage_groups = _audit_content_duplicates(samples, state)
    globally_used = {annotation.class_id for sample in samples for annotation in sample.annotations}
    for class_id, class_name in mapping.items():
        if class_id not in globally_used:
            state.issue(
                "ERROR", "dangling_class_mapping",
                f"Declared class {class_id} ({class_name}) has no annotations",
            )
            continue
        for split in SPLIT_DIRECTORIES:
            if split_class_counts[split][class_id] == 0:
                state.issue(
                    "WARNING", "class_missing_from_split",
                    f"Class {class_id} ({class_name}) has no annotations in split {split!r}",
                )

    issues = tuple(sorted(state.issues, key=_issue_sort_key))
    split_counts = {
        split: {
            "images": split_image_counts[split],
            "annotations": split_annotation_counts[split],
            "empty_samples": split_empty_counts[split],
        }
        for split in SPLIT_DIRECTORIES
    }
    class_counts = {
        str(class_id): {
            "class_name": class_name,
            "total": sum(split_class_counts[split][class_id] for split in SPLIT_DIRECTORIES),
            **{split: split_class_counts[split][class_id] for split in SPLIT_DIRECTORIES},
        }
        for class_id, class_name in mapping.items()
    }
    audit = DatasetAudit(
        total_images=split_image_counts.total(),
        valid_samples=len(samples) - sum(1 for sample in samples if sample.image_path in state.invalid_image_paths),
        invalid_samples=split_image_counts.total()
        - (len(samples) - sum(1 for sample in samples if sample.image_path in state.invalid_image_paths)),
        total_annotations=split_annotation_counts.total(),
        split_counts=split_counts,
        class_counts=class_counts,
        empty_samples=split_empty_counts.total(),
        corrupt_images=state.corrupt_images,
        missing_labels=state.missing_labels,
        orphan_labels=state.orphan_labels,
        duplicate_groups=duplicate_groups,
        cross_split_leakage=leakage_groups,
        issues=issues,
    )
    if not audit.passed:
        return DatasetBuildResult(mapping, audit, None)
    statistics = _statistics_for_samples(samples, mapping)
    manifest = DatasetManifest(
        schema_version=SCHEMA_VERSION,
        annotation_format=ANNOTATION_FORMAT,
        allow_empty=allow_empty,
        class_mapping=mapping,
        samples=tuple(samples),
        statistics=statistics,
        fingerprint="",
    )
    manifest = replace(manifest, fingerprint=_manifest_fingerprint(manifest))
    return DatasetBuildResult(mapping, audit, manifest)


def build_manifest(
    dataset_root: str | Path,
    output_path: str | Path,
    *,
    audit_path: str | Path | None = None,
    allow_empty: bool = True,
    runtime_class_names: Sequence[str] | None = None,
) -> DatasetManifest:
    """Run strict audit and atomically expose no manifest unless it passes."""

    result = audit_dataset(dataset_root, allow_empty=allow_empty, runtime_class_names=runtime_class_names)
    if audit_path is not None:
        _write_json(Path(audit_path), result.audit.to_dict())
    if result.manifest is None:
        raise DatasetValidationError(result.audit)
    result.manifest.save(output_path)
    return result.manifest


def _parse_yolo_label(
    label_path: Path,
    label_relative: str,
    image_relative: str,
    mapping: Mapping[int, str],
    state: _AuditState,
) -> tuple[Annotation, ...] | None:
    try:
        lines = label_path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        state.issue(
            "ERROR", "unreadable_annotation", str(exc), path=label_relative,
            invalid_image=image_relative,
        )
        return None
    annotations: list[Annotation] = []
    valid = True
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            state.issue(
                "ERROR", "malformed_annotation", "YOLO bbox requires exactly 5 fields",
                path=label_relative, line=line_number, invalid_image=image_relative,
            )
            valid = False
            continue
        class_token = fields[0]
        try:
            class_id = int(class_token)
        except ValueError:
            class_id = -1
        if class_token != str(class_id):
            state.issue(
                "ERROR", "invalid_class_id", "YOLO class ID must be a canonical integer",
                path=label_relative, line=line_number, invalid_image=image_relative,
            )
            valid = False
            continue
        if class_id not in mapping:
            state.issue(
                "ERROR", "unknown_class_id", f"Unknown class ID: {class_id}",
                path=label_relative, line=line_number, invalid_image=image_relative,
            )
            valid = False
            continue
        try:
            values = tuple(float(value) for value in fields[1:])
        except ValueError:
            state.issue(
                "ERROR", "malformed_annotation", "YOLO bbox coordinates must be numeric",
                path=label_relative, line=line_number, invalid_image=image_relative,
            )
            valid = False
            continue
        bbox = BoundingBox(*values)
        try:
            _validate_bbox(bbox)
        except DatasetContractError as exc:
            state.issue(
                "ERROR", "invalid_bbox", str(exc), path=label_relative, line=line_number,
                invalid_image=image_relative,
            )
            valid = False
            continue
        annotations.append(Annotation(class_id, mapping[class_id], bbox))
    return tuple(annotations) if valid else None


def _validate_bbox(bbox: BoundingBox) -> None:
    values = (bbox.x_center, bbox.y_center, bbox.width, bbox.height)
    if not all(math.isfinite(value) for value in values):
        raise DatasetContractError("bbox values must be finite")
    if not 0 <= bbox.x_center <= 1 or not 0 <= bbox.y_center <= 1:
        raise DatasetContractError("bbox center must be in [0, 1]")
    if not 0 < bbox.width <= 1 or not 0 < bbox.height <= 1:
        raise DatasetContractError("bbox width and height must be in (0, 1]")
    if (
        bbox.x_center - bbox.width / 2 < -1e-12
        or bbox.y_center - bbox.height / 2 < -1e-12
        or bbox.x_center + bbox.width / 2 > 1 + 1e-12
        or bbox.y_center + bbox.height / 2 > 1 + 1e-12
    ):
        raise DatasetContractError("bbox extends outside normalized image bounds")


def _discover_pair_files(
    root: Path,
    images_dir: Path,
    labels_dir: Path,
    state: _AuditState,
) -> tuple[list[Path], list[Path]]:
    image_files: list[Path] = []
    for path in sorted(images_dir.iterdir(), key=lambda item: item.name):
        if not path.is_file():
            continue
        relative = _relative_posix(root, path)
        if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            state.issue(
                "ERROR", "unsupported_image_extension",
                f"Unsupported image extension: {path.suffix or '<none>'}", path=relative,
            )
            continue
        if not _is_resolved_under_root(root, path):
            state.issue("ERROR", "path_escape", "Image resolves outside dataset root", path=relative)
            continue
        image_files.append(path)
    label_files: list[Path] = []
    for path in sorted(labels_dir.iterdir(), key=lambda item: item.name):
        if not path.is_file():
            continue
        relative = _relative_posix(root, path)
        if path.suffix.lower() != ".txt":
            state.issue(
                "ERROR", "unsupported_annotation_extension",
                f"Unsupported annotation extension: {path.suffix or '<none>'}", path=relative,
            )
            continue
        if not _is_resolved_under_root(root, path):
            state.issue("ERROR", "path_escape", "Annotation resolves outside dataset root", path=relative)
            continue
        label_files.append(path)
    return image_files, label_files


def _unique_by_stem(
    root: Path,
    paths: Sequence[Path],
    kind: str,
    state: _AuditState,
) -> dict[str, Path]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        grouped[path.stem].append(path)
    result: dict[str, Path] = {}
    for stem, matches in sorted(grouped.items()):
        if len(matches) > 1:
            for path in matches:
                state.issue(
                    "ERROR", f"ambiguous_{kind}_stem",
                    f"Multiple {kind} files share stem {stem!r}", path=_relative_posix(root, path),
                )
            continue
        result[stem] = matches[0]
    return result


def _audit_content_duplicates(
    samples: Sequence[DatasetSample], state: _AuditState,
) -> tuple[tuple[tuple[str, ...], ...], tuple[tuple[str, ...], ...]]:
    by_hash: dict[str, list[DatasetSample]] = defaultdict(list)
    for sample in samples:
        by_hash[sample.content_sha256].append(sample)
    duplicate_groups: list[tuple[str, ...]] = []
    leakage_groups: list[tuple[str, ...]] = []
    for matches in by_hash.values():
        if len(matches) < 2:
            continue
        paths = tuple(sorted(sample.image_path for sample in matches))
        duplicate_groups.append(paths)
        splits = {sample.split for sample in matches}
        if len(splits) > 1:
            leakage_groups.append(paths)
            state.issue(
                "ERROR", "cross_split_leakage",
                f"Identical image content occurs across splits: {', '.join(paths)}",
            )
        else:
            state.issue(
                "ERROR", "duplicate_image_content",
                f"Identical image content occurs more than once: {', '.join(paths)}",
            )
        state.invalid_image_paths.update(paths)
    return tuple(sorted(duplicate_groups)), tuple(sorted(leakage_groups))


def _audit_metadata_paths(root: Path, metadata: Mapping[str, str], state: _AuditState) -> None:
    aliases = {"train": "train", "val": "val", "test": "test"}
    for metadata_key, canonical in aliases.items():
        value = metadata.get(metadata_key)
        if value is None:
            state.issue(
                "WARNING", "metadata_split_missing",
                f"data.yaml does not declare canonical split {canonical!r}; physical layout is authoritative",
                path="data.yaml",
            )
            continue
        candidate = (root / value).resolve()
        if not _is_relative_to(candidate, root) or not candidate.is_dir():
            state.issue(
                "WARNING", "metadata_split_path_unusable",
                f"data.yaml {metadata_key!r} path {value!r} is outside/missing; physical layout is authoritative",
                path="data.yaml",
            )


def _audit_runtime_mapping(
    mapping: Mapping[int, str], runtime_class_names: Sequence[str] | None, state: _AuditState,
) -> None:
    if runtime_class_names is None:
        return
    dataset_names = list(mapping.values())
    runtime_names = list(runtime_class_names)
    if dataset_names == runtime_names:
        return
    if [name.casefold() for name in dataset_names] == [name.casefold() for name in runtime_names]:
        message = "Dataset class names differ from runtime names by casing; no alias or rename was applied"
    else:
        message = f"Dataset classes {dataset_names!r} differ from runtime classes {runtime_names!r}; no alias was applied"
    state.issue("WARNING", "runtime_class_mapping_mismatch", message, path="data.yaml")


def _mapping_from_manifest(raw: object) -> dict[int, str]:
    if not isinstance(raw, list):
        raise DatasetContractError("class_mapping must be a list")
    entries: list[tuple[object, object]] = []
    for item in raw:
        if not isinstance(item, dict) or set(item) != {"class_id", "class_name"}:
            raise DatasetContractError("class_mapping entries require class_id and class_name")
        entries.append((item["class_id"], item["class_name"]))
    return validate_class_mapping(entries)


def _validate_sample_collection(samples: Sequence[DatasetSample], *, allow_empty: bool) -> None:
    ids: set[str] = set()
    paths: set[str] = set()
    hashes: set[str] = set()
    for sample in samples:
        if sample.sample_id in ids:
            raise DatasetContractError(f"duplicate sample_id: {sample.sample_id}")
        if sample.image_path in paths:
            raise DatasetContractError(f"duplicate image_path: {sample.image_path}")
        if sample.content_sha256 in hashes:
            raise DatasetContractError(f"duplicate image content: {sample.content_sha256}")
        if not allow_empty and not sample.annotations:
            raise DatasetContractError(f"empty annotations are disabled for sample {sample.sample_id}")
        if sample.sample_id != f"{sample.split}:{sample.image_path}":
            raise DatasetContractError(f"sample_id does not match split/image_path: {sample.sample_id}")
        split_directory = SPLIT_DIRECTORIES[sample.split]
        image_path = PurePosixPath(sample.image_path)
        annotation_path = PurePosixPath(sample.annotation_path)
        if image_path.parent != PurePosixPath(split_directory, "images"):
            raise DatasetContractError(f"image_path does not match split layout: {sample.image_path}")
        if (
            annotation_path.parent != PurePosixPath(split_directory, "labels")
            or annotation_path.stem != image_path.stem
        ):
            raise DatasetContractError(f"annotation_path does not pair with image_path: {sample.annotation_path}")
        if image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS or annotation_path.suffix.lower() != ".txt":
            raise DatasetContractError("sample paths use unsupported extensions")
        ids.add(sample.sample_id)
        paths.add(sample.image_path)
        hashes.add(sample.content_sha256)


def _statistics_for_samples(
    samples: Sequence[DatasetSample], mapping: Mapping[int, str],
) -> dict[str, object]:
    split_image_counts = Counter(sample.split for sample in samples)
    split_annotation_counts = Counter()
    split_empty_counts = Counter()
    split_class_counts: dict[str, Counter[int]] = {split: Counter() for split in SPLIT_DIRECTORIES}
    for sample in samples:
        split_annotation_counts[sample.split] += len(sample.annotations)
        if not sample.annotations:
            split_empty_counts[sample.split] += 1
        split_class_counts[sample.split].update(annotation.class_id for annotation in sample.annotations)
    split_counts = {
        split: {
            "images": split_image_counts[split],
            "annotations": split_annotation_counts[split],
            "empty_samples": split_empty_counts[split],
        }
        for split in SPLIT_DIRECTORIES
    }
    class_counts = {
        str(class_id): {
            "class_name": class_name,
            "total": sum(split_class_counts[split][class_id] for split in SPLIT_DIRECTORIES),
            **{split: split_class_counts[split][class_id] for split in SPLIT_DIRECTORIES},
        }
        for class_id, class_name in mapping.items()
    }
    return {
        "total_images": len(samples),
        "total_annotations": sum(len(sample.annotations) for sample in samples),
        "empty_samples": sum(not sample.annotations for sample in samples),
        "split_counts": split_counts,
        "class_counts": class_counts,
    }


def _sample_sort_key(sample: DatasetSample) -> tuple[int, str]:
    return SPLIT_ORDER[sample.split], sample.image_path


def _issue_sort_key(issue: AuditIssue) -> tuple[str, str, int, str]:
    return issue.severity, issue.path or "", issue.line or 0, issue.code


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _manifest_fingerprint(manifest: DatasetManifest) -> str:
    payload = manifest.to_dict(include_fingerprint=False)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _validate_relative_path(raw: object, field_name: str) -> str:
    if not isinstance(raw, str) or not raw:
        raise DatasetContractError(f"{field_name} must be a non-empty relative path")
    normalized = raw.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or "." in path.parts or normalized != path.as_posix():
        raise DatasetContractError(f"{field_name} must be a normalized path inside dataset root")
    return normalized


def _resolve_under_root(root: Path, relative: str) -> Path:
    normalized = _validate_relative_path(relative, "manifest path")
    resolved_root = root.resolve()
    resolved = (resolved_root / Path(*PurePosixPath(normalized).parts)).resolve()
    if not _is_relative_to(resolved, resolved_root):
        raise DatasetContractError(f"manifest path escapes dataset root: {relative!r}")
    return resolved


def _relative_posix(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _is_resolved_under_root(root: Path, path: Path) -> bool:
    return _is_relative_to(path.resolve(), root.resolve())


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical_json(value) + "\n", encoding="utf-8")


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit a Roboflow YOLO dataset and build its canonical manifest")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("outputs/dataset_manifest.json"))
    parser.add_argument("--audit-report", type=Path, default=Path("outputs/dataset_audit.json"))
    parser.add_argument("--runtime-config", type=Path, help="Compare dataset names with runtime color names")
    parser.add_argument("--require-objects", action="store_true", help="Reject empty/negative label files")
    args = parser.parse_args(argv)
    try:
        runtime_class_names = None
        if args.runtime_config is not None:
            from .config import load_config

            runtime_class_names = [color.name for color in load_config(args.runtime_config).colors]
        manifest = build_manifest(
            args.dataset_root,
            args.manifest,
            audit_path=args.audit_report,
            allow_empty=not args.require_objects,
            runtime_class_names=runtime_class_names,
        )
    except DatasetValidationError as exc:
        print(json.dumps(exc.audit.to_dict(), ensure_ascii=False, indent=2))
        return 1
    except DatasetContractError as exc:
        print(f"dataset contract error: {exc}")
        return 2
    print(f"manifest={args.manifest}")
    print(f"fingerprint={manifest.fingerprint}")
    print(f"samples={len(manifest.samples)} annotations={manifest.statistics['total_annotations']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
