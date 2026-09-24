"""Safe, deterministic intake for untrusted external bottle-cap datasets.

The intake boundary deliberately keeps every object in ``PENDING`` review state.
Color scores are deterministic ranking evidence, never probabilities or approved
ground truth.  Only YOLO detection annotations are supported until a real source
requires another adapter.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import zipfile

import cv2
import numpy as np

from .dataset import BoundingBox, DatasetContractError, load_yolo_class_mapping


SUPPORTED_IMAGES = frozenset({".jpg", ".jpeg", ".png", ".webp"})
CANONICAL_TARGETS = {"lilac": 3, "yellow": 9}
GENERIC_CAP_NAMES = frozenset({"bottlecap", "bottle-cap", "bottle_cap", "bottle cap", "cap"})
LILAC_ALIASES = frozenset({"lavender", "light purple", "medium lavender"})
REVIEW_STATES = ("PENDING", "APPROVED_YELLOW", "APPROVED_LILAC", "REJECTED", "AMBIGUOUS")
_RF_PATTERN = re.compile(r"^(?P<base>.+)\.rf\.[^.]+$", re.IGNORECASE)


class ExternalIntakeError(ValueError):
    """Raised when an external source cannot be handled safely."""


class UnsafeArchiveError(ExternalIntakeError):
    """Raised before extracting a ZIP containing an unsafe member path."""


class ManualDownloadRequired(ExternalIntakeError):
    """Raised when a web source cannot be acquired without user action."""


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    source_type: str
    source: str
    license: str = "unknown"
    source_name: str | None = None
    source_page: str | None = None
    attribution: str | None = None
    notes: str = ""


@dataclass(frozen=True)
class IntakeOptions:
    output_root: Path
    canonical_dataset_root: Path | None = None
    target_classes: tuple[str, ...] = ("yellow", "lilac")
    max_images: int | None = None
    dry_run: bool = False
    allow_web: bool = False


@dataclass
class IntakeResult:
    source: dict[str, object]
    discovery: dict[str, object]
    summary: dict[str, object]
    image_records: list[dict[str, object]] = field(default_factory=list)
    object_records: list[dict[str, object]] = field(default_factory=list)
    issues: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "discovery": self.discovery,
            "summary": self.summary,
            "issues": self.issues,
        }


def stable_staging_id(source_id: str, relative_path: str) -> str:
    """Return an order-independent ID tied to source identity and original path."""

    payload = f"{source_id}\n{PurePosixPath(relative_path).as_posix()}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def infer_source_group(relative_path: str, source_id: str) -> tuple[str, bool]:
    """Infer Roboflow augmentation lineage without claiming exact duplication."""

    stem = PurePosixPath(relative_path).stem
    match = _RF_PATTERN.match(stem)
    if match:
        return f"{source_id}:{match.group('base')}", True
    return f"{source_id}:{stem}", False


def pixel_bbox_to_yolo(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    image_width: int,
    image_height: int,
    *,
    clip: bool = True,
) -> BoundingBox:
    """Convert a pixel corner box, explicitly clipping edge overflow if enabled."""

    values = (xmin, ymin, xmax, ymax)
    if image_width <= 0 or image_height <= 0 or not all(math.isfinite(value) for value in values):
        raise ExternalIntakeError("pixel bbox and image dimensions must be finite and positive")
    if clip:
        xmin = min(max(xmin, 0.0), float(image_width))
        xmax = min(max(xmax, 0.0), float(image_width))
        ymin = min(max(ymin, 0.0), float(image_height))
        ymax = min(max(ymax, 0.0), float(image_height))
    if xmin < 0 or ymin < 0 or xmax > image_width or ymax > image_height or xmax <= xmin or ymax <= ymin:
        raise ExternalIntakeError("pixel bbox is empty or outside the image")
    bbox = BoundingBox(
        (xmin + xmax) / 2.0 / image_width,
        (ymin + ymax) / 2.0 / image_height,
        (xmax - xmin) / image_width,
        (ymax - ymin) / image_height,
    )
    validate_canonical_bbox(bbox)
    return bbox


def validate_canonical_bbox(bbox: BoundingBox) -> None:
    values = (bbox.x_center, bbox.y_center, bbox.width, bbox.height)
    if not all(math.isfinite(value) for value in values):
        raise ExternalIntakeError("bbox values must be finite")
    if not (0 <= bbox.x_center <= 1 and 0 <= bbox.y_center <= 1):
        raise ExternalIntakeError("bbox center must be normalized")
    if not (0 < bbox.width <= 1 and 0 < bbox.height <= 1):
        raise ExternalIntakeError("bbox size must be normalized and positive")
    epsilon = 1e-9
    if (
        bbox.x_center - bbox.width / 2 < -epsilon
        or bbox.x_center + bbox.width / 2 > 1 + epsilon
        or bbox.y_center - bbox.height / 2 < -epsilon
        or bbox.y_center + bbox.height / 2 > 1 + epsilon
    ):
        raise ExternalIntakeError("bbox extends outside the image")


def safe_extract_zip(archive: Path, destination: Path) -> list[Path]:
    """Extract only regular ZIP members whose resolved paths remain contained."""

    destination = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        planned: list[tuple[zipfile.ZipInfo, Path]] = []
        for info in sorted(handle.infolist(), key=lambda item: item.filename):
            member = PurePosixPath(info.filename.replace("\\", "/"))
            if member.is_absolute() or ".." in member.parts or not member.parts:
                raise UnsafeArchiveError(f"unsafe ZIP member path: {info.filename!r}")
            if info.is_dir():
                continue
            target = destination.joinpath(*member.parts).resolve()
            if not _is_relative_to(target, destination):
                raise UnsafeArchiveError(f"ZIP member escapes extraction root: {info.filename!r}")
            planned.append((info, target))
        extracted: list[Path] = []
        for info, target in planned:
            target.parent.mkdir(parents=True, exist_ok=True)
            with handle.open(info) as source_handle, target.open("wb") as target_handle:
                shutil.copyfileobj(source_handle, target_handle)
            extracted.append(target)
        return extracted


def inspect_zip(archive: Path) -> dict[str, int]:
    with zipfile.ZipFile(archive) as handle:
        files = [item for item in handle.infolist() if not item.is_dir()]
        for item in files:
            member = PurePosixPath(item.filename.replace("\\", "/"))
            if member.is_absolute() or ".." in member.parts:
                raise UnsafeArchiveError(f"unsafe ZIP member path: {item.filename!r}")
        return {
            "files": len(files),
            "images": sum(PurePosixPath(item.filename).suffix.lower() in SUPPORTED_IMAGES for item in files),
            "annotations": sum(PurePosixPath(item.filename).suffix.lower() == ".txt" for item in files),
        }


def analyze_color_candidate(crop: np.ndarray) -> dict[str, object]:
    """Rank yellow/lilac evidence after excluding dark, grey and near-white pixels.

    OpenCV hue ranges are 0..179.  Yellow uses H=20..38, S>=70, V>=70.
    Lilac uses H=125..165, S>=30, V>=90.  Scores are target-pixel ratios
    among chromatic pixels, not probabilities.
    """

    if crop.size == 0:
        return _empty_color_evidence()
    if crop.ndim == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    elif crop.shape[2] == 4:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGRA2BGR)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, saturation, value = (hsv[:, :, index] for index in range(3))
    chromatic = (saturation >= 30) & (value >= 50) & ~((saturation < 45) & (value >= 235))
    valid_count = int(np.count_nonzero(chromatic))
    total = int(hue.size)
    if not valid_count:
        return _empty_color_evidence(total)
    valid_hues = hue[chromatic]
    histogram = np.bincount(valid_hues, minlength=180)
    dominant_hue = int(np.argmax(histogram))
    yellow = chromatic & (hue >= 20) & (hue <= 38) & (saturation >= 70) & (value >= 70)
    lilac = chromatic & (hue >= 125) & (hue <= 165) & (saturation >= 30) & (value >= 90)
    yellow_ratio = float(np.count_nonzero(yellow) / valid_count)
    lilac_ratio = float(np.count_nonzero(lilac) / valid_count)
    proposed: str | None = None
    score = 0.0
    if yellow_ratio >= 0.35 and yellow_ratio >= lilac_ratio:
        proposed, score = "yellow", yellow_ratio
    elif lilac_ratio >= 0.35:
        proposed, score = "lilac", lilac_ratio
    return {
        "dominant_hue": dominant_hue,
        "saturation_median": float(np.median(saturation[chromatic])),
        "value_median": float(np.median(value[chromatic])),
        "chromatic_pixel_ratio": valid_count / total,
        "yellow_pixel_ratio": yellow_ratio,
        "lilac_pixel_ratio": lilac_ratio,
        "heuristic_score": score,
        "proposed_class": proposed,
        "score_semantics": "candidate-ranking ratio; not probability or model confidence",
    }


def _empty_color_evidence(total: int = 0) -> dict[str, object]:
    return {
        "dominant_hue": None,
        "saturation_median": None,
        "value_median": None,
        "chromatic_pixel_ratio": 0.0 if total else 0.0,
        "yellow_pixel_ratio": 0.0,
        "lilac_pixel_ratio": 0.0,
        "heuristic_score": 0.0,
        "proposed_class": None,
        "score_semantics": "candidate-ranking ratio; not probability or model confidence",
    }


def parse_yolo_annotations(path: Path, class_mapping: Mapping[int, str]) -> tuple[list[dict[str, object]], list[str]]:
    objects: list[dict[str, object]] = []
    errors: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        return [], [f"cannot read annotation: {exc}"]
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"line {line_number}: expected 5 fields")
            continue
        try:
            if not re.fullmatch(r"[0-9]+", parts[0]):
                raise ValueError
            class_id = int(parts[0])
            values = [float(value) for value in parts[1:]]
            bbox = BoundingBox(*values)
            validate_canonical_bbox(bbox)
        except (ValueError, ExternalIntakeError):
            errors.append(f"line {line_number}: invalid YOLO annotation")
            continue
        if class_id not in class_mapping:
            errors.append(f"line {line_number}: unknown source class ID {class_id}")
            continue
        objects.append({"source_class_id": class_id, "source_class": class_mapping[class_id], "bbox": bbox})
    return objects, errors


def discover_yolo_dataset(root: Path) -> dict[str, object]:
    """Discover images and conservatively pair labels by structural relative path."""

    root = root.resolve()
    image_roots = _discover_image_roots(root)
    image_set = {
        path
        for image_root in image_roots
        for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGES
    }
    image_set.update(
        path for path in root.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGES
    )
    images = sorted(image_set, key=lambda path: path.relative_to(root).as_posix().casefold())
    mapping, mapping_source = _discover_class_mapping(root)
    label_roots = {image_root.parent / "labels" for image_root in image_roots}
    label_files = {
        path for label_root in label_roots if label_root.is_dir() for path in label_root.rglob("*.txt")
    }
    label_files.update(path for path in root.glob("*.txt") if path.is_file())
    metadata_names = {"classes.txt", "readme.txt", "readme.dataset.txt", "readme.roboflow.txt"}
    label_files = sorted(
        (path for path in label_files if path.name.casefold() not in metadata_names),
        key=lambda path: path.relative_to(root).as_posix().casefold(),
    )
    labels_by_key: dict[str, list[Path]] = {}
    for label in label_files:
        relative = label.relative_to(root)
        parts = list(relative.parts)
        parts = ["images" if part.casefold() == "labels" else part for part in parts]
        key = PurePosixPath(*parts).with_suffix("").as_posix().casefold()
        labels_by_key.setdefault(key, []).append(label)
    samples: list[dict[str, object]] = []
    for image in images:
        relative = image.relative_to(root)
        key = PurePosixPath(relative.as_posix()).with_suffix("").as_posix().casefold()
        matches = labels_by_key.get(key, [])
        if not matches:
            same_stem = [path for path in label_files if path.stem.casefold() == image.stem.casefold()]
            matches = same_stem if len(same_stem) == 1 else []
        samples.append({"image": image, "relative_path": relative.as_posix(), "label": matches[0] if len(matches) == 1 else None})
    annotation_count = sum(sample["label"] is not None for sample in samples)
    return {
        "format": "YOLO" if annotation_count else "IMAGE_ONLY",
        "images": samples,
        "annotation_files": annotation_count,
        "unpaired_annotation_files": max(0, len(label_files) - annotation_count),
        "class_mapping": mapping,
        "mapping_source": mapping_source,
    }


def _discover_image_roots(root: Path) -> list[Path]:
    metadata_parents = {path.parent.resolve() for path in root.rglob("data.yaml")}
    conventional_parents = {"train", "training", "valid", "validation", "val", "test", "dataset"}
    candidates: list[Path] = []
    for directory in root.rglob("*"):
        if not directory.is_dir() or directory.name.casefold() != "images":
            continue
        parent = directory.parent.resolve()
        has_labels = (parent / "labels").is_dir()
        has_metadata = parent in metadata_parents or parent.parent in metadata_parents
        conventional = parent.name.casefold() in conventional_parents or parent == root
        if has_labels or has_metadata or conventional:
            candidates.append(directory.resolve())
    return sorted(candidates, key=lambda path: path.relative_to(root).as_posix().casefold())


def _discover_class_mapping(root: Path) -> tuple[dict[int, str], str | None]:
    candidates = sorted(root.rglob("data.yaml"), key=lambda path: (len(path.parts), path.as_posix()))
    for candidate in candidates:
        try:
            mapping, _ = load_yolo_class_mapping(candidate)
        except DatasetContractError:
            continue
        return mapping, candidate.relative_to(root).as_posix()
    candidates = sorted(root.rglob("classes.txt"), key=lambda path: (len(path.parts), path.as_posix()))
    for candidate in candidates:
        names = [line.strip() for line in candidate.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if names:
            return dict(enumerate(names)), candidate.relative_to(root).as_posix()
    return {}, None


def _mapping_decision(source_class: str, evidence: Mapping[str, object]) -> tuple[str | None, str]:
    normalized = " ".join(source_class.casefold().strip().split())
    if normalized in CANONICAL_TARGETS:
        return normalized, "EXACT"
    if normalized in LILAC_ALIASES:
        return "lilac", "AMBIGUOUS"
    if normalized in {"purple", "violet", "magenta"}:
        return ("lilac", "AMBIGUOUS") if evidence["proposed_class"] == "lilac" else (None, "UNSUPPORTED")
    if normalized in GENERIC_CAP_NAMES:
        proposed = evidence["proposed_class"]
        return (str(proposed), "CANDIDATE") if proposed in CANONICAL_TARGETS else (None, "UNSUPPORTED")
    return None, "UNSUPPORTED"


def tree_fingerprint(root: Path | None) -> str | None:
    """Hash relative names and contents to prove a canonical tree stayed unchanged."""

    if root is None or not root.exists():
        return None
    digest = hashlib.sha256()
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def stage_external_source(spec: SourceSpec, options: IntakeOptions) -> IntakeResult:
    """Ingest one source while preserving raw artifacts and canonical immutability."""

    _validate_inputs(spec, options)
    canonical_before = tree_fingerprint(options.canonical_dataset_root)
    if options.dry_run:
        result = _dry_run(spec, options, canonical_before)
        canonical_after = tree_fingerprint(options.canonical_dataset_root)
        if canonical_before != canonical_after:
            raise ExternalIntakeError("canonical dataset changed during dry-run")
        return result

    output_root = options.output_root.resolve()
    raw_root = output_root / "raw" / spec.source_id
    raw_original = raw_root / "original"
    extracted_root = output_root / "extracted" / spec.source_id
    reports_root = output_root / "reports"
    raw_root.mkdir(parents=True, exist_ok=True)
    acquired_status = "INGESTED"
    with tempfile.TemporaryDirectory(prefix="bcv-external-") as temp_name:
        temp = Path(temp_name)
        dataset_root = _acquire_source(spec, options, raw_original, temp)
        discovery = discover_yolo_dataset(dataset_root)
        if not discovery["images"]:
            raise ExternalIntakeError(
                "source contains no supported dataset images in a conservative images/labels or flat-root layout"
            )
        samples = list(discovery["images"])
        if options.max_images is not None:
            samples = samples[: options.max_images]
        source_manifest = _source_manifest(spec, acquired_status, "INGESTING")
        _write_json(raw_root / "source_manifest.json", source_manifest)
        image_records, object_records, issues, conversion = _process_samples(
            spec, options, dataset_root, samples, discovery, extracted_root
        )
    source_manifest = _source_manifest(spec, acquired_status, "COMPLETE")
    _write_json(raw_root / "source_manifest.json", source_manifest)
    normalized_manifest = {
        "source": source_manifest,
        "annotation_format": discovery["format"],
        "mapping_source": discovery["mapping_source"],
        "source_classes": discovery["class_mapping"],
        "images": image_records,
        "objects": object_records,
        "issues": issues,
    }
    _write_json(extracted_root / "normalized_source_manifest.json", normalized_manifest)
    _materialize_candidates(output_root, spec.source_id, image_records, object_records)
    all_images = _load_all_normalized_images(output_root)
    duplicate_report = _duplicate_report(all_images, options.canonical_dataset_root)
    canonical_hashes = set(duplicate_report.pop("_canonical_hashes"))
    for record in image_records:
        record["canonical_duplicate"] = record["sha256"] in canonical_hashes
    for record in object_records:
        image = next(item for item in image_records if item["staging_id"] == record["staging_id"])
        record["canonical_duplicate"] = image["canonical_duplicate"]
    _write_json(extracted_root / "normalized_source_manifest.json", normalized_manifest)
    _update_candidate_manifests(output_root, spec.source_id, object_records)
    review_records = _update_review_outputs(output_root, spec.source_id, object_records, image_records)
    _generate_contact_sheets(output_root, object_records)
    source_group_report = _source_group_report(all_images, object_records)
    provenance_report = _provenance_report(spec, image_records, object_records)
    summary = _summary(spec, discovery, image_records, object_records, issues, duplicate_report, review_records)
    conversion["source_id"] = spec.source_id
    reports_root.mkdir(parents=True, exist_ok=True)
    _write_json(reports_root / "staging_summary.json", summary)
    _write_json(reports_root / "conversion_report.json", conversion)
    _write_json(reports_root / "duplicate_report.json", duplicate_report)
    _write_json(reports_root / "source_group_report.json", source_group_report)
    _write_json(reports_root / "provenance_report.json", provenance_report)
    canonical_after = tree_fingerprint(options.canonical_dataset_root)
    if canonical_before != canonical_after:
        raise ExternalIntakeError("canonical dataset was modified during external intake")
    summary["canonical_dataset_fingerprint_before"] = canonical_before
    summary["canonical_dataset_fingerprint_after"] = canonical_after
    summary["canonical_dataset_modified"] = False
    _write_json(reports_root / "staging_summary.json", summary)
    return IntakeResult(source_manifest, _public_discovery(discovery), summary, image_records, object_records, issues)


def _validate_inputs(spec: SourceSpec, options: IntakeOptions) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", spec.source_id):
        raise ExternalIntakeError("source_id must be a safe filesystem identifier")
    supported_types = {"zip", "local_zip", "directory", "local_directory", "direct_image_list", "web_archive"}
    if spec.source_type not in supported_types:
        raise ExternalIntakeError(f"unsupported source type: {spec.source_type!r}")
    if not options.target_classes or any(name not in CANONICAL_TARGETS for name in options.target_classes):
        raise ExternalIntakeError("target_classes may contain only yellow and lilac")
    if options.max_images is not None and options.max_images <= 0:
        raise ExternalIntakeError("max_images must be positive")
    output = options.output_root.resolve()
    canonical = options.canonical_dataset_root.resolve() if options.canonical_dataset_root else None
    if canonical and (_is_relative_to(output, canonical) or _is_relative_to(canonical, output)):
        raise ExternalIntakeError("output root and canonical dataset root must not overlap")
    if spec.source_type in {"zip", "local_zip", "directory", "local_directory", "direct_image_list"}:
        source = Path(spec.source).resolve()
        if _is_relative_to(source, output) or _is_relative_to(output, source):
            raise ExternalIntakeError("local source and output root must not overlap")


def _dry_run(spec: SourceSpec, options: IntakeOptions, canonical_fingerprint: str | None) -> IntakeResult:
    source_path = Path(spec.source)
    if spec.source_type in {"zip", "local_zip"}:
        if not source_path.is_file():
            raise ExternalIntakeError(f"source ZIP does not exist: {source_path}")
        counts = inspect_zip(source_path)
        discovery = {"format": "YOLO_OR_IMAGE_ONLY", **counts, "source_classes": []}
    elif spec.source_type in {"directory", "local_directory"}:
        if not source_path.is_dir():
            raise ExternalIntakeError(f"source directory does not exist: {source_path}")
        found = discover_yolo_dataset(source_path)
        discovery = _public_discovery(found)
    elif spec.source_type == "direct_image_list":
        urls = _read_url_list(source_path)
        discovery = {"format": "IMAGE_ONLY", "images": len(urls), "annotation_files": 0, "source_classes": []}
    else:
        discovery = {"format": "UNKNOWN_UNTIL_DOWNLOAD", "images": None, "annotation_files": None, "source_classes": []}
    source = _source_manifest(spec, "PLANNED", "DRY_RUN")
    summary = {
        "dry_run": True,
        "planned_actions": ["acquire immutable raw artifact", "discover dataset", "normalize", "build review queue"],
        "canonical_dataset_fingerprint_before": canonical_fingerprint,
        "canonical_dataset_modified": False,
    }
    return IntakeResult(source, discovery, summary)


def _acquire_source(spec: SourceSpec, options: IntakeOptions, raw_original: Path, temp: Path) -> Path:
    source_path = Path(spec.source)
    if spec.source_type in {"directory", "local_directory"}:
        if not source_path.is_dir():
            raise ExternalIntakeError(f"source directory does not exist: {source_path}")
        _copy_directory_immutable(source_path.resolve(), raw_original)
        return raw_original
    if spec.source_type in {"zip", "local_zip"}:
        if not source_path.is_file():
            raise ExternalIntakeError(f"source ZIP does not exist: {source_path}")
        raw_original.mkdir(parents=True, exist_ok=True)
        cached = raw_original / source_path.name
        _copy_file_immutable(source_path.resolve(), cached)
        extracted = temp / "archive"
        safe_extract_zip(cached, extracted)
        return extracted
    if spec.source_type == "direct_image_list":
        urls = _read_url_list(source_path)
        if not options.allow_web:
            raise ManualDownloadRequired("direct image URLs require --allow-web")
        raw_original.mkdir(parents=True, exist_ok=True)
        manifest_path = raw_original / "download_manifest.json"
        existing_manifest: list[dict[str, object]] = []
        if manifest_path.exists():
            raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(raw_manifest, list) or not all(isinstance(item, dict) for item in raw_manifest):
                raise ExternalIntakeError("direct-image raw cache manifest is invalid")
            existing_manifest = raw_manifest
            if [item.get("url") for item in existing_manifest] != urls:
                raise ExternalIntakeError("direct-image source list differs from immutable raw cache")
        downloaded: list[dict[str, object]] = []
        for index, url in enumerate(urls):
            suffix = Path(urlparse(url).path).suffix.lower()
            suffix = suffix if suffix in SUPPORTED_IMAGES else ".jpg"
            target = raw_original / f"{index:06d}-{hashlib.sha256(url.encode()).hexdigest()[:12]}{suffix}"
            _download_immutable(url, target)
            digest = _sha256(target)
            if existing_manifest and existing_manifest[index].get("sha256") != digest:
                raise ExternalIntakeError(f"direct-image raw cache hash mismatch: {target.name}")
            downloaded.append({"filename": target.name, "url": url, "sha256": digest})
        if not manifest_path.exists():
            _write_json(manifest_path, downloaded)
        return raw_original
    if spec.source_type == "web_archive":
        if not options.allow_web:
            raise ManualDownloadRequired("web archive requires --allow-web; download it manually and use --source-type zip")
        raw_original.mkdir(parents=True, exist_ok=True)
        name = Path(urlparse(spec.source).path).name or "source.zip"
        if not name.lower().endswith(".zip"):
            name += ".zip"
        cached = raw_original / name
        _download_immutable(spec.source, cached)
        extracted = temp / "archive"
        safe_extract_zip(cached, extracted)
        return extracted
    raise AssertionError("validated source type was not handled")


def _copy_directory_immutable(source: Path, target: Path) -> None:
    if target.exists():
        if tree_fingerprint(source) != tree_fingerprint(target):
            raise ExternalIntakeError(f"raw cache differs and will not be overwritten: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)


def _copy_file_immutable(source: Path, target: Path) -> None:
    if target.exists():
        if _sha256(source) != _sha256(target):
            raise ExternalIntakeError(f"raw cache differs and will not be overwritten: {target}")
        return
    shutil.copy2(source, target)


def _download_immutable(url: str, target: Path) -> None:
    if target.exists():
        return
    request = Request(url, headers={"User-Agent": "BottleCapVision-external-intake/1.0"})
    partial = target.with_suffix(target.suffix + ".partial")
    try:
        with urlopen(request, timeout=30) as response, partial.open("wb") as handle:
            if getattr(response, "status", 200) not in {200, 206}:
                raise ManualDownloadRequired(f"source returned HTTP {response.status}")
            shutil.copyfileobj(response, handle)
        partial.replace(target)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        partial.unlink(missing_ok=True)
        raise ManualDownloadRequired(
            f"source requires manual download ({exc}); place the archive locally and use --source-type zip"
        ) from exc


def _read_url_list(path: Path) -> list[str]:
    if not path.is_file():
        raise ExternalIntakeError(f"URL list does not exist: {path}")
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".json":
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ExternalIntakeError(f"invalid URL list JSON: {exc}") from exc
        if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
            raise ExternalIntakeError("URL list JSON must be an array of strings")
        urls = raw
    else:
        urls = [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    for url in urls:
        if urlparse(url).scheme not in {"http", "https"}:
            raise ExternalIntakeError(f"unsupported image URL: {url!r}")
    return sorted(set(urls))


def _process_samples(
    spec: SourceSpec,
    options: IntakeOptions,
    dataset_root: Path,
    samples: Sequence[dict[str, object]],
    discovery: Mapping[str, object],
    extracted_root: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    images_dir = extracted_root / "images"
    labels_dir = extracted_root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    image_records: list[dict[str, object]] = []
    object_records: list[dict[str, object]] = []
    issues: list[dict[str, object]] = []
    conversion = {
        "original_annotation_format": discovery["format"],
        "mapping_source": discovery["mapping_source"],
        "total_boxes": 0,
        "converted_boxes": 0,
        "rejected_invalid_boxes": 0,
        "unknown_classes": 0,
        "image_annotation_mismatch": discovery["unpaired_annotation_files"],
    }
    mapping = discovery["class_mapping"]
    url_by_filename: dict[str, str] = {}
    download_manifest = dataset_root / "download_manifest.json"
    if download_manifest.is_file():
        raw_downloads = json.loads(download_manifest.read_text(encoding="utf-8"))
        if isinstance(raw_downloads, list):
            url_by_filename = {
                str(item["filename"]): str(item["url"])
                for item in raw_downloads
                if isinstance(item, dict) and "filename" in item and "url" in item
            }
    for sample in samples:
        image_path = Path(sample["image"])
        relative = str(sample["relative_path"])
        staging_id = stable_staging_id(spec.source_id, relative)
        decoded = _decode_image(image_path)
        if decoded is None:
            issues.append({"code": "INVALID_IMAGE", "path": relative, "message": "image decode failed"})
            continue
        height, width = decoded.shape[:2]
        group, suspected_augmented = infer_source_group(relative, spec.source_id)
        staged_image = images_dir / f"{staging_id}{image_path.suffix.lower()}"
        shutil.copy2(image_path, staged_image)
        source_objects: list[dict[str, object]] = []
        annotation_errors: list[str] = []
        label_path = sample["label"]
        if label_path is not None:
            source_objects, annotation_errors = parse_yolo_annotations(Path(label_path), mapping)
            conversion["total_boxes"] += len(source_objects) + len(annotation_errors)
            conversion["rejected_invalid_boxes"] += len(annotation_errors)
            for message in annotation_errors:
                if "unknown source class" in message:
                    conversion["unknown_classes"] += 1
                issues.append({"code": "INVALID_ANNOTATION", "path": str(label_path), "message": message})
        image_object_records: list[dict[str, object]] = []
        output_lines: list[str] = []
        for index, source_object in enumerate(source_objects):
            bbox = source_object["bbox"]
            assert isinstance(bbox, BoundingBox)
            crop = _crop_bbox(decoded, bbox)
            evidence = analyze_color_candidate(crop)
            proposed, status = _mapping_decision(str(source_object["source_class"]), evidence)
            if proposed not in options.target_classes:
                proposed = None
            object_id = f"{staging_id}-{index:04d}"
            category = proposed if status in {"EXACT", "CANDIDATE"} and proposed else "ambiguous" if proposed else "irrelevant"
            if status == "AMBIGUOUS" and proposed:
                category = "ambiguous"
            record = {
                "object_id": object_id,
                "staging_id": staging_id,
                "source_id": spec.source_id,
                "source_object_class_id": source_object["source_class_id"],
                "source_object_class": source_object["source_class"],
                "source_bbox": _bbox_dict(bbox),
                "canonical_bbox": _bbox_dict(bbox),
                "proposed_class_id": CANONICAL_TARGETS.get(proposed) if proposed else None,
                "proposed_class_name": proposed,
                "heuristic_score": evidence["heuristic_score"],
                "color_evidence": evidence,
                "mapping_status": status,
                "review_required": True,
                "review_status": "PENDING",
                "category": category,
                "crop_path": None,
                "source_group": group,
                "suspected_augmentation_group": suspected_augmented,
                "license": spec.license,
                "canonical_duplicate": False,
                "conversion_history": ["source YOLO normalized bbox validated", "candidate class proposed; no approval"],
                "notes": "",
            }
            if proposed:
                output_lines.append(_format_yolo(CANONICAL_TARGETS[proposed], bbox))
            image_object_records.append(record)
            object_records.append(record)
            conversion["converted_boxes"] += 1
        staged_label: Path | None = None
        if source_objects:
            staged_label = labels_dir / f"{staging_id}.txt"
            staged_label.write_text("\n".join(output_lines) + ("\n" if output_lines else ""), encoding="utf-8")
        image_records.append({
            "staging_id": staging_id,
            "source_id": spec.source_id,
            "original_filename": image_path.name,
            "original_relative_path": relative,
            "original_source": url_by_filename.get(image_path.name, spec.source),
            "staged_image_path": _relative_output(options.output_root, staged_image),
            "staged_label_path": _relative_output(options.output_root, staged_label) if staged_label else None,
            "width": width,
            "height": height,
            "sha256": _sha256(image_path),
            "objects_total": len(source_objects),
            "candidate_objects": sum(record["proposed_class_name"] is not None for record in image_object_records),
            "target_classes_proposed": sorted({record["proposed_class_name"] for record in image_object_records if record["proposed_class_name"]}),
            "source_group": group,
            "suspected_augmentation_group": suspected_augmented,
            "annotation_status": "ANNOTATED" if label_path is not None else "IMAGE_ONLY",
            "review_state": "PENDING",
            "review_required": True,
            "license": spec.license,
            "canonical_duplicate": False,
            "notes": "",
        })
    return image_records, object_records, issues, conversion


def _materialize_candidates(
    output_root: Path,
    source_id: str,
    images: Sequence[dict[str, object]],
    objects: Sequence[dict[str, object]],
) -> None:
    by_id = {record["staging_id"]: record for record in images}
    for record in objects:
        category = record["category"]
        if category not in {"yellow", "lilac", "ambiguous"}:
            continue
        image = by_id[record["staging_id"]]
        source_image = output_root / str(image["staged_image_path"])
        category_root = output_root / "candidates" / str(category)
        destination_image = category_root / "images" / source_image.name
        destination_image.parent.mkdir(parents=True, exist_ok=True)
        if not destination_image.exists():
            shutil.copy2(source_image, destination_image)
        label_dir = category_root / "labels"
        label_dir.mkdir(parents=True, exist_ok=True)
        if record["proposed_class_id"] is not None:
            line = _format_yolo(int(record["proposed_class_id"]), _bbox_from_dict(record["canonical_bbox"]))
            label_path = label_dir / f"{record['object_id']}.txt"
            label_path.write_text(line + "\n", encoding="utf-8")
        else:
            label_path = None
        decoded = _decode_image(source_image)
        assert decoded is not None
        crop = _crop_bbox(decoded, _bbox_from_dict(record["canonical_bbox"]))
        crop_path = category_root / "crops" / f"{record['object_id']}.jpg"
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(crop_path), crop):
            raise ExternalIntakeError(f"cannot write crop: {crop_path}")
        record["crop_path"] = _relative_output(output_root, crop_path)
        record["candidate_image_path"] = _relative_output(output_root, destination_image)
        record["candidate_label_path"] = _relative_output(output_root, label_path) if label_path else None


def _update_candidate_manifests(output_root: Path, source_id: str, objects: Sequence[dict[str, object]]) -> None:
    for category in ("yellow", "lilac", "ambiguous"):
        path = output_root / "candidates" / category / "candidate_manifest.json"
        existing = _read_json_list(path)
        retained = [item for item in existing if item.get("source_id") != source_id]
        current = [record for record in objects if record["category"] == category]
        _write_json(path, sorted(retained + current, key=lambda item: str(item["object_id"])))


def _update_review_outputs(
    output_root: Path,
    source_id: str,
    objects: Sequence[dict[str, object]],
    images: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    review_root = output_root / "review"
    json_path = review_root / "review_queue.json"
    existing = _read_json_list(json_path)
    retained = [item for item in existing if item.get("source_id") != source_id]
    image_by_id = {item["staging_id"]: item for item in images}
    current: list[dict[str, object]] = []
    for record in objects:
        if record["category"] not in {"yellow", "lilac", "ambiguous"}:
            continue
        image = image_by_id[record["staging_id"]]
        current.append({
            "staging_id": record["staging_id"],
            "object_id": record["object_id"],
            "source_id": source_id,
            "image_path": image["staged_image_path"],
            "crop_path": record["crop_path"],
            "label_path": record.get("candidate_label_path"),
            "source_class": record["source_object_class"],
            "proposed_class": record["proposed_class_name"],
            "proposed_class_id": record["proposed_class_id"],
            "heuristic_score": record["heuristic_score"],
            "mapping_status": record["mapping_status"],
            "source_group": record["source_group"],
            "canonical_duplicate": record["canonical_duplicate"],
            "license": record["license"],
            "review_required": True,
            "review_status": "PENDING",
            "notes": "",
            "provenance": {
                "original_filename": image["original_filename"],
                "original_relative_path": image["original_relative_path"],
                "original_source": image["original_source"],
                "canonical_bbox": record["canonical_bbox"],
                "conversion_history": record["conversion_history"],
            },
        })
    candidate_image_ids = {str(item["staging_id"]) for item in current}
    for image in images:
        if image["annotation_status"] != "IMAGE_ONLY" or str(image["staging_id"]) in candidate_image_ids:
            continue
        current.append({
            "staging_id": image["staging_id"],
            "object_id": "",
            "source_id": source_id,
            "image_path": image["staged_image_path"],
            "crop_path": None,
            "label_path": None,
            "source_class": None,
            "proposed_class": None,
            "proposed_class_id": None,
            "heuristic_score": 0.0,
            "mapping_status": "IMAGE_ONLY",
            "source_group": image["source_group"],
            "canonical_duplicate": image["canonical_duplicate"],
            "license": image["license"],
            "review_required": True,
            "review_status": "PENDING",
            "notes": "manual bbox and class labeling required; no annotation was fabricated",
            "provenance": {
                "original_filename": image["original_filename"],
                "original_relative_path": image["original_relative_path"],
                "original_source": image["original_source"],
                "canonical_bbox": None,
                "conversion_history": [],
            },
        })
    combined = sorted(retained + current, key=lambda item: (str(item["source_id"]), str(item["object_id"])))
    _write_json(json_path, combined)
    fields = [
        "staging_id", "object_id", "source_id", "image_path", "crop_path", "label_path",
        "source_class", "proposed_class", "proposed_class_id", "heuristic_score", "mapping_status",
        "source_group", "canonical_duplicate", "license", "review_required", "review_status", "notes",
    ]
    csv_path = review_root / "review_queue.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in combined:
            writer.writerow({field: item.get(field) for field in fields})
    notes = review_root / "reviewer_notes_template.md"
    if not notes.exists():
        notes.write_text(
            "# External candidate review\n\nAllowed states: " + ", ".join(REVIEW_STATES) +
            "\n\nAll intake rows start PENDING. Heuristic scores are ranking evidence, not probabilities.\n",
            encoding="utf-8",
        )
    return combined


def _generate_contact_sheets(output_root: Path, objects: Sequence[dict[str, object]], batch_size: int = 25) -> None:
    for category in ("yellow", "lilac", "ambiguous"):
        selected = sorted((item for item in objects if item["category"] == category), key=lambda item: str(item["object_id"]))
        directory = output_root / "review" / f"contact_sheet_{category}"
        directory.mkdir(parents=True, exist_ok=True)
        for batch_index in range(0, len(selected), batch_size):
            batch = selected[batch_index : batch_index + batch_size]
            cells: list[np.ndarray] = []
            for item in batch:
                crop = _decode_image(output_root / str(item["crop_path"]))
                if crop is None:
                    continue
                thumb = np.full((150, 180, 3), 245, dtype=np.uint8)
                resized = _fit_image(crop, 130, 120)
                y = (120 - resized.shape[0]) // 2
                x = (180 - resized.shape[1]) // 2
                thumb[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
                label = f"{item['object_id'][-8:]} {item['proposed_class_name'] or 'amb'} {float(item['heuristic_score']):.2f}"
                group = str(item["source_group"]).rsplit(":", 1)[-1][-18:]
                cv2.putText(thumb, label, (4, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(thumb, group, (4, 147), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)
                cells.append(thumb)
            if cells:
                rows = []
                for offset in range(0, len(cells), 5):
                    row = cells[offset : offset + 5]
                    row += [np.full_like(cells[0], 245)] * (5 - len(row))
                    rows.append(np.hstack(row))
                cv2.imwrite(str(directory / f"sheet_{batch_index // batch_size:04d}.jpg"), np.vstack(rows))


def _load_all_normalized_images(output_root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted((output_root / "extracted").glob("*/normalized_source_manifest.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(raw, dict) and isinstance(raw.get("images"), list):
            records.extend(item for item in raw["images"] if isinstance(item, dict))
    return records


def _duplicate_report(images: Sequence[dict[str, object]], canonical_root: Path | None) -> dict[str, object]:
    by_hash: dict[str, list[dict[str, object]]] = {}
    for image in images:
        by_hash.setdefault(str(image["sha256"]), []).append(image)
    groups = []
    for digest, members in sorted(by_hash.items()):
        if len(members) > 1:
            source_ids = sorted({str(item["source_id"]) for item in members})
            groups.append({
                "sha256": digest,
                "images": [str(item["staged_image_path"]) for item in members],
                "source_ids": source_ids,
                "cross_source": len(source_ids) > 1,
            })
    canonical_hashes: set[str] = set()
    canonical_matches: list[dict[str, object]] = []
    if canonical_root and canonical_root.is_dir():
        for path in sorted(item for item in canonical_root.rglob("*") if item.is_file() and item.suffix.lower() in SUPPORTED_IMAGES):
            digest = _sha256(path)
            canonical_hashes.add(digest)
        for image in images:
            if image["sha256"] in canonical_hashes:
                canonical_matches.append({"source_id": image["source_id"], "staging_id": image["staging_id"], "sha256": image["sha256"]})
    return {
        "exact_duplicate_groups": groups,
        "internal_exact_groups": sum(len(group["source_ids"]) == 1 for group in groups),
        "cross_source_groups": sum(bool(group["cross_source"]) for group in groups),
        "canonical_dataset_exact_duplicates": canonical_matches,
        "canonical_overlap_count": len(canonical_matches),
        "policy": "duplicates are reported, never deleted; source groups are separate from exact hashes",
        "_canonical_hashes": sorted(canonical_hashes),
    }


def _source_group_report(images: Sequence[dict[str, object]], objects: Sequence[dict[str, object]]) -> dict[str, object]:
    groups: dict[str, dict[str, object]] = {}
    for image in images:
        group = str(image["source_group"])
        current = groups.setdefault(group, {"source_group": group, "files": [], "candidate_objects": 0, "suspected_augmentation": False})
        current["files"].append(image["staged_image_path"])
        current["suspected_augmentation"] = bool(current["suspected_augmentation"] or image["suspected_augmentation_group"])
    for obj in objects:
        if obj["proposed_class_name"]:
            groups[str(obj["source_group"])]["candidate_objects"] += 1
    ordered = sorted(groups.values(), key=lambda item: str(item["source_group"]))
    for group in ordered:
        group["files"] = sorted(group["files"])
        digest = int(hashlib.sha256(str(group["source_group"]).encode()).hexdigest(), 16)
        group["split_suggestion"] = "VAL_CANDIDATE" if digest % 5 == 0 else "TRAIN_CANDIDATE"
    train_groups = sum(item["split_suggestion"] == "TRAIN_CANDIDATE" for item in ordered)
    val_groups = len(ordered) - train_groups
    limitations: list[str] = []
    if ordered and val_groups == 0:
        limitations.append("no VAL_CANDIDATE source group; manual split review is required before promotion")
    if len(ordered) == 1:
        limitations.append("one source group cannot provide independent train and validation support")
    return {
        "total_source_groups": len(ordered),
        "suspected_augmentation_groups": sum(bool(item["suspected_augmentation"]) for item in ordered),
        "unknown_provenance_groups": 0,
        "groups": ordered,
        "split_policy": "deterministic by source_group; all files in a group stay together; canonical test is unchanged",
        "split_suggestion_counts": {"TRAIN_CANDIDATE": train_groups, "VAL_CANDIDATE": val_groups},
        "limitations": limitations,
    }


def _provenance_report(spec: SourceSpec, images: Sequence[dict[str, object]], objects: Sequence[dict[str, object]]) -> dict[str, object]:
    required = ("source_id", "original_source", "original_filename", "source_group", "license")
    complete = sum(all(record.get(field) not in {None, ""} for field in required) for record in images)
    return {
        "source_id": spec.source_id,
        "source_uri_or_path": spec.source,
        "source_page": spec.source_page,
        "license": spec.license,
        "attribution": spec.attribution,
        "complete_image_records": complete,
        "incomplete_image_records": len(images) - complete,
        "complete_object_records": sum(bool(item.get("conversion_history")) and bool(item.get("source_object_class")) for item in objects),
        "unknown_license_records": len(images) if spec.license == "unknown" else 0,
    }


def _summary(
    spec: SourceSpec,
    discovery: Mapping[str, object],
    images: Sequence[dict[str, object]],
    objects: Sequence[dict[str, object]],
    issues: Sequence[dict[str, object]],
    duplicate_report: Mapping[str, object],
    review: Sequence[dict[str, object]],
) -> dict[str, object]:
    categories = {name: [obj for obj in objects if obj["category"] == name] for name in ("yellow", "lilac", "ambiguous")}
    return {
        "sources_attempted": 1,
        "sources_ingested": 1,
        "source_id": spec.source_id,
        "images_discovered": len(discovery["images"]),
        "annotations_discovered": discovery["annotation_files"],
        "objects_discovered": len(objects),
        "images_decoded": len(images),
        "invalid_images": sum(item["code"] == "INVALID_IMAGE" for item in issues),
        "target_candidate_objects": sum(len(items) for items in categories.values()),
        "yellow_candidates": len(categories["yellow"]),
        "lilac_candidates": len(categories["lilac"]),
        "ambiguous_candidates": len(categories["ambiguous"]),
        "canonical_duplicates": duplicate_report["canonical_overlap_count"],
        "unique_source_groups": len({image["source_group"] for image in images}),
        "license_unknown_count": len(images) if spec.license == "unknown" else 0,
        "review_queue_size": len(review),
        "ready_for_manual_review": bool(review),
        "ready_for_p13_reaudit": False,
    }


def _source_manifest(spec: SourceSpec, acquisition_status: str, ingestion_status: str) -> dict[str, object]:
    return {
        "source_id": spec.source_id,
        "source_type": spec.source_type,
        "source_uri_or_path": spec.source,
        "source_name": spec.source_name,
        "source_page": spec.source_page,
        "license": spec.license,
        "attribution": spec.attribution,
        "acquisition_status": acquisition_status,
        "ingestion_status": ingestion_status,
        "notes": spec.notes,
    }


def _public_discovery(discovery: Mapping[str, object]) -> dict[str, object]:
    return {
        "annotation_format": discovery["format"],
        "images": len(discovery["images"]),
        "annotation_files": discovery["annotation_files"],
        "unpaired_annotation_files": discovery["unpaired_annotation_files"],
        "source_classes": [discovery["class_mapping"][key] for key in sorted(discovery["class_mapping"])],
        "mapping_source": discovery["mapping_source"],
    }


def _decode_image(path: Path) -> np.ndarray | None:
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0 or image.shape[0] <= 0 or image.shape[1] <= 0:
        return None
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.shape[2] != 3:
        return None
    return image


def _crop_bbox(image: np.ndarray, bbox: BoundingBox) -> np.ndarray:
    height, width = image.shape[:2]
    xmin = max(0, int(math.floor((bbox.x_center - bbox.width / 2) * width)))
    ymin = max(0, int(math.floor((bbox.y_center - bbox.height / 2) * height)))
    xmax = min(width, int(math.ceil((bbox.x_center + bbox.width / 2) * width)))
    ymax = min(height, int(math.ceil((bbox.y_center + bbox.height / 2) * height)))
    return image[ymin:ymax, xmin:xmax]


def _fit_image(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    scale = min(max_width / image.shape[1], max_height / image.shape[0])
    width = max(1, int(round(image.shape[1] * scale)))
    height = max(1, int(round(image.shape[0] * scale)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _bbox_dict(bbox: BoundingBox) -> dict[str, float]:
    return {"x_center": bbox.x_center, "y_center": bbox.y_center, "width": bbox.width, "height": bbox.height}


def _bbox_from_dict(raw: object) -> BoundingBox:
    if not isinstance(raw, dict):
        raise ExternalIntakeError("internal bbox record is invalid")
    return BoundingBox(float(raw["x_center"]), float(raw["y_center"]), float(raw["width"]), float(raw["height"]))


def _format_yolo(class_id: int, bbox: BoundingBox) -> str:
    return f"{class_id} {bbox.x_center:.10g} {bbox.y_center:.10g} {bbox.width:.10g} {bbox.height:.10g}"


def _relative_output(root: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    return path.resolve().relative_to(root.resolve()).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_list(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalIntakeError(f"cannot read existing staging manifest {path}: {exc}") from exc
    if not isinstance(raw, list) or not all(isinstance(item, dict) for item in raw):
        raise ExternalIntakeError(f"existing staging manifest must be a JSON array: {path}")
    return raw


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
