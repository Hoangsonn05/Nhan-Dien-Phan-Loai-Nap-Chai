from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import zipfile

import cv2
import numpy as np
import pytest

from bottle_cap_vision.dataset import BoundingBox
from bottle_cap_vision.external_intake import (
    ExternalIntakeError,
    IntakeOptions,
    SourceSpec,
    UnsafeArchiveError,
    analyze_color_candidate,
    discover_yolo_dataset,
    infer_source_group,
    parse_yolo_annotations,
    pixel_bbox_to_yolo,
    safe_extract_zip,
    stable_staging_id,
    stage_external_source,
    tree_fingerprint,
    validate_canonical_bbox,
)


CLASS_NAMES = ("bottle-cap", "yellow", "lilac", "purple", "other")


def _image(path: Path, bgr: tuple[int, int, int], *, size: tuple[int, int] = (60, 80)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((size[0], size[1], 3), bgr, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def _dataset(root: Path) -> None:
    (root / "train/images").mkdir(parents=True)
    (root / "train/labels").mkdir(parents=True)
    root.joinpath("data.yaml").write_text(
        "train: train/images\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {list(CLASS_NAMES)!r}\n",
        encoding="utf-8",
    )


def _sample(root: Path, name: str, bgr: tuple[int, int, int], label: str | None) -> Path:
    path = root / "train/images" / f"{name}.jpg"
    _image(path, bgr)
    if label is not None:
        (root / "train/labels" / f"{name}.txt").write_text(label, encoding="utf-8")
    return path


def _options(output: Path, canonical: Path | None = None, **kwargs) -> IntakeOptions:
    return IntakeOptions(output_root=output, canonical_dataset_root=canonical, **kwargs)


def _spec(source: Path, source_id: str = "source-001", source_type: str = "directory") -> SourceSpec:
    return SourceSpec(
        source_id=source_id,
        source_type=source_type,
        source=str(source),
        license="CC-BY-4.0",
        source_name="test fixture caps",
        source_page="https://example.invalid/caps",
        attribution="Fixture Author",
    )


def _zip_tree(source: Path, archive: Path) -> None:
    with zipfile.ZipFile(archive, "w") as handle:
        for path in sorted(item for item in source.rglob("*") if item.is_file()):
            handle.write(path, path.relative_to(source).as_posix())


def test_safe_zip_ingest_discovers_yolo_and_preserves_raw(tmp_path):
    source = tmp_path / "source"
    _dataset(source)
    _sample(source, "yellow_cap", (0, 220, 240), "1 0.5 0.5 1 1\n")
    archive = tmp_path / "caps.zip"
    _zip_tree(source, archive)

    result = stage_external_source(_spec(archive, source_type="zip"), _options(tmp_path / "staging"))

    assert result.discovery["annotation_format"] == "YOLO"
    assert result.discovery["source_classes"] == list(CLASS_NAMES)
    assert result.summary["yellow_candidates"] == 1
    assert (tmp_path / "staging/raw/source-001/original/caps.zip").read_bytes() == archive.read_bytes()
    assert (tmp_path / "staging/extracted/source-001/normalized_source_manifest.json").is_file()


def test_zip_path_traversal_is_rejected_before_extraction(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("safe/file.txt", "safe")
        handle.writestr("../../escaped.txt", "bad")
    destination = tmp_path / "extract"
    with pytest.raises(UnsafeArchiveError, match="unsafe ZIP member"):
        safe_extract_zip(archive, destination)
    assert not destination.exists()
    assert not (tmp_path / "escaped.txt").exists()


def test_yolo_discovery_supports_split_layout_and_mapping(tmp_path):
    _dataset(tmp_path)
    image = _sample(tmp_path, "cap", (0, 255, 255), "1 0.5 0.5 0.5 0.5\n")
    discovered = discover_yolo_dataset(tmp_path)
    assert discovered["format"] == "YOLO"
    assert discovered["mapping_source"] == "data.yaml"
    assert discovered["class_mapping"][1] == "yellow"
    assert discovered["images"] == [{
        "image": image.resolve(),
        "relative_path": "train/images/cap.jpg",
        "label": (tmp_path / "train/labels/cap.txt").resolve(),
    }]


def test_discovery_does_not_treat_training_plots_as_dataset_images(tmp_path):
    _image(tmp_path / "runs/train/results.png", (0, 255, 255))
    (tmp_path / "runs/train/metrics.txt").write_text("not an annotation", encoding="utf-8")
    discovered = discover_yolo_dataset(tmp_path)
    assert discovered["images"] == []
    assert discovered["annotation_files"] == 0
    with pytest.raises(ExternalIntakeError, match="no supported dataset images"):
        stage_external_source(_spec(tmp_path), _options(tmp_path.parent / "staging"))


def test_yolo_parse_validates_boxes_and_reports_unknown_class(tmp_path):
    path = tmp_path / "labels.txt"
    path.write_text(
        "1 0.5 0.5 0.25 0.4\n"
        "99 0.5 0.5 0.2 0.2\n"
        "1 0.95 0.5 0.2 0.2\n",
        encoding="utf-8",
    )
    objects, errors = parse_yolo_annotations(path, {1: "yellow"})
    assert len(objects) == 1
    assert objects[0]["source_class"] == "yellow"
    assert errors == [
        "line 2: unknown source class ID 99",
        "line 3: invalid YOLO annotation",
    ]


def test_pixel_bbox_conversion_clips_and_validates():
    bbox = pixel_bbox_to_yolo(-0.01, 10, 100.01, 50, 100, 80)
    assert bbox == BoundingBox(0.5, 0.375, 1.0, 0.5)
    validate_canonical_bbox(bbox)
    with pytest.raises(ExternalIntakeError, match="extends outside"):
        validate_canonical_bbox(BoundingBox(0.95, 0.5, 0.2, 0.2))
    with pytest.raises(ExternalIntakeError, match="empty or outside"):
        pixel_bbox_to_yolo(10, 10, 10, 20, 100, 80)


@pytest.mark.parametrize(
    "bgr, expected",
    [
        ((0, 220, 240), "yellow"),
        ((216, 170, 216), "lilac"),
        ((40, 40, 40), None),
        ((255, 255, 255), None),
    ],
)
def test_color_candidate_filter_is_deterministic_and_ignores_neutral_pixels(bgr, expected):
    crop = np.full((40, 40, 3), bgr, dtype=np.uint8)
    first = analyze_color_candidate(crop)
    second = analyze_color_candidate(crop.copy())
    assert first == second
    assert first["proposed_class"] == expected
    assert "not probability" in first["score_semantics"]


def test_exact_generic_ambiguous_and_unknown_mappings_remain_pending(tmp_path):
    source = tmp_path / "source"
    _dataset(source)
    _sample(source, "generic_yellow", (0, 220, 240), "0 0.5 0.5 1 1\n")
    _sample(source, "exact_yellow", (30, 30, 30), "1 0.5 0.5 1 1\n")
    _sample(source, "exact_lilac", (30, 30, 30), "2 0.5 0.5 1 1\n")
    _sample(source, "purple", (216, 170, 216), "3 0.5 0.5 1 1\n")
    _sample(source, "unknown", (0, 220, 240), "4 0.5 0.5 1 1\n")

    result = stage_external_source(_spec(source), _options(tmp_path / "staging"))
    records = {item["source_object_class"] + ":" + item["staging_id"]: item for item in result.object_records}
    generic = next(item for item in records.values() if item["source_object_class"] == "bottle-cap")
    yellow = next(item for item in records.values() if item["source_object_class"] == "yellow")
    lilac = next(item for item in records.values() if item["source_object_class"] == "lilac")
    purple = next(item for item in records.values() if item["source_object_class"] == "purple")
    unknown = next(item for item in records.values() if item["source_object_class"] == "other")
    assert (generic["proposed_class_name"], generic["mapping_status"]) == ("yellow", "CANDIDATE")
    assert (yellow["proposed_class_id"], yellow["mapping_status"]) == (9, "EXACT")
    assert (lilac["proposed_class_id"], lilac["mapping_status"]) == (3, "EXACT")
    assert (purple["category"], purple["mapping_status"]) == ("ambiguous", "AMBIGUOUS")
    assert (unknown["proposed_class_name"], unknown["mapping_status"]) == (None, "UNSUPPORTED")
    assert all(item["review_status"] == "PENDING" and item["review_required"] for item in result.object_records)
    assert not any(str(item["review_status"]).startswith("APPROVED") for item in result.object_records)


def test_corrupt_images_and_invalid_annotations_are_reported_not_hidden(tmp_path):
    source = tmp_path / "source"
    _dataset(source)
    (source / "train/images/broken.jpg").write_bytes(b"not an image")
    (source / "train/labels/broken.txt").write_text("1 0.5 0.5 1 1\n", encoding="utf-8")
    _sample(source, "bad_label", (0, 255, 255), "99 0.5 0.5 1 1\n")
    result = stage_external_source(_spec(source), _options(tmp_path / "staging"))
    assert result.summary["invalid_images"] == 1
    assert {item["code"] for item in result.issues} == {"INVALID_IMAGE", "INVALID_ANNOTATION"}
    conversion = json.loads((tmp_path / "staging/reports/conversion_report.json").read_text(encoding="utf-8"))
    assert conversion["unknown_classes"] == 1
    assert conversion["rejected_invalid_boxes"] == 1


def test_image_only_source_enters_review_without_fabricated_bbox(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _image(source / "unlabelled.png", (0, 220, 240))
    result = stage_external_source(_spec(source), _options(tmp_path / "staging"))
    assert result.object_records == []
    queue = json.loads((tmp_path / "staging/review/review_queue.json").read_text(encoding="utf-8"))
    assert len(queue) == 1
    assert queue[0]["mapping_status"] == "IMAGE_ONLY"
    assert queue[0]["label_path"] is None
    assert queue[0]["provenance"]["canonical_bbox"] is None


def test_exact_and_canonical_duplicates_are_reported_without_deletion(tmp_path):
    source = tmp_path / "source"
    _dataset(source)
    first = _sample(source, "same_a", (0, 220, 240), "1 0.5 0.5 1 1\n")
    second = source / "train/images/same_b.jpg"
    shutil.copy2(first, second)
    (source / "train/labels/same_b.txt").write_text("1 0.5 0.5 1 1\n", encoding="utf-8")
    canonical = tmp_path / "canonical/train/images"
    canonical.mkdir(parents=True)
    shutil.copy2(first, canonical / "existing.jpg")
    before = tree_fingerprint(tmp_path / "canonical")

    result = stage_external_source(_spec(source), _options(tmp_path / "staging", tmp_path / "canonical"))

    report = json.loads((tmp_path / "staging/reports/duplicate_report.json").read_text(encoding="utf-8"))
    assert report["internal_exact_groups"] == 1
    assert report["canonical_overlap_count"] == 2
    assert all(item["canonical_duplicate"] for item in result.image_records)
    assert tree_fingerprint(tmp_path / "canonical") == before
    assert first.exists() and second.exists()


def test_cross_source_duplicate_is_detected_on_second_intake(tmp_path):
    source_a = tmp_path / "source_a"
    source_b = tmp_path / "source_b"
    _dataset(source_a)
    _dataset(source_b)
    first = _sample(source_a, "cap_a", (0, 220, 240), "1 0.5 0.5 1 1\n")
    target = source_b / "train/images/cap_b.jpg"
    shutil.copy2(first, target)
    (source_b / "train/labels/cap_b.txt").write_text("1 0.5 0.5 1 1\n", encoding="utf-8")
    output = tmp_path / "staging"
    stage_external_source(_spec(source_a, "source-a"), _options(output))
    stage_external_source(_spec(source_b, "source-b"), _options(output))
    report = json.loads((output / "reports/duplicate_report.json").read_text(encoding="utf-8"))
    assert report["cross_source_groups"] == 1
    assert report["exact_duplicate_groups"][0]["source_ids"] == ["source-a", "source-b"]


def test_roboflow_source_group_and_split_suggestion_keep_variants_together(tmp_path):
    first, suspected_first = infer_source_group("WIN_001_jpg.rf.aaaa.jpg", "roboflow")
    second, suspected_second = infer_source_group("WIN_001_jpg.rf.bbbb.jpg", "roboflow")
    assert first == second == "roboflow:WIN_001_jpg"
    assert suspected_first and suspected_second

    source = tmp_path / "source"
    _dataset(source)
    _sample(source, "WIN_001_jpg.rf.aaaa", (0, 220, 240), "1 0.5 0.5 1 1\n")
    _sample(source, "WIN_001_jpg.rf.bbbb", (0, 220, 240), "1 0.5 0.5 1 1\n")
    stage_external_source(_spec(source, "roboflow"), _options(tmp_path / "staging"))
    report = json.loads((tmp_path / "staging/reports/source_group_report.json").read_text(encoding="utf-8"))
    assert report["total_source_groups"] == 1
    assert report["suspected_augmentation_groups"] == 1
    assert len(report["groups"][0]["files"]) == 2
    assert report["groups"][0]["split_suggestion"] in {"TRAIN_CANDIDATE", "VAL_CANDIDATE"}


def test_review_csv_json_provenance_and_contact_sheet_are_generated(tmp_path):
    source = tmp_path / "source"
    _dataset(source)
    _sample(source, "yellow", (0, 220, 240), "1 0.5 0.5 1 1\n")
    output = tmp_path / "staging"
    result = stage_external_source(_spec(source), _options(output))
    with (output / "review/review_queue.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    structured = json.loads((output / "review/review_queue.json").read_text(encoding="utf-8"))
    provenance = json.loads((output / "reports/provenance_report.json").read_text(encoding="utf-8"))
    assert len(rows) == len(structured) == 1
    assert rows[0]["review_status"] == "PENDING"
    assert structured[0]["provenance"]["original_filename"] == "yellow.jpg"
    assert provenance["complete_image_records"] == 1
    assert provenance["unknown_license_records"] == 0
    assert list((output / "review/contact_sheet_yellow").glob("*.jpg"))
    assert result.summary["ready_for_manual_review"] is True
    assert result.summary["ready_for_p13_reaudit"] is False


def test_staging_ids_are_path_based_not_order_based():
    first = stable_staging_id("source", "train/images/a.jpg")
    assert first == stable_staging_id("source", "train/images/a.jpg")
    assert first != stable_staging_id("source", "train/images/b.jpg")
    assert first != stable_staging_id("other", "train/images/a.jpg")


def test_dry_run_is_deterministic_and_makes_no_staging_mutation(tmp_path):
    source = tmp_path / "source"
    _dataset(source)
    _sample(source, "yellow", (0, 220, 240), "1 0.5 0.5 1 1\n")
    output = tmp_path / "staging"
    options = _options(output, dry_run=True)
    first = stage_external_source(_spec(source), options)
    second = stage_external_source(_spec(source), options)
    assert first.to_dict() == second.to_dict()
    assert first.discovery["images"] == 1
    assert not output.exists()


def test_output_and_canonical_roots_must_not_overlap(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ExternalIntakeError, match="must not overlap"):
        stage_external_source(
            _spec(source),
            _options(tmp_path / "canonical/external", tmp_path / "canonical", dry_run=True),
        )


def test_source_and_output_roots_must_not_overlap(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ExternalIntakeError, match="must not overlap"):
        stage_external_source(_spec(source), _options(source / "staging", dry_run=True))
