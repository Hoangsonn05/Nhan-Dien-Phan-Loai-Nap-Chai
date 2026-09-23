from __future__ import annotations

import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import pytest

from bottle_cap_vision.dataset import (
    DatasetContractError,
    DatasetManifest,
    DatasetValidationError,
    audit_dataset,
    build_manifest,
    validate_class_mapping,
)


SPLIT_DIRS = {"train": "train", "val": "valid", "test": "test"}


def _write_image(path: Path, seed: int) -> None:
    image = np.full((24, 32, 3), seed % 251, dtype=np.uint8)
    image[seed % 24, seed % 32] = ((seed * 3) % 251, (seed * 5) % 251, (seed * 7) % 251)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    path.write_bytes(encoded.tobytes())


def _create_layout(root: Path, names: tuple[str, ...] = ("blue", "red")) -> None:
    for directory in SPLIT_DIRS.values():
        (root / directory / "images").mkdir(parents=True)
        (root / directory / "labels").mkdir(parents=True)
    root.joinpath("data.yaml").write_text(
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n\n"
        f"nc: {len(names)}\n"
        f"names: {list(names)!r}\n",
        encoding="utf-8",
    )


def _add_sample(
    root: Path,
    split: str,
    name: str,
    label: str,
    *,
    seed: int,
) -> tuple[Path, Path]:
    directory = SPLIT_DIRS[split]
    image = root / directory / "images" / f"{name}.jpg"
    annotation = root / directory / "labels" / f"{name}.txt"
    _write_image(image, seed)
    annotation.write_text(label, encoding="utf-8")
    return image, annotation


def _valid_dataset(root: Path) -> None:
    _create_layout(root)
    _add_sample(root, "train", "z_train", "0 0.5 0.5 0.2 0.2\n", seed=11)
    _add_sample(root, "train", "a_train", "0 0.4 0.4 0.2 0.2\n", seed=12)
    _add_sample(root, "val", "validation", "1 0.5 0.5 0.25 0.25\n", seed=13)
    _add_sample(root, "test", "negative", "", seed=14)


def _issue_codes(root: Path, **kwargs) -> set[str]:
    return {issue.code for issue in audit_dataset(root, **kwargs).audit.errors}


def test_valid_dataset_builds_loadable_deterministic_portable_manifest(tmp_path):
    root = tmp_path / "dữ liệu"
    _valid_dataset(root)
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"

    first = build_manifest(root, first_path, audit_path=tmp_path / "audit.json")
    second = build_manifest(root, second_path)
    loaded = DatasetManifest.load(first_path)

    assert first.fingerprint == second.fingerprint == loaded.fingerprint
    assert first_path.read_bytes() == second_path.read_bytes()
    assert list(loaded.class_mapping.items()) == [(0, "blue"), (1, "red")]
    assert [sample.sample_id for sample in loaded.samples] == [
        "train:train/images/a_train.jpg",
        "train:train/images/z_train.jpg",
        "val:valid/images/validation.jpg",
        "test:test/images/negative.jpg",
    ]
    assert len(loaded.split("train")) == 2
    assert loaded.statistics["total_images"] == 4
    assert loaded.statistics["total_annotations"] == 3
    assert loaded.statistics["split_counts"]["test"]["empty_samples"] == 1
    assert loaded.statistics["class_counts"]["0"]["total"] == 2
    assert loaded.statistics["class_counts"]["1"]["val"] == 1
    assert all(not Path(sample.image_path).is_absolute() for sample in loaded.samples)
    assert loaded.resolve_image(root, loaded.samples[0]).is_file()


def test_missing_annotation_is_an_error(tmp_path):
    _valid_dataset(tmp_path)
    (tmp_path / "train/labels/a_train.txt").unlink()
    assert "missing_annotation" in _issue_codes(tmp_path)


def test_orphan_annotation_detects_missing_image(tmp_path):
    _valid_dataset(tmp_path)
    (tmp_path / "test/labels/orphan.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    assert "orphan_annotation" in _issue_codes(tmp_path)


def test_corrupt_image_is_an_error(tmp_path):
    _valid_dataset(tmp_path)
    (tmp_path / "train/images/a_train.jpg").write_bytes(b"not a jpeg")
    assert "corrupt_image" in _issue_codes(tmp_path)


def test_unknown_class_id_is_an_error(tmp_path):
    _valid_dataset(tmp_path)
    (tmp_path / "train/labels/a_train.txt").write_text("9 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    assert "unknown_class_id" in _issue_codes(tmp_path)


@pytest.mark.parametrize(
    "entries, message",
    [
        ([(0, "blue"), (0, "red")], "duplicate class ID"),
        ([(0, "blue"), (1, "blue")], "duplicate class name"),
        ([(True, "blue")], "integer"),
        ([(0, "")], "non-empty"),
        ([(1, "blue")], "contiguous"),
    ],
)
def test_invalid_class_mapping_is_rejected(entries, message):
    with pytest.raises(DatasetContractError, match=message):
        validate_class_mapping(entries)


@pytest.mark.parametrize(
    "bad_label, expected_code",
    [
        ("0 0.5 0.5 0.2\n", "malformed_annotation"),
        ("0 x 0.5 0.2 0.2\n", "malformed_annotation"),
        ("0.0 0.5 0.5 0.2 0.2\n", "invalid_class_id"),
        ("0 nan 0.5 0.2 0.2\n", "invalid_bbox"),
        ("0 0.5 0.5 0 0.2\n", "invalid_bbox"),
        ("0 0.95 0.5 0.2 0.2\n", "invalid_bbox"),
        ("0 1.1 0.5 0.2 0.2\n", "invalid_bbox"),
    ],
)
def test_malformed_and_invalid_yolo_annotations_are_errors(tmp_path, bad_label, expected_code):
    _valid_dataset(tmp_path)
    (tmp_path / "train/labels/a_train.txt").write_text(bad_label, encoding="utf-8")
    assert expected_code in _issue_codes(tmp_path)


def test_duplicate_image_content_within_split_is_an_error(tmp_path):
    _valid_dataset(tmp_path)
    shutil.copyfile(tmp_path / "train/images/a_train.jpg", tmp_path / "train/images/z_train.jpg")
    result = audit_dataset(tmp_path)
    assert "duplicate_image_content" in {issue.code for issue in result.audit.errors}
    assert len(result.audit.duplicate_groups) == 1
    assert not result.audit.cross_split_leakage


def test_cross_split_content_leakage_is_a_strict_error(tmp_path):
    _valid_dataset(tmp_path)
    shutil.copyfile(tmp_path / "train/images/a_train.jpg", tmp_path / "test/images/negative.jpg")
    result = audit_dataset(tmp_path)
    assert "cross_split_leakage" in {issue.code for issue in result.audit.errors}
    assert len(result.audit.cross_split_leakage) == 1
    with pytest.raises(DatasetValidationError):
        build_manifest(tmp_path, tmp_path / "must-not-exist.json")
    assert not (tmp_path / "must-not-exist.json").exists()


def test_empty_annotation_policy_is_explicit(tmp_path):
    _valid_dataset(tmp_path)
    assert audit_dataset(tmp_path, allow_empty=True).audit.passed
    assert "empty_annotation" in _issue_codes(tmp_path, allow_empty=False)


def test_empty_dataset_fails_clearly(tmp_path):
    _create_layout(tmp_path, names=("blue",))
    assert "empty_dataset" in _issue_codes(tmp_path)


def test_unsupported_image_extension_is_an_error(tmp_path):
    _valid_dataset(tmp_path)
    (tmp_path / "train/images/unexpected.gif").write_bytes(b"GIF89a")
    assert "unsupported_image_extension" in _issue_codes(tmp_path)


def test_ambiguous_stem_pairing_is_an_error(tmp_path):
    _valid_dataset(tmp_path)
    jpg = tmp_path / "train/images/a_train.jpg"
    jpg.with_suffix(".png").write_bytes(jpg.read_bytes())
    assert "ambiguous_image_stem" in _issue_codes(tmp_path)


def test_manifest_rejects_path_escape(tmp_path):
    _valid_dataset(tmp_path)
    path = tmp_path / "manifest.json"
    build_manifest(tmp_path, path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["samples"][0]["image_path"] = "../../outside.jpg"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(DatasetContractError, match="inside dataset root"):
        DatasetManifest.load(path)


def test_manifest_rejects_duplicate_image_path(tmp_path):
    _valid_dataset(tmp_path)
    path = tmp_path / "manifest.json"
    build_manifest(tmp_path, path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["samples"][1]["image_path"] = raw["samples"][0]["image_path"]
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(DatasetContractError, match="duplicate image_path"):
        DatasetManifest.load(path)


def test_class_mapping_comes_from_metadata_not_filesystem_order(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _valid_dataset(first)
    _create_layout(second)
    _add_sample(second, "test", "negative", "", seed=14)
    _add_sample(second, "val", "validation", "1 0.5 0.5 0.25 0.25\n", seed=13)
    _add_sample(second, "train", "a_train", "0 0.4 0.4 0.2 0.2\n", seed=12)
    _add_sample(second, "train", "z_train", "0 0.5 0.5 0.2 0.2\n", seed=11)
    assert audit_dataset(first).class_mapping == audit_dataset(second).class_mapping == {0: "blue", 1: "red"}


def test_runtime_name_case_mismatch_is_reported_without_normalization(tmp_path):
    _valid_dataset(tmp_path)
    result = audit_dataset(tmp_path, runtime_class_names=["Blue", "Red"])
    assert result.audit.passed
    assert result.class_mapping == {0: "blue", 1: "red"}
    assert "runtime_class_mapping_mismatch" in {issue.code for issue in result.audit.warnings}


def test_dangling_mapping_is_an_error(tmp_path):
    _valid_dataset(tmp_path)
    (tmp_path / "valid/labels/validation.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    assert "dangling_class_mapping" in _issue_codes(tmp_path)
