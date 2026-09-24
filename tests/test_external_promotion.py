from __future__ import annotations

import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import pytest

import bottle_cap_vision.external_promotion as promotion
from bottle_cap_vision.external_intake import tree_fingerprint
from bottle_cap_vision.external_promotion import (
    ExternalPromotionError,
    approve_annotation,
    create_promotion_plan,
    generate_annotation_overlay,
    promote_approved,
    record_visual_review,
    save_proposed_annotation,
    validate_annotation,
)


CLASS_NAMES = [
    "blue", "brown", "green", "lilac", "orange",
    "pink", "purple", "red", "white", "yellow",
]


def _write_image(path: Path, color: tuple[int, int, int] = (0, 220, 240)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((60, 80, 3), color, dtype=np.uint8)
    image[5, 7] = (12, 34, 56)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    path.write_bytes(encoded.tobytes())


def _workspace(tmp_path: Path) -> tuple[Path, Path]:
    staging = tmp_path / "staging"
    dataset = tmp_path / "dataset"
    for split in ("train", "valid", "test"):
        (dataset / split / "images").mkdir(parents=True)
        (dataset / split / "labels").mkdir(parents=True)
    dataset.joinpath("data.yaml").write_text(
        "train: train/images\nval: valid/images\ntest: test/images\n"
        f"nc: 10\nnames: {CLASS_NAMES!r}\n",
        encoding="utf-8",
    )
    _write_image(dataset / "test/images/benchmark.jpg", (100, 100, 100))
    (dataset / "test/labels/benchmark.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    _add_queue_item(staging, "item-a")
    source_manifest = {
        "source_id": "source-001",
        "license": "CC BY-SA 4.0",
        "attribution": "Fixture Author",
        "source_page": "https://example.invalid/source",
    }
    path = staging / "raw/source-001/source_manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(source_manifest), encoding="utf-8")
    return staging, dataset


def _add_queue_item(staging: Path, staging_id: str, *, source_group: str = "source-001:session-a") -> None:
    image = staging / f"extracted/source-001/images/{staging_id}.jpg"
    _write_image(image)
    queue_path = staging / "review/review_queue.json"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue = json.loads(queue_path.read_text(encoding="utf-8")) if queue_path.exists() else []
    queue.append({
        "staging_id": staging_id,
        "object_id": "",
        "source_id": "source-001",
        "image_path": image.relative_to(staging).as_posix(),
        "label_path": None,
        "mapping_status": "IMAGE_ONLY",
        "source_group": source_group,
        "canonical_duplicate": False,
        "license": "CC BY-SA 4.0",
        "review_required": True,
        "review_status": "PENDING",
        "annotation_status": "IMAGE_ONLY",
        "provenance": {
            "original_filename": f"{staging_id}.jpg",
            "original_relative_path": f"{staging_id}.jpg",
            "original_source": f"https://example.invalid/{staging_id}.jpg",
            "canonical_bbox": None,
            "conversion_history": [],
        },
    })
    queue_path.write_text(json.dumps(queue), encoding="utf-8")


def _approve(staging: Path, staging_id: str = "item-a") -> None:
    record_visual_review(
        staging, staging_id, decision="APPROVED_FOR_ANNOTATION",
        reviewer="reviewer", notes="yellow bottle cap with a clear boundary",
    )
    save_proposed_annotation(staging, staging_id, [(9, 10, 10, 70, 50)])
    generate_annotation_overlay(staging, staging_id)
    approve_annotation(staging, staging_id, reviewer="reviewer", notes="overlay box and class verified")


def _plan(staging: Path, dataset: Path, assignments: dict[str, str] | None = None) -> Path:
    path = staging / "review/promotion_plan.json"
    create_promotion_plan(staging, dataset, assignments or {"item-a": "train"}, path)
    return path


def test_pending_item_cannot_enter_a_promotion_plan(tmp_path):
    staging, dataset = _workspace(tmp_path)
    with pytest.raises(ExternalPromotionError, match="fully APPROVED"):
        _plan(staging, dataset)
    assert not (staging / "review/promotion_plan.json").exists()


def test_approved_valid_pair_promotes_with_manifest_attribution_and_unchanged_test(tmp_path):
    staging, dataset = _workspace(tmp_path)
    _approve(staging)
    test_before = tree_fingerprint(dataset / "test")
    plan_path = _plan(staging, dataset)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["items"][0]["canonical_class_ids"] == [9]
    manifest_path = staging / "review/promotion_manifest.json"
    attribution_path = tmp_path / "EXTERNAL_DATA_ATTRIBUTION.md"

    manifest = promote_approved(staging, dataset, plan_path, manifest_path, attribution_path)

    image = dataset / plan["items"][0]["destination_image_path"]
    label = dataset / plan["items"][0]["destination_label_path"]
    assert image.is_file() and label.is_file()
    assert label.read_text(encoding="utf-8").startswith("9 ")
    assert manifest_path.is_file() and manifest["promoted_files"] == plan["items"]
    assert "Fixture Author" in attribution_path.read_text(encoding="utf-8")
    assert tree_fingerprint(dataset / "test") == test_before


@pytest.mark.parametrize(
    "boxes, message",
    [
        ([(9, 20, 10, 20, 40)], "empty or outside"),
        ([(9, 10, 10, 90, 40)], "empty or outside"),
        ([(99, 10, 10, 70, 50)], "unknown canonical class"),
        ([(9, 10, 10, 70, 50), (9, 10, 10, 70, 50)], "duplicate identical bbox"),
    ],
)
def test_invalid_unknown_and_duplicate_manual_boxes_are_rejected(tmp_path, boxes, message):
    staging, _ = _workspace(tmp_path)
    record_visual_review(
        staging, "item-a", decision="APPROVED_FOR_ANNOTATION",
        reviewer="reviewer", notes="clear yellow cap",
    )
    with pytest.raises(ExternalPromotionError, match=message):
        save_proposed_annotation(staging, "item-a", boxes)


def test_unknown_class_in_approved_label_blocks_plan(tmp_path):
    staging, dataset = _workspace(tmp_path)
    _approve(staging)
    (staging / "review/annotations/item-a.txt").write_text("99 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    with pytest.raises(ExternalPromotionError, match="unknown canonical class"):
        _plan(staging, dataset)


def test_exact_canonical_duplicate_blocks_plan(tmp_path):
    staging, dataset = _workspace(tmp_path)
    _approve(staging)
    shutil.copy2(
        staging / "extracted/source-001/images/item-a.jpg",
        dataset / "train/images/existing.jpg",
    )
    (dataset / "train/labels/existing.txt").write_text("9 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    with pytest.raises(ExternalPromotionError, match="exact duplicate"):
        _plan(staging, dataset)


def test_destination_collision_blocks_plan(tmp_path):
    staging, dataset = _workspace(tmp_path)
    _approve(staging)
    collision = dataset / "train/images/ext_source-001_item-a.jpg"
    _write_image(collision, (20, 30, 40))
    with pytest.raises(ExternalPromotionError, match="destination collision"):
        _plan(staging, dataset)


def test_test_destination_is_forbidden(tmp_path):
    staging, dataset = _workspace(tmp_path)
    _approve(staging)
    with pytest.raises(ExternalPromotionError, match="test is forbidden"):
        _plan(staging, dataset, {"item-a": "test"})


def test_same_source_group_cannot_cross_train_and_val(tmp_path):
    staging, dataset = _workspace(tmp_path)
    _add_queue_item(staging, "item-b", source_group="source-001:session-a")
    _approve(staging, "item-a")
    _approve(staging, "item-b")
    with pytest.raises(ExternalPromotionError, match="would cross train/val"):
        _plan(staging, dataset, {"item-a": "train", "item-b": "val"})


def test_copy_failure_leaves_no_orphan_pair_or_manifest(tmp_path, monkeypatch):
    staging, dataset = _workspace(tmp_path)
    _approve(staging)
    plan_path = _plan(staging, dataset)
    real_copy = shutil.copy2
    calls = 0

    def failing_copy(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected label-copy failure")
        return real_copy(source, destination)

    monkeypatch.setattr(promotion.shutil, "copy2", failing_copy)
    manifest = staging / "review/promotion_manifest.json"
    attribution = tmp_path / "EXTERNAL_DATA_ATTRIBUTION.md"
    with pytest.raises(OSError, match="injected"):
        promote_approved(staging, dataset, plan_path, manifest, attribution)
    assert not list((dataset / "train/images").glob("ext_*"))
    assert not list((dataset / "train/labels").glob("ext_*"))
    assert not list(dataset.rglob("*.p13b.tmp"))
    assert not manifest.exists() and not attribution.exists()


def test_revoked_queue_approval_cannot_promote_an_old_valid_plan(tmp_path):
    staging, dataset = _workspace(tmp_path)
    _approve(staging)
    plan_path = _plan(staging, dataset)
    queue_path = staging / "review/review_queue.json"
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    queue[0]["review_status"] = "PENDING"
    queue_path.write_text(json.dumps(queue), encoding="utf-8")
    with pytest.raises(ExternalPromotionError, match="approval was revoked"):
        promote_approved(
            staging, dataset, plan_path,
            staging / "review/promotion_manifest.json",
            tmp_path / "EXTERNAL_DATA_ATTRIBUTION.md",
        )


def test_annotation_pairing_is_exact(tmp_path):
    staging, _ = _workspace(tmp_path)
    image = staging / "extracted/source-001/images/item-a.jpg"
    label = staging / "review/annotations/different.txt"
    label.parent.mkdir(parents=True)
    label.write_text("9 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    with pytest.raises(ExternalPromotionError, match="stems do not pair"):
        validate_annotation(image, label)


def test_relative_staging_and_dataset_roots_build_portable_plan(tmp_path, monkeypatch):
    staging, dataset = _workspace(tmp_path)
    _approve(staging)
    monkeypatch.chdir(tmp_path)
    plan_path = Path("staging/review/promotion_plan.json")
    plan = create_promotion_plan(Path("staging"), Path("dataset"), {"item-a": "train"}, plan_path)
    assert plan["items"][0]["source_image_path"] == "extracted/source-001/images/item-a.jpg"
    assert plan_path.is_file()
