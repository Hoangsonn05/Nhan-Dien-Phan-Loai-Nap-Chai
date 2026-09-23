from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from bottle_cap_vision.dataset import DatasetManifest, build_manifest
from bottle_cap_vision.training import (
    BackendInfo,
    BackendTrainingOutput,
    TrainingAdapter,
    TrainingConfig,
    TrainingContractError,
)


class FakeBackend:
    def __init__(self, *, available: bool = True, cuda: bool = False) -> None:
        self.available = available
        self.cuda = cuda
        self.requests = []

    def probe(self) -> BackendInfo:
        return BackendInfo(
            name="fake-detector",
            version="1.2.3" if self.available else None,
            available=self.available,
            python_version="3.test",
            torch_version="2.test" if self.available else None,
            cuda_available=self.cuda,
            cuda_version="test" if self.cuda else None,
            device_count=1 if self.cuda else 0,
            detail=None if self.available else "not installed",
        )

    def train(self, request) -> BackendTrainingOutput:
        self.requests.append(request)
        yaml = request.data_config.read_text(encoding="utf-8")
        assert "test:" not in yaml
        assert "0: \"blue\"" in yaml
        weights = request.backend_output_dir / "weights"
        weights.mkdir(parents=True)
        best = weights / "framework-best.pt"
        last = weights / "framework-last.pt"
        best.write_bytes(b"real-backend-shape-best")
        last.write_bytes(b"real-backend-shape-last")
        return BackendTrainingOutput(
            best_checkpoint=best,
            last_checkpoint=last,
            backend_output_dir=request.backend_output_dir,
            epochs_completed=request.config.epochs,
            best_selection_rule="fake validation fitness",
            effective_arguments={"seed": request.config.seed, "device": request.resolved_device},
            metrics_references=("history.csv",),
        )


def _add_sample(root: Path, split: str, stem: str, label: str, value: int) -> None:
    physical = "valid" if split == "val" else split
    image = np.full((12, 16, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(root / physical / "images" / f"{stem}.jpg"), image)
    (root / physical / "labels" / f"{stem}.txt").write_text(label, encoding="utf-8")


def _dataset(tmp_path: Path) -> tuple[Path, Path, DatasetManifest]:
    root = tmp_path / "dataset"
    for split in ("train", "valid", "test"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir()
    (root / "data.yaml").write_text(
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "nc: 3\n"
        "names: ['blue', 'lilac', 'yellow']\n",
        encoding="utf-8",
    )
    _add_sample(root, "train", "blue", "0 0.5 0.5 0.2 0.2\n", 20)
    _add_sample(root, "train", "yellow", "2 0.4 0.4 0.2 0.2\n", 40)
    _add_sample(root, "val", "blue_val", "0 0.5 0.5 0.2 0.2\n", 60)
    _add_sample(root, "val", "lilac_val", "1 0.5 0.5 0.2 0.2\n", 80)
    _add_sample(root, "test", "held_out", "1 0.5 0.5 0.2 0.2\n", 100)
    manifest_path = tmp_path / "manifest.json"
    build_manifest(root, manifest_path)
    return root, manifest_path, DatasetManifest.load(manifest_path)


def _config(tmp_path: Path, **overrides) -> TrainingConfig:
    values = {
        "model": "fake.pt",
        "epochs": 1,
        "batch_size": 2,
        "image_size": 64,
        "device": "cpu",
        "seed": 17,
        "workers": 0,
        "output_root": str(tmp_path / "experiments"),
        "experiment_name": "unit_run",
        "low_support_threshold": 5,
    }
    values.update(overrides)
    return TrainingConfig(**values)


def test_training_config_valid_round_trip(tmp_path):
    raw = _config(tmp_path).to_dict()
    assert TrainingConfig.from_dict(raw).to_dict() == raw


@pytest.mark.parametrize(
    "field,value",
    [
        ("epochs", 0),
        ("epochs", -1),
        ("epochs", "1"),
        ("batch_size", 0),
        ("image_size", False),
        ("workers", -1),
        ("device", "gpu"),
        ("experiment_name", "../escape"),
        ("learning_rate", float("nan")),
    ],
)
def test_training_config_rejects_invalid_values(tmp_path, field, value):
    with pytest.raises(TrainingContractError):
        _config(tmp_path, **{field: value})


def test_training_config_rejects_unknown_framework_options():
    with pytest.raises(TrainingContractError, match="unknown training config fields"):
        TrainingConfig.from_dict({"epochs": 1, "framework_magic": True})


def test_preflight_uses_manifest_mapping_and_reports_support(tmp_path):
    _, _, manifest = _dataset(tmp_path)
    report = TrainingAdapter(FakeBackend()).preflight(manifest, _config(tmp_path))

    assert report.class_mapping == {0: "blue", 1: "lilac", 2: "yellow"}
    assert report.train_sample_count == report.val_sample_count == 2
    assert report.per_class_support == {
        0: {"train": 1, "val": 1},
        1: {"train": 0, "val": 1},
        2: {"train": 1, "val": 0},
    }
    zero = [issue for issue in report.issues if issue.code == "zero_train_support"]
    low = [issue for issue in report.issues if issue.code == "low_train_support"]
    assert [(issue.severity, issue.class_name) for issue in zero] == [("HIGH", "lilac")]
    assert [issue.class_name for issue in low] == ["blue", "yellow"]


def test_preflight_plan_is_deterministic_and_excludes_test(tmp_path):
    _, _, manifest = _dataset(tmp_path)
    config = _config(tmp_path, train_sample_limit=1, val_sample_limit=1)
    adapter = TrainingAdapter(FakeBackend())

    first = adapter.preflight(manifest, config)
    second = adapter.preflight(manifest, config)

    assert first.to_dict() == second.to_dict()
    selected = first.selected_train_sample_ids + first.selected_val_sample_ids
    assert all(not sample_id.startswith("test:") for sample_id in selected)
    assert len(first.selected_train_sample_ids) == len(first.selected_val_sample_ids) == 1


def test_tampered_manifest_fingerprint_is_rejected(tmp_path):
    _, _, manifest = _dataset(tmp_path)
    tampered = replace(manifest, fingerprint="0" * 64)
    with pytest.raises(TrainingContractError, match="fingerprint"):
        TrainingAdapter(FakeBackend()).preflight(tampered, _config(tmp_path))


def test_backend_unavailable_and_invalid_cuda_are_clear_preflight_errors(tmp_path):
    _, _, manifest = _dataset(tmp_path)
    unavailable = TrainingAdapter(FakeBackend(available=False)).preflight(
        manifest, _config(tmp_path)
    )
    no_cuda = TrainingAdapter(FakeBackend()).preflight(
        manifest, _config(tmp_path, device="cuda:0")
    )

    assert [issue.code for issue in unavailable.errors] == ["backend_unavailable"]
    assert [issue.code for issue in no_cuda.errors] == ["cuda_unavailable"]
    with pytest.raises(TrainingContractError, match="CUDA device"):
        TrainingAdapter(FakeBackend()).train(
            manifest=manifest,
            config=_config(tmp_path, device="cuda"),
            dataset_root=tmp_path / "dataset",
        )


def test_fake_backend_training_writes_canonical_artifacts_and_complete_metadata(tmp_path):
    root, manifest_path, manifest = _dataset(tmp_path)
    backend = FakeBackend()
    config = _config(tmp_path)

    result = TrainingAdapter(backend).train(
        manifest=manifest,
        config=config,
        dataset_root=root,
        manifest_path=manifest_path,
    )

    assert result.status == "completed"
    assert result.best_checkpoint.is_file() and result.best_checkpoint.read_bytes().endswith(b"best")
    assert result.last_checkpoint.is_file() and result.last_checkpoint.read_bytes().endswith(b"last")
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["dataset"]["manifest_fingerprint"] == manifest.fingerprint
    assert metadata["dataset"]["class_mapping"] == [
        {"class_id": 0, "class_name": "blue"},
        {"class_id": 1, "class_name": "lilac"},
        {"class_id": 2, "class_name": "yellow"},
    ]
    assert metadata["dataset"]["test_split_usage"] == "excluded"
    assert metadata["training_config"] == config.to_dict()
    assert metadata["reproducibility"]["seed"] == 17
    assert metadata["backend"] == {
        "model_identifier": "fake.pt",
        "name": "fake-detector",
        "version": "1.2.3",
    }
    assert metadata["checkpoints"]["best"] == "checkpoints/best.pt"
    assert metadata["checkpoints"]["last"] == "checkpoints/last.pt"
    checkpoint_metadata = json.loads(
        (result.output_dir / "checkpoints/checkpoint_metadata.json").read_text(encoding="utf-8")
    )
    assert checkpoint_metadata["dataset_fingerprint"] == manifest.fingerprint
    assert checkpoint_metadata["class_mapping"] == metadata["dataset"]["class_mapping"]
    view = result.output_dir / "dataset_view"
    assert len(list((view / "train/images").iterdir())) == 2
    assert len(list((view / "val/images").iterdir())) == 2
    assert not (view / "test").exists()


def test_existing_experiment_directory_is_never_overwritten(tmp_path):
    root, _, manifest = _dataset(tmp_path)
    config = _config(tmp_path)
    target = Path(config.output_root) / config.experiment_name
    target.mkdir(parents=True)
    sentinel = target / "keep.txt"
    sentinel.write_text("user data", encoding="utf-8")

    with pytest.raises(TrainingContractError, match="already exists"):
        TrainingAdapter(FakeBackend()).train(
            manifest=manifest, config=config, dataset_root=root
        )
    assert sentinel.read_text(encoding="utf-8") == "user data"


def test_source_image_drift_fails_closed_and_records_failed_termination(tmp_path):
    root, _, manifest = _dataset(tmp_path)
    config = _config(tmp_path)
    (root / "train/images/blue.jpg").write_bytes(b"changed after manifest")

    with pytest.raises(TrainingContractError, match="hash mismatch"):
        TrainingAdapter(FakeBackend()).train(
            manifest=manifest, config=config, dataset_root=root
        )

    metadata = json.loads(
        (Path(config.output_root) / config.experiment_name / "experiment_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["status"] == "failed"
    assert metadata["termination"]["status"] == "failed"
