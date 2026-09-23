"""Manifest-driven, offline training lifecycle for BottleCapVision.

The generic orchestration in this module owns experiment identity, dataset
membership, metadata and canonical checkpoint names. Framework-specific code
is isolated behind :class:`TrainingBackend`; the production implementation is
the lazy-imported :class:`UltralyticsBackend`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import shutil
import time
from typing import Mapping, Protocol, Sequence

from .dataset import ANNOTATION_FORMAT, SCHEMA_VERSION, DatasetManifest, DatasetSample


TRAINING_SCHEMA_VERSION = "1.0"
PROJECT_VERSION = "P9"
_EXPERIMENT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_CUDA_DEVICE = re.compile(r"^cuda(?::([0-9]+))?$")


class TrainingContractError(ValueError):
    """Raised when a training request violates the P9 contract."""


class TrainingBackendError(RuntimeError):
    """Raised when the selected ML backend cannot complete training."""


@dataclass(frozen=True)
class TrainingConfig:
    model: str = "yolo11n.pt"
    epochs: int = 1
    batch_size: int = 8
    image_size: int = 640
    learning_rate: float | None = None
    device: str = "cpu"
    seed: int = 0
    workers: int = 0
    output_root: str = "outputs/experiments"
    experiment_name: str = "p9_training"
    train_sample_limit: int | None = None
    val_sample_limit: int | None = None
    low_support_threshold: int = 5

    def __post_init__(self) -> None:
        _require_nonempty_string(self.model, "model")
        _require_int_at_least(self.epochs, "epochs", 1)
        _require_int_at_least(self.batch_size, "batch_size", 1)
        _require_int_at_least(self.image_size, "image_size", 1)
        _require_int_at_least(self.workers, "workers", 0)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TrainingContractError("seed must be an integer")
        if self.learning_rate is not None:
            if (
                isinstance(self.learning_rate, bool)
                or not isinstance(self.learning_rate, (int, float))
                or not math.isfinite(self.learning_rate)
                or self.learning_rate <= 0
            ):
                raise TrainingContractError("learning_rate must be finite and greater than zero")
        if not isinstance(self.device, str) or not (
            self.device in {"cpu", "auto"} or _CUDA_DEVICE.fullmatch(self.device)
        ):
            raise TrainingContractError("device must be 'cpu', 'auto', 'cuda' or 'cuda:N'")
        _require_nonempty_string(self.output_root, "output_root")
        if not isinstance(self.experiment_name, str) or not _EXPERIMENT_NAME.fullmatch(
            self.experiment_name
        ) or self.experiment_name in {".", ".."}:
            raise TrainingContractError(
                "experiment_name must be a safe 1-64 character name"
            )
        for value, name in (
            (self.train_sample_limit, "train_sample_limit"),
            (self.val_sample_limit, "val_sample_limit"),
        ):
            if value is not None:
                _require_int_at_least(value, name, 1)
        _require_int_at_least(self.low_support_threshold, "low_support_threshold", 1)

    @property
    def is_smoke(self) -> bool:
        return self.train_sample_limit is not None or self.val_sample_limit is not None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: object) -> "TrainingConfig":
        if not isinstance(raw, dict):
            raise TrainingContractError("training config root must be an object")
        allowed = set(cls.__dataclass_fields__)
        unknown = set(raw) - allowed
        if unknown:
            raise TrainingContractError(f"unknown training config fields: {sorted(unknown)!r}")
        try:
            return cls(**raw)
        except TypeError as exc:
            raise TrainingContractError(f"invalid training config: {exc}") from exc

    @classmethod
    def load(cls, path: str | Path) -> "TrainingConfig":
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TrainingContractError(f"cannot load training config: {exc}") from exc
        return cls.from_dict(raw)


@dataclass(frozen=True)
class TrainingIssue:
    severity: str
    code: str
    message: str
    class_id: int | None = None
    class_name: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True)
class BackendInfo:
    name: str
    version: str | None
    available: bool
    python_version: str
    torch_version: str | None
    cuda_available: bool
    cuda_version: str | None
    device_count: int
    detail: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingPreflight:
    manifest_fingerprint: str
    manifest_schema_version: str
    class_mapping: dict[int, str]
    train_sample_count: int
    val_sample_count: int
    selected_train_sample_ids: tuple[str, ...]
    selected_val_sample_ids: tuple[str, ...]
    per_class_support: dict[int, dict[str, int]]
    issues: tuple[TrainingIssue, ...]
    backend: BackendInfo
    requested_device: str
    resolved_device: str | None
    output_dir: str
    smoke: bool

    @property
    def errors(self) -> tuple[TrainingIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "ERROR")

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_fingerprint": self.manifest_fingerprint,
            "manifest_schema_version": self.manifest_schema_version,
            "class_mapping": _mapping_list(self.class_mapping),
            "train_sample_count": self.train_sample_count,
            "val_sample_count": self.val_sample_count,
            "selected_train_sample_ids": list(self.selected_train_sample_ids),
            "selected_val_sample_ids": list(self.selected_val_sample_ids),
            "per_class_support": [
                {
                    "class_id": class_id,
                    "class_name": self.class_mapping[class_id],
                    **support,
                }
                for class_id, support in self.per_class_support.items()
            ],
            "issues": [issue.to_dict() for issue in self.issues],
            "backend": self.backend.to_dict(),
            "requested_device": self.requested_device,
            "resolved_device": self.resolved_device,
            "output_dir": self.output_dir,
            "smoke": self.smoke,
        }


@dataclass(frozen=True)
class BackendTrainingRequest:
    data_config: Path
    backend_output_dir: Path
    config: TrainingConfig
    resolved_device: str


@dataclass(frozen=True)
class BackendTrainingOutput:
    best_checkpoint: Path
    last_checkpoint: Path
    backend_output_dir: Path
    epochs_completed: int
    best_selection_rule: str
    effective_arguments: Mapping[str, object]
    metrics_references: tuple[str, ...] = ()


class TrainingBackend(Protocol):
    def probe(self) -> BackendInfo: ...

    def train(self, request: BackendTrainingRequest) -> BackendTrainingOutput: ...


@dataclass(frozen=True)
class TrainingResult:
    experiment_id: str
    status: str
    output_dir: Path
    best_checkpoint: Path
    last_checkpoint: Path
    metadata_path: Path
    backend_output_dir: Path
    duration_seconds: float
    epochs_completed: int

    def to_dict(self) -> dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "status": self.status,
            "output_dir": str(self.output_dir),
            "best_checkpoint": str(self.best_checkpoint),
            "last_checkpoint": str(self.last_checkpoint),
            "metadata_path": str(self.metadata_path),
            "backend_output_dir": str(self.backend_output_dir),
            "duration_seconds": self.duration_seconds,
            "epochs_completed": self.epochs_completed,
        }


class UltralyticsBackend:
    """Ultralytics YOLO adapter, imported only when training is requested."""

    name = "ultralytics"

    def probe(self) -> BackendInfo:
        try:
            version = importlib.metadata.version("ultralytics")
            import torch
        except Exception as exc:
            return BackendInfo(
                name=self.name,
                version=None,
                available=False,
                python_version=platform.python_version(),
                torch_version=None,
                cuda_available=False,
                cuda_version=None,
                device_count=0,
                detail=str(exc),
            )
        return BackendInfo(
            name=self.name,
            version=version,
            available=True,
            python_version=platform.python_version(),
            torch_version=str(torch.__version__),
            cuda_available=bool(torch.cuda.is_available()),
            cuda_version=torch.version.cuda,
            device_count=int(torch.cuda.device_count()),
        )

    def train(self, request: BackendTrainingRequest) -> BackendTrainingOutput:
        try:
            import numpy as np
            import torch
            from ultralytics import YOLO
        except ImportError as exc:
            raise TrainingBackendError(f"Ultralytics backend is unavailable: {exc}") from exc

        random.seed(request.config.seed)
        np.random.seed(request.config.seed)
        torch.manual_seed(request.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(request.config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        device_argument: str | int = request.resolved_device
        if request.resolved_device.startswith("cuda:"):
            device_argument = int(request.resolved_device.split(":", 1)[1])
        arguments: dict[str, object] = {
            "data": str(request.data_config),
            "epochs": request.config.epochs,
            "batch": request.config.batch_size,
            "imgsz": request.config.image_size,
            "device": device_argument,
            "workers": request.config.workers,
            "seed": request.config.seed,
            "deterministic": True,
            "project": str(request.backend_output_dir.parent),
            "name": request.backend_output_dir.name,
            "exist_ok": False,
            "plots": False,
            "verbose": True,
        }
        if request.config.learning_rate is not None:
            arguments["lr0"] = request.config.learning_rate

        try:
            model = YOLO(request.config.model)
            model.train(**arguments)
        except Exception as exc:
            raise TrainingBackendError(f"Ultralytics training failed: {exc}") from exc

        weights = request.backend_output_dir / "weights"
        metrics = tuple(
            path.name
            for path in (request.backend_output_dir / "results.csv",)
            if path.is_file()
        )
        return BackendTrainingOutput(
            best_checkpoint=weights / "best.pt",
            last_checkpoint=weights / "last.pt",
            backend_output_dir=request.backend_output_dir,
            epochs_completed=request.config.epochs,
            best_selection_rule="Ultralytics validation fitness",
            effective_arguments=arguments,
            metrics_references=metrics,
        )


class TrainingAdapter:
    """Own the framework-independent P9 experiment lifecycle."""

    def __init__(self, backend: TrainingBackend) -> None:
        self.backend = backend

    def preflight(self, manifest: DatasetManifest, config: TrainingConfig) -> TrainingPreflight:
        _verify_manifest_identity(manifest)
        info = self.backend.probe()
        train_samples = manifest.split("train")
        val_samples = manifest.split("val")
        selected_train = _select_samples(train_samples, config.train_sample_limit, config.seed)
        selected_val = _select_samples(val_samples, config.val_sample_limit, config.seed)
        support = _class_support(manifest)
        issues: list[TrainingIssue] = []
        for class_id, class_name in manifest.class_mapping.items():
            train_support = support[class_id]["train"]
            if train_support == 0:
                issues.append(
                    TrainingIssue(
                        "HIGH",
                        "zero_train_support",
                        f"model has no supervised train examples for class {class_name}",
                        class_id,
                        class_name,
                    )
                )
            elif train_support < config.low_support_threshold:
                issues.append(
                    TrainingIssue(
                        "WARNING",
                        "low_train_support",
                        f"class {class_name} has only {train_support} supervised train examples",
                        class_id,
                        class_name,
                    )
                )
        if not train_samples:
            issues.append(TrainingIssue("ERROR", "empty_train_split", "train split is empty"))
        if not val_samples:
            issues.append(TrainingIssue("ERROR", "empty_val_split", "val split is empty"))
        if not info.available:
            issues.append(
                TrainingIssue(
                    "ERROR", "backend_unavailable", f"backend {info.name} is unavailable: {info.detail}"
                )
            )
        resolved_device = _resolve_device(config.device, info, issues)
        return TrainingPreflight(
            manifest_fingerprint=manifest.fingerprint,
            manifest_schema_version=manifest.schema_version,
            class_mapping=dict(manifest.class_mapping),
            train_sample_count=len(train_samples),
            val_sample_count=len(val_samples),
            selected_train_sample_ids=tuple(sample.sample_id for sample in selected_train),
            selected_val_sample_ids=tuple(sample.sample_id for sample in selected_val),
            per_class_support=support,
            issues=tuple(issues),
            backend=info,
            requested_device=config.device,
            resolved_device=resolved_device,
            output_dir=str(Path(config.output_root) / config.experiment_name),
            smoke=config.is_smoke,
        )

    def train(
        self,
        *,
        manifest: DatasetManifest,
        config: TrainingConfig,
        dataset_root: str | Path,
        manifest_path: str | Path | None = None,
    ) -> TrainingResult:
        preflight = self.preflight(manifest, config)
        if preflight.errors:
            detail = "; ".join(issue.message for issue in preflight.errors)
            raise TrainingContractError(f"training preflight failed: {detail}")
        if preflight.resolved_device is None:
            raise TrainingContractError("training preflight did not resolve a device")

        output_dir = Path(preflight.output_dir).resolve()
        if output_dir.exists():
            raise TrainingContractError(f"experiment directory already exists: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        created_at = _utc_now()
        metadata_path = output_dir / "experiment_metadata.json"
        snapshot_path = output_dir / "dataset_manifest.json"
        config_path = output_dir / "training_config.json"
        manifest.save(snapshot_path)
        _write_json(config_path, config.to_dict())

        metadata = _base_metadata(
            manifest=manifest,
            config=config,
            preflight=preflight,
            created_at=created_at,
            manifest_path=manifest_path,
        )
        _write_json(metadata_path, metadata)
        try:
            selected = {
                "train": _samples_by_id(manifest, preflight.selected_train_sample_ids),
                "val": _samples_by_id(manifest, preflight.selected_val_sample_ids),
            }
            data_config = _materialize_data_view(
                manifest=manifest,
                dataset_root=Path(dataset_root),
                samples=selected,
                target=output_dir / "dataset_view",
            )
            request = BackendTrainingRequest(
                data_config=data_config,
                backend_output_dir=output_dir / "backend" / "train",
                config=config,
                resolved_device=preflight.resolved_device,
            )
            backend_result = self.backend.train(request)
            _require_checkpoint(backend_result.best_checkpoint, "best")
            _require_checkpoint(backend_result.last_checkpoint, "last")
            checkpoint_dir = output_dir / "checkpoints"
            checkpoint_dir.mkdir()
            best = checkpoint_dir / f"best{backend_result.best_checkpoint.suffix}"
            last = checkpoint_dir / f"last{backend_result.last_checkpoint.suffix}"
            shutil.copy2(backend_result.best_checkpoint, best)
            shutil.copy2(backend_result.last_checkpoint, last)
            duration = time.monotonic() - started
            checkpoint_metadata = {
                "training_schema_version": TRAINING_SCHEMA_VERSION,
                "experiment_id": config.experiment_name,
                "backend": {
                    "name": preflight.backend.name,
                    "version": preflight.backend.version,
                    "model_identifier": config.model,
                },
                "class_mapping": _mapping_list(manifest.class_mapping),
                "dataset_fingerprint": manifest.fingerprint,
                "image_size": config.image_size,
                "training_config": "../training_config.json",
                "experiment_metadata": "../experiment_metadata.json",
                "best_checkpoint": best.name,
                "last_checkpoint": last.name,
                "best_selection_rule": backend_result.best_selection_rule,
                "smoke_checkpoint_not_final_model": config.is_smoke,
            }
            _write_json(checkpoint_dir / "checkpoint_metadata.json", checkpoint_metadata)
            metadata.update(
                {
                    "completed_at": _utc_now(),
                    "duration_seconds": duration,
                    "status": "completed",
                    "termination": {
                        "status": "completed",
                        "epochs_completed": backend_result.epochs_completed,
                    },
                    "backend_training": {
                        "output_dir": _relative(output_dir, backend_result.backend_output_dir),
                        "effective_arguments": dict(backend_result.effective_arguments),
                        "effective_arguments_reference": (
                            _relative(output_dir, backend_result.backend_output_dir / "args.yaml")
                            if (backend_result.backend_output_dir / "args.yaml").is_file()
                            else None
                        ),
                        "best_selection_rule": backend_result.best_selection_rule,
                        "metrics_references": list(backend_result.metrics_references),
                        "telemetry_is_not_p8_evaluation": True,
                    },
                    "checkpoints": {
                        "best": _relative(output_dir, best),
                        "last": _relative(output_dir, last),
                        "metadata": "checkpoints/checkpoint_metadata.json",
                    },
                }
            )
            _write_json(metadata_path, metadata)
            return TrainingResult(
                experiment_id=config.experiment_name,
                status="completed",
                output_dir=output_dir,
                best_checkpoint=best,
                last_checkpoint=last,
                metadata_path=metadata_path,
                backend_output_dir=backend_result.backend_output_dir,
                duration_seconds=duration,
                epochs_completed=backend_result.epochs_completed,
            )
        except Exception as exc:
            metadata.update(
                {
                    "completed_at": _utc_now(),
                    "duration_seconds": time.monotonic() - started,
                    "status": "failed",
                    "termination": {"status": "failed", "error": str(exc)},
                }
            )
            _write_json(metadata_path, metadata)
            raise


def _base_metadata(
    *,
    manifest: DatasetManifest,
    config: TrainingConfig,
    preflight: TrainingPreflight,
    created_at: str,
    manifest_path: str | Path | None,
) -> dict[str, object]:
    return {
        "training_schema_version": TRAINING_SCHEMA_VERSION,
        "project_version": PROJECT_VERSION,
        "experiment_id": config.experiment_name,
        "created_at": created_at,
        "status": "running",
        "smoke_checkpoint_not_final_model": config.is_smoke,
        "dataset": {
            "manifest_fingerprint": manifest.fingerprint,
            "manifest_schema_version": manifest.schema_version,
            "annotation_format": manifest.annotation_format,
            "manifest_reference": str(manifest_path) if manifest_path is not None else None,
            "manifest_snapshot": "dataset_manifest.json",
            "class_mapping": _mapping_list(manifest.class_mapping),
            "train_sample_count": preflight.train_sample_count,
            "val_sample_count": preflight.val_sample_count,
            "selected_train_sample_count": len(preflight.selected_train_sample_ids),
            "selected_val_sample_count": len(preflight.selected_val_sample_ids),
            "per_class_support": preflight.to_dict()["per_class_support"],
            "test_split_usage": "excluded",
        },
        "training_config": config.to_dict(),
        "preflight_issues": [issue.to_dict() for issue in preflight.issues],
        "reproducibility": {
            "seed": config.seed,
            "determinism_mode": (
                "deterministic data plan and seeded backend; bitwise GPU reproducibility not guaranteed"
            ),
            "python_version": preflight.backend.python_version,
            "torch_version": preflight.backend.torch_version,
            "cuda_version": preflight.backend.cuda_version,
            "cuda_available": preflight.backend.cuda_available,
            "requested_device": config.device,
            "resolved_device": preflight.resolved_device,
        },
        "backend": {
            "name": preflight.backend.name,
            "version": preflight.backend.version,
            "model_identifier": config.model,
        },
    }


def _materialize_data_view(
    *,
    manifest: DatasetManifest,
    dataset_root: Path,
    samples: Mapping[str, Sequence[DatasetSample]],
    target: Path,
) -> Path:
    """Create an exact backend view without copying image bytes or raw labels."""

    for split in ("train", "val"):
        image_dir = target / split / "images"
        label_dir = target / split / "labels"
        image_dir.mkdir(parents=True)
        label_dir.mkdir()
        for sample in samples[split]:
            if sample.split != split:
                raise TrainingContractError(f"sample {sample.sample_id} is not in {split} split")
            source = manifest.resolve_image(dataset_root, sample)
            if not source.is_file():
                raise TrainingContractError(f"manifest image does not exist: {source}")
            if _file_sha256(source) != sample.content_sha256:
                raise TrainingContractError(f"manifest image hash mismatch: {sample.image_path}")
            destination = image_dir / source.name
            try:
                os.link(source, destination)
            except OSError as exc:
                raise TrainingContractError(
                    "cannot hard-link manifest images into the experiment view; "
                    "choose an output_root on the same filesystem as the dataset"
                ) from exc
            lines = [
                " ".join(
                    (
                        str(annotation.class_id),
                        _number(annotation.bbox.x_center),
                        _number(annotation.bbox.y_center),
                        _number(annotation.bbox.width),
                        _number(annotation.bbox.height),
                    )
                )
                for annotation in sample.annotations
            ]
            content = "\n".join(lines) + ("\n" if lines else "")
            (label_dir / f"{source.stem}.txt").write_text(content, encoding="utf-8")

    yaml_lines = [
        f"path: {json.dumps(str(target.resolve()))}",
        "train: train/images",
        "val: val/images",
        f"nc: {len(manifest.class_mapping)}",
        "names:",
    ]
    yaml_lines.extend(
        f"  {class_id}: {json.dumps(class_name, ensure_ascii=False)}"
        for class_id, class_name in manifest.class_mapping.items()
    )
    config_path = target / "data.yaml"
    config_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    return config_path


def _select_samples(
    samples: Sequence[DatasetSample], limit: int | None, seed: int
) -> tuple[DatasetSample, ...]:
    ordered = sorted(
        samples,
        key=lambda sample: (
            hashlib.sha256(f"{seed}:{sample.sample_id}".encode("utf-8")).hexdigest(),
            sample.sample_id,
        ),
    )
    if limit is None or limit >= len(ordered):
        return tuple(sorted(samples, key=lambda sample: sample.sample_id))
    selected: list[DatasetSample] = []
    selected_ids: set[str] = set()
    uncovered = {annotation.class_id for sample in ordered for annotation in sample.annotations}
    while uncovered and len(selected) < limit:
        candidate = next(
            (
                sample
                for sample in ordered
                if sample.sample_id not in selected_ids
                and uncovered.intersection(annotation.class_id for annotation in sample.annotations)
            ),
            None,
        )
        if candidate is None:
            break
        selected.append(candidate)
        selected_ids.add(candidate.sample_id)
        uncovered.difference_update(annotation.class_id for annotation in candidate.annotations)
    for sample in ordered:
        if len(selected) >= limit:
            break
        if sample.sample_id not in selected_ids:
            selected.append(sample)
            selected_ids.add(sample.sample_id)
    return tuple(sorted(selected, key=lambda sample: sample.sample_id))


def _class_support(manifest: DatasetManifest) -> dict[int, dict[str, int]]:
    counters = {split: Counter() for split in ("train", "val")}
    for split in counters:
        for sample in manifest.split(split):
            counters[split].update(annotation.class_id for annotation in sample.annotations)
    return {
        class_id: {"train": counters["train"][class_id], "val": counters["val"][class_id]}
        for class_id in manifest.class_mapping
    }


def _resolve_device(
    requested: str, info: BackendInfo, issues: list[TrainingIssue]
) -> str | None:
    if not info.available:
        return None
    if requested == "cpu":
        return "cpu"
    if requested == "auto":
        return "cuda:0" if info.cuda_available else "cpu"
    match = _CUDA_DEVICE.fullmatch(requested)
    assert match is not None
    index = int(match.group(1) or 0)
    if not info.cuda_available or index >= info.device_count:
        issues.append(
            TrainingIssue(
                "ERROR", "cuda_unavailable", f"requested CUDA device {index} is unavailable"
            )
        )
        return None
    return f"cuda:{index}"


def _verify_manifest_identity(manifest: DatasetManifest) -> None:
    if manifest.schema_version != SCHEMA_VERSION or manifest.annotation_format != ANNOTATION_FORMAT:
        raise TrainingContractError("unsupported DatasetManifest schema or annotation format")
    payload = manifest.to_dict(include_fingerprint=False)
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if manifest.fingerprint != expected:
        raise TrainingContractError("manifest fingerprint does not match its semantic content")


def _samples_by_id(
    manifest: DatasetManifest, sample_ids: Sequence[str]
) -> tuple[DatasetSample, ...]:
    by_id = {sample.sample_id: sample for sample in manifest.samples}
    try:
        return tuple(by_id[sample_id] for sample_id in sample_ids)
    except KeyError as exc:
        raise TrainingContractError(f"unknown manifest sample: {exc.args[0]}") from exc


def _require_checkpoint(path: Path, kind: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise TrainingBackendError(f"backend did not produce a non-empty {kind} checkpoint: {path}")


def _mapping_list(mapping: Mapping[int, str]) -> list[dict[str, object]]:
    return [
        {"class_id": class_id, "class_name": mapping[class_id]}
        for class_id in sorted(mapping)
    ]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: float) -> str:
    return format(value, ".17g")


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_int_at_least(value: object, name: str, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TrainingContractError(f"{name} must be an integer >= {minimum}")


def _require_nonempty_string(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise TrainingContractError(f"{name} must be a non-empty, trimmed string")
