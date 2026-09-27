"""Controlled camera-frame capture and learned-inference boundary diagnosis."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bottle_cap_vision.camera_demo import DEFAULT_CONFIG
from bottle_cap_vision.config import AppConfig
from bottle_cap_vision.detector import create_detector
from bottle_cap_vision.inference import LearnedDetector, PredictionAdapter
from bottle_cap_vision.models import Detection, SourceMetadata
from bottle_cap_vision.pipeline import BottleCapPipeline
from bottle_cap_vision.practical_demo import PracticalDemoError, load_learned_config
from bottle_cap_vision.sources import CameraSource
from bottle_cap_vision.visualization import annotate_detections


DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "diagnostics" / "camera_fp"
WINDOW_NAME = "BottleCapVision - Controlled Camera FP Capture"
QUIT_KEYS = {ord("q"), ord("Q"), 27}
SAVE_KEYS = {ord("s"), ord("S")}


class DiagnosticError(RuntimeError):
    """A controlled-capture or replay contract could not be satisfied."""


def frame_sha256(frame: np.ndarray) -> str:
    """Hash the exact ndarray byte sequence supplied by the camera/decoder."""
    return hashlib.sha256(frame.tobytes(order="C")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_png(path: str | Path) -> np.ndarray:
    """Read a PNG through bytes so Unicode workspace paths remain diagnostic-safe."""
    image_path = Path(path)
    try:
        encoded = np.frombuffer(image_path.read_bytes(), dtype=np.uint8)
    except OSError as exc:
        raise DiagnosticError(f"Cannot read replay frame {image_path}: {exc}") from exc
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise DiagnosticError(f"Replay frame is not a decodable image: {image_path}")
    return frame


def write_lossless_png(path: Path, frame: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", frame)
    if not ok:
        raise DiagnosticError(f"OpenCV could not encode lossless PNG: {path}")
    path.write_bytes(encoded.tobytes())
    decoded = read_png(path)
    if decoded.shape != frame.shape or decoded.dtype != frame.dtype or not np.array_equal(decoded, frame):
        raise DiagnosticError(f"Saved PNG is not pixel-equivalent to the raw frame: {path}")


def serialize_detection(detection: Detection, index: int) -> dict[str, object]:
    x, y, width, height = detection.bbox
    return {
        "index": index,
        "class_id": detection.class_id,
        "class_name": detection.color,
        "confidence": detection.confidence,
        "bbox_xywh": [x, y, width, height],
        "bbox_xyxy": [x, y, x + width, y + height],
    }


def serialize_detections(detections: Sequence[Detection]) -> list[dict[str, object]]:
    return [serialize_detection(detection, index) for index, detection in enumerate(detections)]


def diagnostic_overlay(frame: np.ndarray, detections: Sequence[Detection]) -> np.ndarray:
    overlay = annotate_detections(frame, list(detections))
    for index, detection in enumerate(detections):
        x, y, _width, _height = detection.bbox
        cv2.putText(
            overlay,
            f"#{index}",
            (max(0, x), min(overlay.shape[0] - 1, max(18, y + 18))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            f"#{index}",
            (max(0, x), min(overlay.shape[0] - 1, max(18, y + 18))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return overlay


def load_diagnostic_detector(
    config_path: str | Path,
) -> tuple[AppConfig, Path, Path, Path, LearnedDetector]:
    config, resolved_config, checkpoint, metadata = load_learned_config(config_path)
    detector = create_detector(config)
    if not isinstance(detector, LearnedDetector):
        raise DiagnosticError("Diagnostic requires the learned production detector")
    return config, resolved_config, checkpoint, metadata, detector


def _assert_frame_unchanged(frame: np.ndarray, before_hash: str, boundary: str) -> None:
    after_hash = frame_sha256(frame)
    if after_hash != before_hash:
        raise DiagnosticError(f"{boundary} mutated the source frame")


def save_capture(
    *,
    output_dir: Path,
    frame: np.ndarray,
    detections: Sequence[Detection],
    camera_index: int,
    source_metadata: SourceMetadata,
    config_path: Path,
    checkpoint: Path,
    checkpoint_hash: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_path = output_dir / "frame.png"
    raw_hash = frame_sha256(frame)
    write_lossless_png(frame_path, frame)
    if frame_sha256(read_png(frame_path)) != raw_hash:
        raise DiagnosticError("Decoded capture hash differs from the raw camera frame")

    metadata = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "frame": str(frame_path.resolve()),
        "frame_shape": list(frame.shape),
        "frame_dtype": str(frame.dtype),
        "frame_sha256": raw_hash,
        "camera_index": camera_index,
        "source_width": source_metadata.width,
        "source_height": source_metadata.height,
        "source_fps": source_metadata.fps,
        "config_path": str(config_path.resolve()),
        "config_sha256": file_sha256(config_path),
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "raw_unannotated_lossless_png": True,
    }
    predictions = {
        "boundary": "live_direct_detector",
        "frame_sha256": raw_hash,
        "predictions": serialize_detections(detections),
    }
    _write_json(output_dir / "frame_metadata.json", metadata)
    _write_json(output_dir / "predictions.json", predictions)
    return frame_path


def capture_one_frame(
    *,
    camera_index: int,
    config_path: str | Path,
    output_dir: str | Path,
) -> Path | None:
    config, resolved_config, checkpoint, _metadata, detector = load_diagnostic_detector(config_path)
    del config
    checkpoint_hash = file_sha256(checkpoint)
    source = CameraSource(camera_index)
    source.open()
    try:
        for packet in source:
            raw_frame = packet.frame
            before_hash = frame_sha256(raw_frame)
            detections = detector.detect(raw_frame)
            _assert_frame_unchanged(raw_frame, before_hash, "live detector inference")

            cv2.imshow(WINDOW_NAME, diagnostic_overlay(raw_frame, detections))
            key = cv2.waitKey(1) & 0xFF
            if key in SAVE_KEYS:
                return save_capture(
                    output_dir=Path(output_dir),
                    frame=raw_frame,
                    detections=detections,
                    camera_index=camera_index,
                    source_metadata=source.metadata,
                    config_path=resolved_config,
                    checkpoint=checkpoint,
                    checkpoint_hash=checkpoint_hash,
                )
            if key in QUIT_KEYS:
                return None
    finally:
        source.close()
        try:
            cv2.destroyWindow(WINDOW_NAME)
        except cv2.error:
            # Inference can fail before the first imshow creates the window.
            pass
    return None


def _as_numpy(value: object) -> np.ndarray:
    converted = value
    if hasattr(converted, "detach"):
        converted = converted.detach()
    if hasattr(converted, "cpu"):
        converted = converted.cpu()
    if hasattr(converted, "numpy"):
        converted = converted.numpy()
    return np.asarray(converted)


def raw_predict(detector: LearnedDetector, frame: np.ndarray) -> object:
    """Run the detector-owned Ultralytics model with production inference options."""
    try:
        try:
            import torch

            context = torch.inference_mode()
        except ImportError:
            context = nullcontext()
        with context:
            results = detector._model.predict(
                source=frame,
                imgsz=detector.metadata.image_size,
                device=detector.device,
                conf=detector.confidence_threshold,
                verbose=False,
            )
    except Exception as exc:
        raise DiagnosticError(f"Raw Ultralytics inference failed: {exc}") from exc
    if not isinstance(results, (list, tuple)) or len(results) != 1:
        raise DiagnosticError("Raw Ultralytics inference did not return exactly one result")
    return results[0]


def serialize_raw_result(
    result: object,
    adapter: PredictionAdapter,
    frame_shape: tuple[int, ...],
) -> tuple[list[dict[str, object]], list[Detection]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        raise DiagnosticError("Raw Ultralytics result has no boxes")
    xyxy = _as_numpy(getattr(boxes, "xyxy", None))
    classes = _as_numpy(getattr(boxes, "cls", None)).reshape(-1)
    confidences = _as_numpy(getattr(boxes, "conf", None)).reshape(-1)
    adapter_detections = adapter.to_detections(result, frame_shape)
    if len(xyxy) != len(classes) or len(xyxy) != len(confidences):
        raise DiagnosticError("Raw Ultralytics prediction arrays have different lengths")
    if len(xyxy) != len(adapter_detections):
        raise DiagnosticError("Raw and adapter prediction counts differ for the same result")

    records: list[dict[str, object]] = []
    for index, (raw_box, raw_class, confidence, adapted) in enumerate(
        zip(xyxy, classes, confidences, adapter_detections)
    ):
        records.append(
            {
                "index": index,
                "class_id": int(raw_class),
                "class_name": adapted.color,
                "confidence": float(confidence),
                "bbox_xyxy": [float(value) for value in raw_box],
                "canonical_bbox_xywh": list(adapted.bbox),
            }
        )
    return records, adapter_detections


def _prediction_matches(target: dict[str, object], candidate: dict[str, object]) -> bool:
    if target.get("class_id") != candidate.get("class_id"):
        return False
    target_box = target.get("bbox_xywh")
    candidate_box = candidate.get("bbox_xywh", candidate.get("canonical_bbox_xywh"))
    if not isinstance(target_box, list) or not isinstance(candidate_box, list):
        return False
    if len(target_box) != 4 or len(candidate_box) != 4:
        return False
    if any(abs(float(left) - float(right)) > 2.0 for left, right in zip(target_box, candidate_box)):
        return False
    target_confidence = target.get("confidence")
    candidate_confidence = candidate.get("confidence")
    return (
        isinstance(target_confidence, (int, float))
        and isinstance(candidate_confidence, (int, float))
        and abs(float(target_confidence) - float(candidate_confidence)) <= 1e-3
    )


def _contains_prediction(target: dict[str, object], predictions: Sequence[dict[str, object]]) -> bool:
    return any(_prediction_matches(target, candidate) for candidate in predictions)


def _same_predictions(
    left: Sequence[dict[str, object]], right: Sequence[dict[str, object]]
) -> bool:
    if len(left) != len(right):
        return False
    unmatched = list(right)
    for prediction in left:
        for index, candidate in enumerate(unmatched):
            if _prediction_matches(prediction, candidate):
                unmatched.pop(index)
                break
        else:
            return False
    return True


def _load_capture_evidence(frame_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    metadata_path = frame_path.parent / "frame_metadata.json"
    predictions_path = frame_path.parent / "predictions.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DiagnosticError(f"Capture metadata/predictions are required beside replay PNG: {exc}") from exc
    if not isinstance(metadata, dict) or not isinstance(predictions, dict):
        raise DiagnosticError("Capture evidence JSON roots must be objects")
    return metadata, predictions


def _verify_provenance(
    *,
    frame: np.ndarray,
    metadata: dict[str, object],
    config_path: Path,
    checkpoint: Path,
    checkpoint_hash: str,
    config_hash: str,
) -> None:
    expected = {
        "frame_sha256": frame_sha256(frame),
        "config_path": str(config_path.resolve()),
        "config_sha256": config_hash,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
    }
    mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
    if mismatches:
        raise DiagnosticError(
            "Replay provenance differs from capture for: " + ", ".join(mismatches)
        )


def replay_frame(
    *,
    frame_path: str | Path,
    config_path: str | Path,
    runs: int = 3,
    fp_index: int | None = None,
    actual_object: str | None = None,
    no_fp: bool = False,
) -> dict[str, object]:
    if runs <= 0:
        raise DiagnosticError("Replay runs must be positive")
    replay_path = Path(frame_path).resolve()
    frame = read_png(replay_path)
    capture_metadata, capture_predictions = _load_capture_evidence(replay_path)
    config, resolved_config, checkpoint, _metadata, detector = load_diagnostic_detector(config_path)
    checkpoint_hash = file_sha256(checkpoint)
    config_hash = file_sha256(resolved_config)
    _verify_provenance(
        frame=frame,
        metadata=capture_metadata,
        config_path=resolved_config,
        checkpoint=checkpoint,
        checkpoint_hash=checkpoint_hash,
        config_hash=config_hash,
    )

    raw_runs: list[dict[str, object]] = []
    adapter_runs: list[dict[str, object]] = []
    detector_runs: list[dict[str, object]] = []
    frame_hash = frame_sha256(frame)
    for run_index in range(1, runs + 1):
        raw_result = raw_predict(detector, frame)
        _assert_frame_unchanged(frame, frame_hash, f"raw replay run {run_index}")
        raw_records, adapter_detections = serialize_raw_result(
            raw_result, detector._adapter, frame.shape
        )
        _assert_frame_unchanged(frame, frame_hash, f"adapter replay run {run_index}")
        direct_detections = detector.detect(frame)
        _assert_frame_unchanged(frame, frame_hash, f"direct detector replay run {run_index}")
        raw_runs.append({"run": run_index, "predictions": raw_records})
        adapter_runs.append(
            {"run": run_index, "predictions": serialize_detections(adapter_detections)}
        )
        detector_runs.append(
            {"run": run_index, "predictions": serialize_detections(direct_detections)}
        )

    pipeline_result = BottleCapPipeline(config, detector=detector).process_frame(frame)
    _assert_frame_unchanged(frame, frame_hash, "pipeline confirmation")
    pipeline_predictions = serialize_detections(pipeline_result.detections)
    output_dir = replay_path.parent
    common = {
        "frame": str(replay_path),
        "frame_sha256": frame_hash,
        "config_path": str(resolved_config),
        "config_sha256": config_hash,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "confidence_threshold": detector.confidence_threshold,
        "image_size": detector.metadata.image_size,
        "device": detector.device,
    }
    _write_json(output_dir / "raw_predictions.json", {**common, "runs": raw_runs})
    _write_json(output_dir / "adapter_predictions.json", {**common, "runs": adapter_runs})
    _write_json(output_dir / "detector_predictions.json", {**common, "runs": detector_runs})
    _write_json(
        output_dir / "pipeline_predictions.json",
        {**common, "predictions": pipeline_predictions},
    )

    selected: dict[str, object] | None = None
    origin = "UNCLASSIFIED"
    reproduction: dict[str, object] | None = None
    if no_fp:
        origin = "NO_FP_CAPTURED"
    elif fp_index is not None:
        live_predictions = capture_predictions.get("predictions")
        if not isinstance(live_predictions, list) or not 0 <= fp_index < len(live_predictions):
            raise DiagnosticError(f"FP index {fp_index} is not present in live predictions")
        if not actual_object or not actual_object.strip():
            raise DiagnosticError("--actual-object is required when --fp-index is used")
        selected = dict(live_predictions[fp_index])
        selected["actual_object_or_background"] = actual_object.strip()
        selected["user_confirmed_fp"] = True
        raw_count = sum(
            _contains_prediction(selected, run["predictions"]) for run in raw_runs
        )
        adapter_count = sum(
            _contains_prediction(selected, run["predictions"]) for run in adapter_runs
        )
        detector_count = sum(
            _contains_prediction(selected, run["predictions"]) for run in detector_runs
        )
        pipeline_present = _contains_prediction(selected, pipeline_predictions)
        reproduction = {
            "raw_ultralytics": f"{raw_count}/{runs}",
            "adapter": f"{adapter_count}/{runs}",
            "direct_detector": f"{detector_count}/{runs}",
            "pipeline": pipeline_present,
        }
        if raw_count:
            origin = "RAW_MODEL"
        elif adapter_count:
            origin = "ADAPTER"
        elif pipeline_present and detector_count == 0:
            origin = "PIPELINE"
        else:
            origin = "NOT_REPRODUCIBLE"

    comparisons = []
    for raw_run, adapter_run, detector_run in zip(raw_runs, adapter_runs, detector_runs):
        raw_canonical = [
            {
                **prediction,
                "bbox_xywh": prediction["canonical_bbox_xywh"],
            }
            for prediction in raw_run["predictions"]
        ]
        comparisons.append(
            {
                "run": raw_run["run"],
                "raw_equals_adapter": _same_predictions(
                    raw_canonical, adapter_run["predictions"]
                ),
                "adapter_equals_direct_detector": _same_predictions(
                    adapter_run["predictions"], detector_run["predictions"]
                ),
            }
        )

    summary = {
        **common,
        "frame_shape": list(frame.shape),
        "frame_dtype": str(frame.dtype),
        "frame_mutated": False,
        "runs": runs,
        "comparisons": comparisons,
        "pipeline_equals_last_direct_detector": _same_predictions(
            detector_runs[-1]["predictions"], pipeline_predictions
        ),
        "selected_fp": selected,
        "reproduction": reproduction,
        "origin": origin,
    }
    _write_json(output_dir / "replay_summary.json", summary)
    return summary


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture one raw camera FP frame or replay it across learned boundaries"
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--replay", help="Replay a controlled frame.png instead of opening camera")
    parser.add_argument("--runs", type=positive_int, default=3)
    identification = parser.add_mutually_exclusive_group()
    identification.add_argument("--fp-index", type=non_negative_int)
    identification.add_argument("--no-fp", action="store_true")
    parser.add_argument(
        "--actual-object",
        help="What the selected false-positive box actually covers (required with --fp-index)",
    )
    args = parser.parse_args(argv)
    if not args.replay and (args.fp_index is not None or args.no_fp or args.actual_object):
        parser.error("FP identification options require --replay")
    if args.fp_index is not None and not args.actual_object:
        parser.error("--actual-object is required with --fp-index")
    if args.actual_object and args.fp_index is None:
        parser.error("--actual-object requires --fp-index")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.replay:
            summary = replay_frame(
                frame_path=args.replay,
                config_path=args.config,
                runs=args.runs,
                fp_index=args.fp_index,
                actual_object=args.actual_object,
                no_fp=args.no_fp,
            )
            print(f"Replay complete: {args.replay}")
            for comparison in summary["comparisons"]:
                print(
                    f"run {comparison['run']}: "
                    f"raw≈adapter={comparison['raw_equals_adapter']} "
                    f"adapter≈detector={comparison['adapter_equals_direct_detector']}"
                )
            print(f"origin: {summary['origin']}")
            if summary["origin"] == "UNCLASSIFIED":
                print("Identify the FP with --fp-index N --actual-object DESCRIPTION, or --no-fp")
            return 0

        print("Controlled camera capture started")
        print("S: save exactly one raw frame for diagnosis")
        print("Q / Esc: quit without capture")
        frame_path = capture_one_frame(
            camera_index=args.camera_index,
            config_path=args.config,
            output_dir=args.output_dir,
        )
        if frame_path is None:
            print("Exited without capture")
        else:
            print(f"Saved raw lossless frame: {frame_path}")
            print(f"Live predictions: {frame_path.parent / 'predictions.json'}")
            print(
                "Replay and identify: --replay "
                f'"{frame_path}" --fp-index N --actual-object "description"'
            )
        return 0
    except (DiagnosticError, PracticalDemoError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
