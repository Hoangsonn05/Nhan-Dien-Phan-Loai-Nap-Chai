"""Folder-image practical demo built on the canonical learned detector boundary."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from time import perf_counter
from typing import Callable, Protocol

import cv2
import numpy as np

from .config import AppConfig, load_config
from .detector import create_detector
from .models import Detection
from .visualization import annotate_detections


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
WINDOW_TITLE = "BottleCapVision - Practical Image Test"
DISPLAY_MAX_WIDTH = 1800
DISPLAY_MAX_HEIGHT = 950
HEADER_HEIGHT = 56
FOOTER_LINE_HEIGHT = 28
FOOTER_PADDING = 12


class PracticalDemoError(RuntimeError):
    """User-facing failure at the practical-demo boundary."""


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]: ...


class DisplayMode(str, Enum):
    SIDE_BY_SIDE = "SIDE_BY_SIDE"
    RESULT_ONLY = "RESULT_ONLY"
    ORIGINAL_ONLY = "ORIGINAL_ONLY"


_DISPLAY_MODE_ORDER = tuple(DisplayMode)


def cycle_display_mode(mode: DisplayMode) -> DisplayMode:
    index = _DISPLAY_MODE_ORDER.index(mode)
    return _DISPLAY_MODE_ORDER[(index + 1) % len(_DISPLAY_MODE_ORDER)]


@dataclass(frozen=True)
class ProcessedImage:
    source_path: Path
    annotated_path: Path
    relative_name: str
    detection_count: int = 0
    counts_by_color: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class DemoReport:
    predictions: dict[str, object]
    summary: dict[str, object]
    processed_images: list[ProcessedImage]
    output_dir: Path


def discover_images(input_dir: str | Path, *, recursive: bool = False) -> list[Path]:
    root = Path(input_dir)
    if not root.is_dir():
        raise PracticalDemoError(f"Input image folder does not exist: {root}")
    candidates = root.rglob("*") if recursive else root.iterdir()
    images = [
        path
        for path in candidates
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        and not any(part.startswith(".") for part in path.relative_to(root).parts)
    ]
    images.sort(key=lambda path: (path.relative_to(root).as_posix().casefold(), path.relative_to(root).as_posix()))
    if not images:
        raise PracticalDemoError(f"Khong tim thay anh test trong {root}")
    return images


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_learned_config(config_path: str | Path) -> tuple[AppConfig, Path, Path, Path]:
    resolved_config = Path(config_path).resolve()
    if not resolved_config.is_file():
        raise PracticalDemoError(f"Config file does not exist: {resolved_config}")
    try:
        config = load_config(resolved_config)
    except Exception as exc:
        raise PracticalDemoError(f"Cannot load config {resolved_config}: {exc}") from exc
    if config.detector.backend != "learned":
        raise PracticalDemoError(
            f"Practical demo requires detector.backend='learned': {resolved_config}"
        )
    assert config.detector.checkpoint is not None
    assert config.detector.metadata is not None
    checkpoint = _resolve_project_path(config.detector.checkpoint)
    metadata = _resolve_project_path(config.detector.metadata)
    if not checkpoint.is_file():
        raise PracticalDemoError(f"Full learned checkpoint does not exist: {checkpoint}")
    if not metadata.is_file():
        raise PracticalDemoError(f"Checkpoint metadata does not exist: {metadata}")
    config.detector.checkpoint = str(checkpoint)
    config.detector.metadata = str(metadata)
    return config, resolved_config, checkpoint, metadata


def _output_names(images: list[Path], input_dir: Path) -> dict[Path, str]:
    frequencies: dict[str, int] = {}
    for path in images:
        frequencies[path.name.casefold()] = frequencies.get(path.name.casefold(), 0) + 1
    names: dict[Path, str] = {}
    for path in images:
        if frequencies[path.name.casefold()] == 1:
            names[path] = path.name
            continue
        relative = path.relative_to(input_dir).as_posix()
        import hashlib

        suffix = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:10]
        names[path] = f"{path.stem}__{suffix}{path.suffix.lower()}"
    return names


def _detection_record(detection: Detection) -> dict[str, object]:
    return {
        "class_id": detection.class_id,
        "color": detection.color,
        "confidence": detection.confidence,
        "bbox": list(detection.bbox),
        "centroid": list(detection.centroid),
    }


def _write_csv(path: Path, image_records: list[dict[str, object]]) -> None:
    fields = [
        "image", "detection_index", "class_id", "color", "confidence",
        "x", "y", "width", "height", "centroid_x", "centroid_y",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for image in image_records:
            if image["status"] != "OK":
                continue
            for index, detection in enumerate(image["detections"], start=1):
                x, y, width, height = detection["bbox"]
                centroid_x, centroid_y = detection["centroid"]
                writer.writerow({
                    "image": image["file"],
                    "detection_index": index,
                    "class_id": detection["class_id"],
                    "color": detection["color"],
                    "confidence": detection["confidence"],
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "centroid_x": centroid_x,
                    "centroid_y": centroid_y,
                })


def process_images(
    input_dir: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
    *,
    recursive: bool = False,
    detector_factory: Callable[[AppConfig], Detector] | None = None,
) -> DemoReport:
    """Process independent images with one learned detector instance."""
    input_root = Path(input_dir).resolve()
    images = discover_images(input_root, recursive=recursive)
    config, resolved_config, checkpoint, _metadata = _load_learned_config(config_path)
    factory = detector_factory or create_detector
    print(f"Loading full learned model: {checkpoint}")
    try:
        detector = factory(config)
    except Exception as exc:
        raise PracticalDemoError(
            f"Full learned model failed compatibility/load validation ({checkpoint}): {exc}"
        ) from exc

    output_root = Path(output_dir).resolve()
    annotated_dir = output_root / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    output_names = _output_names(images, input_root)
    image_records: list[dict[str, object]] = []
    processed: list[ProcessedImage] = []
    counts = {color.name: 0 for color in config.colors}
    successful_inference_seconds: list[float] = []

    for image_path in images:
        relative_name = image_path.relative_to(input_root).as_posix()
        try:
            frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if frame is None or frame.size == 0:
                raise PracticalDemoError("OpenCV could not decode the image")
            started = perf_counter()
            detections = detector.detect(frame)
            inference_seconds = perf_counter() - started
            annotated = annotate_detections(frame, detections)
            annotated_path = annotated_dir / output_names[image_path]
            if not cv2.imwrite(str(annotated_path), annotated):
                raise PracticalDemoError(f"OpenCV could not write {annotated_path}")
            image_counts: dict[str, int] = {}
            for detection in detections:
                counts[detection.color] = counts.get(detection.color, 0) + 1
                image_counts[detection.color] = image_counts.get(detection.color, 0) + 1
            successful_inference_seconds.append(inference_seconds)
            image_records.append({
                "file": relative_name,
                "width": int(frame.shape[1]),
                "height": int(frame.shape[0]),
                "status": "OK",
                "inference_ms": inference_seconds * 1000.0,
                "detections": [_detection_record(item) for item in detections],
            })
            processed.append(ProcessedImage(
                image_path,
                annotated_path,
                relative_name,
                len(detections),
                tuple(sorted(image_counts.items())),
            ))
        except Exception as exc:
            image_records.append({
                "file": relative_name,
                "width": None,
                "height": None,
                "status": "ERROR",
                "error": str(exc),
                "detections": [],
            })
            print(f"ERROR {relative_name}: {exc}")

    success_count = len(processed)
    failed_count = len(images) - success_count
    total_detections = sum(counts.values())
    average_ms = (
        sum(successful_inference_seconds) * 1000.0 / len(successful_inference_seconds)
        if successful_inference_seconds
        else None
    )
    predictions: dict[str, object] = {
        "model": {
            "checkpoint": str(checkpoint),
            "confidence_threshold": config.detector.confidence_threshold,
        },
        "images": image_records,
    }
    summary: dict[str, object] = {
        "images_discovered": len(images),
        "images_successfully_processed": success_count,
        "failed_images": failed_count,
        "total_detections": total_detections,
        "detections_by_color": counts,
        "average_inference_ms_per_processed_image": average_ms,
        "checkpoint": str(checkpoint),
        "config": str(resolved_config),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "predictions.json").write_text(
        json.dumps(predictions, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_csv(output_root / "predictions.csv", image_records)
    return DemoReport(predictions, summary, processed, output_root)


def _fit_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0], 1.25)
    resized_width = max(1, round(image.shape[1] * scale))
    resized_height = max(1, round(image.shape[0] * scale))
    if (resized_width, resized_height) == (image.shape[1], image.shape[0]):
        return image.copy()
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)


def build_footer_lines(
    *,
    filename: str,
    index: int,
    total: int,
    detection_count: int,
    counts_by_color: tuple[tuple[str, int], ...] = (),
    mode: DisplayMode = DisplayMode.SIDE_BY_SIDE,
) -> tuple[str, ...]:
    image_line = f"Image {index} / {total} | {filename}"
    summary_parts = [f"Detected caps: {detection_count}"]
    summary_parts.extend(f"{color}: {count}" for color, count in counts_by_color)
    summary_lines: list[str] = []
    current_summary = summary_parts[0]
    for part in summary_parts[1:]:
        candidate = f"{current_summary} | {part}"
        if len(candidate) > 110:
            summary_lines.append(current_summary)
            current_summary = part
        else:
            current_summary = candidate
    summary_lines.append(current_summary)
    controls = (
        "A/Left Prev | D/Right Next | M/Tab Mode | Q/Esc Exit"
        f" | Mode: {mode.value}"
    )
    return image_line, *summary_lines, controls


def build_display_board(
    original: np.ndarray,
    annotated: np.ndarray,
    *,
    filename: str,
    index: int,
    total: int,
    detection_count: int = 0,
    counts_by_color: tuple[tuple[str, int], ...] = (),
    mode: DisplayMode = DisplayMode.SIDE_BY_SIDE,
    max_width: int = DISPLAY_MAX_WIDTH,
    max_height: int = DISPLAY_MAX_HEIGHT,
) -> np.ndarray:
    footer_lines = build_footer_lines(
        filename=filename,
        index=index,
        total=total,
        detection_count=detection_count,
        counts_by_color=counts_by_color,
        mode=mode,
    )
    footer_height = FOOTER_PADDING * 2 + FOOTER_LINE_HEIGHT * len(footer_lines)
    content_height_limit = max_height - HEADER_HEIGHT - footer_height
    gap = 12

    if mode is DisplayMode.SIDE_BY_SIDE:
        panel_width = (max_width - gap) // 2
        panels = (
            ("ANH GOC", _fit_image(original, panel_width, content_height_limit)),
            ("KET QUA NHAN DIEN", _fit_image(annotated, panel_width, content_height_limit)),
        )
        content_height = max(image.shape[0] for _, image in panels)
        panel_offsets = (0, panel_width + gap)
    else:
        title = "KET QUA NHAN DIEN" if mode is DisplayMode.RESULT_ONLY else "ANH GOC"
        source = annotated if mode is DisplayMode.RESULT_ONLY else original
        panels = ((title, _fit_image(source, max_width, content_height_limit)),)
        content_height = panels[0][1].shape[0]
        panel_width = max_width
        panel_offsets = (0,)

    board_height = HEADER_HEIGHT + content_height + footer_height
    board = np.full((board_height, max_width, 3), 28, np.uint8)
    for (title, image), panel_x in zip(panels, panel_offsets):
        image_x = panel_x + (panel_width - image.shape[1]) // 2
        image_y = HEADER_HEIGHT + (content_height - image.shape[0]) // 2
        board[image_y:image_y + image.shape[0], image_x:image_x + image.shape[1]] = image
        cv2.putText(
            board,
            title,
            (panel_x + 20, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.88,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    footer_top = HEADER_HEIGHT + content_height
    cv2.line(board, (0, footer_top), (max_width - 1, footer_top), (85, 85, 85), 1)
    for line_index, line in enumerate(footer_lines):
        text_y = footer_top + FOOTER_PADDING + 20 + line_index * FOOTER_LINE_HEIGHT
        cv2.putText(
            board,
            line,
            (20, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60 if line_index < 2 else 0.54,
            (255, 255, 255) if line_index < 2 else (220, 220, 220),
            2 if line_index == 0 else 1,
            cv2.LINE_AA,
        )
    return board


def build_comparison_board(
    original: np.ndarray,
    annotated: np.ndarray,
    *,
    filename: str,
    index: int,
    total: int,
    detection_count: int = 0,
    counts_by_color: tuple[tuple[str, int], ...] = (),
    max_width: int = DISPLAY_MAX_WIDTH,
    max_height: int = DISPLAY_MAX_HEIGHT,
) -> np.ndarray:
    return build_display_board(
        original,
        annotated,
        filename=filename,
        index=index,
        total=total,
        detection_count=detection_count,
        counts_by_color=counts_by_color,
        mode=DisplayMode.SIDE_BY_SIDE,
        max_width=max_width,
        max_height=max_height,
    )


def show_results(images: list[ProcessedImage]) -> None:
    if not images:
        raise PracticalDemoError("No successfully processed images are available for the GUI")
    current = 0
    mode = DisplayMode.SIDE_BY_SIDE
    created = False
    try:
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        created = True
        while True:
            selected = images[current]
            original = cv2.imread(str(selected.source_path), cv2.IMREAD_COLOR)
            annotated = cv2.imread(str(selected.annotated_path), cv2.IMREAD_COLOR)
            if original is None or annotated is None:
                raise PracticalDemoError(f"Cannot open GUI image: {selected.relative_name}")
            board = build_display_board(
                original,
                annotated,
                filename=selected.relative_name,
                index=current + 1,
                total=len(images),
                detection_count=selected.detection_count,
                counts_by_color=selected.counts_by_color,
                mode=mode,
            )
            cv2.imshow(WINDOW_TITLE, board)
            key = cv2.waitKeyEx(0)
            key_char = key & 0xFF
            if key in (27,) or key_char in (27, ord("q"), ord("Q")):
                break
            if key in (2424832, 65361, 81) or key_char in (ord("a"), ord("A"), ord("p"), ord("P")):
                current = (current - 1) % len(images)
            elif key in (2555904, 65363, 83) or key_char in (ord("d"), ord("D"), ord("n"), ord("N"), 13, 32):
                current = (current + 1) % len(images)
            elif key_char in (9, ord("m"), ord("M")):
                mode = cycle_display_mode(mode)
    except cv2.error as exc:
        raise PracticalDemoError(
            "OpenCV GUI is unavailable. Run: python scripts/demo_images.py --no-gui"
        ) from exc
    finally:
        if created:
            cv2.destroyWindow(WINDOW_TITLE)


def print_summary(report: DemoReport) -> None:
    summary = report.summary
    print("\nBottleCapVision Practical Demo\n")
    print(f"Images processed: {summary['images_successfully_processed']}")
    print(f"Failed: {summary['failed_images']}")
    print(f"Detections: {summary['total_detections']}")
    for color, count in summary["detections_by_color"].items():
        print(f"{color}: {count}")
    print(f"\nResults: {report.output_dir}")
