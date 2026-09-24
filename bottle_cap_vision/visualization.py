from __future__ import annotations

import cv2
import numpy as np

from .config import AppConfig
from .models import Detection, FrameResult, Track


_DETECTION_COLORS: dict[str, tuple[int, int, int]] = {
    "blue": (235, 115, 25),
    "brown": (45, 95, 150),
    "green": (65, 195, 85),
    "lilac": (220, 155, 220),
    "orange": (20, 145, 250),
    "pink": (190, 125, 250),
    "purple": (185, 75, 180),
    "red": (55, 55, 235),
    "white": (235, 235, 235),
    "yellow": (35, 210, 245),
}
_FALLBACK_COLORS = tuple(_DETECTION_COLORS.values())


def _detection_color(class_name: str) -> tuple[int, int, int]:
    normalized = class_name.strip().casefold()
    if normalized in _DETECTION_COLORS:
        return _DETECTION_COLORS[normalized]
    palette_index = sum(normalized.encode("utf-8")) % len(_FALLBACK_COLORS)
    return _FALLBACK_COLORS[palette_index]


def _contrasting_text_color(background: tuple[int, int, int]) -> tuple[int, int, int]:
    blue, green, red = background
    luminance = 0.114 * blue + 0.587 * green + 0.299 * red
    return (20, 20, 20) if luminance >= 150 else (255, 255, 255)


def annotate_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Return a frame-local detection view without tracker/counting semantics."""
    annotated = frame.copy()
    image_height, image_width = annotated.shape[:2]
    box_thickness = max(2, min(4, round(min(image_height, image_width) / 400)))
    font_scale = 0.68
    text_thickness = 2
    padding = 4
    for detection in detections:
        x, y, width, height = detection.bbox
        box_color = _detection_color(detection.color)
        cv2.rectangle(
            annotated,
            (x, y),
            (x + width, y + height),
            box_color,
            box_thickness,
        )
        label = detection.color
        if detection.confidence is not None:
            label = f"{label} {detection.confidence:.0%}"

        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
        )
        label_width = text_width + 2 * padding
        label_height = text_height + baseline + 2 * padding
        label_x = max(0, min(x, image_width - label_width))
        label_y = y - label_height if y >= label_height else max(0, min(y, image_height - label_height))
        label_right = min(image_width - 1, label_x + label_width)
        label_bottom = min(image_height - 1, label_y + label_height)
        cv2.rectangle(
            annotated,
            (label_x, label_y),
            (label_right, label_bottom),
            box_color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (label_x + padding, min(label_bottom - baseline - padding, label_y + padding + text_height)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            _contrasting_text_color(box_color),
            text_thickness,
            cv2.LINE_AA,
        )
    return annotated


def draw_line(frame: np.ndarray, cfg: AppConfig) -> None:
    if not cfg.counting.enabled:
        return
    h, w = frame.shape[:2]
    if cfg.counting.orientation == "vertical":
        x = int(w * cfg.counting.line_position)
        cv2.line(frame, (x, 0), (x, h - 1), (0, 255, 255), 2)
    else:
        y = int(h * cfg.counting.line_position)
        cv2.line(frame, (0, y), (w - 1, y), (0, 255, 255), 2)


def draw_tracks(frame: np.ndarray, tracks: list[Track]) -> None:
    for tr in tracks:
        x, y, w, h = tr.bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"ID {tr.track_id} | {tr.color}"
        cv2.putText(frame, label, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.circle(frame, tr.centroid, 4, (255, 255, 255), -1)


def draw_panel(
    frame: np.ndarray,
    total_count: int,
    count_by_color: dict[str, int],
    fps: float = 0.0,
) -> None:
    h, w = frame.shape[:2]
    panel_w = min(320, w)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, min(h, 240)), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
    y = 28
    cv2.putText(frame, "Bottle Cap Counter", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    y += 30
    cv2.putText(frame, f"TOTAL: {total_count}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    y += 28
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    y += 25
    for color, count in sorted(count_by_color.items()):
        if y > min(h, 235):
            break
        cv2.putText(frame, f"{color}: {count}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1)
        y += 20


def annotate_frame(
    frame: np.ndarray,
    result: FrameResult,
    cfg: AppConfig,
    fps: float = 0.0,
) -> np.ndarray:
    """Return an annotated copy without mutating the source-owned frame."""
    annotated = frame.copy()
    draw_line(annotated, cfg)
    if result.tracks:
        draw_tracks(annotated, result.tracks)
    if cfg.counting.enabled:
        draw_panel(annotated, result.total_count, result.count_by_color, fps)
    return annotated
