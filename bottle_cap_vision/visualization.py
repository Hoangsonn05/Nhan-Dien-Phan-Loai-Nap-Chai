from __future__ import annotations

import cv2
import numpy as np

from .config import AppConfig
from .models import FrameResult, Track


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
