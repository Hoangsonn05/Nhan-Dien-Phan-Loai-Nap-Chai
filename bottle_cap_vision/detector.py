from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol
import cv2
import numpy as np

from .config import AppConfig, ColorConfig
from .models import Detection


_DUPLICATE_IOU_THRESHOLD = 0.5
_DUPLICATE_CONTAINMENT_THRESHOLD = 0.8


class DetectorProtocol(Protocol):
    """The only detector contract visible to the runtime core."""

    def detect(self, frame: np.ndarray) -> list[Detection]: ...


@dataclass(frozen=True)
class _Candidate:
    contour: np.ndarray
    bbox: tuple[int, int, int, int]
    centroid: tuple[int, int]
    area: float
    circularity: float
    solidity: float


def _bbox_intersection_area(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> float:
    left_x, left_y, left_w, left_h = left
    right_x, right_y, right_w, right_h = right
    intersection_w = max(0, min(left_x + left_w, right_x + right_w) - max(left_x, right_x))
    intersection_h = max(0, min(left_y + left_h, right_y + right_h) - max(left_y, right_y))
    return float(intersection_w * intersection_h)


def bbox_iou(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> float:
    """Return intersection-over-union for ``(x, y, width, height)`` boxes."""
    left_area = max(0, left[2]) * max(0, left[3])
    right_area = max(0, right[2]) * max(0, right[3])
    intersection = _bbox_intersection_area(left, right)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _bbox_containment(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> float:
    """Return the intersection as a fraction of the smaller box."""
    smaller_area = min(
        max(0, left[2]) * max(0, left[3]),
        max(0, right[2]) * max(0, right[3]),
    )
    if smaller_area == 0:
        return 0.0
    return _bbox_intersection_area(left, right) / smaller_area


def _is_duplicate(left: Detection, right: Detection) -> bool:
    return (
        bbox_iou(left.bbox, right.bbox) >= _DUPLICATE_IOU_THRESHOLD
        or _bbox_containment(left.bbox, right.bbox) >= _DUPLICATE_CONTAINMENT_THRESHOLD
    )


def _winner_key(detection: Detection) -> tuple:
    return (
        -detection.color_ratio,
        -detection.area,
        detection.color,
        detection.bbox,
        detection.centroid,
    )


class ColorCapDetector:
    """Classical OpenCV detector used as the first software baseline.

    Pipeline: BGR -> HSV -> color masks -> morphology -> connected contours ->
    geometric filters -> centroid/bounding box.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        k = max(3, int(config.detection.morphology_kernel) | 1)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    def _roi(self, frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        roi = self.config.roi
        if not roi.enabled:
            return frame, (0, 0)
        h, w = frame.shape[:2]
        x1, y1 = max(0, min(roi.x1, w - 1)), max(0, min(roi.y1, h - 1))
        x2, y2 = max(x1 + 1, min(roi.x2, w)), max(y1 + 1, min(roi.y2, h))
        return frame[y1:y2, x1:x2], (x1, y1)

    @staticmethod
    def _shape_metrics(contour: np.ndarray) -> tuple[float, float]:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = (4.0 * np.pi * area / (perimeter * perimeter)) if perimeter > 1e-6 else 0.0
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        solidity = area / hull_area if hull_area > 1e-6 else 0.0
        return circularity, solidity

    def _mask_for_color(self, hsv: np.ndarray, color_cfg: ColorConfig) -> np.ndarray:
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for rng in color_cfg.hsv_ranges:
            lo = np.array(rng.lower, dtype=np.uint8)
            hi = np.array(rng.upper, dtype=np.uint8)
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=self.config.detection.open_iterations)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=self.config.detection.close_iterations)
        return mask

    def _extract_candidates(self, mask: np.ndarray) -> list[_Candidate]:
        candidates: list[_Candidate] = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dcfg = self.config.detection
        for contour in contours:
            if contour.size == 0:
                continue
            area = float(cv2.contourArea(contour))
            if not (dcfg.min_area <= area <= dcfg.max_area):
                continue
            circularity, solidity = self._shape_metrics(contour)
            if circularity < dcfg.min_circularity or solidity < dcfg.min_solidity:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue
            moments = cv2.moments(contour)
            if abs(moments["m00"]) < 1e-6:
                centroid = (x + w // 2, y + h // 2)
            else:
                centroid = (
                    int(moments["m10"] / moments["m00"]),
                    int(moments["m01"] / moments["m00"]),
                )
            candidates.append(_Candidate(
                contour=contour,
                bbox=(x, y, w, h),
                centroid=centroid,
                area=area,
                circularity=circularity,
                solidity=solidity,
            ))
        return candidates

    def _classify_candidate(
        self,
        candidate: _Candidate,
        mask: np.ndarray,
        color: str,
        offset: tuple[int, int],
    ) -> Detection | None:
        component_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(component_mask, [candidate.contour], -1, 255, -1)
        color_pixels = cv2.countNonZero(cv2.bitwise_and(mask, mask, mask=component_mask))
        total_pixels = cv2.countNonZero(component_mask)
        color_ratio = color_pixels / total_pixels if total_pixels else 0.0
        if color_ratio < self.config.detection.min_color_ratio:
            return None

        ox, oy = offset
        x, y, w, h = candidate.bbox
        cx, cy = candidate.centroid
        return Detection(
            color=color,
            centroid=(cx + ox, cy + oy),
            bbox=(x + ox, y + oy, w, h),
            area=candidate.area,
            circularity=candidate.circularity,
            solidity=candidate.solidity,
            color_ratio=color_ratio,
        )

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if frame is None or frame.size == 0:
            return []
        work, (ox, oy) = self._roi(frame)
        hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
        detections: list[Detection] = []

        for color_cfg in self.config.colors:
            mask = self._mask_for_color(hsv, color_cfg)
            for candidate in self._extract_candidates(mask):
                detection = self._classify_candidate(candidate, mask, color_cfg.name, (ox, oy))
                if detection is not None:
                    detections.append(detection)
        return self._deduplicate(detections)

    @staticmethod
    def _deduplicate(detections: List[Detection]) -> List[Detection]:
        # NMS-style suppression uses bbox evidence only. Centroid proximity alone
        # is not evidence that two frame-local observations are the same object.
        result: list[Detection] = []
        for det in sorted(detections, key=_winner_key):
            if any(_is_duplicate(det, previous) for previous in result):
                continue
            result.append(det)
        return result


# Explicit backend name while preserving the established public class name.
HSVDetector = ColorCapDetector


def create_detector(config: AppConfig) -> DetectorProtocol:
    """Resolve a detector once at the composition boundary."""
    backend = config.detector.backend
    if backend == "hsv":
        return HSVDetector(config)
    if backend == "learned":
        if config.detector.checkpoint is None or config.detector.metadata is None:
            raise ValueError("learned detector requires checkpoint and metadata paths")
        # Lazy import keeps HSV-only runtime independent from the ML framework.
        from .inference import LearnedDetector

        return LearnedDetector(
            checkpoint=config.detector.checkpoint,
            metadata=config.detector.metadata,
            device=config.detector.device,
            confidence_threshold=config.detector.confidence_threshold,
            runtime_class_names=[color.name for color in config.colors],
        )
    raise ValueError(f"Unsupported detector backend: {backend!r}")
