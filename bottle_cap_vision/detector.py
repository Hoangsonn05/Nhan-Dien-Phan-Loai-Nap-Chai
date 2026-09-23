from __future__ import annotations

from typing import List
import cv2
import numpy as np

from .config import AppConfig, ColorConfig
from .models import Detection


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

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if frame is None or frame.size == 0:
            return []
        work, (ox, oy) = self._roi(frame)
        hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
        detections: list[Detection] = []
        dcfg = self.config.detection

        for color_cfg in self.config.colors:
            mask = self._mask_for_color(hsv, color_cfg)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = float(cv2.contourArea(contour))
                if not (dcfg.min_area <= area <= dcfg.max_area):
                    continue
                circularity, solidity = self._shape_metrics(contour)
                if circularity < dcfg.min_circularity or solidity < dcfg.min_solidity:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                component_mask = np.zeros(mask.shape, dtype=np.uint8)
                cv2.drawContours(component_mask, [contour], -1, 255, -1)
                color_pixels = cv2.countNonZero(cv2.bitwise_and(mask, mask, mask=component_mask))
                total_pixels = cv2.countNonZero(component_mask)
                ratio = color_pixels / total_pixels if total_pixels else 0.0
                if ratio < dcfg.min_color_ratio:
                    continue
                m = cv2.moments(contour)
                if abs(m["m00"]) < 1e-6:
                    cx, cy = x + w // 2, y + h // 2
                else:
                    cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
                detections.append(Detection(
                    color=color_cfg.name,
                    centroid=(cx + ox, cy + oy),
                    bbox=(x + ox, y + oy, w, h),
                    area=area,
                    circularity=circularity,
                    solidity=solidity,
                    color_ratio=ratio,
                ))
        return self._deduplicate(detections)

    @staticmethod
    def _deduplicate(detections: List[Detection]) -> List[Detection]:
        # Different hue masks can overlap slightly (e.g. purple/lilac). Keep the
        # strongest candidate by color ratio when centroids are very close.
        result: list[Detection] = []
        for det in sorted(detections, key=lambda d: d.color_ratio, reverse=True):
            if any(np.linalg.norm(np.array(det.centroid) - np.array(prev.centroid)) < 12 for prev in result):
                continue
            result.append(det)
        return result
