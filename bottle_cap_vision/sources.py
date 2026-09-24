from __future__ import annotations

import math
from pathlib import Path
from typing import Iterator

import cv2

from .models import FramePacket, SourceMetadata


class ImageSource:
    """Read exactly one image from a filesystem path."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._frame = None
        self.metadata = SourceMetadata()

    def open(self) -> None:
        frame = cv2.imread(str(self.path))
        if frame is None:
            raise RuntimeError(f"Cannot read image: {self.path}")
        self._frame = frame
        height, width = frame.shape[:2]
        self.metadata = SourceMetadata(width=width, height=height)

    def __iter__(self) -> Iterator[FramePacket]:
        if self._frame is None:
            raise RuntimeError("ImageSource must be opened before iteration")
        yield FramePacket(frame=self._frame, frame_index=0)

    def close(self) -> None:
        self._frame = None


class VideoSource:
    """Sequential OpenCV video-file source with explicit ownership."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._capture = None
        self._expected_frames: int | None = None
        self.metadata = SourceMetadata()

    def open(self) -> None:
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Cannot open video: {self.path}")
        self._capture = capture
        fps_value = float(capture.get(cv2.CAP_PROP_FPS))
        fps = fps_value if math.isfinite(fps_value) and fps_value > 0 else None
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            capture.release()
            self._capture = None
            raise RuntimeError(
                f"Video has invalid dimensions ({width}x{height}): {self.path}"
            )
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self._expected_frames = (
            int(round(frame_count))
            if math.isfinite(frame_count) and frame_count > 0
            else None
        )
        self.metadata = SourceMetadata(fps=fps, width=width, height=height)

    def __iter__(self) -> Iterator[FramePacket]:
        if self._capture is None:
            raise RuntimeError("VideoSource must be opened before iteration")
        frame_index = 0
        while True:
            ok, frame = self._capture.read()
            if not ok:
                if self._expected_frames is not None and frame_index < self._expected_frames:
                    raise RuntimeError(
                        "Video decoding stopped before the container-reported frame count: "
                        f"decoded={frame_index}, expected={self._expected_frames}, path={self.path}"
                    )
                break
            if frame is None or frame.size == 0:
                raise RuntimeError(f"Video decoded an empty frame at index {frame_index}: {self.path}")
            height, width = frame.shape[:2]
            if (width, height) != (self.metadata.width, self.metadata.height):
                raise RuntimeError(
                    "Decoded frame dimensions differ from video metadata at "
                    f"index {frame_index}: frame={width}x{height}, "
                    f"metadata={self.metadata.width}x{self.metadata.height}"
                )
            yield FramePacket(frame=frame, frame_index=frame_index)
            frame_index += 1

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._expected_frames = None


class CameraSource:
    """Synchronous OpenCV camera source with explicit ownership."""

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self._capture = None
        self.metadata = SourceMetadata()

    def open(self) -> None:
        capture = cv2.VideoCapture(self.camera_index)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Cannot open camera index {self.camera_index}")

        try:
            width_value = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height_value = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_value = float(capture.get(cv2.CAP_PROP_FPS))
        except BaseException:
            capture.release()
            raise

        width = int(width_value) if math.isfinite(width_value) else 0
        height = int(height_value) if math.isfinite(height_value) else 0
        if width <= 0 or height <= 0:
            capture.release()
            raise RuntimeError(
                f"Camera index {self.camera_index} has invalid dimensions "
                f"({width}x{height})"
            )

        fps = fps_value if math.isfinite(fps_value) and fps_value > 0 else None
        self._capture = capture
        self.metadata = SourceMetadata(fps=fps, width=width, height=height)

    def __iter__(self) -> Iterator[FramePacket]:
        if self._capture is None:
            raise RuntimeError("CameraSource must be opened before iteration")
        frame_index = 0
        while True:
            ok, frame = self._capture.read()
            if not ok:
                raise RuntimeError(
                    f"Camera read failed at frame index {frame_index} "
                    f"(camera index {self.camera_index})"
                )
            if frame is None or frame.size == 0:
                raise RuntimeError(
                    f"Camera returned an empty frame at index {frame_index} "
                    f"(camera index {self.camera_index})"
                )
            yield FramePacket(frame=frame, frame_index=frame_index)
            frame_index += 1

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
