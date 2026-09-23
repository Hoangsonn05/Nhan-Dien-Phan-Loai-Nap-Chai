from __future__ import annotations

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
        self.metadata = SourceMetadata()

    def open(self) -> None:
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Cannot open video: {self.path}")
        self._capture = capture
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.metadata = SourceMetadata(fps=float(fps), width=width, height=height)

    def __iter__(self) -> Iterator[FramePacket]:
        if self._capture is None:
            raise RuntimeError("VideoSource must be opened before iteration")
        frame_index = 0
        while True:
            ok, frame = self._capture.read()
            if not ok:
                break
            yield FramePacket(frame=frame, frame_index=frame_index)
            frame_index += 1

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
