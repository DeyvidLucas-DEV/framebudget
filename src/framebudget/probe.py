"""Video metadata, read without decoding the whole file."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from .errors import UnreadableVideoError

__all__ = ["VideoInfo", "probe"]


@dataclass(frozen=True)
class VideoInfo:
    """Container-level facts about a video file."""

    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"


def probe(path: str | Path) -> VideoInfo:
    """Read dimensions, frame rate and duration from a video file."""
    path = Path(path)
    if not path.exists():
        raise UnreadableVideoError(f"no such file: {path}")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise UnreadableVideoError(
            f"could not open {path}; the container or codec may be unsupported"
        )
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()

    if width <= 0 or height <= 0:
        raise UnreadableVideoError(f"{path} reports no usable video stream")

    # VFR files and some streamed containers report 0 or NaN. Falling back to 30
    # is fine, timestamps are only used for spacing samples, never for seeking.
    if not fps or fps != fps or fps <= 0:
        fps = 30.0

    duration = frame_count / fps if frame_count > 0 else 0.0
    return VideoInfo(
        path=path,
        width=width,
        height=height,
        fps=fps,
        frame_count=max(frame_count, 0),
        duration=duration,
    )
