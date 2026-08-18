"""The entry point most people will use."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .errors import BudgetTooSmallError, UnreadableVideoError
from .frame import Frame
from .messages import build_messages
from .probe import probe
from .report import Report
from .selection import select
from .signals import scan
from .targets import Target, fit_within, resolve_target

__all__ = ["Frame", "Result", "extract"]

# 85 is the usual knee in the JPEG curve. Above it the file grows fast and the
# model sees nothing new, below it compression artefacts start showing up in OCR
# style tasks.
_JPEG_QUALITY = 85


@dataclass(frozen=True)
class Result:
    """Selected frames plus the numbers behind the selection."""

    frames: list[Frame]
    report: Report
    target: Target

    def to_messages(self, with_timestamps: bool = True) -> list[dict[str, Any]]:
        """Content blocks ready to drop into the target provider's API call."""
        return build_messages(self.frames, self.target.name, with_timestamps)

    def __len__(self) -> int:
        return len(self.frames)


def extract(
    path: str | Path,
    budget: int = 50_000,
    target: str | Target = "claude",
    analysis_fps: float = 2.0,
    min_distance: float = 0.02,
    sensitivity: float = 4.0,
    max_dimension: int | None = None,
) -> Result:
    """Pick the frames that fit ``budget`` tokens and say the most about the video.

    Args:
        path: video file to read.
        budget: token ceiling for the images. Text and prompt are not counted.
        target: ``"claude"``, ``"openai"``, ``"gemini"`` or a custom ``Target``.
        analysis_fps: how densely to look at the video while deciding. Higher
            catches shorter events and costs more wall clock, nothing else.
        min_distance: redundancy floor in [0, 1]. Below this two frames are the
            same picture and the second is never worth paying for. Fitting the
            budget is the allocator's job, not this one.
        sensitivity: how many times above the running level of the current scene
            a sample must jump to count as a cut. Lower finds more scenes.
        max_dimension: longest output edge. Defaults to whatever the target
            downsamples to anyway, since going above that only costs tokens.

    Returns:
        A ``Result`` holding the frames and a ``Report``.

    Raises:
        UnreadableVideoError: the file could not be opened or decoded.
        BudgetTooSmallError: the budget does not cover a single frame.
    """
    resolved = resolve_target(target)
    limit = max_dimension or resolved.max_dimension

    info = probe(path)
    out_width, out_height = fit_within(info.width, info.height, limit)
    per_frame = resolved.tokens_for(out_width, out_height)

    if per_frame > budget:
        raise BudgetTooSmallError(
            f"one frame at {out_width}x{out_height} costs {per_frame} tokens on "
            f"{resolved.name}, budget is {budget}. Lower max_dimension or raise "
            f"the budget."
        )

    budget_frames = budget // per_frame
    scan_result = scan(info, analysis_fps=analysis_fps)
    selection = select(
        scan_result,
        budget_frames=budget_frames,
        min_distance=min_distance,
        sensitivity=sensitivity,
    )

    wanted = scan_result.frame_indices[selection.chosen]
    timestamps = scan_result.timestamps[selection.chosen]
    images = _decode(info.path, wanted, (out_width, out_height))

    # Keyed by frame index rather than zipped positionally. A frame that fails to
    # decode would otherwise shift every timestamp after it onto the wrong image,
    # and nothing downstream would notice.
    frames = [
        Frame(
            index=int(frame_index),
            timestamp=float(timestamp),
            width=out_width,
            height=out_height,
            jpeg=images[int(frame_index)],
        )
        for frame_index, timestamp in zip(wanted, timestamps, strict=True)
        if int(frame_index) in images
    ]

    baseline_frames = max(1, int(info.duration)) if info.duration else len(frames)
    report = Report(
        duration=info.duration,
        analysis_fps=analysis_fps,
        scanned=selection.scanned,
        unique=int(selection.kept_after_dedup.size),
        scenes=len(selection.scenes),
        selected=len(frames),
        tokens=len(frames) * per_frame,
        baseline_frames=baseline_frames,
        baseline_tokens=baseline_frames * per_frame,
    )
    return Result(frames=frames, report=report, target=resolved)


def _decode(path: Path, wanted: np.ndarray, size: tuple[int, int]) -> dict[int, bytes]:
    """Second pass over the file, decoding only the frames that were chosen.

    Sequential grab is faster and far more reliable than seeking. Seeking on a
    long GOP codec lands on the nearest keyframe, not the frame that was asked
    for, so the timestamps would quietly stop matching the images.
    """
    targets = {int(index) for index in wanted}
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise UnreadableVideoError(f"could not reopen {path} for extraction")

    encoded: dict[int, bytes] = {}
    try:
        position = 0
        last = max(targets) if targets else -1
        while position <= last:
            if not capture.grab():
                break
            if position in targets:
                ok, frame = capture.retrieve()
                if ok:
                    encoded[position] = _encode(frame, size)
            position += 1
    finally:
        capture.release()

    return encoded


def _encode(frame: np.ndarray, size: tuple[int, int]) -> bytes:
    width, height = size
    if (frame.shape[1], frame.shape[0]) != size:
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY]
    )
    if not ok:
        raise UnreadableVideoError("frame could not be encoded to JPEG")
    return bytes(buffer)
