"""Cheap per-frame descriptors and the novelty signal built from them.

Two descriptors per frame instead of raw pixels: a normalised intensity
thumbnail for structure, and a coarse colour histogram for palette. Both are
free next to the cost of decoding, and together they tell "the shot changed"
apart from "the encoder noise changed".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from .errors import UnreadableVideoError
from .probe import VideoInfo

__all__ = ["Scan", "distances_to", "scan"]

_THUMB = 16
_HIST_BINS = 32

# Added to the standard deviation before normalising. On a flat wall or a blank
# slide the real deviation is almost zero, and dividing by it would amplify
# sensor noise into a completely different descriptor every frame. This is the
# failure mode that sinks difference hashes on smooth footage: comparing two
# nearly equal cells is a coin flip, so half the bits change for no reason.
_FLAT_GUARD = 1.0

# Mean absolute difference between two normalised thumbnails that counts as
# completely different. Past this the frames share nothing worth measuring.
_STRUCTURE_SCALE = 2.0


@dataclass(frozen=True)
class Scan:
    """Sampled descriptors for one video, in time order."""

    info: VideoInfo
    frame_indices: npt.NDArray[np.int64]
    timestamps: npt.NDArray[np.float64]
    structure: npt.NDArray[np.float32]
    """Shape (n, 256): brightness normalised 16x16 thumbnail per sample."""
    histograms: npt.NDArray[np.float32]
    """Shape (n, 96): normalised colour distribution."""
    novelty: npt.NDArray[np.float64]
    """Distance from the previous sample, in [0, 1]. First entry is 1.0."""

    def __len__(self) -> int:
        return int(self.frame_indices.size)


def scan(info: VideoInfo, analysis_fps: float = 2.0) -> Scan:
    """Walk the video once, describing it at ``analysis_fps`` samples per second."""
    step = max(1, round(info.fps / analysis_fps))

    capture = cv2.VideoCapture(str(info.path))
    if not capture.isOpened():
        raise UnreadableVideoError(f"could not reopen {info.path} for scanning")

    indices: list[int] = []
    structures: list[npt.NDArray[np.float32]] = []
    histograms: list[npt.NDArray[np.float32]] = []
    try:
        position = 0
        while True:
            # grab() parses the packet but skips colour conversion, so skipping is
            # much cheaper than decoding everything and throwing most of it away.
            if not capture.grab():
                break
            if position % step == 0:
                ok, frame = capture.retrieve()
                if not ok:
                    break
                indices.append(position)
                structures.append(_structure(frame))
                histograms.append(_colour_histogram(frame))
            position += 1
    finally:
        capture.release()

    if not indices:
        raise UnreadableVideoError(f"{info.path} yielded no decodable frames")

    frame_indices = np.asarray(indices, dtype=np.int64)
    structure_matrix = np.asarray(structures, dtype=np.float32)
    histogram_matrix = np.asarray(histograms, dtype=np.float32)

    return Scan(
        info=info,
        frame_indices=frame_indices,
        timestamps=frame_indices / info.fps,
        structure=structure_matrix,
        histograms=histogram_matrix,
        novelty=_novelty(structure_matrix, histogram_matrix),
    )


def _structure(frame: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
    """Brightness normalised thumbnail, flattened.

    Area interpolation averages each block, which is what kills the noise: a few
    thousand noisy pixels collapse into one stable number. Normalising by mean
    and spread then makes the descriptor ignore exposure changes and react to
    layout instead.
    """
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(grey, (_THUMB, _THUMB), interpolation=cv2.INTER_AREA)
    small = small.astype(np.float32)
    normalised = (small - small.mean()) / (small.std() + _FLAT_GUARD)
    return normalised.flatten().astype(np.float32)


def _colour_histogram(frame: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
    """32 bins per BGR channel, concatenated and normalised to sum to 1."""
    channels = [
        cv2.calcHist([frame], [channel], None, [_HIST_BINS], [0, 256]).flatten()
        for channel in range(3)
    ]
    histogram = np.concatenate(channels).astype(np.float32)
    total = float(histogram.sum())
    return histogram / total if total else histogram


def _combine(
    structural: npt.NDArray[np.float64], chromatic: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Weighted blend of the two distances, clipped to [0, 1]."""
    return np.clip(0.6 * structural + 0.4 * chromatic, 0.0, 1.0)


def _novelty(
    structure: npt.NDArray[np.float32], histograms: npt.NDArray[np.float32]
) -> npt.NDArray[np.float64]:
    """Distance between consecutive samples."""
    count = structure.shape[0]
    novelty = np.ones(count, dtype=np.float64)
    if count < 2:
        return novelty

    structural = np.abs(structure[1:] - structure[:-1]).mean(axis=1) / _STRUCTURE_SCALE
    # Half the L1 distance between two distributions is total variation distance,
    # already bounded to [0, 1], so nothing needs calibrating.
    chromatic = np.abs(histograms[1:] - histograms[:-1]).sum(axis=1) / 2.0

    novelty[1:] = _combine(structural, chromatic)
    return novelty


def distances_to(
    scan_result: Scan, index: int, candidates: npt.NDArray[np.int64]
) -> npt.NDArray[np.float64]:
    """Distance from one sample to many, same scale as ``Scan.novelty``."""
    structural = (
        np.abs(scan_result.structure[candidates] - scan_result.structure[index]).mean(
            axis=1
        )
        / _STRUCTURE_SCALE
    )
    chromatic = (
        np.abs(scan_result.histograms[candidates] - scan_result.histograms[index]).sum(
            axis=1
        )
        / 2.0
    )
    return _combine(structural, chromatic)
