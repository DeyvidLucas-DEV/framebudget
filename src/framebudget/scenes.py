"""Cutting the sample sequence into scenes."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import numpy.typing as npt

__all__ = ["Scene", "segment"]

# Below this two samples differ by compression noise and nothing else. Keeps a
# single continuous shot from being shredded into hundreds of scenes, which is
# what happens on locked off footage where the threshold has nothing to grip.
_NOISE_FLOOR = 0.02

# How many samples back to look when measuring what the current scene has been
# doing. Long enough to be stable, short enough that a scene which changes pace
# is judged on its recent behaviour rather than how it started.
_BASELINE_WINDOW = 21

# A scene needs at least this many samples before it is allowed to be cut again.
# Without it the first samples after a cut are compared against an empty history
# and every one of them looks like a jump. Two, not three: on quickly cut footage
# a whole scene can be two samples long, and three made those invisible.
_MIN_SCENE = 2

# The running level is a low percentile, not the median. On footage that cuts
# every second the history fills up with the cuts themselves, and a median sits
# high enough to hide the next one. A low percentile tracks how the scene behaves
# when it is not cutting, which is the thing a cut has to stand out against.
_BASELINE_PERCENTILE = 25


@dataclass(frozen=True)
class Scene:
    """Half open run of samples ``[start, end)`` between two cuts."""

    start: int
    end: int
    weight: float
    """How much budget this scene deserves, before rounding."""

    @property
    def length(self) -> int:
        return self.end - self.start


def segment(novelty: npt.NDArray[np.float64], sensitivity: float = 4.0) -> list[Scene]:
    """Find cuts and weight each scene.

    A cut is a sample that jumps far beyond what the current scene has been
    doing. Not beyond a fixed number, and not beyond the video average: beyond
    the scene it is sitting in, measured from the samples since the last cut.

    That framing is what makes this survive both a locked off interview and a
    handheld chase. In the interview the running level is near zero, so a cut
    towers over it. In the chase the running level is already high, so the same
    absolute jump is correctly read as more of the same motion. Any single
    threshold gets one of those two badly wrong.

    ``sensitivity`` is how many times above the running level a sample has to sit
    to count as a cut. Lower finds more scenes.
    """
    count = int(novelty.size)
    if count == 0:
        return []
    if count < 3:
        return [Scene(start=0, end=count, weight=1.0)]

    boundaries = _find_cuts(novelty, sensitivity)

    scenes: list[Scene] = []
    for start, end in pairwise(boundaries):
        span = novelty[start:end]
        # Length says how much time the scene takes up, mean novelty says how much
        # actually happens in it. A long static shot should not get the same
        # number of frames as a short busy one.
        scenes.append(
            Scene(
                start=start,
                end=end,
                weight=float(span.size) * (1.0 + float(span.mean())),
            )
        )
    return scenes


def _find_cuts(novelty: npt.NDArray[np.float64], sensitivity: float) -> list[int]:
    """Walk the signal forward, cutting whenever it leaves the current scene."""
    count = int(novelty.size)
    boundaries = [0]
    # Skip index 0. Its novelty is 1.0 by convention, not measurement.
    history: list[float] = []

    for index in range(1, count):
        value = float(novelty[index])
        if len(history) >= _MIN_SCENE:
            baseline = float(
                np.percentile(history[-_BASELINE_WINDOW:], _BASELINE_PERCENTILE)
            )
            if value >= max(baseline * sensitivity, _NOISE_FLOOR):
                boundaries.append(index)
                history = []
                continue
        history.append(value)

    boundaries.append(count)
    return boundaries
