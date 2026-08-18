from __future__ import annotations

from pathlib import Path

import numpy as np

from framebudget.probe import probe
from framebudget.signals import distances_to, scan


def test_scan_samples_at_the_requested_rate(cut_video: Path) -> None:
    info = probe(cut_video)
    result = scan(info, analysis_fps=2.0)
    # 20 seconds at 2 samples per second, give or take one at the tail.
    assert 39 <= len(result) <= 41


def test_novelty_is_bounded(cut_video: Path) -> None:
    result = scan(probe(cut_video), analysis_fps=2.0)
    assert result.novelty.min() >= 0.0
    assert result.novelty.max() <= 1.0


def test_static_footage_produces_near_zero_novelty(
    static_video: Path, cut_video: Path
) -> None:
    # This is the regression that matters. A difference hash compares nearly
    # equal cells on flat footage and flips half its bits on noise alone, which
    # reads as constant change where there is none. It scored 0.17 here.
    #
    # Asserted against real change rather than a fixed number on purpose. How
    # much noise survives encoding depends on the OpenCV build, so an absolute
    # threshold passes on one machine and fails on CI while the code is fine.
    still = float(np.median(scan(probe(static_video), 2.0).novelty[1:]))
    changing = float(scan(probe(cut_video), 2.0).novelty[1:].max())

    assert still < 0.05
    assert still * 10 < changing


def test_cuts_stand_out_against_their_neighbours(cut_video: Path) -> None:
    result = scan(probe(cut_video), analysis_fps=2.0)
    quiet = float(np.median(result.novelty[1:]))
    assert result.novelty[1:].max() > quiet * 10


def test_distance_to_itself_is_zero(cut_video: Path) -> None:
    result = scan(probe(cut_video), analysis_fps=2.0)
    assert distances_to(result, 0, np.asarray([0]))[0] == 0.0


def test_distances_match_the_novelty_signal(cut_video: Path) -> None:
    result = scan(probe(cut_video), analysis_fps=2.0)
    direct = distances_to(result, 0, np.asarray([1]))[0]
    assert direct == float(result.novelty[1])
