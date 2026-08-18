from __future__ import annotations

import numpy as np

from framebudget.scenes import segment


def test_empty_signal_has_no_scenes() -> None:
    assert segment(np.zeros(0)) == []


def test_flat_signal_stays_one_scene() -> None:
    novelty = np.full(200, 0.001)
    novelty[0] = 1.0
    assert len(segment(novelty)) == 1


def test_isolated_spike_becomes_a_cut() -> None:
    novelty = np.full(200, 0.001)
    novelty[0] = 1.0
    novelty[100] = 0.8
    scenes = segment(novelty)
    assert len(scenes) == 2
    assert scenes[1].start == 100


def test_sustained_motion_is_not_a_cut() -> None:
    # Every sample is far from the last one, but nothing stands out against the
    # rest, so this is one continuous shot.
    novelty = np.full(200, 0.3)
    novelty[0] = 1.0
    assert len(segment(novelty)) == 1


def test_a_cut_is_found_even_inside_busy_footage() -> None:
    novelty = np.full(200, 0.1)
    novelty[0] = 1.0
    novelty[120] = 0.95
    scenes = segment(novelty)
    assert [scene.start for scene in scenes] == [0, 120]


def test_lower_sensitivity_finds_more_scenes() -> None:
    rng = np.random.default_rng(0)
    novelty = rng.uniform(0.02, 0.05, 300)
    novelty[0] = 1.0
    novelty[[60, 150, 240]] = 0.5
    assert len(segment(novelty, sensitivity=1.5)) >= len(
        segment(novelty, sensitivity=8.0)
    )


def test_busy_scenes_weigh_more_than_quiet_ones_of_equal_length() -> None:
    novelty = np.concatenate(([1.0], np.full(99, 0.001), [0.9], np.full(99, 0.05)))
    quiet, busy = segment(novelty)
    assert busy.length == quiet.length
    assert busy.weight > quiet.weight
