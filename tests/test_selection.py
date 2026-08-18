from __future__ import annotations

from pathlib import Path

from framebudget.probe import probe
from framebudget.selection import select
from framebudget.signals import scan


def test_selection_respects_the_frame_budget(cut_video: Path) -> None:
    result = scan(probe(cut_video), analysis_fps=2.0)
    selection = select(result, budget_frames=5)
    assert len(selection.chosen) <= 5


def test_static_footage_collapses_to_almost_nothing(static_video: Path) -> None:
    result = scan(probe(static_video), analysis_fps=2.0)
    selection = select(result, budget_frames=100)
    # Ten seconds of one unchanging shot. Paying for twenty copies of it is the
    # exact waste this library exists to stop.
    assert selection.redundant > len(selection.chosen)


def test_every_scene_gets_a_frame_when_the_budget_allows(cut_video: Path) -> None:
    result = scan(probe(cut_video), analysis_fps=2.0)
    selection = select(result, budget_frames=20)
    covered = {
        scene.start
        for scene in selection.scenes
        for index in selection.chosen
        if scene.start <= index < scene.end
    }
    assert len(covered) == len(selection.scenes)


def test_chosen_frames_stay_in_time_order(cut_video: Path) -> None:
    result = scan(probe(cut_video), analysis_fps=2.0)
    chosen = select(result, budget_frames=8).chosen
    assert list(chosen) == sorted(chosen)


def test_a_tiny_budget_still_returns_something(cut_video: Path) -> None:
    result = scan(probe(cut_video), analysis_fps=2.0)
    assert len(select(result, budget_frames=1).chosen) == 1


def test_continuous_motion_is_sampled_across_the_whole_shot(
    motion_video: Path,
) -> None:
    result = scan(probe(motion_video), analysis_fps=2.0)
    selection = select(result, budget_frames=6)
    chosen = selection.chosen
    # Farthest point sampling should reach the end of the pan, not cluster at the
    # start where the seed frame sits.
    assert int(chosen.max()) > len(result) * 0.6


def test_budget_is_a_ceiling_not_a_target(cut_video: Path) -> None:
    result = scan(probe(cut_video), analysis_fps=2.0)
    # Budget far beyond anything the video justifies. Spending all of it on four
    # static shots would be the waste this library exists to remove.
    selection = select(result, budget_frames=10_000)
    assert len(selection.chosen) <= len(selection.scenes) * 3


def test_fast_cuts_are_all_covered(fast_cut_video: Path) -> None:
    from framebudget.signals import auto_scan

    result = auto_scan(probe(fast_cut_video))
    selection = select(result, budget_frames=100)
    # Eight distinct shots in four seconds. Finding only one or two means the
    # scene detector collapsed, which is what happened before auto rate.
    assert len(selection.scenes) >= 6
