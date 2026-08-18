"""Turning a token budget into an actual list of frames.

Three stages: cut the video into scenes, throw away samples that repeat what
came before, then split the frame budget across scenes and pick frames inside
each one.

The last stage is where uniform sampling loses. Spacing frames evenly in time
spends the budget on whatever happened to be on screen at fixed intervals.
Spacing them by appearance spends it on distinct content instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .scenes import Scene, segment
from .signals import Scan, distances_to

__all__ = ["Selection", "select"]


@dataclass(frozen=True)
class Selection:
    """What survived each stage, as indices into the original scan."""

    chosen: npt.NDArray[np.int64]
    kept_after_dedup: npt.NDArray[np.int64]
    scenes: list[Scene]
    scanned: int

    @property
    def redundant(self) -> int:
        return self.scanned - int(self.kept_after_dedup.size)


def select(
    scan_result: Scan,
    budget_frames: int,
    min_distance: float = 0.02,
    sensitivity: float = 4.0,
) -> Selection:
    """Choose at most ``budget_frames`` samples that best represent the video."""
    scanned = len(scan_result)

    # Scenes come from the full novelty signal, before dedup. A cut only stands
    # out because everything around it is flat, so cutting the flat parts first
    # would hide the very thing we are looking for.
    scene_list = segment(scan_result.novelty, sensitivity)
    kept = _deduplicate(scan_result, min_distance)

    if kept.size <= budget_frames:
        return Selection(
            chosen=kept,
            kept_after_dedup=kept,
            scenes=scene_list,
            scanned=scanned,
        )

    populated = [(scene, _within(kept, scene)) for scene in scene_list]
    populated = [(scene, members) for scene, members in populated if members.size]
    quotas = _allocate([scene for scene, _ in populated], budget_frames)

    chosen: list[int] = []
    for (_, members), quota in zip(populated, quotas, strict=True):
        if quota > 0:
            chosen.extend(_spread(scan_result, members, quota))

    return Selection(
        chosen=np.asarray(sorted(chosen), dtype=np.int64),
        kept_after_dedup=kept,
        scenes=scene_list,
        scanned=scanned,
    )


def _within(kept: npt.NDArray[np.int64], scene: Scene) -> npt.NDArray[np.int64]:
    """The kept samples that fall inside this scene."""
    return kept[(kept >= scene.start) & (kept < scene.end)]


def _deduplicate(scan_result: Scan, min_distance: float) -> npt.NDArray[np.int64]:
    """Drop samples that repeat the last one kept.

    This is a redundancy floor, not the budget. Anything below the threshold is
    genuinely the same picture and never worth paying for. Fitting the budget is
    the allocator's job, and it does a better one, since it can see the whole
    video at once instead of deciding frame by frame.

    Compares against the last kept frame rather than the previous sample. A slow
    pan moves very little between two samples but ends up somewhere completely
    different, and comparing neighbours would discard all of it.
    """
    total = len(scan_result)
    if total == 0:
        return np.zeros(0, dtype=np.int64)

    kept = [0]
    anchor = 0
    for index in range(1, total):
        distance = float(
            distances_to(scan_result, anchor, np.asarray([index], dtype=np.int64))[0]
        )
        if distance >= min_distance:
            kept.append(index)
            anchor = index
    return np.asarray(kept, dtype=np.int64)


def _allocate(scenes: list[Scene], budget_frames: int) -> list[int]:
    """Split the budget across scenes, covering every one of them if possible."""
    if not scenes:
        return []

    # More scenes than frames. Covering all of them is impossible, so put the
    # budget on the heaviest ones and let the report show the gap, instead of
    # spreading it so thin that nothing is usable.
    if len(scenes) >= budget_frames:
        order = sorted(range(len(scenes)), key=lambda i: scenes[i].weight, reverse=True)
        quotas = [0] * len(scenes)
        for position in order[:budget_frames]:
            quotas[position] = 1
        return quotas

    quotas = [1] * len(scenes)
    remaining = budget_frames - len(scenes)
    if remaining <= 0:
        return quotas

    total_weight = sum(scene.weight for scene in scenes) or 1.0
    exact = [scene.weight / total_weight * remaining for scene in scenes]
    floors = [int(value) for value in exact]

    # Largest remainder, so the total lands exactly on the budget. Leftovers go to
    # whichever scenes lost the most when their float got truncated.
    leftover = remaining - sum(floors)
    order = sorted(range(len(scenes)), key=lambda i: exact[i] - floors[i], reverse=True)
    for position in order[:leftover]:
        floors[position] += 1

    for position, extra in enumerate(floors):
        quotas[position] = quotas[position] + extra
    return quotas


def _spread(
    scan_result: Scan, candidates: npt.NDArray[np.int64], quota: int
) -> list[int]:
    """Pick ``quota`` candidates that look as little like each other as possible.

    Farthest point sampling, seeded at the first frame of the scene. That frame
    is the cut itself, which is the moment the content changed and the whole
    reason this scene exists.
    """
    if quota >= candidates.size:
        return [int(index) for index in candidates]

    chosen = [int(candidates[0])]
    closest = distances_to(scan_result, chosen[0], candidates)
    for _ in range(quota - 1):
        position = int(np.argmax(closest))
        if closest[position] <= 0:
            break
        pick = int(candidates[position])
        chosen.append(pick)
        closest = np.minimum(closest, distances_to(scan_result, pick, candidates))
    return chosen
