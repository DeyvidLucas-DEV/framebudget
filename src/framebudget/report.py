"""What the selection cost, and what it would have cost the naive way."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Report"]


@dataclass(frozen=True)
class Report:
    """Numbers worth showing after a run.

    The baseline is uniform sampling at 1 fps, because that is what almost
    everyone does the first time they wire a video into a model.
    """

    duration: float
    analysis_fps: float
    scanned: int
    unique: int
    """Samples left after dropping the ones that repeat what came before."""
    scenes: int
    selected: int
    tokens: int
    baseline_frames: int
    baseline_tokens: int

    @property
    def redundant(self) -> int:
        return self.scanned - self.unique

    @property
    def coverage(self) -> float:
        """Share of the distinct content that made it into the budget.

        This is the number to watch, not the time between frames. A long gap in a
        static shot means nothing happened, which is the whole point. A low
        coverage means real content was dropped because the budget ran out.
        """
        if self.unique <= 0:
            return 0.0
        return self.selected / self.unique

    @property
    def saved(self) -> float:
        """Fraction of the baseline token spend that was avoided.

        Goes negative when the video holds more distinct content than 1 fps would
        capture and the budget was large enough to pay for it. That is a real
        result, not an error: more tokens, and nothing missed.
        """
        if self.baseline_tokens <= 0:
            return 0.0
        return 1.0 - self.tokens / self.baseline_tokens

    def summary(self) -> str:
        """One block of plain text for a terminal or a log line."""
        lines = [
            f"duration     {self.duration:.1f}s",
            f"scanned      {self.scanned} samples at {self.analysis_fps:g} fps",
            f"unique       {self.unique} ({self.redundant} redundant dropped)",
            f"scenes       {self.scenes}",
            f"selected     {self.selected} frames, "
            f"{self.coverage:.0%} of distinct content",
            f"tokens       {self.tokens:,}",
            f"baseline     {self.baseline_tokens:,} "
            f"({self.baseline_frames} frames at 1 fps)",
            f"saved        {self.saved:+.1%}",
        ]
        return "\n".join(lines)
