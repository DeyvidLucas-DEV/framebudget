"""Token cost per frame for each provider."""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["TARGETS", "Target", "fit_within", "resolve_target"]

# Numbers below follow each provider's published behaviour as of 2026. They are
# estimates, not billing. Providers round internally and change encodings without
# telling anyone, so use these to plan, then check a real invoice.


@dataclass(frozen=True)
class Target:
    """Cost model and size limits for one provider."""

    name: str
    max_dimension: int

    def tokens_for(self, width: int, height: int) -> int:
        """Token cost of a single frame at these pixel dimensions."""
        raise NotImplementedError


@dataclass(frozen=True)
class _AnthropicTarget(Target):
    def tokens_for(self, width: int, height: int) -> int:
        w, h = fit_within(width, height, self.max_dimension)
        return math.ceil(w * h / 750)


@dataclass(frozen=True)
class _OpenAITarget(Target):
    short_edge: int = 768
    base_tokens: int = 85
    tile_tokens: int = 170

    def tokens_for(self, width: int, height: int) -> int:
        w, h = fit_within(width, height, self.max_dimension)
        # The short edge gets rescaled to a fixed size before tiling. Side effect:
        # a wide frame and a tall frame with the same area can cost different
        # amounts.
        scale = self.short_edge / min(w, h)
        w, h = round(w * scale), round(h * scale)
        tiles = math.ceil(w / 512) * math.ceil(h / 512)
        return self.base_tokens + self.tile_tokens * tiles


@dataclass(frozen=True)
class _FixedCostTarget(Target):
    cost: int = 258

    def tokens_for(self, width: int, height: int) -> int:
        # Flat rate per frame no matter the resolution, so frame count is the only
        # lever worth pulling here.
        return self.cost


def fit_within(width: int, height: int, limit: int) -> tuple[int, int]:
    """Scale down so the longest edge fits in ``limit``. Never scales up."""
    longest = max(width, height)
    if longest <= limit:
        return width, height
    scale = limit / longest
    return max(1, round(width * scale)), max(1, round(height * scale))


TARGETS: dict[str, Target] = {
    "claude": _AnthropicTarget(name="claude", max_dimension=1568),
    "openai": _OpenAITarget(name="openai", max_dimension=2048),
    "gemini": _FixedCostTarget(name="gemini", max_dimension=768),
}


def resolve_target(target: str | Target) -> Target:
    """Look up a target by name, or pass an already built one straight through."""
    if isinstance(target, Target):
        return target
    try:
        return TARGETS[target.lower()]
    except KeyError:
        known = ", ".join(sorted(TARGETS))
        raise ValueError(
            f"unknown target {target!r}; expected one of: {known}"
        ) from None
