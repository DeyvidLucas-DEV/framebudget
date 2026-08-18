from __future__ import annotations

import pytest

from framebudget.targets import TARGETS, fit_within, resolve_target


def test_resolve_is_case_insensitive() -> None:
    assert resolve_target("CLAUDE") is TARGETS["claude"]


def test_resolve_passes_through_a_target() -> None:
    target = TARGETS["openai"]
    assert resolve_target(target) is target


def test_unknown_target_lists_the_valid_ones() -> None:
    with pytest.raises(ValueError, match="claude"):
        resolve_target("gpt5")


def test_gemini_cost_ignores_resolution() -> None:
    gemini = TARGETS["gemini"]
    assert gemini.tokens_for(320, 240) == gemini.tokens_for(1920, 1080)


def test_anthropic_cost_grows_with_area() -> None:
    claude = TARGETS["claude"]
    assert claude.tokens_for(1000, 1000) > claude.tokens_for(500, 500)


def test_oversized_frames_are_capped_at_the_provider_limit() -> None:
    claude = TARGETS["claude"]
    assert claude.tokens_for(8000, 8000) == claude.tokens_for(1568, 1568)


def test_fit_within_never_scales_up() -> None:
    assert fit_within(100, 50, 1568) == (100, 50)


def test_fit_within_keeps_aspect_ratio() -> None:
    width, height = fit_within(4000, 2000, 1000)
    assert (width, height) == (1000, 500)
