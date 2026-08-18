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


def test_openai_does_not_upscale_small_frames() -> None:
    openai = TARGETS["openai"]
    # Measured against the API: a 854x480 frame bills around 425 tokens. Scaling
    # its short edge up to 768 before tiling predicted 1105, which is what this
    # library reported for months of nothing but synthetic tests.
    assert openai.tokens_for(854, 480) == 425


def test_openai_downscales_large_frames() -> None:
    openai = TARGETS["openai"]
    # Same frame count either way once the short edge is above the limit.
    assert openai.tokens_for(1280, 682) == openai.tokens_for(2560, 1364)
