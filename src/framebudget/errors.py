"""Exceptions raised by framebudget."""

from __future__ import annotations

__all__ = ["BudgetTooSmallError", "FrameBudgetError", "UnreadableVideoError"]


class FrameBudgetError(Exception):
    """Base class for every error raised by this package."""


class UnreadableVideoError(FrameBudgetError):
    """The video could not be opened or contains no decodable frames."""


class BudgetTooSmallError(FrameBudgetError):
    """The token budget cannot fit a single frame at the requested size."""
