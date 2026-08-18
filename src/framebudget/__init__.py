"""Fit a video into a token budget for multimodal LLMs.

from framebudget import extract

result = extract("talk.mp4", budget=50_000, target="claude")
print(result.report.summary())
blocks = result.to_messages()
"""

from __future__ import annotations

from .errors import BudgetTooSmallError, FrameBudgetError, UnreadableVideoError
from .extract import Result, extract
from .frame import Frame
from .probe import VideoInfo, probe
from .report import Report
from .targets import TARGETS, Target

__version__ = "0.1.0"

__all__ = [
    "TARGETS",
    "BudgetTooSmallError",
    "Frame",
    "FrameBudgetError",
    "Report",
    "Result",
    "Target",
    "UnreadableVideoError",
    "VideoInfo",
    "extract",
    "probe",
]
