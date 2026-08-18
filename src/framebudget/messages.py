"""Formatting selected frames into the shape each provider's API expects."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .frame import Frame

__all__ = ["build_messages"]


def build_messages(
    frames: list[Frame], target_name: str, with_timestamps: bool = True
) -> list[dict[str, Any]]:
    """Build the content blocks for one user message.

    Timestamps go in as text right before each image. Without them the model sees
    a pile of stills with no idea what order or how far apart they are, which
    kills any question about sequence or duration.
    """
    builder = _BUILDERS.get(target_name, _claude_blocks)
    blocks: list[dict[str, Any]] = []
    for frame in frames:
        if with_timestamps:
            blocks.append(_text_block(target_name, f"[{_clock(frame.timestamp)}]"))
        blocks.append(builder(frame))
    return blocks


def _clock(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _text_block(target_name: str, text: str) -> dict[str, Any]:
    if target_name == "gemini":
        return {"text": text}
    return {"type": "text", "text": text}


def _claude_blocks(frame: Frame) -> dict[str, Any]:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": frame.as_base64(),
        },
    }


def _openai_blocks(frame: Frame) -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{frame.as_base64()}"},
    }


def _gemini_blocks(frame: Frame) -> dict[str, Any]:
    return {"inline_data": {"mime_type": "image/jpeg", "data": frame.as_base64()}}


_BUILDERS: dict[str, Callable[[Frame], dict[str, Any]]] = {
    "claude": _claude_blocks,
    "openai": _openai_blocks,
    "gemini": _gemini_blocks,
}
