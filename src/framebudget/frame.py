"""The unit that comes out of a run."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Frame"]


@dataclass(frozen=True)
class Frame:
    """One selected frame, already encoded and sized for the target."""

    index: int
    timestamp: float
    width: int
    height: int
    jpeg: bytes

    def as_base64(self) -> str:
        return base64.b64encode(self.jpeg).decode("ascii")

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_bytes(self.jpeg)
        return path
