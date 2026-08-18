"""Synthetic videos with known structure, so tests can assert on real numbers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

FPS = 30
WIDTH = 320
HEIGHT = 240


def _writer(path: Path) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT)
    )
    if not writer.isOpened():
        pytest.skip("no mp4v encoder available in this OpenCV build")
    return writer


def _panel(colour: tuple[int, int, int], marker: int) -> np.ndarray:
    frame = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    frame[:] = colour
    cv2.rectangle(frame, (20, 20), (20 + 40 * marker, 90), (255, 255, 255), -1)
    return frame


@pytest.fixture(scope="session")
def cut_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Four static shots of five seconds each, cutting at 5, 10 and 15 seconds."""
    path = tmp_path_factory.mktemp("videos") / "cuts.mp4"
    writer = _writer(path)
    rng = np.random.default_rng(3)
    palette = [(200, 60, 40), (40, 180, 70), (40, 40, 210), (180, 180, 40)]
    for marker, colour in enumerate(palette, start=1):
        base = _panel(colour, marker)
        for _ in range(5 * FPS):
            noise = rng.integers(0, 3, (HEIGHT, WIDTH, 3), dtype=np.uint8)
            writer.write(cv2.add(base, noise))
    writer.release()
    return path


@pytest.fixture(scope="session")
def static_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Ten seconds of one unchanging shot. Nothing here deserves a second frame."""
    path = tmp_path_factory.mktemp("videos") / "static.mp4"
    writer = _writer(path)
    rng = np.random.default_rng(5)
    base = _panel((120, 120, 120), 2)
    for _ in range(10 * FPS):
        writer.write(
            cv2.add(base, rng.integers(0, 3, (HEIGHT, WIDTH, 3), dtype=np.uint8))
        )
    writer.release()
    return path


@pytest.fixture(scope="session")
def motion_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Ten seconds of continuous motion with no cut anywhere in it."""
    path = tmp_path_factory.mktemp("videos") / "motion.mp4"
    writer = _writer(path)
    total = 10 * FPS
    for index in range(total):
        frame = np.full((HEIGHT, WIDTH, 3), 50, np.uint8)
        x = int(20 + (WIDTH - 100) * index / total)
        cv2.circle(frame, (x, HEIGHT // 2), 35, (40, 200, 240), -1)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture(scope="session")
def fast_cut_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Eight shots in four seconds, cutting twice a second.

    Modelled on real edited footage. At 2 samples per second every sample lands
    in a different shot, so nothing can be told apart from anything else and the
    scene detector needs a higher rate to see the cuts at all.
    """
    path = tmp_path_factory.mktemp("videos") / "fastcuts.mp4"
    writer = _writer(path)
    palette = [
        (200, 60, 40),
        (40, 180, 70),
        (40, 40, 210),
        (180, 180, 40),
        (200, 40, 190),
        (60, 200, 200),
        (120, 40, 90),
        (30, 120, 200),
    ]
    for marker, colour in enumerate(palette, start=1):
        base = _panel(colour, (marker % 4) + 1)
        for _ in range(FPS // 2):
            writer.write(base)
    writer.release()
    return path
