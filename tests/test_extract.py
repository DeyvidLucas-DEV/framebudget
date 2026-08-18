from __future__ import annotations

import base64
from pathlib import Path

import pytest

from framebudget import BudgetTooSmallError, UnreadableVideoError, extract
from framebudget.targets import TARGETS


def test_extract_stays_inside_the_token_budget(cut_video: Path) -> None:
    result = extract(cut_video, budget=6_000, target="claude")
    assert result.report.tokens <= 6_000


def test_frames_carry_usable_jpeg_data(cut_video: Path) -> None:
    result = extract(cut_video, budget=6_000)
    assert result.frames
    for frame in result.frames:
        assert frame.jpeg.startswith(b"\xff\xd8")
        assert base64.b64decode(frame.as_base64()) == frame.jpeg


def test_timestamps_increase(cut_video: Path) -> None:
    result = extract(cut_video, budget=20_000)
    stamps = [frame.timestamp for frame in result.frames]
    assert stamps == sorted(stamps)


def test_frames_are_capped_to_the_target_dimension(cut_video: Path) -> None:
    result = extract(cut_video, budget=20_000, max_dimension=64)
    assert all(max(frame.width, frame.height) <= 64 for frame in result.frames)


def test_a_budget_below_one_frame_is_rejected(cut_video: Path) -> None:
    with pytest.raises(BudgetTooSmallError):
        extract(cut_video, budget=10, target="claude")


def test_missing_file_is_reported_clearly(tmp_path: Path) -> None:
    with pytest.raises(UnreadableVideoError):
        extract(tmp_path / "nope.mp4")


def test_report_counts_add_up(cut_video: Path) -> None:
    report = extract(cut_video, budget=20_000).report
    assert report.unique + report.redundant == report.scanned
    assert report.selected <= report.unique
    assert 0.0 <= report.coverage <= 1.0


def test_claude_messages_are_shaped_for_the_api(cut_video: Path) -> None:
    result = extract(cut_video, budget=6_000, target="claude")
    blocks = result.to_messages()
    assert len(blocks) == 2 * len(result.frames)
    assert blocks[0]["type"] == "text"
    assert blocks[1]["source"]["media_type"] == "image/jpeg"


def test_openai_messages_use_data_urls(cut_video: Path) -> None:
    result = extract(cut_video, budget=6_000, target="openai")
    image = result.to_messages()[1]
    assert image["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_gemini_messages_use_inline_data(cut_video: Path) -> None:
    result = extract(cut_video, budget=6_000, target="gemini")
    assert "inline_data" in result.to_messages()[1]


def test_timestamps_can_be_left_out(cut_video: Path) -> None:
    result = extract(cut_video, budget=6_000)
    assert len(result.to_messages(with_timestamps=False)) == len(result.frames)


def test_every_target_produces_frames(cut_video: Path) -> None:
    for name in TARGETS:
        assert extract(cut_video, budget=20_000, target=name).frames


def test_frame_can_be_written_to_disk(cut_video: Path, tmp_path: Path) -> None:
    frame = extract(cut_video, budget=6_000).frames[0]
    written = frame.save(tmp_path / "frame.jpg")
    assert written.read_bytes() == frame.jpeg
