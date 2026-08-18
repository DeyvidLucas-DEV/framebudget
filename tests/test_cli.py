from __future__ import annotations

import json
from pathlib import Path

import pytest

from framebudget.cli import main


def test_cli_prints_a_summary(
    cut_video: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(cut_video), "--budget", "6000"]) == 0
    assert "selected" in capsys.readouterr().out


def test_cli_emits_valid_json(
    cut_video: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(cut_video), "--budget", "6000", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["selected"] == len(payload["frames"])


def test_cli_writes_frames(cut_video: Path, tmp_path: Path) -> None:
    out = tmp_path / "frames"
    assert main([str(cut_video), "--budget", "6000", "--out", str(out)]) == 0
    assert list(out.glob("*.jpg"))


def test_cli_reports_failure_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(tmp_path / "missing.mp4")]) == 1
    assert "framebudget:" in capsys.readouterr().err
