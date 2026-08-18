"""Command line front end."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .errors import FrameBudgetError
from .extract import extract
from .targets import TARGETS

__all__ = ["main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="framebudget",
        description="Fit a video into a token budget for multimodal LLMs.",
    )
    parser.add_argument("video", type=Path, help="video file to read")
    parser.add_argument(
        "-b",
        "--budget",
        type=int,
        default=50_000,
        help="token ceiling for images (default: 50000)",
    )
    parser.add_argument(
        "-t",
        "--target",
        choices=sorted(TARGETS),
        default="claude",
        help="provider cost model (default: claude)",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        help="directory to write the selected frames as JPEG",
    )
    parser.add_argument(
        "--analysis-fps",
        type=float,
        default=2.0,
        help="samples per second while deciding (default: 2.0)",
    )
    parser.add_argument(
        "--min-distance",
        type=float,
        default=0.02,
        help="redundancy floor in [0, 1] (default: 0.02)",
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=4.0,
        help="cut threshold, times above the scene baseline (default: 4.0)",
    )
    parser.add_argument(
        "--max-dimension",
        type=int,
        help="longest output edge in pixels (default: target dependent)",
    )
    parser.add_argument(
        "--json", action="store_true", help="print the report as JSON instead of text"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        result = extract(
            args.video,
            budget=args.budget,
            target=args.target,
            analysis_fps=args.analysis_fps,
            min_distance=args.min_distance,
            sensitivity=args.sensitivity,
            max_dimension=args.max_dimension,
        )
    except FrameBudgetError as error:
        print(f"framebudget: {error}", file=sys.stderr)
        return 1

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for position, frame in enumerate(result.frames):
            frame.save(args.out / f"{position:04d}_{frame.timestamp:08.2f}s.jpg")

    if args.json:
        payload = asdict(result.report)
        payload["coverage"] = result.report.coverage
        payload["saved"] = result.report.saved
        payload["redundant"] = result.report.redundant
        payload["frames"] = [
            {"index": frame.index, "timestamp": frame.timestamp}
            for frame in result.frames
        ]
        print(json.dumps(payload, indent=2))
    else:
        print(result.report.summary())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
