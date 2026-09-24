"""One-command practical realtime camera demo for BottleCapVision."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bottle_cap_vision.camera_demo import DEFAULT_CONFIG, run_camera
from bottle_cap_vision.models import RunSummary
from bottle_cap_vision.practical_demo import PracticalDemoError


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BottleCapVision practical realtime camera demo")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--max-frames", type=positive_int)
    return parser.parse_args(argv)


def print_summary(summary: RunSummary) -> None:
    print("Run summary")
    print(f"frames_processed: {summary.frames_processed}")
    print(f"average_processing_fps: {summary.average_processing_fps:.2f}")
    print(f"total_count: {summary.total_count}")
    print(f"count_by_color: {summary.count_by_color}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print("Camera realtime started")
    print("Q / Esc: Exit")
    try:
        summary = run_camera(
            camera_index=args.camera_index,
            config_path=args.config,
            max_frames=args.max_frames,
        )
    except (PracticalDemoError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
