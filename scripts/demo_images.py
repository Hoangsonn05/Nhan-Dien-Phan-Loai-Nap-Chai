"""One-command practical still-image demo for BottleCapVision."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bottle_cap_vision.practical_demo import (
    PracticalDemoError,
    print_summary,
    process_images,
    show_results,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BottleCapVision practical folder-image demo")
    parser.add_argument("--input", default=str(ROOT / "data" / "input" / "images"))
    parser.add_argument("--output", default=str(ROOT / "outputs" / "practical_demo"))
    parser.add_argument("--config", default=str(ROOT / "configs" / "config.learned-p9-full.json"))
    parser.add_argument("--no-gui", action="store_true", help="Generate outputs without opening a window")
    parser.add_argument("--recursive", action="store_true", help="Discover images in subdirectories")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = process_images(
            args.input,
            args.output,
            args.config,
            recursive=args.recursive,
        )
        print_summary(report)
        if not args.no_gui:
            show_results(report.processed_images)
        return 0
    except PracticalDemoError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
