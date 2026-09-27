"""Unified interactive launcher for BottleCapVision practical demos."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bottle_cap_vision.practical_launcher import DEFAULT_CONFIG, run_launcher


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BottleCapVision unified practical launcher")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_launcher(config_path=args.config)


if __name__ == "__main__":
    raise SystemExit(main())
