"""Convenience wrapper for processing one image.

Run from the project root:
    python scripts/run_image.py data/input/images/sample.jpg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bottle_cap_vision.config import load_config
from bottle_cap_vision.pipeline import BottleCapPipeline
from bottle_cap_vision.runtime import run
from bottle_cap_vision.sinks import ImageSink
from bottle_cap_vision.sources import ImageSource


def main() -> int:
    parser = argparse.ArgumentParser(description="Process one image with BottleCapVision")
    parser.add_argument("image")
    parser.add_argument("--config", default=str(ROOT / "configs" / "config.json"))
    parser.add_argument("--output", default=str(ROOT / "outputs" / "image_result.jpg"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    run(
        ImageSource(args.image),
        BottleCapPipeline(cfg),
        [ImageSink(args.output, cfg)],
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
