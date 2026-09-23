from __future__ import annotations

import argparse
from pathlib import Path

from bottle_cap_vision.config import AppConfig, load_config, save_config
from bottle_cap_vision.pipeline import BottleCapPipeline


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bottle Cap Vision - Python/OpenCV core baseline")
    p.add_argument("--source", help="Path to image or video")
    p.add_argument("--config", default="configs/config.json", help="JSON configuration")
    p.add_argument("--output", help="Annotated output image/video")
    p.add_argument("--csv", help="Per-frame CSV log")
    p.add_argument("--json", dest="json_output", help="Final JSON result")
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--save-default-config", action="store_true")
    p.add_argument("--max-frames", type=int)
    return p.parse_args()


def is_image(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_SUFFIXES


def load_or_create_config(path: str | Path) -> AppConfig:
    cfg_path = Path(path)
    if cfg_path.exists():
        return load_config(cfg_path)
    cfg = AppConfig.default()
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    save_config(cfg, cfg_path)
    print(f"Created starter config: {cfg_path}")
    return cfg


def main() -> int:
    args = parse_args()
    cfg_path = Path(args.config)
    cfg = load_or_create_config(cfg_path)

    if args.save_default_config:
        save_config(cfg, cfg_path)
        print(f"Saved config: {cfg_path}")
        if not args.source:
            return 0

    if not args.source:
        print("Use --source <image-or-video>")
        return 2

    pipeline = BottleCapPipeline(cfg)
    if is_image(args.source):
        out = Path(args.output) if args.output else Path("outputs") / "image_result.jpg"
        result = pipeline.run_image(args.source, out)
        detections = result["detections"]
        print(f"Detections: {len(detections)}")
        for det in detections:
            print(f"  {det.color:8s} centroid={det.centroid} area={det.area:.0f}")
        print(f"Saved: {out}")
        return 0

    result = pipeline.run_video(
        args.source,
        output=args.output,
        csv_output=args.csv,
        display=not args.no_display,
        max_frames=args.max_frames,
    )
    if args.json_output:
        pipeline.export_result_json(args.json_output, source=args.source)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
