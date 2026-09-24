"""CLI for the BottleCapVision external-data staging boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bottle_cap_vision.external_intake import (
    ExternalIntakeError,
    IntakeOptions,
    ManualDownloadRequired,
    SourceSpec,
    stage_external_source,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage external yellow/lilac bottle-cap candidates for manual review (never auto-promotes)",
    )
    parser.add_argument(
        "--source-type",
        required=True,
        choices=("zip", "local_zip", "directory", "local_directory", "direct_image_list", "web_archive"),
    )
    parser.add_argument("--source", required=True, help="Local path, URL-list path, or public ZIP URL")
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/external_staging"))
    parser.add_argument("--canonical-dataset-root", type=Path, default=Path("data/dataset/archive"))
    parser.add_argument("--target-classes", nargs="+", choices=("yellow", "lilac"), default=("yellow", "lilac"))
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-web", action="store_true")
    parser.add_argument("--license", default="unknown")
    parser.add_argument("--source-name")
    parser.add_argument("--source-page")
    parser.add_argument("--attribution")
    parser.add_argument("--notes", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = SourceSpec(
        source_id=args.source_id,
        source_type=args.source_type,
        source=args.source,
        license=args.license,
        source_name=args.source_name,
        source_page=args.source_page,
        attribution=args.attribution,
        notes=args.notes,
    )
    options = IntakeOptions(
        output_root=args.output_root,
        canonical_dataset_root=args.canonical_dataset_root,
        target_classes=tuple(args.target_classes),
        max_images=args.max_images,
        dry_run=args.dry_run,
        allow_web=args.allow_web,
    )
    try:
        result = stage_external_source(spec, options)
    except ManualDownloadRequired as exc:
        print(f"SOURCE STATUS: REQUIRES MANUAL DOWNLOAD\n{exc}", file=sys.stderr)
        return 3
    except ExternalIntakeError as exc:
        print(f"external intake error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
