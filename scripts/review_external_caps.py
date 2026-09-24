"""Small CLI for P13B manual review, annotation and controlled promotion."""

from __future__ import annotations

import argparse
from pathlib import Path

from bottle_cap_vision.external_promotion import (
    ExternalPromotionError,
    approve_annotation,
    create_promotion_plan,
    generate_annotation_overlay,
    promote_approved,
    record_visual_review,
    save_proposed_annotation,
)


DEFAULT_STAGING = Path("data/external_staging")
DEFAULT_DATASET = Path("data/dataset/archive")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Review and promote staged external bottle-cap images")
    parser.add_argument("--staging-root", type=Path, default=DEFAULT_STAGING)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    subparsers = parser.add_subparsers(dest="command", required=True)

    review = subparsers.add_parser("review", help="record manual visual review")
    review.add_argument("staging_id")
    review.add_argument("--decision", required=True, choices=("APPROVED_FOR_ANNOTATION", "REJECTED", "AMBIGUOUS"))
    review.add_argument("--reviewer", required=True)
    review.add_argument("--notes", required=True)
    review.add_argument("--source-group")
    review.add_argument("--source-group-basis")

    annotate = subparsers.add_parser("annotate", help="save manual pixel bboxes")
    annotate.add_argument("staging_id")
    annotate.add_argument(
        "--box", nargs=5, action="append", metavar=("CLASS_ID", "XMIN", "YMIN", "XMAX", "YMAX"), required=True,
    )

    overlay = subparsers.add_parser("overlay", help="render proposed bboxes")
    overlay.add_argument("staging_id")

    approve = subparsers.add_parser("approve", help="approve a reviewed overlay")
    approve.add_argument("staging_id")
    approve.add_argument("--reviewer", required=True)
    approve.add_argument("--notes", required=True)

    plan = subparsers.add_parser("plan", help="create a validated promotion plan")
    plan.add_argument("--assign", action="append", required=True, metavar="STAGING_ID=SPLIT")
    plan.add_argument("--output", type=Path, default=DEFAULT_STAGING / "review/promotion_plan.json")

    promote = subparsers.add_parser("promote", help="execute a validated promotion plan")
    promote.add_argument("--plan", type=Path, default=DEFAULT_STAGING / "review/promotion_plan.json")
    promote.add_argument("--manifest", type=Path, default=DEFAULT_STAGING / "review/promotion_manifest.json")
    promote.add_argument("--attribution", type=Path, default=Path("data/EXTERNAL_DATA_ATTRIBUTION.md"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "review":
            record_visual_review(
                args.staging_root, args.staging_id, decision=args.decision, reviewer=args.reviewer,
                notes=args.notes, source_group=args.source_group, source_group_basis=args.source_group_basis,
            )
        elif args.command == "annotate":
            boxes = []
            for raw in args.box:
                try:
                    class_id = int(raw[0])
                    if str(class_id) != raw[0]:
                        raise ValueError
                    boxes.append((class_id, *(float(value) for value in raw[1:])))
                except ValueError as exc:
                    raise ExternalPromotionError(f"invalid --box values: {raw!r}") from exc
            print(save_proposed_annotation(args.staging_root, args.staging_id, boxes))
        elif args.command == "overlay":
            print(generate_annotation_overlay(args.staging_root, args.staging_id))
        elif args.command == "approve":
            approve_annotation(
                args.staging_root, args.staging_id, reviewer=args.reviewer, notes=args.notes,
            )
        elif args.command == "plan":
            assignments: dict[str, str] = {}
            for raw in args.assign:
                staging_id, separator, split = raw.partition("=")
                if not separator or not staging_id or staging_id in assignments:
                    raise ExternalPromotionError(f"invalid or duplicate assignment: {raw!r}")
                assignments[staging_id] = split
            result = create_promotion_plan(args.staging_root, args.dataset_root, assignments, args.output)
            print(f"plan={args.output} items={len(result['items'])} run_id={result['promotion_run_id']}")
        elif args.command == "promote":
            result = promote_approved(
                args.staging_root, args.dataset_root, args.plan, args.manifest, args.attribution,
            )
            print(f"manifest={args.manifest} promoted={len(result['promoted_files'])}")
    except ExternalPromotionError as exc:
        print(f"external promotion error: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
