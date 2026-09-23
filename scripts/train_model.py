from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bottle_cap_vision.dataset import DatasetContractError, DatasetManifest
from bottle_cap_vision.training import (
    TrainingAdapter,
    TrainingBackendError,
    TrainingConfig,
    TrainingContractError,
    UltralyticsBackend,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a detector from a canonical P7 DatasetManifest"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print preflight without starting optimization"
    )
    args = parser.parse_args()
    try:
        manifest = DatasetManifest.load(args.manifest)
        config = TrainingConfig.load(args.config)
        adapter = TrainingAdapter(UltralyticsBackend())
        preflight = adapter.preflight(manifest, config)
        if args.dry_run:
            print(json.dumps(preflight.to_dict(), ensure_ascii=False, indent=2))
            return 2 if preflight.errors else 0
        result = adapter.train(
            manifest=manifest,
            config=config,
            dataset_root=args.dataset_root,
            manifest_path=args.manifest,
        )
    except (DatasetContractError, TrainingContractError, TrainingBackendError) as exc:
        print(f"training error: {exc}")
        return 2
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
