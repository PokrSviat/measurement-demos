"""Log the published bake-off when tracking is turned on."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from bakeoff.tracking import log_bakeoff

ROOT = Path(__file__).resolve().parents[2]


def tracking_enabled(flag: bool) -> bool:
    return flag or os.environ.get("MLFLOW_TRACKING") == "1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Log the bake-off that is already on disk.")
    parser.add_argument("--track", action="store_true", help="Write a run to ./mlruns.")
    args = parser.parse_args(argv)
    if not tracking_enabled(args.track):
        print("Tracking is off. Pass --track or set MLFLOW_TRACKING=1.")
        return 0
    try:
        run_id = log_bakeoff(ROOT, ROOT / "mlruns")
    except ImportError:
        print(
            'MLflow is not installed. Install the tracking extra: pip install -e "./mmm-evaluation-bakeoff[tracking]"',
            file=sys.stderr,
        )
        return 1
    print(run_id)
    print("mlflow ui --backend-store-uri ./mlruns")
    return 0


if __name__ == "__main__":
    sys.exit(main())
