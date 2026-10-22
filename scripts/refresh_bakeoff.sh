#!/usr/bin/env bash
# Rebuild the bake-off from the PyMC export: copy it, refit Robyn and Meridian, rebuild the table and memo.
# Needs R with the packages in mmm-evaluation-bakeoff/robyn/versions.txt and a Python
# with mmm-evaluation-bakeoff/meridian/requirements.lock.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
MERIDIAN_PYTHON="${MERIDIAN_PYTHON:-/tmp/meridian-venv/bin/python}"
export ROBYN_TRIALS="${ROBYN_TRIALS:-5}" ROBYN_ITERS="${ROBYN_ITERS:-2000}"
FRAMEWORK="$ROOT/measurement-framework"
BAKEOFF="$ROOT/mmm-evaluation-bakeoff"
WEEKLY="$FRAMEWORK/data/dt_simulated_weekly.csv"
HOLIDAYS="$FRAMEWORK/data/dt_prophet_holidays_de.csv"
MANIFEST="$BAKEOFF/artifacts/manifest.json"

cp "$FRAMEWORK/artifacts/pymc_export.json" "$FRAMEWORK/artifacts/manifest.json" "$BAKEOFF/artifacts/"
cd "$BAKEOFF"
Rscript robyn/run_robyn.R "$WEEKLY" "$HOLIDAYS" "$MANIFEST"
"$MERIDIAN_PYTHON" meridian/run_meridian.py --weekly "$WEEKLY" --manifest "$MANIFEST"
"$MERIDIAN_PYTHON" meridian/run_meridian.py --weekly "$WEEKLY" --manifest "$MANIFEST" --sensitivity
"$MERIDIAN_PYTHON" meridian/run_meridian.py --weekly "$WEEKLY" --manifest "$MANIFEST" --seed-check "$(mktemp -d)"
"$PYTHON" -m bakeoff.compare
