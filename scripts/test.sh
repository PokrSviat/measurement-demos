#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install -r requirements-ci.txt
"$PYTHON" -m pip install -e ./budget-rule
"$PYTHON" -m pip install -e "./geo-holdout-case[dev]"
"$PYTHON" -m pip install -e "./measurement-framework[dev]"
"$PYTHON" -m pip install -e "./measurement-reconciliation[dev]"
"$PYTHON" -m pip install -e "./mmm-evaluation-bakeoff[dev]"
for package in budget-rule geo-holdout-case measurement-framework measurement-reconciliation mmm-evaluation-bakeoff; do
  (cd "$package" && "$PYTHON" -m pytest -q)
done
