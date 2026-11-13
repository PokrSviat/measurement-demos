#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
docker build -t measurement-demos-geo .
mkdir -p out/geo
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "$ROOT/out/geo:/app/geo-holdout-case/results" \
  measurement-demos-geo
PYTHON="${PYTHON:-python3}"
"$PYTHON" scripts/check_geo_result.py out/geo/case_result.json geo-holdout-case/results/case_result.json
