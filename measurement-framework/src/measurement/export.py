"""Manifest shared with the evaluation bake-off."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def write_manifest(
    path: Path,
    *,
    data_path: Path,
    target: str,
    date_min: str,
    date_max: str,
    regressors: list[str],
    holdout_weeks: int,
    seed: int,
) -> dict[str, object]:
    payload = {
        "dataset_sha256": file_sha256(data_path),
        "target": target,
        "date_min": date_min,
        "date_max": date_max,
        "regressors": regressors,
        "holdout_weeks": holdout_weeks,
        "seed": seed,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
