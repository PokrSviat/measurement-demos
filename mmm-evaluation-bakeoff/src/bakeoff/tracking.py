"""Log the bake-off that is already on disk.

This module does not import mlflow. The caller imports it only when tracking is on.
"""

from __future__ import annotations

import os
from pathlib import Path

from bakeoff.compare import holdout_mape, split_exports


def metrics_from_exports(exports: list[dict]) -> dict[str, float]:
    """Numbers already carried by the export JSON, with MAPE from the shared formula."""
    metrics: dict[str, float] = {}
    for export in exports:
        model = str(export["model"])
        metrics[f"{model}_holdout_mape"] = holdout_mape(export["holdout_actual"], export["holdout_pred"])
        for channel, roi in export["channel_roi"].items():
            metrics[f"{model}_roi_{channel}"] = float(roi)
    return metrics


def log_bakeoff(root: Path, store: Path) -> str:
    """Write one run to a local file store. Returns the run id.

    Reads the exports and memo.txt. Does not refit and does not rewrite results/.
    """
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    import mlflow

    included, _excluded = split_exports(root)
    manifest = included[0]["manifest"]
    store.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(store.resolve().as_uri())
    mlflow.set_experiment("mmm-bakeoff")
    params = {
        "target": str(manifest["target"]),
        "date_min": str(manifest["date_min"]),
        "date_max": str(manifest["date_max"]),
        "holdout_weeks": str(manifest["holdout_weeks"]),
        "models": ",".join(str(export["model"]) for export in included),
    }
    if "seed" in manifest:
        params["seed"] = str(manifest["seed"])
    with mlflow.start_run(run_name="bakeoff-" + str(manifest["dataset_sha256"])[:8]) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics_from_exports(included))
        mlflow.log_artifact(str(root / "results" / "memo.txt"))
        for export in included:
            mlflow.log_artifact(str(root / "artifacts" / f"{export['model']}_export.json"))
        return run.info.run_id
