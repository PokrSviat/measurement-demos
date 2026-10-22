"""Build the bake-off table from the exports that exist.

The harness computes holdout MAPE from each export's holdout path with one
formula. Each ROI comes with the interval its model can give, and the table
keeps posterior intervals and model spread in separate columns.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from budget_rule.experiment import PAYBACK, experiment_priority, next_experiment
from budget_rule.memo_pdf import write_memo_pdf

MANIFEST_KEYS = ("dataset_sha256", "target", "date_min", "date_max", "regressors", "holdout_weeks")
ESTIMAND = "marginal_roi_next_ruble"
EXPORT_NAMES = ("pymc_export.json", "robyn_export.json", "meridian_export.json")
REQUIRED_KEYS = (
    "model",
    "manifest",
    "holdout_actual",
    "holdout_pred",
    "channel_roi",
    "channel_roi_low",
    "channel_roi_high",
    "interval_kind",
)
INTERVAL_KINDS = ("posterior", "model_spread")
SPEND_SOURCE = "pymc"
R_HAT_GATE = 1.01


def load_export(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    missing = [key for key in REQUIRED_KEYS if key not in payload]
    if missing:
        raise ValueError(f"{path} is missing {missing}.")
    if payload["interval_kind"] not in INTERVAL_KINDS:
        raise ValueError(f"{path} has interval_kind {payload['interval_kind']!r}, not one of {INTERVAL_KINDS}.")
    return payload


def holdout_mape(actual: list[float], predicted: list[float]) -> float:
    """Mean absolute error over max(|actual|, 1). The same formula for every model."""
    observed = np.asarray(actual, dtype=float)
    forecast = np.asarray(predicted, dtype=float)
    if observed.shape != forecast.shape or observed.size == 0:
        raise ValueError("The holdout path needs matching, non-empty actual and predicted weeks.")
    return float(np.mean(np.abs(forecast - observed) / np.maximum(np.abs(observed), 1.0)))


def assert_same_manifest(exports: list[dict]) -> None:
    base = exports[0]["manifest"]
    for export in exports[1:]:
        for key in MANIFEST_KEYS:
            if export["manifest"][key] != base[key]:
                raise ValueError(f"Manifest field {key} differs for {export['model']}.")


def assert_same_estimand(exports: list[dict]) -> None:
    for export in exports:
        if export.get("estimand") != ESTIMAND:
            raise ValueError(f"{export['model']} is not the {ESTIMAND} estimand.")


def assert_same_holdout(exports: list[dict]) -> None:
    weeks = int(exports[0]["manifest"]["holdout_weeks"])
    base = np.asarray(exports[0]["holdout_actual"], dtype=float)
    for export in exports:
        actual = np.asarray(export["holdout_actual"], dtype=float)
        if len(actual) != weeks or len(export["holdout_pred"]) != weeks:
            raise ValueError(f"{export['model']} does not export {weeks} holdout weeks.")
        if not np.allclose(actual, base, rtol=1e-6):
            raise ValueError(f"{export['model']} holds out different weeks.")


def comparison_table(exports: list[dict]) -> pd.DataFrame:
    if len(exports) < 2:
        raise ValueError("A comparison needs at least two exports.")
    assert_same_manifest(exports)
    assert_same_estimand(exports)
    assert_same_holdout(exports)
    rows = []
    for export in exports:
        if export["model"] == "meridian" and export.get("r_hat_max") is None:
            raise ValueError("The Meridian export must include r_hat_max.")
        cells = json.dumps(_roi_cells(export), sort_keys=True)
        contraction = export.get("channel_contraction")
        rows.append(
            {
                "model": export["model"],
                "interval_kind": export["interval_kind"],
                "holdout_mape": holdout_mape(export["holdout_actual"], export["holdout_pred"]),
                "r_hat_max": export.get("r_hat_max"),
                "prior_sensitivity_relative": export.get("prior_sensitivity_relative"),
                "interval": export.get("interval"),
                "posterior_roi": cells if export["interval_kind"] == "posterior" else None,
                "model_spread_roi": cells if export["interval_kind"] == "model_spread" else None,
                "channel_contraction": None if contraction is None else json.dumps(contraction, sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def _roi_cells(export: dict) -> dict[str, list[float]]:
    return {
        channel: [export["channel_roi_low"][channel], export["channel_roi"][channel], export["channel_roi_high"][channel]]
        for channel in sorted(export["channel_roi"])
    }


def _roi_order(export: dict) -> list[str]:
    return [
        channel
        for channel, _ in sorted(export["channel_roi"].items(), key=lambda item: (-item[1], item[0]))
    ]


def unsettled_channels(exports: list[dict]) -> list[str]:
    """Channels whose interval contains payback in at least one model."""
    channels = sorted(exports[0]["channel_roi"])
    return [
        channel
        for channel in channels
        if any(export["channel_roi_low"][channel] <= PAYBACK <= export["channel_roi_high"][channel] for export in exports)
    ]


def earning_channels(exports: list[dict]) -> list[str]:
    """Channels whose interval sits above payback in every model."""
    channels = sorted(exports[0]["channel_roi"])
    return [channel for channel in channels if all(export["channel_roi_low"][channel] > PAYBACK for export in exports)]


def spend_shares(exports: list[dict]) -> dict[str, float]:
    """Training spend shares from the PyMC export, which reads the same manifest as the others."""
    source = next((export for export in exports if export["model"] == SPEND_SOURCE), None)
    if source is None or "channel_spend_share" not in source:
        raise ValueError(f"The {SPEND_SOURCE} export must carry channel_spend_share.")
    return {channel: float(share) for channel, share in source["channel_spend_share"].items()}


def experiment_by_model(exports: list[dict]) -> dict[str, str | None]:
    """The shared lift-test rule on each model's own interval, with PyMC spend shares."""
    shares = spend_shares(exports)
    return {
        export["model"]: next_experiment(shares, export["channel_roi_low"], export["channel_roi_high"])
        for export in exports
    }


def experiment_priorities(exports: list[dict]) -> dict[str, dict[str, float]]:
    shares = spend_shares(exports)
    return {
        export["model"]: experiment_priority(shares, export["channel_roi_low"], export["channel_roi_high"])
        for export in exports
    }


def experiment_channel(exports: list[dict]) -> str | None:
    """Majority model pick; a tie goes to the largest spend share among the picks."""
    picks = experiment_by_model(exports)
    voted = [channel for channel in picks.values() if channel is not None]
    if not voted:
        return None
    counts = {channel: voted.count(channel) for channel in set(voted)}
    top = max(counts.values())
    leaders = [channel for channel, count in counts.items() if count == top]
    if len(leaders) == 1:
        return leaders[0]
    shares = spend_shares(exports)
    return max(leaders, key=lambda channel: (shares[channel], channel))


def recommendation(exports: list[dict]) -> str:
    if not exports:
        raise ValueError("A recommendation needs an export.")
    orders = [_roi_order(export) for export in exports]
    if all(order == orders[0] for order in orders):
        order_sentence = f"The ROI order is the same: {', '.join(orders[0])}."
    else:
        parts = [f"{export['model']} {' > '.join(_roi_order(export))}" for export in exports]
        order_sentence = "The ROI order differs: " + "; ".join(parts) + "."
    unsettled = unsettled_channels(exports)
    unsettled_sentence = (
        f"In at least one model the interval for {', '.join(unsettled)} contains 1."
        if unsettled
        else "No model has an interval that contains 1."
    )
    earning = earning_channels(exports)
    if earning:
        unsettled_sentence += f" Every model puts the interval for {', '.join(earning)} above 1."
    sensitivity = [
        f"{export['model']} facebook ROI moves {export['prior_sensitivity_relative']:+.0%} with a wider prior"
        for export in exports
        if export.get("prior_sensitivity_relative") is not None
    ]
    sensitivity_sentence = (" " + "; ".join(sensitivity) + ".") if sensitivity else ""
    return (
        "Do not reallocate. These data do not identify the channel returns.\n"
        f"{order_sentence} {unsettled_sentence}{sensitivity_sentence}\n"
        f"{_experiment_sentence(exports)}"
    )


def _experiment_sentence(exports: list[dict]) -> str:
    picks = experiment_by_model(exports)
    kinds = {export["model"]: export["interval_kind"].replace("_", " ") for export in exports}
    by_model = ", ".join(
        f"{model} ({kinds[model]}) "
        + ("no channel" if channel is None else channel)
        for model, channel in picks.items()
    )
    kinds_used = {export["interval_kind"] for export in exports}
    vote = (
        "The models do not share an interval kind, so they are not voted. "
        if len(kinds_used) > 1
        else "Per-model value of information is listed and is not a test to run. "
    )
    return (
        f"{vote}Per-model value of information: {by_model}.\n"
        "No lift test is recommended."
    )


def _contraction_text(export: dict) -> str:
    contraction = export.get("channel_contraction")
    if not contraction:
        return ""
    parameter = export.get("contraction_parameter", "channel coefficient")
    cells = ", ".join(f"{channel} {value:.2f}" for channel, value in sorted(contraction.items()))
    return f" Prior contraction on {parameter}: {cells}."


def _model_line(export: dict) -> str:
    cells = ", ".join(
        f"{channel} {mean:.2f} ({low:.2f} to {high:.2f})" for channel, (low, mean, high) in _roi_cells(export).items()
    )
    interval = export.get("interval") or "interval source not stated"
    kind = export["interval_kind"].replace("_", " ")
    r_hat = export.get("r_hat_max")
    convergence = "" if r_hat is None else f" r_hat_max {r_hat:.3f}{'' if r_hat < R_HAT_GATE else ', at or above 1.01'}."
    return (
        f"{export['model']}\n"
        f"holdout MAPE {holdout_mape(export['holdout_actual'], export['holdout_pred']):.3f}.{convergence}\n"
        f"ROI {cells}.\n"
        f"Interval ({kind}): {interval}.{_contraction_text(export)}"
    )


def seed_sentence(seed_check: dict | None) -> str:
    if not seed_check:
        return ""
    spread = seed_check["point_roi_spread"]
    widest = max(spread, key=lambda channel: (spread[channel], channel))
    seeds = ", ".join(str(seed) for seed in seed_check["seeds"])
    return f" Meridian refit on seeds {seeds}: the point ROI moves by at most {spread[widest]:.2f}, on {widest}."


def memo_text(exports: list[dict], *, excluded: list[str] | None = None, seed_check: dict | None = None) -> str:
    lines = ["Should holdout error or channel ROI from three models move the mix?", ""]
    for export in exports:
        lines.extend([_model_line(export), ""])
    lines.extend(
        [
            "Harness",
            "The harness computes every MAPE from the exported holdout weeks with one formula. "
            "Each ROI is one estimand, the marginal ROI of the next ruble at observed spend. "
            "The Robyn interval is a spread across models, not a posterior, so the intervals are not the same kind "
            "and the table keeps them in separate columns. Prior contraction is one minus posterior over prior "
            "variance; near 0 the data did not narrow that prior, and below 0 the posterior is wider than the prior."
            f"{seed_sentence(seed_check)}",
            "",
            "Recommendation",
            recommendation(exports),
        ]
    )
    if excluded:
        lines.extend(
            [
                "",
                f"Not in the ROI column: {', '.join(excluded)}. The stored ROI is not the marginal ROI of the next ruble.",
            ]
        )
    elif not any(export["model"] == "meridian" for export in exports):
        lines.extend(["", "Meridian is not in this table until an export with r_hat_max exists."])
    return "\n".join(lines)


def write_comparison(exports: list[dict], path: Path) -> pd.DataFrame:
    table = comparison_table(exports)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return table


def split_exports(root: Path) -> tuple[list[dict], list[str]]:
    included: list[dict] = []
    excluded: list[str] = []
    for name in EXPORT_NAMES:
        path = root / "artifacts" / name
        if not path.exists():
            continue
        export = load_export(path)
        if export.get("estimand") == ESTIMAND:
            included.append(export)
        else:
            excluded.append(str(export["model"]))
    if len(included) < 2:
        raise ValueError("The ROI column needs two exports of the next-ruble estimand.")
    return included, excluded


def published_exports(root: Path) -> list[dict]:
    included, _excluded = split_exports(root)
    return included


def load_seed_check(root: Path) -> dict | None:
    path = root / "artifacts" / "meridian_seed_check.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def publish(root: Path) -> None:
    exports, excluded = split_exports(root)
    write_comparison(exports, root / "artifacts" / "comparison.csv")
    memo = root / "results" / "memo.txt"
    memo.parent.mkdir(parents=True, exist_ok=True)
    text = memo_text(exports, excluded=excluded, seed_check=load_seed_check(root))
    memo.write_text(text + "\n", encoding="utf-8")
    write_memo_pdf(text, root / "results" / "memo.pdf")


def main() -> None:
    publish(Path(__file__).resolve().parents[2])


if __name__ == "__main__":
    main()
