"""Planning-proxy geo lift for the MMM channel that the shared rule names.

The noise comes from the public geo panel's control geos, which never received
the published tactic. Spend and the point ROI come from the weekly revenue
series. This is a planning proxy, not a test of search on that panel.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from geo_holdout.design import volume_strata
from geo_holdout.estimators import (
    cuped_total,
    cuped_totals,
    effect_profile,
    location_window_means,
    picks_to_masks,
    stratified_picks,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "search_lift.yaml"
OUTPUT = ROOT / "results" / "search_lift_design.json"
NOTE = (
    "planning proxy: noise from a simulated public geo panel; "
    "one-sided randomization test of incremental = extra spend, not a full inverted interval; "
    "no cell is a test to run"
)
PAYBACK = 1.0
POWER_TARGET = 0.8
N_TREATED = 25
N_GEOS = 50


@dataclass(frozen=True)
class LiftWindow:
    pre: np.ndarray
    delta: np.ndarray
    days: np.ndarray
    stratum_ids: np.ndarray


def spend_to_revenue(weekly: pd.DataFrame, spend_column: str, holdout_weeks: int) -> float:
    train = weekly.iloc[:-holdout_weeks]
    revenue = float(train["revenue"].mean())
    if revenue <= 0:
        raise ValueError("Training revenue must be positive.")
    return float(train[spend_column].mean()) / revenue


def control_window(
    panel: pd.DataFrame,
    assignment: pd.DataFrame,
    *,
    treated_group: int,
    pre_days: int,
    duration_days: int,
    stratum_size: int,
) -> LiftWindow:
    """Last `duration_days` of the control geos, with `pre_days` before them as the covariate."""
    control = assignment.loc[assignment["group"] != treated_group, "location"]
    frame = panel.loc[panel["location"].isin(control)].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    dates = np.array(sorted(frame["date"].unique()))
    if len(dates) < pre_days + duration_days:
        raise ValueError("The control panel is shorter than the planned pre-period plus test.")
    post_end = pd.Timestamp(dates[-1])
    post_start = pd.Timestamp(dates[-duration_days])
    pre_end = post_start - pd.Timedelta(days=1)
    pre_start = pre_end - pd.Timedelta(days=pre_days - 1)
    earlier = frame.loc[(frame["date"] >= pre_start) & (frame["date"] <= pre_end)]
    strata = volume_strata(earlier, stratum_size)
    means = location_window_means(
        frame.loc[frame["date"] >= pre_start],
        pre_end=pre_end.strftime("%Y-%m-%d"),
        post_start=post_start.strftime("%Y-%m-%d"),
        post_end=post_end.strftime("%Y-%m-%d"),
    )
    ids = means["location"].map(strata.set_index("location")["stratum"])
    if ids.isna().any():
        raise ValueError("A control geo in the lift window has no volume stratum.")
    return LiftWindow(
        pre=means["pre"].to_numpy(dtype=float),
        delta=means["delta"].to_numpy(dtype=float),
        days=means["n_post_days"].to_numpy(dtype=float),
        stratum_ids=ids.astype(int).to_numpy(),
    )


def extra_spend(pre: np.ndarray, days: np.ndarray, treated: np.ndarray, *, increase: float, ratio: float) -> float:
    return float(increase * ratio * np.sum(pre[treated] * days[treated]))


def plant_lift(pre: np.ndarray, *, roi: float, increase: float, ratio: float) -> np.ndarray:
    return roi * increase * ratio * pre


def settles(pre: np.ndarray, delta: np.ndarray, days: np.ndarray, treated: np.ndarray, masks: np.ndarray, spend: float) -> bool:
    """One-sided randomization test that incremental equals extra spend, not a two-sided inverted interval."""
    if spend <= 0:
        raise ValueError("Extra spend must be positive.")
    profile = effect_profile(pre, "proportional")
    per_unit = profile[treated] / float(np.sum(profile[treated] * days[treated]))
    adjusted = delta.copy()
    adjusted[treated] -= spend * per_unit
    observed = cuped_total(pre, adjusted, days, treated)
    low, high = np.quantile(cuped_totals(pre, adjusted, days, masks), [0.025, 0.975])
    return bool(observed > high)


def _draws(
    window: LiftWindow,
    *,
    treated_per_stratum: int,
    n_draws: int,
    n_reassignments: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    assignments = picks_to_masks(
        stratified_picks(window.stratum_ids, treated_per_stratum, n_draws, rng),
        len(window.pre),
    )
    out = []
    for index, treated in enumerate(assignments):
        redraws = picks_to_masks(
            stratified_picks(
                window.stratum_ids,
                treated_per_stratum,
                n_reassignments,
                np.random.default_rng(seed + 1 + index),
            ),
            len(window.pre),
        )
        out.append((treated, redraws))
    return out


def settle_share(
    window: LiftWindow,
    draws: list[tuple[np.ndarray, np.ndarray]],
    *,
    roi: float,
    increase: float,
    ratio: float,
) -> float:
    settled = 0
    for treated, masks in draws:
        effect = plant_lift(window.pre, roi=roi, increase=increase, ratio=ratio)
        injected = window.delta.copy()
        injected[treated] += effect[treated]
        spend = extra_spend(window.pre, window.days, treated, increase=increase, ratio=ratio)
        settled += int(settles(window.pre, injected, window.days, treated, masks, spend))
    return settled / len(draws)


def mde_roi(
    window: LiftWindow,
    draws: list[tuple[np.ndarray, np.ndarray]],
    *,
    increase: float,
    ratio: float,
    target: float = POWER_TARGET,
) -> float | None:
    """Smallest ROI at which the settle share reaches `target`, by bisection on shared draws."""
    high = 8.0
    if settle_share(window, draws, roi=high, increase=increase, ratio=ratio) < target:
        return None
    low = PAYBACK
    for _ in range(8):
        mid = (low + high) / 2
        if settle_share(window, draws, roi=mid, increase=increase, ratio=ratio) >= target:
            high = mid
        else:
            low = mid
    return high


def run_design(config: dict, panel: pd.DataFrame, assignment: pd.DataFrame, weekly: pd.DataFrame) -> dict:
    ratio = spend_to_revenue(weekly, config["spend_column"], int(config["holdout_weeks"]))
    if abs(ratio - float(config["spend_to_revenue"])) > 1e-9:
        raise ValueError("search_lift.yaml spend_to_revenue does not match the weekly series.")
    cells = []
    for duration in config["durations_days"]:
        window = control_window(
            panel,
            assignment,
            treated_group=int(config["treated_group"]),
            pre_days=int(config["pre_days"]),
            duration_days=int(duration),
            stratum_size=int(config["stratum_size"]),
        )
        draws = _draws(
            window,
            treated_per_stratum=int(config["treated_per_stratum"]),
            n_draws=int(config["n_draws"]),
            n_reassignments=int(config["n_reassignments"]),
            seed=int(config["seed"]) + 1000 * int(duration),
        )
        for increase in config["increases"]:
            cells.append(
                {
                    "duration_days": int(duration),
                    "increase": float(increase),
                    "power_at_point_roi": settle_share(
                        window, draws, roi=float(config["point_roi"]), increase=float(increase), ratio=ratio
                    ),
                    "power_at_roi_1_5": settle_share(
                        window, draws, roi=1.5, increase=float(increase), ratio=ratio
                    ),
                    "false_settle_at_payback": settle_share(
                        window, draws, roi=1.0, increase=float(increase), ratio=ratio
                    ),
                    "mde_roi_80": mde_roi(window, draws, increase=float(increase), ratio=ratio),
                }
            )
    return {
        "note": NOTE,
        "channel": config["channel"],
        "point_roi": float(config["point_roi"]),
        "spend_to_revenue": ratio,
        "n_geos": N_GEOS,
        "n_treated": N_TREATED,
        "pre_days": int(config["pre_days"]),
        "cells": cells,
    }


def shortest_powered(payload: dict, *, target: float = POWER_TARGET) -> dict | None:
    powered = [cell for cell in payload["cells"] if cell["power_at_point_roi"] >= target]
    if not powered:
        return None
    return min(powered, key=lambda cell: (cell["duration_days"], cell["increase"]))


def prior_sentence(payload: dict) -> str:
    chosen = shortest_powered(payload)
    if chosen is None:
        return (
            f" On the planning-proxy geo panel no design reached {POWER_TARGET:.0%} power at the current "
            f"{payload['channel']} point ROI {payload['point_roi']:.2f}. "
            "That panel is simulated and no cell is a test to run."
        )
    mde = chosen["mde_roi_80"]
    mde_text = "none of the searched ROIs" if mde is None else f"{mde:.2f}"
    return (
        f" The shortest planning-proxy geo design with at least {POWER_TARGET:.0%} power at the current "
        f"{payload['channel']} point ROI {payload['point_roi']:.2f} is {payload['n_treated']} treated geos "
        f"of {payload['n_geos']}, {chosen['duration_days']} days, and a {chosen['increase']:.0%} heavy-up "
        f"(false-settle at payback {chosen['false_settle_at_payback']:.0%}, MDE ROI {mde_text} at 80% power). "
        "A test that returned a 95% ROI interval [L, H] would replace the current search coefficient prior "
        "with a normal whose mean is (L+H)/2 and whose standard deviation is (H-L)/3.92, mapped through the "
        f"current saturation slope at observed search spend. {payload['note']}."
    )


def weekly_path() -> Path:
    return Path(__file__).resolve().parents[3] / "measurement-framework" / "data" / "dt_simulated_weekly.csv"


def main() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    panel = pd.read_csv(ROOT / "data" / "sales_and_cost.csv", parse_dates=["date"], dtype={"location": str})
    assignment = pd.read_csv(ROOT / "data" / "geo_assignment.csv", dtype={"location": str})
    weekly = pd.read_csv(weekly_path())
    payload = run_design(config, panel, assignment, weekly)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
