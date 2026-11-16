"""Write the simulated interrupted-time-series and regression-discontinuity results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from quasi.dgp import simulate_its, simulate_rdd
from quasi.its import fit_its, observed_and_counterfactual, placebo_dates
from quasi.rdd import (
    bandwidth_sensitivity,
    covariate_jump,
    density_check,
    local_linear,
    local_linear_segments,
    placebo_cutoffs,
)

ROOT = Path(__file__).resolve().parents[2]
SEED = 1
RDD_H = 5.0
BANDWIDTHS = [2.0, 5.0, 10.0, 25.0]
PLACEBO_POINTS = list(range(35, 46)) + list(range(55, 66))


def _interval(fit) -> dict[str, float]:
    return {"estimate": fit.estimate, "low": fit.low, "high": fit.high}


def _density(check) -> dict[str, float]:
    return {"estimate": check.estimate, "se": check.se, "p_value": check.p_value}


def _binned_means(spend: np.ndarray, outcome: np.ndarray, cutoff: float, h: float, width: float = 0.5):
    """Bin means on each side of the cutoff. Bins do not cross it."""
    centers: list[float] = []
    means: list[float] = []
    edges = np.arange(cutoff - h, cutoff + h + 1e-9, width)
    for left, right in zip(edges[:-1], edges[1:]):
        if left < cutoff <= right:
            continue
        closed_on_right = right >= cutoff + h - 1e-9
        inside = (spend >= left) & (spend <= right if closed_on_right else spend < right)
        if not np.any(inside):
            continue
        centers.append((left + right) / 2.0)
        means.append(float(outcome[inside].mean()))
    return np.asarray(centers), np.asarray(means)


def _write_its_plot(weeks: np.ndarray, actual: np.ndarray, counterfactual: np.ndarray, t0: int, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(weeks, actual, color="0.65", label="Actual")
    axis.plot(weeks, counterfactual, color="0.1", label="Counterfactual")
    axis.axvline(t0, color="0.2", linestyle="--", label="T0")
    axis.set_xlabel("Week")
    axis.set_ylabel("Sales")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _write_rdd_plot(frame, cutoff: float, h: float, path: Path) -> None:
    spend = frame["spend_prior"].to_numpy(dtype=float)
    outcome = frame["spend_next"].to_numpy(dtype=float)
    centers, means = _binned_means(spend, outcome, cutoff, h)
    left_x, left_y, right_x, right_y = local_linear_segments(frame, cutoff, h)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(centers, means, linestyle="none", marker="o", color="0.45", label="Binned means")
    axis.plot(left_x, left_y, color="0.1", label="Left fit")
    axis.plot(right_x, right_y, color="0.1", linestyle="--", label="Right fit")
    axis.axvline(cutoff, color="0.2", linestyle=":", label="Cutoff")
    axis.set_xlabel("Last-year spend, thousand rubles")
    axis.set_ylabel("Next-year spend")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _memo(its: dict, rdd: dict) -> str:
    level = its["estimate"]["level"]
    slope = its["estimate"]["slope"]
    cumulative = its["estimate"]["cumulative"]
    truth = its["truth"]
    jump = rdd["estimate"]["tau"]
    tenure = rdd["estimate"]["tenure"]
    density = rdd["estimate"]["density"]
    bunched = rdd["bunching_check"]
    return "\n".join(
        [
            "Simulated interrupted time series and sharp regression discontinuity. The effects are known.",
            "",
            "Interrupted time series",
            (
                f"Level truth {truth['level']:.0f}, estimate {level['estimate']:.2f} "
                f"({level['low']:.2f} to {level['high']:.2f}). "
                f"Slope truth {truth['slope']:.1f}, estimate {slope['estimate']:.2f} "
                f"({slope['low']:.2f} to {slope['high']:.2f}). "
                f"Cumulative truth {truth['cumulative']:.0f}, estimate {cumulative['estimate']:.2f} "
                f"({cumulative['low']:.2f} to {cumulative['high']:.2f}). "
                f"Placebo share {its['estimate']['placebo_share']:.2f}."
            ),
            "",
            "Regression discontinuity",
            (
                f"Tau truth {rdd['truth']['tau']:.0f}, h={jump['h']:.0f} estimate {jump['estimate']:.3f} "
                f"({jump['low']:.3f} to {jump['high']:.3f}). "
                f"Tenure truth 0, estimate {tenure['estimate']:.3f} "
                f"({tenure['low']:.3f} to {tenure['high']:.3f}). "
                f"Density jump without bunching {density['estimate']:.3f} "
                f"(se {density['se']:.3f}, p {density['p_value']:.3f}). "
                f"With bunching share {bunched['truth']['bunching_share']:.1f}, "
                f"density jump {bunched['estimate']['estimate']:.3f} "
                f"(se {bunched['estimate']['se']:.3f}, p {bunched['estimate']['p_value']:.3e}). "
                f"Placebo cutoffs containing 0: {rdd['estimate']['placebo_zero_share']:.2f}."
            ),
            "",
            "There is no real panel. The density check is a simplified McCrary test, not rddensity. "
            "The intervals are conventional, not bias-corrected. The rd extra is not used.",
            "",
        ]
    )


def main() -> None:
    output = ROOT / "results"
    output.mkdir(parents=True, exist_ok=True)

    its_frame, its_truth = simulate_its(SEED)
    its_fit = fit_its(its_frame, int(its_truth["t0"]))
    placebo = placebo_dates(its_frame, int(its_truth["t0"]))
    its_payload = {
        "data": "simulated",
        "truth": {**its_truth, "n_weeks": int(len(its_frame))},
        "estimate": {
            "level": _interval(its_fit.level),
            "slope": _interval(its_fit.slope),
            "cumulative": _interval(its_fit.cumulative),
            "placebo_share": placebo.share,
            "cov": "HAC",
        },
    }

    rdd_frame, rdd_truth = simulate_rdd(SEED)
    rdd_fit = local_linear(rdd_frame, float(rdd_truth["cutoff"]), RDD_H)
    bands = bandwidth_sensitivity(rdd_frame, float(rdd_truth["cutoff"]), BANDWIDTHS)
    density = density_check(rdd_frame["spend_prior"], float(rdd_truth["cutoff"]))
    tenure = covariate_jump(rdd_frame, float(rdd_truth["cutoff"]), RDD_H)
    placebos = placebo_cutoffs(rdd_frame, float(rdd_truth["cutoff"]), PLACEBO_POINTS, RDD_H)
    covered = (placebos["ci_low"] <= 0) & (placebos["ci_high"] >= 0)
    bunched_frame, bunched_truth = simulate_rdd(SEED, bunching_share=0.3)
    bunched_density = density_check(bunched_frame["spend_prior"], float(bunched_truth["cutoff"]))
    rdd_payload = {
        "data": "simulated",
        "truth": {**rdd_truth, "tenure_jump": 0.0, "density_jump": 0.0},
        "estimate": {
            "tau": {
                "estimate": rdd_fit.tau,
                "se": rdd_fit.se,
                "low": rdd_fit.ci_low,
                "high": rdd_fit.ci_high,
                "h": rdd_fit.h,
                "n_left": rdd_fit.n_left,
                "n_right": rdd_fit.n_right,
            },
            "bandwidths": bands.to_dict(orient="records"),
            "density": _density(density),
            "tenure": {
                "estimate": tenure.tau,
                "se": tenure.se,
                "low": tenure.ci_low,
                "high": tenure.ci_high,
            },
            "placebo_zero_share": float(covered.mean()),
        },
        "bunching_check": {
            "data": "simulated",
            "truth": {
                "tau": bunched_truth["tau"],
                "cutoff": bunched_truth["cutoff"],
                "bunching_share": bunched_truth["bunching_share"],
            },
            "estimate": _density(bunched_density),
        },
    }

    (output / "its.json").write_text(json.dumps(its_payload, indent=2) + "\n", encoding="utf-8")
    (output / "rdd.json").write_text(json.dumps(rdd_payload, indent=2) + "\n", encoding="utf-8")
    (output / "memo.txt").write_text(_memo(its_payload, rdd_payload), encoding="utf-8")
    weeks, actual, counterfactual = observed_and_counterfactual(its_frame, int(its_truth["t0"]))
    _write_its_plot(weeks, actual, counterfactual, int(its_truth["t0"]), output / "its.png")
    _write_rdd_plot(rdd_frame, float(rdd_truth["cutoff"]), RDD_H, output / "rdd.png")


if __name__ == "__main__":
    main()
