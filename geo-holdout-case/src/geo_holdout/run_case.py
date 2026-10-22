"""Run the public geo experiment under the frozen analysis plan and write JSON plus a one-page PDF."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from geo_holdout.blind import BlindCell, PanelModel, run_blind_check
from geo_holdout.decision import decide
from geo_holdout.design import assert_stratified_assignment, pseudo_experiment, volume_strata
from geo_holdout.diagnostics import (
    control_pre_delta_correlation,
    cooldown_window,
    daily_group_means,
    highest_cost_geos,
    write_path_chart,
)
from geo_holdout.estimators import (
    EffectEstimate,
    cuped_difference_in_differences,
    difference_in_differences,
    location_window_means,
    pretest_weighted_total,
    randomization_interval,
)
from geo_holdout.memo import Window, memo_text, write_memo_pdf
from geo_holdout.rule import decision_interval
from geo_holdout.tail import stratum_effects
from geo_holdout.validation import ValidationCell, run_validation

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CONFIG = ROOT / "config" / "case.yaml"
PLAN = ROOT / "config" / "analysis_plan.yaml"
OUTPUT = ROOT / "results"


def _top5_pretest_share(pretest: pd.DataFrame) -> float:
    totals = pretest.groupby("location")["Y"].sum()
    if float(totals.sum()) <= 0:
        raise ValueError("Pre-test sales must be positive.")
    return float(totals.nlargest(5).sum() / totals.sum())


def _bounds(bounds: tuple[float, float]) -> dict[str, float]:
    return {"low": bounds[0], "high": bounds[1]}


def _stratum_ids(means: pd.DataFrame, strata: pd.DataFrame) -> np.ndarray:
    ids = means["location"].map(strata.set_index("location")["stratum"])
    if ids.isna().any():
        raise ValueError("A geo in the experiment window has no volume stratum.")
    return ids.astype(int).to_numpy()


def _yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def plan_sha256(path: Path = PLAN) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _window_cost(panel: pd.DataFrame, locations: list[str], *, start: str, end: str) -> float:
    mask = (
        panel["location"].isin(locations)
        & (panel["date"] >= pd.Timestamp(start))
        & (panel["date"] <= pd.Timestamp(end))
    )
    return float(panel.loc[mask, "cost"].sum())


def _read_window(
    panel: pd.DataFrame,
    treated: list[str],
    strata: pd.DataFrame,
    *,
    pre_end: str,
    start: str,
    end: str,
    treated_per_stratum: int,
    n_reassignments: int,
    seed: int,
    effect_shape: str = "additive",
) -> tuple[pd.DataFrame, Window]:
    means = location_window_means(panel, pre_end=pre_end, post_start=start, post_end=end)
    estimate = cuped_difference_in_differences(means, treated)
    interval = randomization_interval(
        means,
        treated,
        _stratum_ids(means, strata),
        treated_per_stratum=treated_per_stratum,
        effect_shape=effect_shape,
        n_simulations=n_reassignments,
        seed=seed,
    )
    return means, Window(start, end, estimate, interval)


def main() -> None:
    config = _yaml(CONFIG)
    plan = _yaml(PLAN)
    sha = plan_sha256()
    panel = pd.read_csv(DATA / "sales_and_cost.csv", parse_dates=["date"], dtype={"location": str})
    assignment = pd.read_csv(DATA / "geo_assignment.csv", dtype={"location": str})
    protocol = config["assignment"]
    if protocol["mechanism"] != "stratified_fixed_count" or protocol["volume"] != "pretest_sales":
        raise ValueError("The case only runs the stratified pretest-sales assignment.")
    stratum_size = int(protocol["stratum_size"])
    treated_per_stratum = int(protocol["treated_per_stratum"])
    treated_group = int(protocol["treated_group"])
    treated = assignment.loc[assignment["group"] == treated_group, "location"].tolist()
    control = assignment.loc[assignment["group"] != treated_group, "location"].tolist()
    if len(treated) != int(protocol["n_markets"]):
        raise ValueError("The plan and the assignment must use the same number of test geos.")
    if len(control) < 2:
        raise ValueError("Need at least two control geos.")

    window = plan["window"]
    start, test_end, end = window["test_start"], window["test_end"], window["end"]
    after_start, panel_end = cooldown_window(panel, test_end)
    if end != panel_end or pd.Timestamp(start) > pd.Timestamp(test_end):
        raise ValueError("The plan window must start before the test ends and end on the last panel day.")
    pre_end = (pd.Timestamp(start) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    pretest = panel.loc[panel["date"] <= pd.Timestamp(pre_end)].copy()
    strata = volume_strata(pretest, stratum_size)
    assert_stratified_assignment(assignment, strata, treated_group=treated_group, treated_per_stratum=treated_per_stratum)

    interval_plan = plan["interval"]
    reassignments = int(interval_plan["n_reassignments"])
    shared = {
        "pre_end": pre_end,
        "treated_per_stratum": treated_per_stratum,
        "n_reassignments": reassignments,
    }
    means, additive_cumulative = _read_window(
        panel, treated, strata, start=start, end=end, seed=int(interval_plan["seed"]), **shared
    )
    rule = plan["rule"]
    tail_plan = plan["tail"]
    pre = means["pre"].to_numpy(dtype=float)
    delta = means["delta"].to_numpy(dtype=float)
    treated_mask = means["location"].isin(treated).to_numpy()
    stratum_ids = _stratum_ids(means, strata)
    chosen = decision_interval(
        pre,
        delta,
        means["n_post_days"].to_numpy(dtype=float),
        treated_mask,
        stratum_ids,
        treated_per_stratum=treated_per_stratum,
        alpha=float(rule["shape_switch"]["alpha"]),
        n_slope_reassignments=int(tail_plan["n_reassignments"]),
        n_interval_reassignments=reassignments,
        slope_seed=int(tail_plan["seed"]),
        interval_seed=int(interval_plan["seed"]),
    )
    shape = chosen.shape
    slope = chosen.slope
    cumulative = replace(additive_cumulative, interval=chosen.bounds)
    _, test_window = _read_window(
        panel, treated, strata, start=start, end=test_end, seed=int(interval_plan["test_window_seed"]),
        effect_shape=shape, **shared
    )
    _, after_test = _read_window(
        panel, treated, strata, start=after_start, end=end, seed=int(interval_plan["test_window_seed"]) + 1,
        effect_shape=shape, **shared
    )
    cuped = cumulative.estimate
    did = difference_in_differences(means, treated)

    dropped = highest_cost_geos(panel, treated, start=start, end=test_end)
    kept_means = means.loc[~means["location"].isin(dropped)].reset_index(drop=True)
    kept_treated = [location for location in treated if location not in set(dropped)]
    without_largest = cuped_difference_in_differences(kept_means, kept_treated)
    without_largest_interval = randomization_interval(
        kept_means,
        kept_treated,
        _stratum_ids(kept_means, strata),
        effect_shape=shape,
        n_simulations=reassignments,
        seed=int(interval_plan["without_largest_seed"]),
    )

    max_cost = float(rule["max_cost_per_incremental"])
    test_cost = _window_cost(panel, treated, start=start, end=end) - _window_cost(panel, control, start=start, end=end)
    decision = decide(
        replace(cuped, incremental_low=cumulative.interval[0], incremental_high=cumulative.interval[1]),
        test_cost=test_cost,
        max_cost_per_incremental=max_cost,
        confirmatory=not plan.get("history"),
    )

    test_days = (pd.Timestamp(test_end) - pd.Timestamp(start)).days + 1
    window_days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
    treated_means = means.loc[means["location"].isin(treated)]
    missing_treated = int(window_days * len(treated) - int(treated_means["n_post_days"].sum()))

    checks = plan["validation"]
    validation = run_validation(
        pretest,
        stratum_size=stratum_size,
        treated_per_stratum=treated_per_stratum,
        duration_days=test_days,
        true_costs=[float(cost) for cost in checks["true_costs"]],
        test_cost=test_cost,
        max_cost_per_incremental=max_cost,
        shape_alpha=float(rule["shape_switch"]["alpha"]),
        n_slope_reassignments=int(checks["n_slope_reassignments"]),
        n_draws=int(checks["n_draws"]),
        n_reassignments=int(checks["n_reassignments"]),
        seed=int(checks["seed"]),
    )
    pseudo = pseudo_experiment(pretest, stratum_size=stratum_size, duration_days=test_days)
    blind_plan = plan["blind"]
    panel_model, blind = run_blind_check(
        pseudo,
        stratum_size=stratum_size,
        treated_per_stratum=treated_per_stratum,
        true_costs=[float(cost) for cost in checks["true_costs"]],
        test_cost=test_cost,
        max_cost_per_incremental=max_cost,
        shape_alpha=float(rule["shape_switch"]["alpha"]),
        n_slope_reassignments=int(blind_plan["n_slope_reassignments"]),
        n_panels=int(blind_plan["n_panels"]),
        n_reassignments=int(blind_plan["n_reassignments"]),
        seed=int(blind_plan["seed"]),
    )

    strata_effects = stratum_effects(pre, delta, treated_mask, stratum_ids, theta=float(cuped.theta or 0.0))

    weighted_total = pretest_weighted_total(means, treated)
    control_correlation = control_pre_delta_correlation(means, treated)
    text = memo_text(
        question=config["question"],
        estimand=plan["estimand"],
        plan_version=int(plan["version"]),
        plan_sha256=sha,
        n_treated=len(treated),
        stratum_size=stratum_size,
        treated_per_stratum=treated_per_stratum,
        cumulative=cumulative,
        decision_shape=shape,
        shape_alpha=float(rule["shape_switch"]["alpha"]),
        additive_interval=additive_cumulative.interval,
        test_window=test_window,
        after_test=after_test,
        did=did,
        control_correlation=control_correlation,
        missing_treated_location_days=missing_treated,
        weighted_total=weighted_total,
        median_pre=float(np.median(pre)),
        max_pre=float(pre.max()),
        top5_pretest_share=_top5_pretest_share(pretest),
        without_largest=without_largest,
        without_largest_interval=without_largest_interval,
        strata=strata_effects,
        slope=slope,
        validation=validation,
        validation_reassignments=int(checks["n_reassignments"]),
        pseudo_window_days=test_days,
        pseudo_pre_days=pseudo.n_pre_days,
        blind=blind,
        blind_reassignments=int(blind_plan["n_reassignments"]),
        decision=decision,
        test_cost=test_cost,
        max_cost=max_cost,
        assumption_note=rule["note"],
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_path_chart(
        daily_group_means(panel, treated, control),
        post_start=start,
        post_end=test_end,
        path=OUTPUT / "path.png",
    )
    payload = {
        "author": config["author"],
        "question": config["question"],
        "analysis_plan_version": int(plan["version"]),
        "analysis_plan_sha256": sha,
        "treated_group": treated_group,
        "n_treated": len(treated),
        "n_control": len(control),
        "window": {"start": start, "test_end": test_end, "end": end},
        "test_cost": test_cost,
        "cuped": cuped.as_dict(),
        "decision_interval_shape": shape,
        "randomization_interval": _bounds(cumulative.interval),
        "additive_randomization_interval": _bounds(additive_cumulative.interval),
        "did": did.as_dict(),
        "control_correlation": control_correlation,
        "test_window": _window_payload(test_window),
        "after_test": _window_payload(after_test),
        "missing_treated_location_days": missing_treated,
        "top5_pretest_share": _top5_pretest_share(pretest),
        "weighted_total_descriptive": weighted_total,
        "dropped_geos": dropped,
        "cuped_without_largest": without_largest.as_dict(),
        "without_largest_randomization_interval": _bounds(without_largest_interval),
        "strata": [item.as_dict() for item in strata_effects],
        "slope_test": slope.as_dict(),
        "decision": decision.as_dict(),
        "memo": text,
    }
    (OUTPUT / "case_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_checks(plan, validation, panel_model, blind)
    write_memo_pdf(text, OUTPUT / "memo.pdf", chart_path=OUTPUT / "path.png")
    (OUTPUT / "memo.txt").write_text(text, encoding="utf-8")


def _window_payload(window: Window) -> dict:
    return {
        "start": window.start,
        "end": window.end,
        "cuped": window.estimate.as_dict(),
        "randomization_interval": _bounds(window.interval),
    }


def _write_checks(
    plan: dict,
    validation: list[ValidationCell],
    panel_model: PanelModel,
    blind: list[BlindCell],
) -> None:
    (OUTPUT / "validation.json").write_text(
        json.dumps({**plan["validation"], "cells": [cell.as_dict() for cell in validation]}, indent=2),
        encoding="utf-8",
    )
    (OUTPUT / "blind_check.json").write_text(
        json.dumps(
            {**plan["blind"], "panel_model": dict(panel_model.__dict__), "cells": [cell.as_dict() for cell in blind]},
            indent=2,
        ),
        encoding="utf-8",
    )


def _estimate_from_dict(payload: dict) -> EffectEstimate:
    return EffectEstimate(
        name=str(payload["name"]),
        att_daily=float(payload["att_daily"]),
        se_daily=float(payload["se_daily"]),
        ci_low=float(payload["ci_low"]),
        ci_high=float(payload["ci_high"]),
        incremental=float(payload["incremental"]),
        incremental_low=float(payload["incremental_low"]),
        incremental_high=float(payload["incremental_high"]),
        n_treated=int(payload["n_treated"]),
        n_control=int(payload["n_control"]),
        n_post_days=None if payload["n_post_days"] is None else int(payload["n_post_days"]),
        theta=None if payload["theta"] is None else float(payload["theta"]),
        treated_location_days=(
            None if payload["treated_location_days"] is None else int(payload["treated_location_days"])
        ),
    )


def _window_from_payload(payload: dict) -> Window:
    bounds = payload["randomization_interval"]
    return Window(
        payload["start"],
        payload["end"],
        _estimate_from_dict(payload["cuped"]),
        (float(bounds["low"]), float(bounds["high"])),
    )


def publish_speech() -> None:
    """Rewrite the published decision speech from stored intervals. Does not rerun reassignments."""
    from geo_holdout.tail import SlopeTest, StratumEffect

    config = _yaml(CONFIG)
    plan = _yaml(PLAN)
    payload = json.loads((OUTPUT / "case_result.json").read_text(encoding="utf-8"))
    validation_raw = json.loads((OUTPUT / "validation.json").read_text(encoding="utf-8"))
    blind_raw = json.loads((OUTPUT / "blind_check.json").read_text(encoding="utf-8"))
    cuped = _estimate_from_dict(payload["cuped"])
    bounds = payload["randomization_interval"]
    decision = decide(
        replace(cuped, incremental_low=float(bounds["low"]), incremental_high=float(bounds["high"])),
        test_cost=float(payload["test_cost"]),
        max_cost_per_incremental=float(plan["rule"]["max_cost_per_incremental"]),
        confirmatory=not plan.get("history"),
    )
    window = payload["window"]
    start, test_end, end = window["start"], window["test_end"], window["end"]
    panel = pd.read_csv(DATA / "sales_and_cost.csv", parse_dates=["date"], dtype={"location": str})
    pre_end = (pd.Timestamp(start) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    pretest = panel.loc[panel["date"] <= pd.Timestamp(pre_end)]
    test_days = (pd.Timestamp(test_end) - pd.Timestamp(start)).days + 1
    pseudo = pseudo_experiment(pretest, stratum_size=int(config["assignment"]["stratum_size"]), duration_days=test_days)
    pre = pretest.groupby("location")["Y"].mean()
    validation = [
        ValidationCell(
            cell["shape"],
            cell["true_cost"],
            cell["daily_effect"],
            cell["n_draws"],
            cell["coverage"],
            cell["mean_bias"],
            cell["weighted_mean_bias"],
            cell["scale_share"],
            cell["correct_share"],
            cell["proportional_share"],
        )
        for cell in validation_raw["cells"]
    ]
    blind = [
        BlindCell(
            cell["shape"],
            cell["true_cost"],
            cell["n_panels"],
            cell["scale_share"],
            cell["correct_share"],
            cell["proportional_share"],
        )
        for cell in blind_raw["cells"]
    ]
    additive = payload["additive_randomization_interval"]
    without = payload["without_largest_randomization_interval"]
    slope = payload["slope_test"]
    text = memo_text(
        question=config["question"],
        estimand=plan["estimand"],
        plan_version=int(plan["version"]),
        plan_sha256=str(payload["analysis_plan_sha256"]),
        n_treated=int(payload["n_treated"]),
        stratum_size=int(config["assignment"]["stratum_size"]),
        treated_per_stratum=int(config["assignment"]["treated_per_stratum"]),
        cumulative=Window(start, end, cuped, (float(bounds["low"]), float(bounds["high"]))),
        decision_shape=str(payload["decision_interval_shape"]),
        shape_alpha=float(plan["rule"]["shape_switch"]["alpha"]),
        additive_interval=(float(additive["low"]), float(additive["high"])),
        test_window=_window_from_payload(payload["test_window"]),
        after_test=_window_from_payload(payload["after_test"]),
        did=_estimate_from_dict(payload["did"]),
        control_correlation=float(payload["control_correlation"]),
        missing_treated_location_days=int(payload["missing_treated_location_days"]),
        weighted_total=float(payload["weighted_total_descriptive"]),
        median_pre=float(pre.median()),
        max_pre=float(pre.max()),
        top5_pretest_share=float(payload["top5_pretest_share"]),
        without_largest=_estimate_from_dict(payload["cuped_without_largest"]),
        without_largest_interval=(float(without["low"]), float(without["high"])),
        strata=[StratumEffect(**{key: item[key] for key in StratumEffect.__dataclass_fields__}) for item in payload["strata"]],
        slope=SlopeTest(float(slope["slope"]), float(slope["p_value"]), int(slope["n_reassignments"])),
        validation=validation,
        validation_reassignments=int(validation_raw["n_reassignments"]),
        pseudo_window_days=test_days,
        pseudo_pre_days=pseudo.n_pre_days,
        blind=blind,
        blind_reassignments=int(blind_raw["n_reassignments"]),
        decision=decision,
        test_cost=float(payload["test_cost"]),
        max_cost=float(plan["rule"]["max_cost_per_incremental"]),
        assumption_note=plan["rule"]["note"],
    )
    payload["decision"] = decision.as_dict()
    payload["memo"] = text
    (OUTPUT / "case_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_memo_pdf(text, OUTPUT / "memo.pdf", chart_path=OUTPUT / "path.png")
    (OUTPUT / "memo.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["--speech"]:
        publish_speech()
    else:
        main()
