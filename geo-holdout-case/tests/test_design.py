import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from geo_holdout.blind import BlindCell, blind_cell, calibrate, synthetic_panel
from geo_holdout.decision import BudgetDecision
from geo_holdout.design import assert_stratified_assignment, pseudo_experiment, volume_strata
from geo_holdout.estimators import EffectEstimate, control_theta, effect_profile, randomization_bounds
from geo_holdout.memo import Window, memo_text, shape_warning
from geo_holdout.rule import choose_shape, decision_interval
from geo_holdout.tail import SlopeTest, StratumEffect, slope_test, stratum_effect
from geo_holdout.validation import ValidationCell, daily_effect_for_cost, mc_se, planted_effect, validate_cell

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data" / "sales_and_cost.csv"
PLAN = ROOT / "config" / "analysis_plan.yaml"


def _pretest() -> pd.DataFrame:
    panel = pd.read_csv(PANEL, parse_dates=["date"], dtype={"location": str})
    return panel.loc[panel["date"] < "2015-02-16"]


def test_published_assignment_matches_volume_strata() -> None:
    pretest = _pretest()
    strata = volume_strata(pretest, 10)
    assignment = pd.read_csv(ROOT / "data" / "geo_assignment.csv", dtype={"location": str})
    assert_stratified_assignment(assignment, strata, treated_group=2, treated_per_stratum=5)


def test_effect_for_a_true_cost_puts_the_truth_on_that_cost() -> None:
    effect = daily_effect_for_cost(50000, 0.4, 1400)
    assert 50000 / (effect * 1400) == pytest.approx(0.4)
    with pytest.raises(ValueError):
        daily_effect_for_cost(50000, 0.0, 1400)
    assert mc_se(0.5, 100) == pytest.approx(0.05)


def test_proportional_effect_keeps_the_treated_mean() -> None:
    pre = np.array([1.0, 2.0, 3.0, 10.0])
    treated = np.array([True, False, True, False])
    effect = planted_effect(pre, treated, "proportional", 5.0)
    assert effect[treated].mean() == pytest.approx(5.0)
    assert effect[3] > effect[0]


def test_placebo_without_an_effect_never_scales() -> None:
    pseudo = pseudo_experiment(_pretest(), stratum_size=10, duration_days=28)
    cell = validate_cell(
        pseudo,
        shape="additive",
        true_cost=None,
        treated_per_stratum=5,
        test_cost=50000,
        max_cost_per_incremental=0.5,
        shape_alpha=0.05,
        n_slope_reassignments=200,
        n_draws=12,
        n_reassignments=200,
        seed=7,
    )
    assert cell.scale_share == 0.0
    assert cell.correct_share == 1.0
    assert cell.coverage >= 0.75


def test_a_cheap_tactic_scales_on_the_pre_test() -> None:
    pseudo = pseudo_experiment(_pretest(), stratum_size=10, duration_days=28)
    cell = validate_cell(
        pseudo,
        shape="additive",
        true_cost=0.1,
        treated_per_stratum=5,
        test_cost=50000,
        max_cost_per_incremental=0.5,
        shape_alpha=0.05,
        n_slope_reassignments=200,
        n_draws=8,
        n_reassignments=200,
        seed=8,
    )
    assert cell.scale_share == 1.0


def test_synthetic_panels_keep_the_calibrated_shape() -> None:
    pseudo = pseudo_experiment(_pretest(), stratum_size=10, duration_days=28)
    model = calibrate(pseudo, stratum_size=10)
    panel = synthetic_panel(model, np.random.default_rng(1))
    assert len(panel.pre) == 100
    assert sorted(np.bincount(panel.stratum_ids)) == [10] * 10
    for stratum in range(9):
        assert panel.pre[panel.stratum_ids == stratum].max() < panel.pre[panel.stratum_ids == stratum + 1].min()
    cell = blind_cell(
        model,
        shape="additive",
        true_cost=0.1,
        treated_per_stratum=5,
        test_cost=50000,
        max_cost_per_incremental=0.5,
        shape_alpha=0.05,
        n_slope_reassignments=200,
        n_panels=4,
        n_reassignments=200,
        seed=3,
    )
    assert cell.scale_share == 1.0 and cell.correct_share == 1.0


def test_stratum_interval_covers_a_planted_effect() -> None:
    rng = np.random.default_rng(4)
    values = rng.normal(0, 1, 10)
    treated = np.zeros(10, dtype=bool)
    treated[:5] = True
    effect = stratum_effect(values + 20 * treated, treated, stratum=0, median_pre=1.0)
    assert effect.n_assignments == 252
    assert effect.low < 20 < effect.high
    assert effect.low > 0


def test_slope_test_finds_an_effect_that_grows_with_size() -> None:
    rng = np.random.default_rng(5)
    pre = np.exp(rng.normal(5, 1.2, 100))
    strata = np.argsort(np.argsort(pre)) // 10
    treated = np.zeros(100, dtype=bool)
    for stratum in range(10):
        treated[rng.choice(np.flatnonzero(strata == stratum), 5, replace=False)] = True
    noise = rng.normal(0, 1, 100)
    flat = slope_test(pre, noise + 10 * treated, treated, strata, theta=0.0, n_reassignments=400, seed=1)
    grows = slope_test(pre, noise + 10 * np.log(pre) * treated, treated, strata, theta=0.0, n_reassignments=400, seed=1)
    assert grows.p_value < 0.01 < flat.p_value
    assert grows.slope > 5


def test_published_validation_and_blind_check_cover_the_plan_grid() -> None:
    plan = yaml.safe_load(PLAN.read_text(encoding="utf-8"))
    costs = [float(cost) for cost in plan["validation"]["true_costs"]]
    validation = json.loads((ROOT / "results" / "validation.json").read_text(encoding="utf-8"))
    blind = json.loads((ROOT / "results" / "blind_check.json").read_text(encoding="utf-8"))
    assert {cell["true_cost"] for cell in validation["cells"]} == {None, *costs}
    assert {cell["true_cost"] for cell in blind["cells"]} == set(costs)
    assert blind["seed"] == plan["blind"]["seed"]
    for cell in validation["cells"]:
        assert cell["coverage"] >= 0.9
        if cell["true_cost"] is None or cell["true_cost"] > 0.5:
            assert cell["scale_share"] <= 0.05
    proportional = [cell for cell in validation["cells"] if cell["shape"] == "proportional" and cell["true_cost"]]
    assert all(abs(cell["weighted_mean_bias"]) > abs(cell["mean_bias"]) for cell in proportional)


def _estimate(name: str) -> EffectEstimate:
    return EffectEstimate(
        name=name,
        att_daily=1,
        se_daily=0.1,
        ci_low=0.8,
        ci_high=1.2,
        incremental=30,
        incremental_low=24,
        incremental_high=36,
        n_treated=50,
        n_control=50,
        n_post_days=51,
        theta=0.2,
        treated_location_days=2550,
    )


def test_memo_states_the_rule_and_avoids_employer_claims() -> None:
    estimate = _estimate("cuped")
    text = memo_text(
        question="Should this tactic be scaled after the geo experiment?",
        estimand="Cumulative incremental sales.",
        plan_version=5,
        plan_sha256="abcdef0123456789",
        n_treated=50,
        stratum_size=10,
        treated_per_stratum=5,
        cumulative=Window("2015-02-16", "2015-04-07", estimate, (20.0, 40.0)),
        decision_shape="proportional",
        shape_alpha=0.05,
        additive_interval=(5.0, 60.0),
        test_window=Window("2015-02-16", "2015-03-15", estimate, (15.0, 35.0)),
        after_test=Window("2015-03-16", "2015-04-07", estimate, (-5.0, 7.0)),
        did=_estimate("did"),
        control_correlation=0.8,
        missing_treated_location_days=0,
        weighted_total=120.0,
        median_pre=140,
        max_pre=5300,
        top5_pretest_share=0.4,
        without_largest=estimate,
        without_largest_interval=(10.0, 20.0),
        strata=[StratumEffect(0, 20.0, 3.0, -1.0, 7.0, 252)],
        slope=SlopeTest(12.0, 0.04, 2000),
        validation=[
            ValidationCell("additive", None, 0.0, 300, 0.95, -20, 600, 0.0, 1.0, 0.04),
            ValidationCell("additive", 0.5, 71.0, 300, 0.95, -20, 600, 0.03, 0.03, 0.05),
        ],
        validation_reassignments=500,
        pseudo_window_days=28,
        pseudo_pre_days=14,
        blind=[BlindCell("proportional", 0.25, 200, 0.9, 0.9, 0.97)],
        blind_reassignments=500,
        decision=BudgetDecision(False, "Do not scale.", 50000 / 20, "Above the cap.", False, True),
        test_cost=50000,
        max_cost=0.5,
        assumption_note="Planning assumption for this public case. Not an employer budget.",
    )
    assert f"{50000 / 20:.2f}" in text
    assert "abcdef012345" in text
    assert "4.0 times the decision total" in text
    assert "for reading only" in text
    assert "at the cap either call is defensible" in text
    assert "Monte Carlo standard error" in text
    assert "252 assignments" in text
    assert "p-value 0.040" in text
    assert "3831ec4" in text and "Version 4" in text and "Version 5" in text
    assert "cumulative total" in text
    assert "spread proportional to pre-period sales" in text
    assert "fixed after exploration" in text
    assert "for reading: 5 to 60" in text
    assert "proportional interval chosen in 97%" in text
    lowered = text.lower()
    for forbidden in ("pre-registered", "m.video", "yandex", "hoff", "okkam", "apple", "synthetic control"):
        assert forbidden not in lowered
    assert "Cap 0.5" in text


def test_shape_warning_reports_misses_under_a_growing_effect() -> None:
    validation = [
        ValidationCell("proportional", 0.25, 30.0, 300, 1.0, -2, 600, 0.99, 0.99, 1.0),
        ValidationCell("proportional", 0.4, 20.0, 300, 1.0, -2, 600, 0.0, 0.0, 1.0),
        ValidationCell("proportional", 0.6, 10.0, 300, 1.0, -2, 600, 0.0, 1.0, 1.0),
    ]
    blind = [BlindCell("proportional", 0.4, 200, 0.02, 0.02, 1.0)]
    text = shape_warning(SlopeTest(60.0, 0.0005, 2000), validation, blind, 0.5)
    assert "true cost 0.40" in text
    assert "0% of pre-test draws and 2% of blind panels" in text
    assert "does not rule out a tactic within the cap" in text
    assert shape_warning(SlopeTest(60.0, 0.2, 2000), validation, blind, 0.5) == ""
    assert shape_warning(SlopeTest(-60.0, 0.001, 2000), validation, blind, 0.5) == ""
    powered = [replace(cell, scale_share=0.9) for cell in validation]
    assert shape_warning(SlopeTest(60.0, 0.0005, 2000), powered, blind, 0.5) == ""


def _growing_panel(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pre = np.exp(rng.normal(5, 1.0, 100))
    strata = np.argsort(np.argsort(pre)) // 10
    treated = np.zeros(100, dtype=bool)
    for stratum in range(10):
        treated[rng.choice(np.flatnonzero(strata == stratum), 5, replace=False)] = True
    delta = 0.05 * pre * rng.normal(0, 1, 100)
    return pre, delta, np.full(100, 28.0), treated, strata


def test_proportional_interval_covers_a_proportional_effect_and_is_narrower() -> None:
    pre, delta, days, treated, strata = _growing_panel(11)
    effect = 0.3 * pre
    injected = delta + effect * treated
    truth = float(np.sum(effect[treated] * days[treated]))
    common = dict(treated_per_stratum=5, n_simulations=300, seed=2)
    low_p, high_p = randomization_bounds(pre, injected, days, treated, strata, effect_shape="proportional", **common)
    low_a, high_a = randomization_bounds(pre, injected, days, treated, strata, effect_shape="additive", **common)
    assert low_p <= truth <= high_p
    assert high_p - low_p < high_a - low_a
    with pytest.raises(ValueError, match="positive"):
        effect_profile(np.array([1.0, 0.0]), "proportional")


def test_decision_interval_switches_on_the_slope_test() -> None:
    pre, delta, days, treated, strata = _growing_panel(12)
    common = dict(
        treated_per_stratum=5,
        alpha=0.05,
        n_slope_reassignments=300,
        n_interval_reassignments=200,
        slope_seed=1,
        interval_seed=2,
    )
    grows = decision_interval(pre, delta + 0.3 * pre * treated, days, treated, strata, **common)
    flat = decision_interval(pre, delta + 5.0 * treated, days, treated, strata, **common)
    assert grows.shape == "proportional" and grows.slope.p_value < 0.05
    assert flat.shape == "additive"
    assert choose_shape(SlopeTest(1.0, 0.049, 100), 0.05) == "proportional"
    assert choose_shape(SlopeTest(1.0, 0.05, 100), 0.05) == "additive"
    assert control_theta(pre, 2 * pre, treated) == pytest.approx(2.0)


def test_published_case_uses_the_v5_rule() -> None:
    plan = yaml.safe_load(PLAN.read_text(encoding="utf-8"))
    result = json.loads((ROOT / "results" / "case_result.json").read_text(encoding="utf-8"))
    validation = json.loads((ROOT / "results" / "validation.json").read_text(encoding="utf-8"))
    text = (ROOT / "results" / "memo.txt").read_text(encoding="utf-8")
    pdf = ROOT / "results" / "memo.pdf"
    assert pdf.is_file() and pdf.read_bytes().startswith(b"%PDF")
    assert result["analysis_plan_version"] == plan["version"] == 5
    assert result["decision"]["recommendation"] == "Hold. Confirm on a new panel."
    assert result["decision"]["clears_cap"] is True
    assert "Hold. Confirm on a new panel." in text
    assert "Scale the tactic." not in text
    expected = "proportional" if result["slope_test"]["p_value"] < plan["rule"]["shape_switch"]["alpha"] else "additive"
    assert result["decision_interval_shape"] == expected
    assert "(-0 " not in text
    below = [
        cell
        for cell in validation["cells"]
        if cell["shape"] == "proportional" and cell["true_cost"] and cell["true_cost"] < plan["rule"]["max_cost_per_incremental"]
    ]
    misses = max(below, key=lambda cell: cell["true_cost"])["scale_share"] < 0.8
    warned = "does not rule out a tactic within the cap" in text
    assert warned == (misses and result["slope_test"]["p_value"] < 0.05 and not result["decision"]["clears_cap"])


def test_search_lift_yaml_matches_the_weekly_export() -> None:
    from geo_holdout.lift_design import CONFIG, plant_lift, spend_to_revenue, shortest_powered

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    export = Path(__file__).resolve().parents[2] / "measurement-framework" / "artifacts" / "pymc_export.json"
    weekly = pd.read_csv(Path(__file__).resolve().parents[2] / "measurement-framework" / "data" / "dt_simulated_weekly.csv")
    payload = json.loads(export.read_text(encoding="utf-8"))
    assert config["channel"] == "search"
    assert config["point_roi"] == pytest.approx(payload["channel_roi"]["search"])
    assert spend_to_revenue(weekly, config["spend_column"], config["holdout_weeks"]) == pytest.approx(
        config["spend_to_revenue"]
    )
    effect = plant_lift(np.array([10.0, 20.0]), roi=2.0, increase=0.5, ratio=0.1)
    assert effect == pytest.approx([1.0, 2.0])
    chosen = shortest_powered(
        {
            "cells": [
                {"duration_days": 56, "increase": 0.5, "power_at_point_roi": 0.9},
                {"duration_days": 28, "increase": 2.0, "power_at_point_roi": 0.85},
                {"duration_days": 28, "increase": 0.5, "power_at_point_roi": 0.4},
            ]
        }
    )
    assert chosen is not None and chosen["duration_days"] == 28 and chosen["increase"] == 2.0


def test_published_search_lift_is_the_planning_proxy() -> None:
    path = ROOT / "results" / "search_lift_design.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = yaml.safe_load(PLAN.parent.joinpath("search_lift.yaml").read_text(encoding="utf-8"))
    assert payload["note"].startswith("planning proxy")
    assert payload["channel"] == config["channel"]
    assert payload["point_roi"] == pytest.approx(config["point_roi"])
    assert {cell["duration_days"] for cell in payload["cells"]} == set(config["durations_days"])


def test_no_source_or_result_calls_the_rule_pre_registered() -> None:
    paths = [*ROOT.joinpath("src").rglob("*.py"), ROOT / "README.md", ROOT / "results" / "memo.txt"]
    for path in paths:
        assert "pre-registered" not in path.read_text(encoding="utf-8").lower(), path
