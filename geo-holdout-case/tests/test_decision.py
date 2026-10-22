import json
from pathlib import Path

import pytest
import yaml

from geo_holdout.decision import decide
from geo_holdout.estimators import EffectEstimate

ROOT = Path(__file__).resolve().parents[1]


def _estimate(low: float, point: float, high: float) -> EffectEstimate:
    return EffectEstimate(
        name="cuped",
        att_daily=point,
        se_daily=0.0,
        ci_low=low,
        ci_high=high,
        incremental=point,
        incremental_low=low,
        incremental_high=high,
        n_treated=2,
        n_control=10,
        n_post_days=15,
    )


def test_scale_when_the_cap_clears_at_the_cumulative_lower_bound() -> None:
    decision = decide(_estimate(100, 200, 300), test_cost=1000, max_cost_per_incremental=10)
    assert decision.fund
    assert decision.clears_cap
    assert decision.confirmatory
    assert decision.recommendation == "Scale the tactic."
    assert decision.cost_per_incremental == 10


def test_hold_when_the_cap_clears_on_a_post_exploration_plan() -> None:
    decision = decide(
        _estimate(100, 200, 300),
        test_cost=1000,
        max_cost_per_incremental=10,
        confirmatory=False,
    )
    assert not decision.fund
    assert decision.clears_cap
    assert not decision.confirmatory
    assert decision.recommendation == "Hold. Confirm on a new panel."
    assert "not a confirmatory scale" in decision.reason


def test_do_not_fund_when_the_interval_includes_zero() -> None:
    decision = decide(_estimate(-10, 50, 80), test_cost=100, max_cost_per_incremental=10)
    assert not decision.fund
    assert decision.cost_per_incremental is None
    assert "includes zero" in decision.reason


def test_do_not_fund_when_the_cost_exceeds_the_cap() -> None:
    decision = decide(_estimate(10, 20, 30), test_cost=500, max_cost_per_incremental=10)
    assert not decision.fund
    assert decision.cost_per_incremental == 50
    assert "above the cap" in decision.reason


def test_point_inside_the_cap_does_not_fund_when_the_lower_bound_does_not() -> None:
    decision = decide(_estimate(10, 20, 30), test_cost=200, max_cost_per_incremental=10)
    assert not decision.fund
    assert decision.cost_per_incremental == 20


def test_cooldown_and_power_are_not_decision_inputs() -> None:
    for extra in ({"cooldown_high": 1.0}, {"planned_power": 0.1}):
        with pytest.raises(TypeError):
            decide(_estimate(100, 200, 300), test_cost=1000, max_cost_per_incremental=10, **extra)  # type: ignore[arg-type]


def test_published_result_follows_the_frozen_rule() -> None:
    payload = json.loads((ROOT / "results" / "case_result.json").read_text(encoding="utf-8"))
    plan = yaml.safe_load((ROOT / "config" / "analysis_plan.yaml").read_text(encoding="utf-8"))
    low = float(payload["randomization_interval"]["low"])
    decision = payload["decision"]
    cap = float(plan["rule"]["max_cost_per_incremental"])
    assert cap == 0.5
    assert payload["window"] == {
        "start": plan["window"]["test_start"],
        "test_end": plan["window"]["test_end"],
        "end": plan["window"]["end"],
    }
    clears = low > 0 and float(payload["test_cost"]) / low <= cap
    assert decision["clears_cap"] is clears
    assert decision["confirmatory"] is False
    assert decision["fund"] is False
    assert decision["recommendation"] == "Hold. Confirm on a new panel."
    if low <= 0:
        assert decision["cost_per_incremental"] is None
    else:
        assert decision["cost_per_incremental"] == pytest.approx(float(payload["test_cost"]) / low)


def test_the_published_run_used_the_committed_plan() -> None:
    from geo_holdout.run_case import plan_sha256

    payload = json.loads((ROOT / "results" / "case_result.json").read_text(encoding="utf-8"))
    assert payload["analysis_plan_sha256"] == plan_sha256(ROOT / "config" / "analysis_plan.yaml")
    assert payload["analysis_plan_sha256"][:12] in payload["memo"]


def test_the_cumulative_point_is_the_unweighted_cuped_total_over_both_parts() -> None:
    payload = json.loads((ROOT / "results" / "case_result.json").read_text(encoding="utf-8"))
    low = float(payload["randomization_interval"]["low"])
    high = float(payload["randomization_interval"]["high"])
    point = float(payload["cuped"]["incremental"])
    assert low < point < high
    assert float(payload["weighted_total_descriptive"]) > high
    parts = payload["test_window"]["cuped"]["treated_location_days"] + payload["after_test"]["cuped"]["treated_location_days"]
    assert payload["cuped"]["treated_location_days"] == parts
