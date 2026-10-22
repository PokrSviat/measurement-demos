import json
import math
from pathlib import Path

import pytest

from reconcile.decision import reconcile
from reconcile.dgp import (
    INCREMENT_PER_GEO_WEEK,
    MAX_COST_PER_INCREMENTAL,
    N_TEST_GEO,
    SEED,
    TEST_WEEKS,
    experiment_cost,
    generate,
    omit_tactic,
)
from reconcile.methods import (
    MethodEstimate,
    ols_incremental,
    geo_incremental,
    markov_incremental,
    removal_share,
)
from reconcile.operating import TRUE_COST_GRID, increment_for_cost, operating_curve


def _budget(geo, mmm, mta, test_cost: float = 12.0):
    return reconcile(
        geo,
        mmm,
        mta,
        test_cost=test_cost,
        max_cost_per_incremental=MAX_COST_PER_INCREMENTAL,
    )


def _expect_experiment_budget(decision, geo, test_cost: float = 12.0) -> None:
    assert not decision.agree
    assert decision.driver == "geo_did"
    assert "The experiment drives the go or no-go." in decision.decision
    assert "The models stay diagnostic." in decision.decision
    assert "The intervals agree" not in decision.decision
    if geo.low <= 0 or test_cost / geo.low > MAX_COST_PER_INCREMENTAL:
        assert "Do not scale." in decision.decision
    else:
        assert "Scale." in decision.decision
        assert f"{test_cost / geo.low:.4f}" in decision.decision


def test_the_treated_geos_are_a_seeded_draw() -> None:
    first = generate()
    assert generate().treated_locations == first.treated_locations
    assert len(first.treated_locations) == N_TEST_GEO
    assert first.treated_locations == sorted(first.treated_locations)
    assert any(
        generate(seed=SEED + offset).treated_locations != first.treated_locations
        for offset in (1, 2, 3)
    )


def test_exact_randomization_interval_covers_a_noiseless_effect() -> None:
    world = generate(seed=1, noise_sd=0.0)
    geo = geo_incremental(world)
    expected = world.true_incremental
    assert geo.point == pytest.approx(expected)
    assert geo.low <= expected <= geo.high


def test_panel_methods_recover_the_known_increment_and_markov_does_not() -> None:
    world = generate()
    expected = INCREMENT_PER_GEO_WEEK * N_TEST_GEO * TEST_WEEKS
    geo = geo_incremental(world)
    mmm = ols_incremental(world)
    mta = markov_incremental(world.paths)
    cost = experiment_cost(world)
    assert mmm.method == "ols"
    assert cost == pytest.approx(expected / INCREMENT_PER_GEO_WEEK)
    assert geo.high - geo.low > expected
    assert mta.point > expected * 2
    assert mta.point != pytest.approx(expected, rel=0.2)
    assert math.isfinite(geo.low) and math.isfinite(geo.high)
    assert geo.low < geo.high
    assert mta.low == mta.high
    decision = _budget(geo, mmm, mta, cost)
    _expect_experiment_budget(decision, geo, cost)
    assert decision.test_cost == pytest.approx(cost)
    assert decision.max_cost_per_incremental == pytest.approx(MAX_COST_PER_INCREMENTAL)


def test_a_share_cannot_be_reconciled_as_incremental_conversions() -> None:
    world = generate()
    share = removal_share(world.paths)
    assert share.estimand == "removal_share"
    with pytest.raises(ValueError, match="incremental_conversions"):
        _budget(geo_incremental(world), ols_incremental(world), share)


def test_dropping_the_tactic_from_the_path_log_leaves_the_experiment_in_charge() -> None:
    world = generate()
    geo = geo_incremental(world)
    mmm = ols_incremental(world)
    logged = markov_incremental(omit_tactic(world.paths))
    assert logged.point == pytest.approx(0.0)
    assert logged.low == logged.high
    decision = _budget(geo, mmm, logged, experiment_cost(world))
    _expect_experiment_budget(decision, geo, experiment_cost(world))


def test_disagreement_leaves_the_budget_with_the_experiment() -> None:
    world = generate()
    geo = geo_incremental(world)
    mmm = ols_incremental(world)
    far = MethodEstimate("markov_mta", "incremental_conversions", 0.0, 0.0, 1.0)
    decision = _budget(geo, mmm, far, experiment_cost(world))
    _expect_experiment_budget(decision, geo, experiment_cost(world))
    assert "Cost per incremental conversion" in decision.decision


def test_agreement_does_not_scale_when_the_experiment_includes_zero() -> None:
    geo = MethodEstimate("geo_did", "incremental_conversions", 1.0, -1.0, 2.0)
    mmm = MethodEstimate("ols", "incremental_conversions", 1.0, -1.0, 2.0)
    mta = MethodEstimate("markov_mta", "incremental_conversions", 1.0, 0.5, 1.5)
    decision = _budget(geo, mmm, mta)
    assert decision.agree
    assert decision.cost_per_incremental is None
    assert decision.decision.startswith("The intervals agree. Do not scale.")
    assert "not strictly positive" in decision.decision


def test_a_positive_lower_bound_above_the_cap_does_not_scale() -> None:
    geo = MethodEstimate("geo_did", "incremental_conversions", 100.0, 10.0, 200.0)
    mmm = MethodEstimate("ols", "incremental_conversions", 100.0, 10.0, 200.0)
    mta = MethodEstimate("markov_mta", "incremental_conversions", 100.0, 50.0, 150.0)
    decision = _budget(geo, mmm, mta, test_cost=12.0)
    assert decision.agree
    assert decision.cost_per_incremental == pytest.approx(1.2)
    assert decision.decision.startswith("The intervals agree. Do not scale.")
    assert "1.2000" in decision.decision
    assert "above the cap" in decision.decision


def test_a_positive_lower_bound_inside_the_cap_scales() -> None:
    geo = MethodEstimate("geo_did", "incremental_conversions", 240.0, 200.0, 280.0)
    mmm = MethodEstimate("ols", "incremental_conversions", 240.0, 200.0, 280.0)
    mta = MethodEstimate("markov_mta", "incremental_conversions", 240.0, 200.0, 280.0)
    decision = _budget(geo, mmm, mta, test_cost=12.0)
    assert decision.agree
    assert decision.cost_per_incremental == pytest.approx(0.06)
    assert decision.decision.startswith("The intervals agree. Scale.")
    assert "0.0600" in decision.decision


def test_skipping_the_path_log_leaves_the_panel_unchanged() -> None:
    full = generate(seed=5)
    panel_only = generate(seed=5, include_paths=False)
    assert panel_only.paths == []
    assert panel_only.panel.equals(full.panel)


def test_the_planted_increment_sets_the_true_cost() -> None:
    world = generate(seed=5, increment=increment_for_cost(0.05, 12.0), include_paths=False)
    assert experiment_cost(world) / world.true_incremental == pytest.approx(0.05)


def test_scale_share_does_not_rise_with_the_true_cost() -> None:
    curve = operating_curve(grid=(0.02, 0.05, 0.2), n_seeds=20, first_seed=1)
    shares = [point.scale_share for point in curve]
    assert shares == sorted(shares, reverse=True)
    assert shares[-1] <= 0.1


def test_published_operating_curve_never_scales_far_above_the_cap() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "results" / "reconciliation.json").read_text(encoding="utf-8"))
    curve = payload["operating_characteristics"]
    assert [point["true_cost_per_incremental"] for point in curve] == list(TRUE_COST_GRID)
    assert all(point["n_seeds"] == 300 for point in curve)
    above = [point["scale_share"] for point in curve if point["true_cost_per_incremental"] > MAX_COST_PER_INCREMENTAL]
    assert max(above) <= 0.05
    assert "Operating characteristics" in payload["memo"]
    assert "Coverage is" in payload["memo"] and "same" in payload["memo"]
    assert "standard errors from 95%" in payload["memo"]
    pdf = root / "results" / "memo.pdf"
    assert pdf.is_file() and pdf.read_bytes().startswith(b"%PDF")
    assert (root / "results" / "memo.txt").read_text(encoding="utf-8") == payload["memo"]


def test_a_markov_point_inside_the_experiment_interval_is_not_agreement() -> None:
    geo = MethodEstimate("geo_did", "incremental_conversions", 240.0, 200.0, 280.0)
    mmm = MethodEstimate("ols", "incremental_conversions", 240.0, 200.0, 280.0)
    mta = MethodEstimate("markov_mta", "incremental_conversions", 240.0, 240.0, 240.0)
    decision = _budget(geo, mmm, mta, test_cost=12.0)
    assert not decision.agree
    assert "Scale." in decision.decision
    assert "diagnostic" in decision.decision
