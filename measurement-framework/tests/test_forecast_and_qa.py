import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from measurement.contract import CALENDAR_TERMS
from measurement.memo import (
    AblationCheck,
    BacktestOrigin,
    LiftPlan,
    PriorCheck,
    RecoveryCheck,
    ResidualCheck,
    forecast_notice,
    geo_lift_sentence,
    lift_plan,
    memo_text,
)
from measurement.model import CONTROLS
from measurement.qa import (
    FitSummary,
    channel_payback,
    chi2_sf,
    contraction,
    gated_payback,
    ljung_box,
    prior_moves,
    prior_sensitivity,
    qa_failures,
    refresh_diff,
    residual_acf,
    sensitivity_is_material,
)
from measurement.recovery import ar_noise, coverage_share, delayed_adstock, true_marginal_roi, true_response
from measurement.sql_layer import add_calendar_terms
from measurement.transforms import (
    ChannelCurve,
    PosteriorDraws,
    apply_planned_spend,
    ar_forecast,
    channel_contribution,
    forecast_errors,
    geometric_adstock,
    marginal_roi_draws,
    mix_prediction,
    naive_holdout_predictions,
    predict_revenue,
    roi_draws,
    saturation,
)


def _curve() -> ChannelCurve:
    return ChannelCurve("search", "search_clicks_P", "search_S", alpha=0.5, lam=0.01, beta=100.0)


def test_adstock_is_causal_and_normalized() -> None:
    pulse = np.array([1.0, 0.0, 0.0, 0.0])
    transformed = geometric_adstock(pulse, alpha=0.5, l_max=4)
    assert transformed[0] > transformed[1] > transformed[2] > 0
    assert transformed.sum() == pytest.approx(1.0)


def test_forecast_backtest_is_exact_when_the_curve_is_the_dgp() -> None:
    curve = _curve()
    frame = pd.DataFrame(
        {
            "search_clicks_P": [10.0, 20.0, 0.0, 30.0],
            "search_S": [5.0, 8.0, 0.0, 9.0],
            "newsletter": [0.0, 0.0, 0.0, 0.0],
        }
    )
    actual = predict_revenue(frame, [curve], baseline=50.0, control_coefs={"newsletter": 2.0})
    errors = forecast_errors(actual, actual)
    assert errors["mae"] == pytest.approx(0.0)
    planned = apply_planned_spend(frame, {"search": 16.0}, [curve])
    assert planned.loc[1, "search_S"] == pytest.approx(8.0)
    rate = (10.0 + 20.0 + 30.0) / (5.0 + 8.0 + 9.0)
    assert planned.loc[3, "search_S"] == pytest.approx(16.0)
    assert planned.loc[3, "search_clicks_P"] == pytest.approx(rate * 16.0)
    dead = frame.copy()
    dead["search_S"] = 0.0
    with pytest.raises(ValueError, match="search"):
        apply_planned_spend(dead, {"search": 16.0}, [curve])
    saturated = saturation(np.array([0.0, 10.0]), lam=0.1)
    assert saturated[0] == 0
    assert 0 < saturated[1] < 1


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "search_clicks_P": np.full(8, 10.0),
            "search_S": np.full(8, 5.0),
            "newsletter": np.full(8, 1.0),
        }
    )


def test_mix_requires_shares_that_sum_to_one() -> None:
    curve = _curve()
    history = _history()
    with pytest.raises(ValueError, match="sum to 1"):
        mix_prediction(
            history,
            [curve],
            baseline=10.0,
            control_coefs={"newsletter": 0.0},
            budget=100.0,
            shares={"search": 0.4},
        )
    value = mix_prediction(
        history,
        [curve],
        baseline=10.0,
        control_coefs={"newsletter": 0.0},
        budget=100.0,
        shares={"search": 1.0},
    )
    assert value > 10


def test_one_week_of_history_is_rejected() -> None:
    curve = _curve()
    week = pd.DataFrame({"search_clicks_P": [10.0], "search_S": [5.0], "newsletter": [1.0]})
    with pytest.raises(ValueError, match="at least"):
        mix_prediction(
            week,
            [curve],
            baseline=10.0,
            control_coefs={"newsletter": 0.0},
            budget=100.0,
            shares={"search": 1.0},
            l_max=8,
        )


def test_history_carries_more_adstock_than_a_single_week() -> None:
    level = np.full(4, 10.0)
    carry = geometric_adstock(level, alpha=0.5, l_max=4)[-1]
    alone = geometric_adstock(np.array([10.0]), alpha=0.5, l_max=4)[-1]
    assert carry > alone


def test_naive_forecasts_match_a_flat_history() -> None:
    train = np.ones(60)
    forecasts = naive_holdout_predictions(train, 8)
    assert forecasts["training_mean"] == pytest.approx(np.ones(8))
    assert forecasts["lag_52"] == pytest.approx(np.ones(8))
    assert forecast_errors(np.ones(8), forecasts["training_mean"])["mape"] == pytest.approx(0.0)


def test_marginal_roi_matches_a_finite_difference() -> None:
    curve = _curve()
    media = np.full(20, 5.0)
    spend = np.full(20, 10.0)
    l_max = 8
    last = len(media) - l_max
    step = 1e-4
    rate = float(media[0] / spend[0])
    base = channel_contribution(media, curve)
    changes = []
    for week in range(last + 1):
        bumped = media.copy()
        bumped[week] += step
        delta = float(np.sum(channel_contribution(bumped, curve)) - np.sum(base))
        changes.append(delta / step * rate)
    expected = float(np.average(changes, weights=spend[: last + 1]))
    draw = marginal_roi_draws(
        media,
        spend,
        np.array([curve.alpha]),
        np.array([curve.lam]),
        np.array([curve.beta]),
        l_max=l_max,
    )
    assert draw[0] == pytest.approx(expected, rel=1e-6)


def test_mix_averages_draws_rather_than_mean_parameters() -> None:
    history = pd.DataFrame(
        {
            "search_clicks_P": np.full(8, 5.0),
            "search_S": np.full(8, 10.0),
            "newsletter": np.zeros(8),
        }
    )
    low = ChannelCurve("search", "search_clicks_P", "search_S", alpha=0.2, lam=0.01, beta=50.0)
    high = ChannelCurve("search", "search_clicks_P", "search_S", alpha=0.8, lam=0.05, beta=150.0)
    posterior = PosteriorDraws(
        channels=("search",),
        alpha=np.array([[0.2], [0.8]]),
        lam=np.array([[0.01], [0.05]]),
        beta=np.array([[50.0], [150.0]]),
        baseline=np.array([10.0, 30.0]),
        controls=("newsletter",),
        control_beta=np.zeros((2, 1)),
    )
    averaged = mix_prediction(
        history,
        [low],
        baseline=20.0,
        control_coefs={"newsletter": 0.0},
        budget=80.0,
        shares={"search": 1.0},
        posterior=posterior,
    )
    plug_in = mix_prediction(
        history,
        [ChannelCurve("search", "search_clicks_P", "search_S", alpha=0.5, lam=0.03, beta=100.0)],
        baseline=20.0,
        control_coefs={"newsletter": 0.0},
        budget=80.0,
        shares={"search": 1.0},
    )
    one = mix_prediction(
        history,
        [low],
        baseline=10.0,
        control_coefs={"newsletter": 0.0},
        budget=80.0,
        shares={"search": 1.0},
    )
    other = mix_prediction(
        history,
        [high],
        baseline=30.0,
        control_coefs={"newsletter": 0.0},
        budget=80.0,
        shares={"search": 1.0},
    )
    assert averaged == pytest.approx((one + other) / 2)
    assert averaged != pytest.approx(plug_in)


def test_roi_draw_is_contribution_over_spend() -> None:
    curve = _curve()
    media = np.array([10.0, 0.0, 0.0, 0.0])
    expected = float(np.sum(curve.beta * saturation(geometric_adstock(media, curve.alpha), curve.lam)) / 20.0)
    draws = roi_draws(media, 20.0, np.array([curve.alpha]), np.array([curve.lam]), np.array([curve.beta]))
    assert draws == pytest.approx([expected])


ROI = {"search": 2.0, "ooh": 0.4, "facebook": 3.0}
ROI_LOW = {"search": 1.2, "ooh": 0.1, "facebook": -1.0}
ROI_HIGH = {"search": 3.0, "ooh": 0.7, "facebook": 9.0}


def _memo(prior: PriorCheck, ablation_failures: list[str] | None = None, lift: LiftPlan | None = None) -> str:
    return memo_text(
        question="Which paid channels earn back the next ruble at observed spend?",
        estimand="Marginal ROI of the next ruble.",
        holdout_mape=0.04,
        naive_mape={"training_mean": 0.2, "lag_52": 0.1},
        backtest=[
            BacktestOrigin(0, "2019-09-23", 0.04, {"training_mean": 0.2, "lag_52": 0.1}, True, []),
            BacktestOrigin(
                8, "2019-07-29", 0.06, {"training_mean": 0.3, "lag_52": 0.05}, False, ["r_hat_max 1.020 is at or above 1.01."]
            ),
        ],
        channel_roi=ROI,
        channel_roi_low=ROI_LOW,
        channel_roi_high=ROI_HIGH,
        channel_contraction={"search": 0.9, "ooh": 0.7, "facebook": 0.1},
        residuals=ResidualCheck(0.4, (0.2, 0.6), [0.4, 0.2, 0.1, 0.0], 0.001, [0.05, -0.02, 0.01, 0.0], 0.8),
        prior=prior,
        ablation=AblationCheck("competitor_sales_B", ablation_failures or [], {"search": 2.5, "ooh": 0.3, "facebook": 2.0}),
        recovery=RecoveryCheck(
            n_fits=20,
            n_passed=20,
            coverage={"facebook": 0.95, "ooh": 0.5, "search": 0.9},
            coverage_se={"facebook": 0.05, "ooh": 0.11, "search": 0.07},
            mean_bias={"facebook": 0.3, "ooh": -0.2, "search": 0.1},
            true_roi={"facebook": 2.5, "ooh": 0.5, "search": 1.9},
        ),
        lift=lift,
    )


def test_memo_states_the_rule_the_checks_and_the_next_experiment() -> None:
    lift = LiftPlan("facebook", 2.0, 0.2, 1000.0, 3.0, {0.1: 40, 0.25: 7, 0.5: None})
    text = _memo(PriorCheck(2.0, [], {"facebook": 0.4, "ooh": 0.1, "search": -0.05}), lift=lift)
    lowered = text.lower()
    for forbidden in ("m.video", "yandex", "hoff", "okkam", "apple", "synthetic control"):
        assert forbidden not in lowered
    assert "95% interval" in text
    assert "lift test on facebook" in text
    assert "40 weeks per arm" in text
    assert "lower bound on duration" in text
    assert "cannot settle" not in text
    assert "Without the AR term these correlated weeks" in text
    far = LiftPlan("search", 2.0, 0.1, 400.0, 3.4, {0.1: 9000, 0.25: 1500, 0.5: 400})
    infeasible = _memo(PriorCheck(2.0, [], {"facebook": 0.1}), lift=far)
    assert "a national test over time cannot settle search" in infeasible
    assert "does not calibrate this model" in text
    assert "facebook moved by 25% or more" in text
    assert "Without competitor_sales_B as a control" in text
    assert "failed the QA gate" in text
    assert "beats the lag-52 forecast in 1 of 2 windows" in text
    assert "sanity check" in text
    assert "rho 0.40" in text and "Ljung-Box p 0.001" in text
    assert "Contraction is below 0.5 for facebook" in text
    assert "For ooh coverage is more than two standard errors under 95%" in text
    assert "family-wise" in text
    assert "search: ROI 2.00 (95% interval 1.20 to 3.00), prior contraction 0.90. The interval is above 1 on this fit" in text
    assert "not a mix move" in text
    assert "not labelled" in text and "recovery coverage" in text
    gated = _memo(
        PriorCheck(2.0, [], {"facebook": 0.1}),
        lift=lift,
    )
    assert "Recovery coverage of at least 90%" in gated
    assert gated_payback("earns", 0.8) == "recovery_gate"
    assert gated_payback("earns", 0.9) == "earns"
    assert gated_payback("unsettled", 0.5) == "unsettled"
    text_geo = memo_text(
        question="q",
        estimand="e",
        holdout_mape=0.04,
        naive_mape={"training_mean": 0.2, "lag_52": 0.1},
        backtest=[BacktestOrigin(0, "2019-09-23", 0.04, {"training_mean": 0.2, "lag_52": 0.1}, True, [])],
        channel_roi=ROI,
        channel_roi_low=ROI_LOW,
        channel_roi_high=ROI_HIGH,
        channel_contraction={"search": 0.9, "ooh": 0.7, "facebook": 0.1},
        residuals=ResidualCheck(0.4, (0.2, 0.6), [0.4, 0.2, 0.1, 0.0], 0.001, [0.05, -0.02, 0.01, 0.0], 0.8),
        prior=PriorCheck(2.0, [], {"facebook": 0.1}),
        ablation=AblationCheck("competitor_sales_B", [], {"search": 2.5, "ooh": 0.3, "facebook": 2.0}),
        recovery=RecoveryCheck(20, 20, {"facebook": 0.95, "ooh": 0.95, "search": 0.95}, {"facebook": 0.05, "ooh": 0.05, "search": 0.05}, {"facebook": 0.0, "ooh": 0.0, "search": 0.0}, {"facebook": 2.0, "ooh": 0.5, "search": 2.0}),
        lift=lift,
        geo_lift=geo_lift_sentence(
            {
                "channel": "search",
                "point_roi": 3.44,
                "n_treated": 25,
                "n_geos": 50,
                "note": "planning proxy: noise from a simulated public geo panel",
                "cells": [
                    {
                        "duration_days": 28,
                        "increase": 2.0,
                        "power_at_point_roi": 0.9,
                        "false_settle_at_payback": 0.02,
                        "mde_roi_80": 1.4,
                    }
                ],
            }
        ),
    )
    assert "25 treated geos of 50, 28 days, and a 200% heavy-up" in text_geo
    assert "planning proxy" in text_geo


def test_memo_does_not_publish_a_prior_move_from_a_failed_fit() -> None:
    text = _memo(PriorCheck(2.0, ["r_hat_max 1.010 is at or above 1.01."], {}), ["divergences 3 are above 0."])
    assert "No prior move is published" in text
    assert "without competitor_sales_B did not clear" in text
    assert "no lift test is needed" in text


def test_lift_plan_uses_the_shared_rule_and_sizes_the_test() -> None:
    plan = lift_plan(
        spend_share={"search": 0.5, "ooh": 0.3, "facebook": 0.2},
        weekly_spend={"search": 5000.0, "ooh": 3000.0, "facebook": 2000.0},
        channel_roi=ROI,
        channel_roi_low=ROI_LOW,
        channel_roi_high=ROI_HIGH,
        sigma=50000.0,
        rho=0.3,
    )
    assert plan is not None
    assert plan.channel == "facebook"
    assert plan.priority == pytest.approx(0.2 * 10.0)
    weeks = [plan.weeks[increase] for increase in (0.10, 0.25, 0.50)]
    assert all(value is not None for value in weeks)
    assert weeks[0] > weeks[1] > weeks[2]


def test_residual_diagnostics_and_contraction() -> None:
    rng = np.random.default_rng(3)
    white = rng.normal(0, 1, 400)
    ar = np.empty(400)
    ar[0] = 0.0
    for week in range(1, 400):
        ar[week] = 0.7 * ar[week - 1] + white[week]
    assert residual_acf(ar, 1)[0] > 0.5
    assert abs(residual_acf(white, 1)[0]) < 0.15
    assert ljung_box(ar, 4).p_value < 0.001
    assert ljung_box(white, 4).p_value > 0.01
    assert ljung_box(white, 4, fitted_terms=1).dof == 3
    assert chi2_sf(3.841458820694124, 1) == pytest.approx(0.05, abs=1e-6)
    assert chi2_sf(9.487729036781154, 4) == pytest.approx(0.05, abs=1e-6)
    assert chi2_sf(7.814727903251178, 3) == pytest.approx(0.05, abs=1e-6)
    assert contraction(2.0, 1.0) == pytest.approx(0.75)
    assert contraction(2.0, 2.0) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        contraction(0.0, 1.0)


def test_ar_forecast_decays_the_last_error() -> None:
    path = ar_forecast(np.zeros(3), 8.0, 0.5)
    assert path == pytest.approx([4.0, 2.0, 1.0])
    with pytest.raises(ValueError):
        ar_forecast(np.zeros(3), 1.0, 1.0)


def test_prior_moves_cover_every_channel() -> None:
    moves = prior_moves({"tv": 2.0, "ooh": 0.5}, {"tv": 3.0, "ooh": 0.5})
    assert moves == {"ooh": pytest.approx(0.0), "tv": pytest.approx(0.5)}
    with pytest.raises(ValueError, match="same channels"):
        prior_moves({"tv": 2.0}, {"ooh": 1.0})


def test_calendar_terms_are_a_trend_in_years_and_two_harmonics() -> None:
    frame = pd.DataFrame({"date": pd.date_range("2016-01-04", periods=105, freq="7D")})
    terms = add_calendar_terms(frame)
    assert terms["trend"].iloc[0] == pytest.approx(0.0)
    assert terms["trend"].iloc[-1] == pytest.approx(104 / 52.18)
    for column in ("sin_1", "cos_1", "sin_2", "cos_2"):
        assert terms[column].between(-1, 1).all()
    assert terms["cos_1"].iloc[0] == pytest.approx(1.0)
    shifted = add_calendar_terms(frame.iloc[10:], origin=frame["date"].iloc[0])
    assert shifted["trend"].to_numpy() == pytest.approx(terms["trend"].iloc[10:].to_numpy())
    assert all(term in CONTROLS for term in CALENDAR_TERMS)


def test_published_qa_has_a_non_overlapping_backtest() -> None:
    root = Path(__file__).resolve().parents[1]
    qa = json.loads((root / "artifacts" / "qa.json").read_text(encoding="utf-8"))
    offsets = [entry["offset_weeks"] for entry in qa["backtest"]]
    assert offsets == [0, 8, 16]
    starts = [entry["holdout_start"] for entry in qa["backtest"]]
    assert starts == sorted(starts, reverse=True)
    assert "confirmation_ok" not in qa
    assert not (root / "artifacts" / "qa_confirm.json").exists()
    assert -0.9 < qa["rho"] < 0.9
    assert set(qa["channel_contraction"]) == set(qa["channel_roi"])
    assert sum(qa["channel_spend_share"].values()) == pytest.approx(1.0)
    assert all(entry["beats_lag_52"] == (entry["holdout_mape"] < entry["naive_mape"]["lag_52"]) for entry in qa["backtest"])
    for channel in qa["channel_roi"]:
        expected = channel_payback(qa["channel_roi"][channel], qa["channel_roi_low"][channel], qa["channel_roi_high"][channel])
        assert qa["channel_payback"][channel] == (expected if not qa["failures"] else "withheld")


def test_forecast_notice_uses_the_published_decision() -> None:
    root = Path(__file__).resolve().parents[1]
    qa = json.loads((root / "artifacts" / "qa.json").read_text(encoding="utf-8"))
    prior = json.loads((root / "artifacts" / "prior_sensitivity.json").read_text(encoding="utf-8"))
    moves = (
        {channel: item["relative_change"] for channel, item in prior["channels"].items()}
        if prior["published"]
        else None
    )
    text = forecast_notice(channel_roi=qa["channel_roi"], channel_payback=qa["channel_payback"], prior_moves=moves)
    lowered = text.lower()
    for forbidden in ("m.video", "yandex", "hoff", "okkam", "apple", "synthetic control"):
        assert forbidden not in lowered
    assert "not a budget optimizer" in text
    if moves is None:
        assert "The prior move is not published." in text
    ranked = sorted(qa["channel_roi"], key=qa["channel_roi"].get, reverse=True)
    assert text.index(ranked[0]) < text.index(ranked[1])


def test_forecast_notice_without_a_prior_keeps_the_channels() -> None:
    text = forecast_notice(
        channel_roi={"print": 8.9, "facebook": 8.7, "tv": 3.7},
        channel_payback={"print": "earns", "facebook": "unsettled", "tv": "earns"},
    )
    assert "The prior move is not published." in text
    assert "print" in text
    assert "facebook" in text
    assert "tv" in text
    assert "not a budget optimizer" in text
    assert text.index("print") < text.index("facebook") < text.index("tv")


def test_published_memo_uses_95_percent_intervals_and_names_the_next_test() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "results" / "memo.txt").read_text(encoding="utf-8")
    pdf = root / "results" / "memo.pdf"
    assert pdf.is_file() and pdf.read_bytes().startswith(b"%PDF")
    assert "95% interval" in text
    assert "Backtest on non-overlapping" in text
    assert "No feasible lift test on the available units." in text
    assert "A lift test on search" not in text
    assert "A lift test on ooh" not in text
    assert "Recovery check." in text
    assert "Hill saturation" in text
    assert "AR(1) errors" in text
    assert "adjacent slices" not in text
    recovery = json.loads((Path(__file__).resolve().parents[1] / "artifacts" / "recovery.json").read_text(encoding="utf-8"))
    assert recovery["n_fits"] == 100
    assert "Recovery coverage of at least 90%" in text
    assert "do not license a budget label" in text or "not a mix move" in text
    export = json.loads((Path(__file__).resolve().parents[1] / "artifacts" / "pymc_export.json").read_text(encoding="utf-8"))
    assert export["contraction_parameter"] == "marginal ROI"


def test_forecast_notice_marks_a_large_prior_move() -> None:
    text = forecast_notice(
        channel_roi={"facebook": 2.0, "tv": 1.0},
        channel_payback={"facebook": "unsettled", "tv": "earns"},
        prior_moves={"facebook": 0.6, "tv": 0.1},
    )
    assert "prior shapes this read" in text
    assert "inside the 25% check" in text


def test_misspecified_recovery_truth() -> None:
    pulse = delayed_adstock(np.array([1.0, 0, 0, 0, 0, 0, 0, 0]), 0.6)
    assert pulse.sum() == pytest.approx(1.0)
    assert pulse[1] > pulse[0]
    rng = np.random.default_rng(2)
    train = pd.DataFrame(
        {
            "search_clicks_P": rng.uniform(5, 15, 40),
            "search_S": rng.uniform(4, 6, 40),
        }
    )
    curve = _curve()
    response = true_response(train, curve)
    media = train["search_clicks_P"].to_numpy()
    assert response.contribution(media).sum() == pytest.approx(abs(channel_contribution(media, curve).sum()))
    roi = true_marginal_roi(train, response)
    assert roi > 0
    doubled = true_marginal_roi(train, type(response)(**{**response.__dict__, "scale": 2 * response.scale}))
    assert doubled == pytest.approx(2 * roi)
    noise = ar_noise(2000, sigma=1.0, rho=0.6, rng=np.random.default_rng(1))
    assert residual_acf(noise, 1)[0] == pytest.approx(0.6, abs=0.06)
    assert coverage_share([{"tv": True, "ooh": False}, {"tv": True, "ooh": True}]) == {"ooh": 0.5, "tv": 1.0}


def test_payback_refuses_only_an_interval_below_one() -> None:
    assert channel_payback(0.4, 0.2, 0.8) == "refuse"
    assert channel_payback(0.7, 0.4, 1.2) == "unsettled"
    assert channel_payback(2.0, 1.1, 3.0) == "earns"
    assert channel_payback(11.8, 1.15, 26.8) == "earns"
    assert channel_payback(11.8, 0.88, 27.0) == "unsettled"


def test_qa_gates_and_refresh() -> None:
    clean = FitSummary(1.001, 800, 0, 0.1, {"search": 1.2, "tv": 0.4})
    dirty = FitSummary(1.2, 50, 3, 0.9, {"search": 0.2, "tv": 1.4})
    assert qa_failures(clean) == []
    assert len(qa_failures(dirty)) == 4
    assert qa_failures(FitSummary(1.01, 800, 0, 0.1, {"search": 1.0}))
    assert any("not finite" in item for item in qa_failures(FitSummary(float("nan"), 800, 0, 0.1, {"search": 1.0})))
    change = prior_sensitivity(1.0, 1.4)
    assert sensitivity_is_material(change)
    diff = refresh_diff(clean, dirty)
    assert diff.roi_rank_changed is True
