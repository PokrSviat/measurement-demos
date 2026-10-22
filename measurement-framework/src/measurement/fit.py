"""Fit the PyMC revenue model and write the artifacts the memo reads.

`python -m measurement.fit` fits the primary origin, the two earlier backtest
origins, and publishes. `--wide`, `--ablation` and `--recovery` write their
checks. `--memo` reads every artifact and writes the memo.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from measurement.contract import model_regressors, spend_column_for, validate_regressors
from measurement.export import write_manifest
from measurement.memo import (
    AblationCheck,
    BacktestOrigin,
    PriorCheck,
    RecoveryCheck,
    ResidualCheck,
    lift_plan,
    memo_text,
    write_memo_pdf,
)
from measurement.model import (
    CONTROLS,
    L_MAX,
    channel_roi_draws,
    curves_from_posterior,
    fit_pymc,
    roi_bounds_from_posterior,
    sampling_draws,
)
from measurement.qa import (
    FitSummary,
    channel_payback,
    draw_contraction,
    gated_payback,
    ljung_box,
    prior_moves,
    qa_failures,
    residual_acf,
    sensitivity_is_material,
)
from measurement.sql_layer import add_calendar_terms, add_holiday_flag, connect, model_grain
from measurement.transforms import ar_forecast, forecast_errors, naive_holdout_predictions, prediction_from_draws

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "dt_simulated_weekly.csv"
HOLIDAYS = ROOT / "data" / "dt_prophet_holidays_de.csv"
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"
HOLDOUT_WEEKS = 8
BACKTEST_OFFSETS = (0, 8, 16)
SEED = 20241006
DRAWS, TUNE, CHAINS = 1000, 1000, 4
WIDE_SCALE = 2.0
ACF_LAGS = 4
ABLATED_CONTROL = "competitor_sales_B"
QUESTION = "Which paid channels earn back the next ruble at observed spend?"
ESTIMAND = (
    "Marginal ROI of one extra ruble at observed weekly spend, "
    "spend-weighted across training weeks whose full adstock horizon sits inside the series. "
    "The point is the mean of that derivative across posterior draws. "
    "The interval is the 2.5th to 97.5th percentile of the same draws. "
    "The media coefficient prior is normal and centered at zero, so contribution can be negative. "
    "The model controls for a linear trend and two yearly harmonics and carries AR(1) errors. "
    "A channel whose 95% interval sits entirely above or below 1 is reported as an interval fact. "
    "Recovery coverage and mean bias are diagnostics on a misspecified response. "
    "This national weekly series does not identify the mix, so no channel is scaled or cut. "
    "Average contribution over total spend is not this estimand."
)


def load_grain() -> pd.DataFrame:
    grain = add_holiday_flag(model_grain(connect(DATA)), pd.read_csv(HOLIDAYS))
    grain = add_calendar_terms(grain)
    validate_regressors(grain, model_regressors())
    return grain


def channel_spend(train: pd.DataFrame, regressors: list[str]) -> dict[str, float]:
    """Total paid spend of each channel over the training weeks."""
    from measurement.contract import column_owner

    return {str(column_owner(regressor)): float(train[spend_column_for(regressor)].sum()) for regressor in regressors}


def spend_shares(train: pd.DataFrame, regressors: list[str]) -> dict[str, float]:
    """Each channel's share of paid spend over the training weeks."""
    totals = channel_spend(train, regressors)
    grand = sum(totals.values())
    if grand <= 0:
        raise ValueError("Training spend must be positive.")
    return {channel: total / grand for channel, total in totals.items()}


def _origin(grain: pd.DataFrame, offset: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out eight weeks ending `offset` weeks before the last row. Training is only the rows before."""
    end = len(grain) - offset
    start = end - HOLDOUT_WEEKS
    if start < 52:
        raise ValueError("The origin needs 52 training weeks before the holdout.")
    return grain.iloc[:start].copy(), grain.iloc[start:end].copy()


def _fit_origin(
    grain: pd.DataFrame,
    offset: int,
    *,
    beta_scale: dict[str, float] | None = None,
    controls: tuple[str, ...] = CONTROLS,
    revenue: np.ndarray | None = None,
    seed: int = SEED,
) -> dict:
    """One fit on one origin. `revenue` replaces the observed target, for recovery checks."""
    train, holdout = _origin(grain, offset)
    if revenue is not None:
        train = train.assign(revenue=np.asarray(revenue, dtype=float))
    regressors = model_regressors()
    fitted = fit_pymc(
        train,
        draws=DRAWS,
        tune=TUNE,
        chains=CHAINS,
        seed=seed,
        beta_scale=beta_scale,
        controls=controls,
    )
    idata = fitted["idata"]
    scales = (idata, regressors, fitted["media_scale"], fitted["control_scale"], fitted["controls"])
    curves, baseline, control_coefs = curves_from_posterior(*scales)
    posterior = sampling_draws(*scales)
    rho = float(idata.posterior["rho"].mean())
    observed = train["revenue"].to_numpy(dtype=float)
    structural_residual = observed - prediction_from_draws(train, curves, posterior)
    innovations = structural_residual[1:] - rho * structural_residual[:-1]
    predicted = ar_forecast(prediction_from_draws(holdout, curves, posterior), float(structural_residual[-1]), rho)
    actual = holdout["revenue"].to_numpy(dtype=float)
    errors = forecast_errors(actual, predicted)
    naive_mape = {
        name: forecast_errors(actual, forecast)["mape"]
        for name, forecast in naive_holdout_predictions(
            train["revenue"].to_numpy(dtype=float),
            HOLDOUT_WEEKS,
        ).items()
    }
    roi_bounds = roi_bounds_from_posterior(train, regressors, posterior)
    import arviz as az

    checked = ["alpha", "beta", "lam", "baseline", "control_beta", "sigma", "rho"]
    rhat = np.asarray(az.rhat(idata, var_names=checked).to_array().values, dtype=float)
    ess = np.asarray(az.ess(idata, var_names=checked).to_array().values, dtype=float)
    prior_roi = channel_roi_draws(train, regressors, sampling_draws(fitted["prior"], *scales[1:], group="prior"))
    posterior_roi = channel_roi_draws(train, regressors, posterior)
    channel_contraction = {
        channel: draw_contraction(prior_roi[channel], posterior_roi[channel]) for channel in posterior.channels
    }
    channel_roi = {channel: bounds["mean"] for channel, bounds in roi_bounds.items()}
    summary = FitSummary(
        r_hat_max=float(np.nanmax(rhat)),
        ess_min=float(np.nanmin(ess)),
        divergences=int(idata.sample_stats["diverging"].sum()),
        holdout_mape=errors["mape"],
        channel_roi=channel_roi,
    )
    failures = qa_failures(summary)
    payback = (
        {channel: "withheld" for channel in channel_roi}
        if failures
        else {
            channel: channel_payback(channel_roi[channel], bounds["low"], bounds["high"])
            for channel, bounds in roi_bounds.items()
        }
    )
    return {
        "offset": offset,
        "train": train,
        "holdout": holdout,
        "curves": curves,
        "baseline": baseline,
        "control_coefs": control_coefs,
        "sigma": float(idata.posterior["sigma"].mean()),
        "rho": rho,
        "rho_interval": [float(value) for value in np.quantile(idata.posterior["rho"].values.reshape(-1), [0.025, 0.975])],
        "residuals": {
            "structural_acf": residual_acf(structural_residual, ACF_LAGS),
            "structural_ljung_box": ljung_box(structural_residual, ACF_LAGS).__dict__,
            "innovation_acf": residual_acf(innovations, ACF_LAGS),
            "innovation_ljung_box": ljung_box(innovations, ACF_LAGS, fitted_terms=1).__dict__,
        },
        "channel_contraction": channel_contraction,
        "channel_spend_share": spend_shares(train, regressors),
        "channel_weekly_spend": {
            channel: total / len(train) for channel, total in channel_spend(train, regressors).items()
        },
        "posterior": posterior,
        "naive_mape": naive_mape,
        "roi_bounds": roi_bounds,
        "summary": summary,
        "failures": failures,
        "payback": payback,
        "forecast_mae": errors["mae"],
        "holdout_pred": predicted,
    }


def _qa_payload(origin: dict) -> dict:
    summary = origin["summary"]
    return {
        "r_hat_max": summary.r_hat_max,
        "ess_min": summary.ess_min,
        "divergences": summary.divergences,
        "holdout_mape": summary.holdout_mape,
        "naive_mape": origin["naive_mape"],
        "channel_roi": summary.channel_roi,
        "channel_roi_low": {channel: bounds["low"] for channel, bounds in origin["roi_bounds"].items()},
        "channel_roi_high": {channel: bounds["high"] for channel, bounds in origin["roi_bounds"].items()},
        "failures": origin["failures"],
        "forecast_mae": origin["forecast_mae"],
        "channel_payback": origin["payback"],
        "sigma": origin["sigma"],
        "rho": origin["rho"],
        "rho_interval": origin["rho_interval"],
        "residuals": origin["residuals"],
        "channel_contraction": origin["channel_contraction"],
        "channel_spend_share": origin["channel_spend_share"],
        "channel_weekly_spend": origin["channel_weekly_spend"],
    }


def _backtest_entry(origin: dict) -> dict:
    return {
        "offset_weeks": origin["offset"],
        "holdout_start": str(pd.Timestamp(origin["holdout"]["date"].iloc[0]).date()),
        "holdout_mape": origin["summary"].holdout_mape,
        "naive_mape": origin["naive_mape"],
        "beats_lag_52": origin["summary"].holdout_mape < origin["naive_mape"]["lag_52"],
        "failures": origin["failures"],
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_primary() -> None:
    """Primary fit plus the earlier backtest origins. Only the primary fit publishes curves and labels."""
    grain = load_grain()
    origins = [_fit_origin(grain, offset) for offset in BACKTEST_OFFSETS]
    primary = origins[0]
    qa_payload = _qa_payload(primary)
    qa_payload["backtest"] = [_backtest_entry(origin) for origin in origins]
    _write_json(ARTIFACTS / "qa.json", qa_payload)
    if qa_payload["failures"]:
        _write_unpublished(qa_payload)
        return
    _publish(primary, grain)


def _publish(primary: dict, grain: pd.DataFrame) -> None:
    curves = primary["curves"]
    posterior = primary["posterior"]
    train = primary["train"]
    pd.DataFrame([curve.__dict__ for curve in curves]).to_csv(ARTIFACTS / "curves.csv", index=False)
    np.savez_compressed(
        ARTIFACTS / "posterior_draws.npz",
        channels=np.array(posterior.channels),
        alpha=posterior.alpha,
        lam=posterior.lam,
        beta=posterior.beta,
        baseline=posterior.baseline,
        controls=np.array(posterior.controls),
        control_beta=posterior.control_beta,
    )
    train.tail(L_MAX + 1).to_csv(ARTIFACTS / "reference_week.csv", index=False)
    _write_json(
        ARTIFACTS / "baseline.json",
        {
            "baseline": primary["baseline"],
            "control_coefs": primary["control_coefs"],
            "sigma": primary["sigma"],
            "rho": primary["rho"],
        },
    )
    _write_json(
        ARTIFACTS / "holdout.json",
        {
            "dates": [str(pd.Timestamp(date).date()) for date in primary["holdout"]["date"]],
            "actual": primary["holdout"]["revenue"].astype(float).tolist(),
            "predicted": [float(value) for value in primary["holdout_pred"]],
        },
    )
    write_manifest(
        ARTIFACTS / "manifest.json",
        data_path=DATA,
        target="revenue",
        date_min=str(grain["date"].min().date()),
        date_max=str(grain["date"].max().date()),
        regressors=model_regressors(),
        holdout_weeks=HOLDOUT_WEEKS,
        seed=SEED,
    )
    spend_map = {curve.channel: spend_column_for(curve.regressor) for curve in curves}
    _write_json(ARTIFACTS / "spend_columns.json", spend_map)


def run_wide() -> None:
    """Every media prior twice as wide, primary origin only."""
    grain = load_grain()
    scale = {channel: WIDE_SCALE for channel in _channels()}
    _write_json(ARTIFACTS / "qa_wide.json", _qa_payload(_fit_origin(grain, 0, beta_scale=scale)))


def run_ablation() -> None:
    """The primary fit without one context control. It does not decide."""
    grain = load_grain()
    controls = tuple(name for name in CONTROLS if name != ABLATED_CONTROL)
    payload = _qa_payload(_fit_origin(grain, 0, controls=controls))
    payload["dropped_control"] = ABLATED_CONTROL
    _write_json(ARTIFACTS / "qa_ablation.json", payload)


def _channels() -> list[str]:
    from measurement.contract import column_owner

    return [owner for regressor in model_regressors() if (owner := column_owner(regressor))]


def _write_unpublished(qa_payload: dict) -> None:
    """A failed gate is the published answer. It does not leave the previous payback in place."""
    lines = [
        QUESTION,
        "",
        f"Estimand. {ESTIMAND}",
        "",
        "The fit did not publish. The QA gate failed, so no channel is marked as earning or refused.",
        *[f"- {item}" for item in qa_payload["failures"]],
        "",
        "Curves, the reference history, and payback labels from an earlier fit are not a decision.",
    ]
    text = "\n".join(lines)
    payload = {
        "question": QUESTION,
        "estimand": ESTIMAND,
        "published": False,
        "failures": qa_payload["failures"],
        "channel_roi": qa_payload["channel_roi"],
        "channel_roi_low": qa_payload["channel_roi_low"],
        "channel_roi_high": qa_payload["channel_roi_high"],
        "channel_payback": qa_payload["channel_payback"],
        "memo": text,
    }
    _write_json(RESULTS / "case_result.json", payload)
    (RESULTS / "memo.txt").write_text(text + "\n", encoding="utf-8")
    write_memo_pdf(text, RESULTS / "memo.pdf")


def _read(name: str) -> dict:
    return json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))


def prior_check(base: dict, widened: dict) -> PriorCheck:
    if widened.get("failures"):
        return PriorCheck(scale=WIDE_SCALE, failures=list(widened["failures"]), moves={})
    return PriorCheck(scale=WIDE_SCALE, failures=[], moves=prior_moves(base["channel_roi"], widened["channel_roi"]))


def write_case_memo() -> None:
    base = _read("qa.json")
    if base.get("failures"):
        raise ValueError("A memo is not written from a fit that failed the QA gate.")
    prior = prior_check(base, _read("qa_wide.json"))
    _write_json(
        ARTIFACTS / "prior_sensitivity.json",
        {
            "scale": prior.scale,
            "published": not prior.failures,
            "failures": prior.failures,
            "channels": {
                channel: {"relative_change": move, "material": sensitivity_is_material(move)}
                for channel, move in prior.moves.items()
            },
        },
    )
    ablation_payload = _read("qa_ablation.json")
    ablation = AblationCheck(
        dropped_control=ablation_payload["dropped_control"],
        failures=list(ablation_payload["failures"]),
        channel_roi=ablation_payload["channel_roi"],
    )
    recovery = _read("recovery.json")
    residuals = base["residuals"]
    lift = lift_plan(
        spend_share=base["channel_spend_share"],
        weekly_spend=base["channel_weekly_spend"],
        channel_roi=base["channel_roi"],
        channel_roi_low=base["channel_roi_low"],
        channel_roi_high=base["channel_roi_high"],
        sigma=float(base["sigma"]),
        rho=float(base["rho"]),
    )
    geo_lift, geo_powered = _geo_lift_payload()
    text = memo_text(
        question=QUESTION,
        estimand=ESTIMAND,
        holdout_mape=float(base["holdout_mape"]),
        naive_mape=base["naive_mape"],
        backtest=[BacktestOrigin(**entry) for entry in base["backtest"]],
        channel_roi=base["channel_roi"],
        channel_roi_low=base["channel_roi_low"],
        channel_roi_high=base["channel_roi_high"],
        channel_contraction=base["channel_contraction"],
        residuals=ResidualCheck(
            rho=float(base["rho"]),
            rho_interval=tuple(base["rho_interval"]),
            structural_acf=residuals["structural_acf"],
            structural_p_value=float(residuals["structural_ljung_box"]["p_value"]),
            innovation_acf=residuals["innovation_acf"],
            innovation_p_value=float(residuals["innovation_ljung_box"]["p_value"]),
        ),
        prior=prior,
        ablation=ablation,
        recovery=RecoveryCheck(
            n_fits=int(recovery["n_fits"]),
            n_passed=int(recovery["n_passed"]),
            coverage=recovery["coverage"],
            coverage_se=recovery["coverage_se"],
            mean_bias=recovery["mean_bias"],
            true_roi=recovery["true_roi"],
        ),
        lift=lift,
        geo_lift=geo_lift,
        geo_powered=geo_powered,
    )
    payload = {
        "question": QUESTION,
        "estimand": ESTIMAND,
        "holdout_mape": base["holdout_mape"],
        "naive_mape": base["naive_mape"],
        "backtest": base["backtest"],
        "channel_roi": base["channel_roi"],
        "channel_roi_low": base["channel_roi_low"],
        "channel_roi_high": base["channel_roi_high"],
        "channel_payback": {
            channel: gated_payback(status, recovery["coverage"].get(channel) if recovery["n_passed"] else None)
            for channel, status in base["channel_payback"].items()
        },
        "channel_payback_before_recovery_gate": base["channel_payback"],
        "prior_sensitivity": _read("prior_sensitivity.json"),
        "ablation": {
            "dropped_control": ablation.dropped_control,
            "failures": ablation.failures,
            "channel_roi": None if ablation.failures else ablation.channel_roi,
        },
        "recovery": recovery,
        "rho": base["rho"],
        "residuals": residuals,
        "channel_contraction": base["channel_contraction"],
        "lift_test": None if lift is None else {**lift.__dict__, "weeks": {str(key): value for key, value in lift.weeks.items()}},
        "memo": text,
    }
    _write_json(RESULTS / "case_result.json", payload)
    (RESULTS / "memo.txt").write_text(text, encoding="utf-8")
    write_memo_pdf(text, RESULTS / "memo.pdf")
    write_bakeoff_export(base, prior)


def _geo_lift_payload() -> tuple[str, bool]:
    from budget_rule.experiment import lift_design_powered
    from measurement.memo import geo_lift_sentence

    path = ROOT.parent / "geo-holdout-case" / "results" / "search_lift_design.json"
    if not path.exists():
        return "", False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return geo_lift_sentence(payload), lift_design_powered(payload)


def write_bakeoff_export(base: dict, prior: PriorCheck) -> None:
    """The PyMC row of the bake-off, with the holdout path so the harness computes one MAPE."""
    holdout = _read("holdout.json")
    _write_json(
        ARTIFACTS / "pymc_export.json",
        {
            "model": "pymc",
            "manifest": _read("manifest.json"),
            "estimand": "marginal_roi_next_ruble",
            "holdout_mape": base["holdout_mape"],
            "holdout_actual": holdout["actual"],
            "holdout_pred": holdout["predicted"],
            "channel_roi": base["channel_roi"],
            "channel_roi_low": base["channel_roi_low"],
            "channel_roi_high": base["channel_roi_high"],
            "interval": "2.5th to 97.5th percentile of posterior draws",
            "interval_kind": "posterior",
            "channel_contraction": base["channel_contraction"],
            "contraction_parameter": "marginal ROI",
            "channel_spend_share": base["channel_spend_share"],
            "r_hat_max": base["r_hat_max"],
            "prior_sensitivity_relative": prior.moves.get("facebook"),
        },
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--wide", action="store_true")
    group.add_argument("--ablation", action="store_true")
    group.add_argument("--recovery", action="store_true")
    group.add_argument("--memo", action="store_true")
    args = parser.parse_args()
    if args.memo:
        write_case_memo()
    elif args.wide:
        run_wide()
    elif args.ablation:
        run_ablation()
    elif args.recovery:
        from measurement.recovery import run_recovery

        run_recovery()
    else:
        run_primary()
