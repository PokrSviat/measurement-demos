"""Meridian fit on the same weekly series as the PyMC manifest.

One national geo. Last ``holdout_weeks`` are held out. A second run can widen
the ROI prior scale on facebook only. This script is a bake-off participant.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BAKEOFF = Path(__file__).resolve().parents[1]
BASE_EXPORT = BAKEOFF / "artifacts" / "meridian_export.json"
SENSITIVITY_EXPORT = BAKEOFF / "artifacts" / "meridian_sensitivity.json"
SEED_CHECK_EXPORT = BAKEOFF / "artifacts" / "meridian_seed_check.json"
PRIOR_DRAWS = 1000
SEED_CHECK_COUNT = 3


class RublePlacementError(ValueError):
    """The next-ruble difference cannot be placed on a training week."""


def prior_sensitivity(base_roi: float, widened_roi: float) -> float:
    """Relative ROI change, (widened - base) / abs(base)."""
    if base_roi == 0:
        raise ValueError("Base ROI must be non-zero to measure a relative change.")
    return (widened_roi - base_roi) / abs(base_roi)


def attach_prior_sensitivity(base_path: Path, widened_path: Path) -> float:
    base = json.loads(base_path.read_text(encoding="utf-8"))
    widened = json.loads(widened_path.read_text(encoding="utf-8"))
    change = prior_sensitivity(base["channel_roi"]["facebook"], widened["channel_roi"]["facebook"])
    base["prior_sensitivity_relative"] = change
    base_path.write_text(json.dumps(base, indent=2), encoding="utf-8")
    return change


def draw_contraction(prior: np.ndarray, posterior: np.ndarray) -> float:
    """One minus posterior variance over prior variance, from draws of one parameter."""
    prior_variance = float(np.var(np.ravel(prior)))
    if prior_variance <= 0:
        raise ValueError("Prior draws need a positive variance.")
    return 1.0 - float(np.var(np.ravel(posterior))) / prior_variance


def channel_contraction(prior: np.ndarray, posterior: np.ndarray, channels: list[str]) -> dict[str, float]:
    """Contraction of each channel's last-axis slice of a parameter."""
    if prior.shape[-1] != len(channels) or posterior.shape[-1] != len(channels):
        raise ValueError("Prior and posterior draws must end in one axis per channel.")
    return {
        channel: draw_contraction(prior[..., index], posterior[..., index]) for index, channel in enumerate(channels)
    }


def seed_spread(exports: list[dict]) -> dict[str, float]:
    """Largest minus smallest point ROI per channel across seeds."""
    channels = sorted(exports[0]["channel_roi"])
    return {
        channel: max(export["channel_roi"][channel] for export in exports)
        - min(export["channel_roi"][channel] for export in exports)
        for channel in channels
    }


def prepare(weekly_csv: Path) -> pd.DataFrame:
    frame = pd.read_csv(weekly_csv)
    frame["geo"] = "national"
    frame["time"] = pd.to_datetime(frame["DATE"]).dt.strftime("%Y-%m-%d")
    frame["tv_exposure"] = frame["tv_S"]
    frame["ooh_exposure"] = frame["ooh_S"]
    frame["print_exposure"] = frame["print_S"]
    frame["event_flag"] = np.where(frame["events"].astype(str).str.lower().isin(["na", "none", ""]), 0, 1)
    return frame


def holdout_id(n_times: int, holdout_weeks: int) -> np.ndarray:
    mask = np.zeros(n_times, dtype=bool)
    mask[-holdout_weeks:] = True
    return mask


def _channel_draws(array: np.ndarray, n_channels: int, name: str) -> np.ndarray:
    if array.shape[-1] != n_channels:
        raise ValueError(f"{name} does not end in {n_channels} channels: {array.shape}.")
    if array.ndim == 3:
        return array
    if array.ndim == 2:
        return array.reshape(1, array.shape[0], n_channels)
    raise ValueError(f"{name} shape {array.shape} is not chain, draw, channel.")


def _hill(media: np.ndarray, ec: np.ndarray, slope: np.ndarray) -> np.ndarray:
    powered = np.power(np.maximum(media, 0.0), slope[..., None])
    half = np.power(ec, slope)[..., None]
    return powered / (powered + half)


def summarize_draws(draws: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """Mean and 2.5th to 97.5th percentile of each channel's per-draw next-ruble ROI."""
    return {
        channel: {
            "mean": float(np.mean(values)),
            "low": float(np.quantile(values, 0.025)),
            "high": float(np.quantile(values, 0.975)),
        }
        for channel, values in draws.items()
    }


def next_ruble_channel_roi(
    mmm, report, channels: list[str], holdout_weeks: int, *, group: str = "posterior"
) -> dict[str, np.ndarray]:
    """Per-draw spend-weighted finite difference of one ruble on the training window.

    The hill parameters stay the draws of `group`. A week counts when its spend
    is positive and the model's adstock window stays inside training, so the
    tail does not enter the holdout. The Analyzer check runs only on the posterior.
    """
    from meridian import backend
    from meridian.analysis.tensors import DataTensors
    from meridian.model.adstock_hill import AdstockTransformer, compute_decay_weights

    posterior = getattr(mmm.inference_data, group)
    n_channels = len(channels)
    missing = [name for name in ("alpha_m", "ec_m", "slope_m", "beta_gm") if name not in posterior]
    if missing:
        raise RublePlacementError(
            f"Meridian {group} is missing "
            + ", ".join(missing)
            + ", so one ruble cannot be placed on one training week."
        )
    alpha = _channel_draws(np.asarray(posterior["alpha_m"].values, dtype=float), n_channels, "alpha_m")
    ec = _channel_draws(np.asarray(posterior["ec_m"].values, dtype=float), n_channels, "ec_m")
    slope = _channel_draws(np.asarray(posterior["slope_m"].values, dtype=float), n_channels, "slope_m")
    beta = np.asarray(posterior["beta_gm"].values, dtype=float)
    if beta.ndim == 3:
        beta = beta.reshape(1, *beta.shape)
    media = np.asarray(mmm.media_tensors.media, dtype=float)
    spend = np.asarray(mmm.media_tensors.media_spend, dtype=float)
    scaled = np.asarray(mmm.media_tensors.media_scaled, dtype=float)
    if media.ndim != 3 or spend.shape != media.shape or scaled.shape != media.shape:
        raise RublePlacementError(
            f"Meridian media {media.shape} and spend {spend.shape} are not a weekly tensor, "
            "so one ruble cannot be placed on one training week."
        )
    if media.shape[0] != 1 or media.shape[-1] != n_channels:
        raise ValueError("This bake-off places the ruble on one national geo.")
    n_times = media.shape[1]
    n_train = n_times - holdout_weeks
    max_lag = int(mmm.model_spec.max_lag)
    horizon = max_lag + 1
    if n_train <= horizon:
        raise ValueError("The training window is shorter than the adstock horizon.")
    if beta.shape[-1] != n_channels or beta.ndim != 4 or beta.shape[2] != 1:
        raise ValueError(f"beta_gm shape {beta.shape} is not chain, draw, geo, channel.")
    scale = np.asarray(mmm.media_tensors.media_transformer.scale_factors_gm, dtype=float)
    stdev = float(np.asarray(mmm.model_context.kpi_transformer.population_scaled_stdev))
    population = float(np.asarray(mmm.input_data.population, dtype=float).reshape(-1)[0])
    revenue_per_kpi = mmm.input_data.revenue_per_kpi
    price = None if revenue_per_kpi is None else np.asarray(revenue_per_kpi, dtype=float)
    alpha_t = backend.to_tensor(alpha, dtype=backend.float_dtype)
    window = horizon
    l_range = backend.arange(window - 1, -1, -1, dtype=backend.float_dtype)
    weights = np.asarray(
        compute_decay_weights(alpha_t, l_range, window, "geometric", True),
        dtype=float,
    )
    lag_weights = np.flip(weights, axis=-1)
    transformer = AdstockTransformer(
        alpha=alpha_t,
        max_lag=max_lag,
        n_times_output=n_times,
        decay_functions="geometric",
    )
    base_adstock = np.asarray(
        transformer.forward(backend.to_tensor(scaled, dtype=backend.float_dtype)),
        dtype=float,
    )
    if base_adstock.shape != (alpha.shape[0], alpha.shape[1], 1, n_times, n_channels):
        raise ValueError(f"Adstock shape {base_adstock.shape} does not match the {group}.")
    last = n_train - horizon
    draws: dict[str, np.ndarray] = {}
    check: tuple[int, int, float] | None = None
    for channel_index, channel in enumerate(channels):
        weekly_spend = spend[0, :, channel_index]
        usable = [week for week in range(last + 1) if weekly_spend[week] > 0]
        if not usable:
            raise ValueError(f"No positive-spend training week has a complete horizon for {channel}.")
        deltas = []
        for week in usable:
            rate = media[0, week, channel_index] / weekly_spend[week]
            scaled_delta = rate / scale[0, channel_index]
            window_slice = slice(week, week + horizon)
            level = base_adstock[:, :, 0, window_slice, channel_index]
            bumped = level + scaled_delta * lag_weights[:, :, channel_index, :]
            difference = _hill(bumped, ec[:, :, channel_index], slope[:, :, channel_index]) - _hill(
                level, ec[:, :, channel_index], slope[:, :, channel_index]
            )
            revenue = difference * beta[:, :, 0, channel_index][..., None] * stdev * population
            if price is not None:
                revenue = revenue * price[0, window_slice]
            deltas.append(revenue.sum(axis=-1).reshape(-1))
        draws[channel] = np.average(np.stack(deltas), axis=0, weights=weekly_spend[usable])
        if check is None:
            check = (channel_index, usable[0], float(np.mean(deltas[0])))
    if check is None:
        raise RublePlacementError("One ruble cannot be placed on one training week.")
    if group != "posterior":
        return draws
    channel_index, week, fast_value = check
    base_outcome = np.asarray(
        report.expected_outcome(use_posterior=True, aggregate_geos=True, aggregate_times=False),
        dtype=float,
    )
    bumped_media = media.copy()
    bumped_media[0, week, channel_index] += media[0, week, channel_index] / spend[0, week, channel_index]
    new_outcome = np.asarray(
        report.expected_outcome(
            use_posterior=True,
            new_data=DataTensors(media=backend.to_tensor(bumped_media, dtype=backend.float_dtype)),
            aggregate_geos=True,
            aggregate_times=False,
        ),
        dtype=float,
    )
    if base_outcome.shape != new_outcome.shape or base_outcome.ndim != 3:
        raise ValueError(
            f"Analyzer outcome shapes {base_outcome.shape} and {new_outcome.shape} are not chain, draw, time."
        )
    analyzer_value = float((new_outcome - base_outcome)[..., week : week + horizon].sum(axis=-1).mean())
    if not np.isfinite(fast_value) or not np.isfinite(analyzer_value):
        raise ValueError("The week-local ruble is not finite.")
    if not np.isclose(fast_value, analyzer_value, rtol=1e-3, atol=1e-2):
        raise RuntimeError(
            "The week-local ruble does not match Analyzer.expected_outcome "
            f"({fast_value} vs {analyzer_value})."
        )
    return draws


def holdout_path(report, revenue: np.ndarray, holdout_weeks: int) -> tuple[list[float], list[float]]:
    """Observed revenue and the posterior-mean expected outcome on the holdout weeks."""
    expected = np.asarray(
        report.expected_outcome(use_posterior=True, aggregate_geos=True, aggregate_times=False),
        dtype=float,
    )
    if expected.ndim != 3:
        raise ValueError(f"Analyzer outcome shape {expected.shape} is not chain, draw, time.")
    predicted = expected.mean(axis=(0, 1))[-holdout_weeks:]
    actual = np.asarray(revenue, dtype=float)[-holdout_weeks:]
    return actual.tolist(), predicted.tolist()


def fit(weekly_csv: Path, manifest_path: Path, output_path: Path, *, sensitivity: bool, seed: int | None = None) -> dict:
    from meridian import backend
    from meridian.data import load
    from meridian.model import model, prior_distribution, spec

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_seed = int(manifest["seed"]) if seed is None else seed
    frame = prepare(weekly_csv)
    media = ["tv_exposure", "ooh_exposure", "print_exposure", "facebook_I", "search_clicks_P"]
    spend = ["tv_S", "ooh_S", "print_S", "facebook_S", "search_S"]
    channels = ["tv", "ooh", "print", "facebook", "search"]
    loader = load.DataFrameDataLoader(
        df=frame,
        kpi_type="revenue",
        coord_to_columns=load.CoordToColumns(
            geo="geo",
            time="time",
            kpi="revenue",
            media=media,
            media_spend=spend,
            controls=["competitor_sales_B", "newsletter", "event_flag"],
        ),
        media_to_channel=dict(zip(media, channels)),
        media_spend_to_channel=dict(zip(spend, channels)),
    )
    data = loader.load()
    scales = np.full(len(channels), 0.9, dtype=backend.np_float_dtype)
    if sensitivity:
        scales[channels.index("facebook")] = backend.np_float_dtype(1.8)
    prior = prior_distribution.PriorDistribution(
        roi_m=backend.tfd.LogNormal(
            loc=backend.np_float_dtype(0.2),
            scale=scales,
        )
    )
    model_spec = spec.ModelSpec(
        prior=prior,
        media_prior_type="roi",
        holdout_id=holdout_id(len(frame), int(manifest["holdout_weeks"])),
        max_lag=8,
        adstock_decay_spec="geometric",
    )
    mmm = model.Meridian(input_data=data, model_spec=model_spec)
    mmm.sample_prior(PRIOR_DRAWS, seed=run_seed)
    mmm.sample_posterior(
        n_chains=4,
        n_adapt=1000,
        n_burnin=500,
        n_keep=1000,
        seed=run_seed,
    )
    from meridian.analysis import analyzer

    report = analyzer.Analyzer(model_context=mmm.model_context, inference_data=mmm.inference_data)
    posterior_roi = next_ruble_channel_roi(mmm, report, channels, int(manifest["holdout_weeks"]), group="posterior")
    prior_roi = next_ruble_channel_roi(mmm, report, channels, int(manifest["holdout_weeks"]), group="prior")
    contraction = {channel: draw_contraction(prior_roi[channel], posterior_roi[channel]) for channel in channels}
    accuracy = report.predictive_accuracy()
    holdout_mape = float(
        accuracy.sel(metric="MAPE", geo_granularity="national", evaluation_set="Test")["value"].item()
    )
    rhat_values = [np.ravel(np.asarray(array, dtype=float)) for array in report.get_rhat().values()]
    finite = np.concatenate(rhat_values)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Meridian did not return a finite r_hat.")
    roi = summarize_draws(posterior_roi)
    actual, predicted = holdout_path(report, frame["revenue"].to_numpy(dtype=float), int(manifest["holdout_weeks"]))
    estimand = "marginal_roi_next_ruble"
    export = {
        "model": "meridian",
        "manifest": manifest,
        "estimand": estimand,
        "holdout_mape": holdout_mape,
        "holdout_actual": actual,
        "holdout_pred": predicted,
        "channel_roi": {channel: item["mean"] for channel, item in roi.items()},
        "channel_roi_low": {channel: item["low"] for channel, item in roi.items()},
        "channel_roi_high": {channel: item["high"] for channel, item in roi.items()},
        "interval": "2.5th to 97.5th percentile of posterior draws",
        "interval_kind": "posterior",
        "channel_contraction": contraction,
        "contraction_parameter": "marginal ROI",
        "r_hat_max": float(np.max(finite)),
        "prior_sensitivity_relative": None,
        "sensitivity_run": sensitivity,
        "seed": run_seed,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(export, indent=2), encoding="utf-8")
    if (
        sensitivity
        and estimand == "marginal_roi_next_ruble"
        and output_path.resolve() != BASE_EXPORT.resolve()
        and BASE_EXPORT.exists()
    ):
        base_export = json.loads(BASE_EXPORT.read_text(encoding="utf-8"))
        if base_export.get("estimand") == "marginal_roi_next_ruble":
            attach_prior_sensitivity(BASE_EXPORT, output_path)
    return export


def seed_check(weekly_csv: Path, manifest_path: Path, scratch: Path) -> dict:
    """Refit the base model on consecutive seeds and record how far the point ROI moves."""
    base_seed = int(json.loads(manifest_path.read_text(encoding="utf-8"))["seed"])
    seeds = [base_seed + offset for offset in range(SEED_CHECK_COUNT)]
    exports = [
        fit(weekly_csv, manifest_path, scratch / f"meridian_seed_{seed}.json", sensitivity=False, seed=seed)
        for seed in seeds
    ]
    payload = {
        "seeds": seeds,
        "channel_roi": {str(export["seed"]): export["channel_roi"] for export in exports},
        "r_hat_max": {str(export["seed"]): export["r_hat_max"] for export in exports},
        "holdout_mape": {str(export["seed"]): export["holdout_mape"] for export in exports},
        "point_roi_spread": seed_spread(exports),
    }
    SEED_CHECK_EXPORT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--weekly", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--sensitivity", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="Overrides the manifest seed.")
    parser.add_argument("--seed-check", type=Path, default=None, help="Scratch directory for a three-seed refit.")
    arguments = parser.parse_args()
    if arguments.seed_check is not None:
        arguments.seed_check.mkdir(parents=True, exist_ok=True)
        seed_check(arguments.weekly, arguments.manifest, arguments.seed_check)
    else:
        output = arguments.output
        if output is None:
            output = SENSITIVITY_EXPORT if arguments.sensitivity else BASE_EXPORT
        fit(arguments.weekly, arguments.manifest, output, sensitivity=arguments.sensitivity, seed=arguments.seed)
