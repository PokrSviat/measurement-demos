"""PyMC revenue model. Importing this module does not import PyMC."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from measurement.contract import CALENDAR_TERMS, model_regressors, spend_column_for
from measurement.transforms import ChannelCurve

L_MAX = 8
CONTROLS = ("newsletter", "competitor_sales_B", "event_flag", "holiday_flag", *CALENDAR_TERMS)
PEAK_SCALED_CONTROLS = frozenset({"newsletter", "competitor_sales_B"})
TARGET_ACCEPT = 0.95
RHO_BOUND = 0.9
PRIOR_DRAWS = 1000


def train_holdout_split(frame: pd.DataFrame, holdout_weeks: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if holdout_weeks < 1 or holdout_weeks >= len(frame):
        raise ValueError("holdout_weeks must leave a training window.")
    return frame.iloc[:-holdout_weeks].copy(), frame.iloc[-holdout_weeks:].copy()


def fit_pymc(
    train: pd.DataFrame,
    *,
    draws: int,
    tune: int,
    chains: int,
    seed: int,
    target_accept: float = TARGET_ACCEPT,
    beta_scale: dict[str, float] | None = None,
    controls: tuple[str, ...] = CONTROLS,
) -> dict[str, object]:
    """Fit geometric adstock and negative-exponential saturation in PyMC, with AR(1) errors.

    Media is divided by its training maximum before sampling. Saved saturation
    rates are converted back to the original regressor units. Newsletter and
    competitor sales are scaled the same way. Flags, the trend in years and the
    yearly harmonics are not scaled. The error of each week carries `rho` times
    the error of the week before, and rho = 0 is inside the prior.
    """
    import pymc as pm
    import pytensor.tensor as pt

    regressors = model_regressors()
    media = train[regressors].to_numpy(dtype=float)
    scale = media.max(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    media = media / scale
    control_names = tuple(controls)
    controls = train[list(control_names)].to_numpy(dtype=float)
    control_scale = np.ones(controls.shape[1], dtype=float)
    for index, name in enumerate(control_names):
        if name in PEAK_SCALED_CONTROLS:
            peak = float(controls[:, index].max())
            control_scale[index] = peak if peak > 0 else 1.0
    controls = controls / control_scale
    y = train["revenue"].to_numpy(dtype=float)
    n_obs, n_channels = media.shape

    with pm.Model():
        baseline = pm.Normal("baseline", mu=float(np.mean(y)), sigma=float(np.std(y)))
        alpha = pm.Beta("alpha", alpha=2, beta=2, shape=n_channels)
        lam = pm.HalfNormal("lam", sigma=1.0, shape=n_channels)
        beta_sigma = np.full(n_channels, float(np.std(y)))
        for index, regressor in enumerate(regressors):
            beta_sigma[index] *= float((beta_scale or {}).get(_channel_id(regressor), 1.0))
        # A normal prior lets a channel take a negative contribution. HalfNormal would
        # make "earns" an answer inside a prior that cannot go below zero.
        beta = pm.Normal("beta", mu=0.0, sigma=beta_sigma, shape=n_channels)
        control_beta = pm.Normal("control_beta", mu=0, sigma=float(np.std(y)), shape=controls.shape[1])
        sigma = pm.HalfNormal("sigma", sigma=float(np.std(y)))

        raw_weights = alpha[None, :] ** pt.arange(L_MAX)[:, None]
        weights = raw_weights / pt.sum(raw_weights, axis=0)
        media_tensor = pt.as_tensor_variable(media)
        adstocked = pt.zeros((n_obs, n_channels))
        for lag in range(L_MAX):
            if lag == 0:
                lagged = media_tensor
            else:
                lagged = pt.concatenate(
                    [pt.zeros((lag, n_channels)), media_tensor[:-lag]],
                    axis=0,
                )
            adstocked = adstocked + weights[lag] * lagged
        saturated = beta * (1 - pt.exp(-lam * adstocked))
        mu = baseline + pt.sum(saturated, axis=1) + pt.dot(pt.as_tensor_variable(controls), control_beta)
        rho = pm.Uniform("rho", lower=-RHO_BOUND, upper=RHO_BOUND)
        pm.Normal("revenue_first", mu=mu[0], sigma=sigma / pt.sqrt(1 - rho**2), observed=y[:1])
        pm.Normal("revenue", mu=mu[1:] + rho * (y[:-1] - mu[:-1]), sigma=sigma, observed=y[1:])
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=seed,
            target_accept=target_accept,
            progressbar=False,
        )
        prior = pm.sample_prior_predictive(draws=PRIOR_DRAWS, random_seed=seed)
    return {
        "idata": idata,
        "prior": prior,
        "regressors": regressors,
        "media_scale": scale,
        "control_scale": control_scale,
        "controls": control_names,
        "beta_prior_sd": beta_sigma,
    }


def curves_from_posterior(
    idata: object,
    regressors: list[str],
    media_scale: np.ndarray,
    control_scale: np.ndarray,
    controls: tuple[str, ...] = CONTROLS,
) -> tuple[list[ChannelCurve], float, dict[str, float]]:
    posterior = idata.posterior
    curves: list[ChannelCurve] = []
    for index, regressor in enumerate(regressors):
        lam_scaled = float(posterior["lam"].mean(dim=("chain", "draw")).values[index])
        curves.append(
            ChannelCurve(
                channel=_channel_id(regressor),
                regressor=regressor,
                spend_column=spend_column_for(regressor),
                alpha=float(posterior["alpha"].mean(dim=("chain", "draw")).values[index]),
                lam=lam_scaled / float(media_scale[index]),
                beta=float(posterior["beta"].mean(dim=("chain", "draw")).values[index]),
            )
        )
    baseline = float(posterior["baseline"].mean())
    control_means = posterior["control_beta"].mean(dim=("chain", "draw")).values
    control_coefs = {
        name: float(control_means[index]) / float(control_scale[index])
        for index, name in enumerate(controls)
    }
    return curves, baseline, control_coefs


def sampling_draws(
    idata: object,
    regressors: list[str],
    media_scale: np.ndarray,
    control_scale: np.ndarray,
    controls: tuple[str, ...] = CONTROLS,
    group: str = "posterior",
):
    """Draws of one inference group in original units. Lam and control coefficients are unscaled."""
    from measurement.transforms import PosteriorDraws

    posterior = getattr(idata, group)
    alpha = _sample_matrix(posterior["alpha"])
    lam = _sample_matrix(posterior["lam"]) / media_scale
    beta = _sample_matrix(posterior["beta"])
    baseline = np.asarray(posterior["baseline"].stack(sample=("chain", "draw")).values, dtype=float).reshape(-1)
    control_beta = _sample_matrix(posterior["control_beta"]) / control_scale
    return PosteriorDraws(
        channels=tuple(_channel_id(regressor) for regressor in regressors),
        alpha=alpha,
        lam=lam,
        beta=beta,
        baseline=baseline,
        controls=tuple(controls),
        control_beta=control_beta,
    )


def roi_bounds_from_posterior(
    train: pd.DataFrame,
    regressors: list[str],
    posterior,
) -> dict[str, dict[str, float]]:
    """Mean and 2.5th to 97.5th percentile of the marginal ROI of the next ruble."""
    return {
        channel: {
            "low": float(np.quantile(draws, 0.025)),
            "high": float(np.quantile(draws, 0.975)),
            "mean": float(np.mean(draws)),
        }
        for channel, draws in channel_roi_draws(train, regressors, posterior).items()
    }


def channel_roi_draws(train: pd.DataFrame, regressors: list[str], draws) -> dict[str, np.ndarray]:
    """Marginal ROI of the next ruble for every draw, per channel. Works for prior or posterior draws."""
    from measurement.transforms import marginal_roi_draws

    return {
        draws.channels[index]: marginal_roi_draws(
            train[regressor].to_numpy(dtype=float),
            train[spend_column_for(regressor)].to_numpy(dtype=float),
            draws.alpha[:, index],
            draws.lam[:, index],
            draws.beta[:, index],
        )
        for index, regressor in enumerate(regressors)
    }


def _sample_matrix(data: object) -> np.ndarray:
    dims = tuple(getattr(data, "dims", ()))
    if "chain" in dims and "draw" in dims:
        stacked = data.stack(sample=("chain", "draw"))
    elif "draw" in dims:
        stacked = data.stack(sample=("draw",))
    else:
        values = np.asarray(getattr(data, "values", data), dtype=float)
        return values.reshape(-1, *values.shape[1:]) if values.ndim >= 2 else values
    values = np.asarray(stacked.values, dtype=float)
    sample_axis = list(stacked.dims).index("sample")
    return np.moveaxis(values, sample_axis, 0)


def _channel_id(regressor: str) -> str:
    from measurement.contract import column_owner

    owner = column_owner(regressor)
    if owner is None:
        raise ValueError(regressor)
    return owner


def write_curve_table(curves: list[ChannelCurve], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([curve.__dict__ for curve in curves]).to_csv(path, index=False)
