"""OLS interrupted time series, plus a structural pre-period forecast."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

from quasi.dgp import PERIOD, post_contrast

Z_95 = float(norm.ppf(0.975))


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float


@dataclass(frozen=True)
class ITSFit:
    level: Interval
    slope: Interval
    cumulative: Interval


def newey_west_lags(n: int) -> int:
    """floor(4 * (n/100) ** (2/9)), the usual Newey-West lag rule."""
    return int(math.floor(4 * (n / 100) ** (2 / 9)))


def _design(weeks: np.ndarray, t0: float, n_fourier: int) -> pd.DataFrame:
    post = (weeks >= t0).astype(float)
    frame = pd.DataFrame(
        {
            "intercept": 1.0,
            "week": weeks,
            "level": post,
            "slope": (weeks - t0) * post,
        }
    )
    for harmonic in range(1, n_fourier + 1):
        angle = 2 * np.pi * harmonic * weeks / PERIOD
        frame[f"sin_{harmonic}"] = np.sin(angle)
        frame[f"cos_{harmonic}"] = np.cos(angle)
    return frame


def _interval(estimate: float, se: float) -> Interval:
    half = Z_95 * se
    return Interval(estimate, estimate - half, estimate + half)


def fit_its(
    df: pd.DataFrame,
    t0: int,
    n_fourier: int = 2,
    cov: str = "HAC",
    maxlags: int | None = None,
) -> ITSFit:
    """Level and slope changes from t0, and their sum over the post weeks.

    Rows are ordered by week. A post week is a week at or after t0, and the
    slope column is (week - t0) on those weeks. cov="HAC" uses Newey-West;
    maxlags defaults to floor(4 * (n/100) ** (2/9)). cov="nonrobust" is the
    iid OLS variance. The cumulative standard error is the variance of that
    linear combination. On a short persistent series that lag rule understates
    the level variance. Adding lags does not repair it; a longer series does.
    """
    if cov not in {"HAC", "nonrobust"}:
        raise ValueError('cov must be "HAC" or "nonrobust".')
    frame = df.sort_values("week")
    weeks = frame["week"].to_numpy(dtype=float)
    design = _design(weeks, t0, n_fourier)
    model = sm.OLS(frame["sales"].to_numpy(dtype=float), design)
    if cov == "HAC":
        lags = newey_west_lags(len(frame)) if maxlags is None else maxlags
        result = model.fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    else:
        result = model.fit(cov_type="nonrobust")
    level = float(result.params["level"])
    slope = float(result.params["slope"])
    covariance = result.cov_params()
    n_post, sum_tau = post_contrast(weeks, t0)
    cumulative = n_post * level + sum_tau * slope
    cumulative_var = (
        n_post**2 * float(covariance.loc["level", "level"])
        + sum_tau**2 * float(covariance.loc["slope", "slope"])
        + 2 * n_post * sum_tau * float(covariance.loc["level", "slope"])
    )
    return ITSFit(
        level=_interval(level, math.sqrt(float(covariance.loc["level", "level"]))),
        slope=_interval(slope, math.sqrt(float(covariance.loc["slope", "slope"]))),
        cumulative=_interval(cumulative, math.sqrt(cumulative_var)),
    )


def observed_and_counterfactual(
    df: pd.DataFrame, t0: int, n_fourier: int = 2
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weeks, observed sales, and the OLS fit with the level and slope set to zero."""
    frame = df.sort_values("week")
    weeks = frame["week"].to_numpy(dtype=float)
    design = _design(weeks, t0, n_fourier)
    result = sm.OLS(frame["sales"].to_numpy(dtype=float), design).fit()
    without = design.copy()
    without["level"] = 0.0
    without["slope"] = 0.0
    counterfactual = np.asarray(result.predict(without), dtype=float)
    return weeks, frame["sales"].to_numpy(dtype=float), counterfactual


@dataclass(frozen=True)
class PlaceboDates:
    estimates: tuple[float, ...]
    share: float


def placebo_dates(
    df: pd.DataFrame,
    t0: int,
    first: int = 52,
    last: int | None = None,
    step: int = 2,
) -> PlaceboDates:
    """Refit the level change on pre-t0 weeks at fake intervention dates.

    Fake dates run from `first` through `last`, or through t0 - 26 when `last`
    is omitted. Each date keeps at least 26 post weeks inside the pre-period.
    The share is the fraction of those level estimates whose absolute value is
    at least the absolute value of the real level estimate.
    """
    if step <= 0:
        raise ValueError("step must be positive.")
    latest = t0 - 26 if last is None else min(last, t0 - 26)
    if first > latest:
        raise ValueError("No fake date leaves 26 post weeks before t0.")
    pre = df.loc[df["week"] < t0]
    real = abs(fit_its(df, t0).level.estimate)
    estimates = tuple(
        fit_its(pre, date).level.estimate for date in range(first, latest + 1, step)
    )
    share = float(np.mean([abs(estimate) >= real for estimate in estimates]))
    return PlaceboDates(estimates, share)


@dataclass(frozen=True)
class StructuralForecast:
    counterfactual: tuple[float, ...]
    low: tuple[float, ...]
    high: tuple[float, ...]
    cumulative: float


def _control_matrix(frame: pd.DataFrame, controls) -> np.ndarray | None:
    names = [controls] if isinstance(controls, str) else list(controls)
    if any(not isinstance(name, str) for name in names):
        raise TypeError("controls must be column names.")
    if not names:
        return None
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise ValueError("Unknown controls: " + ", ".join(missing) + ".")
    return frame.loc[:, names].to_numpy(dtype=float)


def its_structural(
    df: pd.DataFrame,
    t0: int,
    controls,
    seasonal: int = PERIOD,
) -> StructuralForecast:
    """CausalImpact-style, Kalman filter and MLE, not Bayesian.

    A local linear trend, a seasonal component, and the exogenous controls
    are fit on weeks before t0. The post-period counterfactual is that
    model's forecast. cumulative is observed post sales minus the forecast.
    """
    if seasonal < 2:
        raise ValueError("seasonal period must be at least 2.")
    frame = df.sort_values("week")
    pre = frame.loc[frame["week"] < t0]
    post = frame.loc[frame["week"] >= t0]
    if pre.empty or post.empty:
        raise ValueError("t0 must leave weeks before it and weeks from it on.")
    exog_pre = _control_matrix(pre, controls)
    exog_post = _control_matrix(post, controls)
    model = sm.tsa.UnobservedComponents(
        pre["sales"].to_numpy(dtype=float),
        level="local linear trend",
        seasonal=seasonal,
        exog=exog_pre,
    )
    result = model.fit(disp=False)
    if exog_post is None:
        forecast = result.get_forecast(steps=len(post))
    else:
        forecast = result.get_forecast(steps=len(post), exog=exog_post)
    counterfactual = np.asarray(forecast.predicted_mean, dtype=float)
    bounds = np.asarray(forecast.conf_int(), dtype=float)
    observed = post["sales"].to_numpy(dtype=float)
    return StructuralForecast(
        tuple(float(value) for value in counterfactual),
        tuple(float(value) for value in bounds[:, 0]),
        tuple(float(value) for value in bounds[:, 1]),
        float(np.sum(observed - counterfactual)),
    )
