"""Adstock, saturation, and the revenue forecast used by the calculator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def geometric_adstock(values: np.ndarray, alpha: float, l_max: int = 8) -> np.ndarray:
    if not 0 <= alpha < 1:
        raise ValueError("alpha must be in [0, 1).")
    weights = np.array([alpha**lag for lag in range(l_max)], dtype=float)
    weights /= weights.sum()
    output = np.zeros(len(values), dtype=float)
    for lag, weight in enumerate(weights):
        if lag == 0:
            output += weight * values
        else:
            output[lag:] += weight * values[:-lag]
    return output


def saturation(adstocked: np.ndarray, lam: float) -> np.ndarray:
    if lam <= 0:
        raise ValueError("lam must be positive.")
    return 1.0 - np.exp(-lam * np.maximum(adstocked, 0.0))


@dataclass(frozen=True)
class ChannelCurve:
    channel: str
    regressor: str
    spend_column: str
    alpha: float
    lam: float
    beta: float


def channel_contribution(regressor: np.ndarray, curve: ChannelCurve) -> np.ndarray:
    return curve.beta * saturation(geometric_adstock(regressor, curve.alpha), curve.lam)


def predict_revenue(
    frame: pd.DataFrame,
    curves: list[ChannelCurve],
    *,
    baseline: float,
    control_coefs: dict[str, float],
) -> np.ndarray:
    prediction = np.full(len(frame), baseline, dtype=float)
    for curve in curves:
        prediction += channel_contribution(frame[curve.regressor].to_numpy(dtype=float), curve)
    for column, coef in control_coefs.items():
        prediction += coef * frame[column].to_numpy(dtype=float)
    return prediction


def apply_planned_spend(
    frame: pd.DataFrame,
    planned_spend: dict[str, float],
    curves: list[ChannelCurve],
) -> pd.DataFrame:
    """Set the last week's exposure from planned spend and the history's exposure per spend.

    Earlier weeks stay observed so adstock still sees carryover. A channel with no positive
    spend in the history cannot be scaled.
    """
    if frame.empty:
        raise ValueError("Spend history is empty.")
    updated = frame.copy()
    for curve in curves:
        planned = planned_spend.get(curve.channel, np.nan)
        if planned is None or (isinstance(planned, float) and np.isnan(planned)):
            continue
        planned = float(planned)
        spend = updated[curve.spend_column].to_numpy(dtype=float)
        regressor = updated[curve.regressor].to_numpy(dtype=float).copy()
        positive = spend > 0
        if not positive.any():
            raise ValueError(
                f"{curve.channel} has no positive spend in the history, so planned spend cannot be converted."
            )
        rate = float(regressor[positive].sum() / spend[positive].sum())
        spend = spend.copy()
        regressor[-1] = rate * planned
        spend[-1] = planned
        updated[curve.regressor] = regressor
        updated[curve.spend_column] = spend
    return updated


def naive_holdout_predictions(train_revenue: np.ndarray, holdout_weeks: int) -> dict[str, np.ndarray]:
    """Training-mean and lag-52 forecasts for the holdout window."""
    if holdout_weeks < 1:
        raise ValueError("holdout_weeks must be positive.")
    if len(train_revenue) < 52:
        raise ValueError("The lag-52 forecast needs 52 training weeks.")
    mean_forecast = np.full(holdout_weeks, float(np.mean(train_revenue)))
    start = len(train_revenue) - 52
    lag_forecast = np.asarray(train_revenue[start : start + holdout_weeks], dtype=float)
    return {"training_mean": mean_forecast, "lag_52": lag_forecast}


@dataclass(frozen=True)
class PosteriorDraws:
    channels: tuple[str, ...]
    alpha: np.ndarray
    lam: np.ndarray
    beta: np.ndarray
    baseline: np.ndarray
    controls: tuple[str, ...]
    control_beta: np.ndarray


def load_posterior(path: Path) -> PosteriorDraws:
    payload = np.load(path, allow_pickle=False)
    return PosteriorDraws(
        channels=tuple(str(item) for item in payload["channels"]),
        alpha=np.asarray(payload["alpha"], dtype=float),
        lam=np.asarray(payload["lam"], dtype=float),
        beta=np.asarray(payload["beta"], dtype=float),
        baseline=np.asarray(payload["baseline"], dtype=float),
        controls=tuple(str(item) for item in payload["controls"]),
        control_beta=np.asarray(payload["control_beta"], dtype=float),
    )


def prediction_from_draws(
    frame: pd.DataFrame,
    curves: list[ChannelCurve],
    posterior: PosteriorDraws,
) -> np.ndarray:
    """Mean revenue path across posterior draws.

    Averaging the saturated response, not evaluating it at the mean parameters.
    """
    order = [posterior.channels.index(curve.channel) for curve in curves]
    total = np.zeros(len(frame), dtype=float)
    n_draws = len(posterior.baseline)
    for index in range(n_draws):
        draw_curves = [
            ChannelCurve(
                curve.channel,
                curve.regressor,
                curve.spend_column,
                float(posterior.alpha[index, order[slot]]),
                float(posterior.lam[index, order[slot]]),
                float(posterior.beta[index, order[slot]]),
            )
            for slot, curve in enumerate(curves)
        ]
        coefs = {
            name: float(posterior.control_beta[index, control_index])
            for control_index, name in enumerate(posterior.controls)
        }
        total += predict_revenue(
            frame,
            draw_curves,
            baseline=float(posterior.baseline[index]),
            control_coefs=coefs,
        )
    return total / n_draws


def ar_forecast(structural: np.ndarray, last_residual: float, rho: float) -> np.ndarray:
    """Structural forecast plus the last training error decayed by rho for each week ahead."""
    if not -1 < rho < 1:
        raise ValueError("rho must be inside (-1, 1).")
    path = np.asarray(structural, dtype=float)
    return path + last_residual * np.power(rho, np.arange(1, len(path) + 1))


def marginal_roi_draws(
    regressor: np.ndarray,
    spend: np.ndarray,
    alpha: np.ndarray,
    lam: np.ndarray,
    beta: np.ndarray,
    l_max: int = 8,
) -> np.ndarray:
    """Spend-weighted marginal ROI of one extra ruble at observed spend.

    A week counts only when its full adstock horizon sits inside the series.
    Average contribution over total spend is a different estimand.
    """
    media = np.asarray(regressor, dtype=float)
    weekly_spend = np.asarray(spend, dtype=float)
    if media.shape != weekly_spend.shape:
        raise ValueError("Spend and the regressor must share a week index.")
    if l_max < 1 or len(media) <= l_max:
        raise ValueError("Marginal ROI needs a week with a complete adstock horizon.")
    last = len(media) - l_max
    usable = weekly_spend[: last + 1] > 0
    if not usable.any():
        raise ValueError("No positive-spend week has a complete adstock horizon.")
    lags = np.arange(l_max)
    draws = np.empty(len(alpha), dtype=float)
    for index, (draw_alpha, draw_lam, draw_beta) in enumerate(zip(alpha, lam, beta, strict=True)):
        weights = np.power(float(draw_alpha), lags)
        weights /= weights.sum()
        adstocked = geometric_adstock(media, float(draw_alpha), l_max)
        slope = float(draw_beta) * float(draw_lam) * np.exp(-float(draw_lam) * np.maximum(adstocked, 0.0))
        weekly = np.zeros(last + 1, dtype=float)
        rate = np.zeros(len(media), dtype=float)
        positive = weekly_spend > 0
        rate[positive] = media[positive] / weekly_spend[positive]
        for week in range(last + 1):
            if weekly_spend[week] <= 0:
                continue
            weekly[week] = rate[week] * float(np.sum(weights * slope[week : week + l_max]))
        draws[index] = float(np.average(weekly[usable], weights=weekly_spend[: last + 1][usable]))
    return draws


def roi_draws(
    regressor: np.ndarray,
    spend: float,
    alpha: np.ndarray,
    lam: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    """One ROI per draw: training contribution divided by training spend."""
    if spend <= 0:
        raise ValueError("Spend must be positive.")
    draws = np.empty(len(alpha), dtype=float)
    media = np.asarray(regressor, dtype=float)
    for index, (draw_alpha, draw_lam, draw_beta) in enumerate(zip(alpha, lam, beta, strict=True)):
        contribution = channel_contribution(
            media,
            ChannelCurve("channel", "regressor", "spend", float(draw_alpha), float(draw_lam), float(draw_beta)),
        )
        draws[index] = float(np.sum(contribution)) / spend
    return draws


def forecast_errors(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    error = predicted - actual
    denom = np.maximum(np.abs(actual), 1.0)
    return {
        "mae": float(np.mean(np.abs(error))),
        "mape": float(np.mean(np.abs(error) / denom)),
    }


def mix_prediction(
    history: pd.DataFrame,
    curves: list[ChannelCurve],
    *,
    baseline: float,
    control_coefs: dict[str, float],
    budget: float,
    shares: dict[str, float],
    l_max: int = 8,
    posterior: PosteriorDraws | None = None,
) -> float:
    """Forecast the last week. With draws, the number is the mean across draws."""
    if budget < 0:
        raise ValueError("budget must be non-negative.")
    if l_max < 1:
        raise ValueError("l_max must be positive.")
    if len(history) < l_max:
        raise ValueError(f"Mix history must cover at least {l_max} weeks.")
    share_sum = float(sum(shares.values()))
    if abs(share_sum - 1.0) > 1e-6:
        raise ValueError("Channel shares must sum to 1.")
    planned = {curve.channel: budget * shares[curve.channel] for curve in curves}
    planned_frame = apply_planned_spend(history, planned, curves)
    if posterior is not None:
        return float(prediction_from_draws(planned_frame, curves, posterior)[-1])
    return float(
        predict_revenue(
            planned_frame,
            curves,
            baseline=baseline,
            control_coefs=control_coefs,
        )[-1]
    )
