"""Python twin of robyn/next_ruble.R. R does not import this module."""

from __future__ import annotations

import numpy as np


def next_ruble_roi(
    media: np.ndarray,
    spend: np.ndarray,
    theta: float,
    alpha: float,
    gamma: float,
    coef: float,
    horizon: int = 8,
) -> float:
    """Spend-weighted finite difference of one ruble. Hill inflexion stays put.

    A week counts when its spend is positive and the horizon fits in the series.
    One ruble adds media at that week's media/spend rate. The difference is
    summed on that week and the following horizon weeks only.
    """
    weekly_media = np.asarray(media, dtype=float)
    weekly_spend = np.asarray(spend, dtype=float)
    if weekly_media.shape != weekly_spend.shape or weekly_media.ndim != 1:
        raise ValueError("Spend and media must share a week index.")
    if horizon < 1 or len(weekly_media) <= horizon:
        raise ValueError("Need a week with a complete horizon.")
    if not all(np.isfinite(value) for value in (theta, alpha, gamma, coef)):
        raise ValueError("Coefficients must be finite.")
    decayed = _geometric_decay(weekly_media, theta)
    inflexion = float(np.max(decayed)) * gamma
    base = coef * _hill(decayed, alpha, inflexion)
    last = len(weekly_media) - horizon
    usable = weekly_spend[: last + 1] > 0
    if not usable.any():
        raise ValueError("No positive-spend week has a complete horizon.")
    deltas = np.empty(int(usable.sum()), dtype=float)
    slot = 0
    for week in range(last + 1):
        if weekly_spend[week] <= 0:
            continue
        bumped = weekly_media.copy()
        bumped[week] = bumped[week] + weekly_media[week] / weekly_spend[week]
        changed = coef * _hill(_geometric_decay(bumped, theta), alpha, inflexion)
        end = week + horizon
        deltas[slot] = float(np.sum(changed[week:end] - base[week:end]))
        slot += 1
    return float(np.average(deltas, weights=weekly_spend[: last + 1][usable]))


def _geometric_decay(media: np.ndarray, theta: float) -> np.ndarray:
    decayed = np.empty(len(media), dtype=float)
    decayed[0] = media[0]
    for index in range(1, len(media)):
        decayed[index] = media[index] + theta * decayed[index - 1]
    return decayed


def _hill(decayed: np.ndarray, alpha: float, inflexion: float) -> np.ndarray:
    powered = np.power(decayed, alpha)
    return powered / (powered + inflexion**alpha)
