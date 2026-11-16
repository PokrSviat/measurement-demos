"""Simulated series with a known interrupted-time-series effect or a known cutoff jump."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

BASELINE = 1000.0
TREND = 0.5
PERIOD = 52
# Two yearly pairs. The fit estimates them; they are not the effect.
FOURIER = ((40.0, 20.0), (15.0, 10.0))


def post_contrast(weeks: np.ndarray, t0: float) -> tuple[float, float]:
    """Number of post weeks and the sum of (week - t0) over them.

    A post week is a week at or after t0. The cumulative effect is
    n_post * level + sum_tau * slope.
    """
    post = weeks >= t0
    return float(np.sum(post)), float(np.sum(weeks[post] - t0))


def _ar1(n_weeks: int, rho: float, noise_sd: float, rng: np.random.Generator) -> np.ndarray:
    if noise_sd <= 0 or not -1 < rho < 1:
        raise ValueError("noise_sd must be positive and rho inside (-1, 1).")
    noise = np.empty(n_weeks, dtype=float)
    # noise_sd is the innovation scale. The first week is drawn from the stationary law.
    noise[0] = rng.normal(0.0, noise_sd / math.sqrt(1 - rho**2))
    for week in range(1, n_weeks):
        noise[week] = rho * noise[week - 1] + rng.normal(0.0, noise_sd)
    return noise


def _seasonality(weeks: np.ndarray) -> np.ndarray:
    seasonal = np.zeros(len(weeks), dtype=float)
    for harmonic, (sin_weight, cos_weight) in enumerate(FOURIER, start=1):
        angle = 2 * np.pi * harmonic * weeks / PERIOD
        seasonal += sin_weight * np.sin(angle) + cos_weight * np.cos(angle)
    return seasonal


def simulate_its(
    seed: int,
    n_weeks: int = 156,
    t0: int = 104,
    level: float = 60.0,
    slope: float = 1.5,
    rho: float = 0.6,
    noise_sd: float = 25.0,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Weekly sales: baseline, trend, two Fourier pairs, an effect from t0, and AR(1) noise.

    Weeks are 0 .. n_weeks-1. level=0 and slope=0 leave the pre-period mean unchanged.
    """
    if n_weeks <= 0 or not 0 <= t0 < n_weeks:
        raise ValueError("t0 must fall inside the week index.")
    rng = np.random.default_rng(seed)
    weeks = np.arange(n_weeks, dtype=float)
    post = weeks >= t0
    effect = level * post + slope * (weeks - t0) * post
    sales = BASELINE + TREND * weeks + _seasonality(weeks) + effect + _ar1(n_weeks, rho, noise_sd, rng)
    n_post, sum_tau = post_contrast(weeks, t0)
    frame = pd.DataFrame({"week": weeks.astype(int), "sales": sales})
    truth = {
        "baseline": BASELINE,
        "trend": TREND,
        "level": float(level),
        "slope": float(slope),
        "rho": float(rho),
        "noise_sd": float(noise_sd),
        "t0": int(t0),
        "cumulative": n_post * float(level) + sum_tau * float(slope),
    }
    return frame, truth


RDD_NOISE_SD = 8.0
TENURE_NOISE_SD = 0.4


def _smooth_spend(spend: np.ndarray) -> np.ndarray:
    """Smooth and nonlinear. There is no jump at the cutoff in this part."""
    centered = spend - 50.0
    return 8.0 + 0.45 * spend + 3.0 * np.sin(centered / 16.0)


def _tenure_years(spend: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return 1.2 + 0.05 * spend - 0.00015 * spend**2 + rng.normal(0.0, TENURE_NOISE_SD, size=len(spend))


def simulate_rdd(
    seed: int,
    n: int = 20000,
    cutoff: float = 50.0,
    tau: float = 2.0,
    bunching_share: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Last-year spend, a gold threshold, next-year spend, and tenure.

    spend_prior is right-skewed, with mass around the cutoff. gold is
    spend_prior >= cutoff. spend_next adds tau only through gold. tenure_years
    follows spend_prior with no jump. bunching_share moves that share of
    customers from [cutoff - 5, cutoff) into [cutoff, cutoff + 2). Their
    next-year spend still follows the pre-move spend, which is the free-delivery
    failure: the running variable was pushed over the threshold.
    """
    if n <= 0 or not 0 <= bunching_share <= 1:
        raise ValueError("n must be positive and bunching_share inside [0, 1].")
    rng = np.random.default_rng(seed)
    latent = rng.lognormal(mean=math.log(48.0), sigma=0.22, size=n)
    spend = latent.copy()
    below = np.flatnonzero((spend >= cutoff - 5.0) & (spend < cutoff))
    n_move = int(round(bunching_share * len(below)))
    if n_move:
        chosen = rng.choice(below, size=n_move, replace=False)
        spend[chosen] = rng.uniform(cutoff, cutoff + 2.0, size=n_move)
    gold = spend >= cutoff
    spend_next = _smooth_spend(latent) + tau * gold.astype(float) + rng.normal(0.0, RDD_NOISE_SD, size=n)
    frame = pd.DataFrame(
        {
            "spend_prior": spend,
            "gold": gold.astype(int),
            "spend_next": spend_next,
            "tenure_years": _tenure_years(spend, rng),
        }
    )
    truth = {
        "tau": float(tau),
        "cutoff": float(cutoff),
        "n": int(n),
        "bunching_share": float(bunching_share),
        "noise_sd": RDD_NOISE_SD,
    }
    return frame, truth
