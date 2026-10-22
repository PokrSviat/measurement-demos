"""QA gates for a fitted revenue model. The gates read a summary, not a live sampler."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FitSummary:
    r_hat_max: float
    ess_min: float
    divergences: int
    holdout_mape: float
    channel_roi: dict[str, float]


def qa_failures(summary: FitSummary, *, mape_cap: float = 0.5) -> list[str]:
    failures: list[str] = []
    if not math.isfinite(summary.r_hat_max):
        failures.append(f"r_hat_max {summary.r_hat_max} is not finite.")
    elif summary.r_hat_max >= 1.01:
        failures.append(f"r_hat_max {summary.r_hat_max:.3f} is at or above 1.01.")
    if not math.isfinite(summary.ess_min):
        failures.append(f"ess_min {summary.ess_min} is not finite.")
    elif summary.ess_min < 400:
        failures.append(f"ess_min {summary.ess_min:.0f} is below 400.")
    if summary.divergences > 0:
        failures.append(f"divergences {summary.divergences} are above 0.")
    if summary.holdout_mape > mape_cap:
        failures.append(f"holdout MAPE {summary.holdout_mape:.3f} is above {mape_cap:.3f}.")
    return failures


def prior_sensitivity(base_roi: float, widened_roi: float) -> float:
    if base_roi == 0:
        raise ValueError("Base ROI must be non-zero to measure a relative change.")
    return (widened_roi - base_roi) / abs(base_roi)


def sensitivity_is_material(relative_change: float, threshold: float = 0.25) -> bool:
    return abs(relative_change) >= threshold


def prior_moves(base_roi: dict[str, float], widened_roi: dict[str, float]) -> dict[str, float]:
    """Relative ROI move of every channel when its prior is widened. Channels must match."""
    if set(base_roi) != set(widened_roi):
        raise ValueError("A prior check needs the same channels in both fits.")
    return {channel: prior_sensitivity(base_roi[channel], widened_roi[channel]) for channel in sorted(base_roi)}


def residual_acf(residuals: np.ndarray, lags: int) -> list[float]:
    """Sample autocorrelation of the residuals at lags 1 through `lags`."""
    values = np.asarray(residuals, dtype=float)
    if lags < 1 or len(values) <= lags:
        raise ValueError("The ACF needs more weeks than lags.")
    centered = values - values.mean()
    denominator = float(np.dot(centered, centered))
    if denominator == 0:
        raise ValueError("Residuals with no variance have no autocorrelation.")
    return [float(np.dot(centered[lag:], centered[:-lag]) / denominator) for lag in range(1, lags + 1)]


def chi2_sf(statistic: float, dof: int) -> float:
    """Upper tail of the chi-square distribution for an integer number of degrees of freedom."""
    if dof < 1:
        raise ValueError("Degrees of freedom must be positive.")
    if statistic <= 0:
        return 1.0
    half = statistic / 2
    if dof % 2 == 0:
        term, total = 1.0, 1.0
        for index in range(1, dof // 2):
            term *= half / index
            total += term
        return float(math.exp(-half) * total)
    root = math.sqrt(statistic)
    tail = math.erfc(root / math.sqrt(2))
    term, total = root, 0.0
    for index in range(1, (dof - 1) // 2 + 1):
        total += term
        term *= statistic / (2 * index + 1)
    return float(tail + math.sqrt(2 / math.pi) * math.exp(-half) * total)


@dataclass(frozen=True)
class LjungBox:
    statistic: float
    dof: int
    p_value: float


def ljung_box(residuals: np.ndarray, lags: int, *, fitted_terms: int = 0) -> LjungBox:
    """Ljung-Box test of no autocorrelation up to `lags`. Fitted AR terms reduce the degrees of freedom."""
    acf = residual_acf(residuals, lags)
    n_weeks = len(residuals)
    statistic = n_weeks * (n_weeks + 2) * sum(value**2 / (n_weeks - lag) for lag, value in enumerate(acf, start=1))
    dof = lags - fitted_terms
    return LjungBox(float(statistic), dof, chi2_sf(statistic, dof))


def contraction(prior_sd: float, posterior_sd: float) -> float:
    """One minus posterior variance over prior variance. Near zero means the data did not move the prior."""
    if prior_sd <= 0 or posterior_sd < 0:
        raise ValueError("Standard deviations must be positive.")
    return 1.0 - (posterior_sd / prior_sd) ** 2


def draw_contraction(prior_draws: np.ndarray, posterior_draws: np.ndarray) -> float:
    """Contraction of one quantity from its prior and posterior draws."""
    return contraction(float(np.std(prior_draws)), float(np.std(posterior_draws)))


PAYBACK = 1.0
RECOVERY_LABEL_FLOOR = 0.90


def gated_payback(status: str, recovery_coverage: float | None, floor: float = RECOVERY_LABEL_FLOOR) -> str:
    """Interval status only. Recovery coverage does not license a budget label."""
    if status not in ("earns", "refuse"):
        return status
    if recovery_coverage is None or recovery_coverage < floor:
        return "recovery_gate"
    return status


def channel_payback(roi: float, roi_low: float, roi_high: float) -> str:
    """Refuse the next ruble only when the whole marginal-ROI interval is below payback.

    A point below payback whose interval includes payback is not a cut.
    The whole interval above payback is the only earns label.
    """
    if roi_high < roi_low:
        raise ValueError("ROI interval is reversed.")
    if roi_high < PAYBACK:
        return "refuse"
    if roi_low > PAYBACK:
        return "earns"
    return "unsettled"


@dataclass(frozen=True)
class RefreshDiff:
    r_hat_delta: float
    roi_rank_changed: bool
    previous_rank: tuple[str, ...]
    refreshed_rank: tuple[str, ...]


def _roi_rank(channel_roi: dict[str, float]) -> tuple[str, ...]:
    return tuple(sorted(channel_roi, key=lambda channel: (-channel_roi[channel], channel)))


def refresh_diff(previous: FitSummary, refreshed: FitSummary) -> RefreshDiff:
    if set(previous.channel_roi) != set(refreshed.channel_roi):
        raise ValueError("A refresh rank needs the same channels.")
    previous_rank = _roi_rank(previous.channel_roi)
    refreshed_rank = _roi_rank(refreshed.channel_roi)
    return RefreshDiff(
        r_hat_delta=refreshed.r_hat_max - previous.r_hat_max,
        roi_rank_changed=previous_rank != refreshed_rank,
        previous_rank=previous_rank,
        refreshed_rank=refreshed_rank,
    )
