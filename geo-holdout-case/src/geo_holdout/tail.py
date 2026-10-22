"""How the effect is spread across geo sizes. Reported, and it does not decide.

Both checks residualize the change on the pre-period mean with the theta of
the full read, held fixed. The stratum intervals enumerate every assignment
inside a stratum. The slope test asks whether the effect per geo-day grows
with log pre-test sales, against reassignments under a constant additive effect.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from geo_holdout.estimators import invert_test, picks_to_masks, stratified_picks


def residualized(pre: np.ndarray, delta: np.ndarray, treated: np.ndarray, theta: float) -> np.ndarray:
    center = float(pre[~treated].mean())
    return delta - theta * (pre - center)


@dataclass(frozen=True)
class StratumEffect:
    stratum: int
    median_pre: float
    daily_effect: float
    low: float
    high: float
    n_assignments: int

    def as_dict(self) -> dict[str, float | int]:
        return dict(self.__dict__)


def _difference(values: np.ndarray, masks: np.ndarray) -> np.ndarray:
    treated = (masks * values).sum(axis=1) / masks.sum(axis=1)
    control = (~masks * values).sum(axis=1) / (~masks).sum(axis=1)
    return treated - control


def stratum_effect(values: np.ndarray, treated: np.ndarray, *, stratum: int, median_pre: float) -> StratumEffect:
    """Daily effect in one stratum with the exact randomization interval over all its assignments."""
    n_treated = int(treated.sum())
    if n_treated < 1 or n_treated >= len(values):
        raise ValueError("A stratum needs treated and control geos.")
    picks = np.array(list(combinations(range(len(values)), n_treated)))
    masks = picks_to_masks(picks, len(values))
    point = float(_difference(values, treated[None, :])[0])

    def covers(tau: float) -> bool:
        shifted = values - tau * treated
        observed = float(_difference(shifted, treated[None, :])[0])
        low, high = np.quantile(_difference(shifted, masks), [0.025, 0.975])
        return bool(low <= observed <= high)

    null = _difference(values, masks)
    span = max(float(null.max() - null.min()), abs(point), 1.0)
    low, high = invert_test(covers, point, span, tolerance=max(0.01, span * 1e-4))
    return StratumEffect(stratum, median_pre, point, low, high, len(masks))


def stratum_effects(
    pre: np.ndarray,
    delta: np.ndarray,
    treated: np.ndarray,
    stratum_ids: np.ndarray,
    *,
    theta: float,
) -> list[StratumEffect]:
    values = residualized(pre, delta, treated, theta)
    return [
        stratum_effect(
            values[stratum_ids == stratum],
            treated[stratum_ids == stratum],
            stratum=int(stratum),
            median_pre=float(np.median(pre[stratum_ids == stratum])),
        )
        for stratum in np.unique(stratum_ids)
    ]


@dataclass(frozen=True)
class SlopeTest:
    slope: float
    p_value: float
    n_reassignments: int

    def as_dict(self) -> dict[str, float | int]:
        return dict(self.__dict__)


def _slopes(values: np.ndarray, log_pre: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Slope of values on log size among treated minus the slope among controls, per mask."""

    def slope(group: np.ndarray) -> np.ndarray:
        count = group.sum(axis=1)
        x_mean = (group * log_pre).sum(axis=1) / count
        y_mean = (group * values).sum(axis=1) / count
        cov = (group * log_pre * values).sum(axis=1) - count * x_mean * y_mean
        var = (group * log_pre**2).sum(axis=1) - count * x_mean**2
        return cov / var

    return slope(masks) - slope(~masks)


def slope_test(
    pre: np.ndarray,
    delta: np.ndarray,
    treated: np.ndarray,
    stratum_ids: np.ndarray,
    *,
    theta: float,
    n_reassignments: int,
    seed: int,
) -> SlopeTest:
    """Two-sided randomization p-value for an effect that changes with log geo size."""
    if np.any(pre <= 0):
        raise ValueError("The slope test needs positive pre-period sales.")
    values = residualized(pre, delta, treated, theta)
    log_pre = np.log(pre)
    observed = float(_slopes(values, log_pre, treated[None, :])[0])
    constant = float(_difference(values, treated[None, :])[0])
    null_values = values - constant * treated
    counts = {int(stratum): int(treated[stratum_ids == stratum].sum()) for stratum in np.unique(stratum_ids)}
    rng = np.random.default_rng(seed)
    masks = picks_to_masks(stratified_picks(stratum_ids, counts, n_reassignments, rng), len(pre))
    null = _slopes(null_values, log_pre, masks)
    p_value = float((1 + np.sum(np.abs(null) >= abs(observed))) / (1 + n_reassignments))
    return SlopeTest(observed, p_value, n_reassignments)
