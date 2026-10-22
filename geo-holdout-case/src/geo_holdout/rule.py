"""The decision interval of analysis plan version 5.

A stratified slope test asks whether the effect per geo-day grows with log
pre-test sales. If it rejects a constant effect at the plan's alpha, the
interval inverts hypotheses whose effect is proportional to pre-period sales;
otherwise it inverts constant effects per location-day. The statistic is the
same unweighted CUPED total either way, so the estimand does not change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geo_holdout.estimators import control_theta, randomization_bounds
from geo_holdout.tail import SlopeTest, slope_test


@dataclass(frozen=True)
class DecisionInterval:
    shape: str
    low: float
    high: float
    slope: SlopeTest

    @property
    def bounds(self) -> tuple[float, float]:
        return self.low, self.high


def choose_shape(slope: SlopeTest, alpha: float) -> str:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be inside (0, 1).")
    return "proportional" if slope.p_value < alpha else "additive"


def decision_interval(
    pre: np.ndarray,
    delta: np.ndarray,
    days: np.ndarray,
    treated: np.ndarray,
    stratum_ids: np.ndarray,
    *,
    treated_per_stratum: int | None,
    alpha: float,
    n_slope_reassignments: int,
    n_interval_reassignments: int,
    slope_seed: int,
    interval_seed: int,
) -> DecisionInterval:
    """Slope test, then the randomization interval for the shape it selects."""
    slope = slope_test(
        pre,
        delta,
        treated,
        stratum_ids,
        theta=control_theta(pre, delta, treated),
        n_reassignments=n_slope_reassignments,
        seed=slope_seed,
    )
    shape = choose_shape(slope, alpha)
    low, high = randomization_bounds(
        pre,
        delta,
        days,
        treated,
        stratum_ids,
        treated_per_stratum=treated_per_stratum,
        effect_shape=shape,
        n_simulations=n_interval_reassignments,
        seed=interval_seed,
    )
    return DecisionInterval(shape, low, high, slope)
