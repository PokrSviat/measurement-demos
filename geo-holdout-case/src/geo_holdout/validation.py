"""Decision power and placebo check of the cap rule on the pre-test, where the effect is known.

Each draw assigns test geos inside pretest-volume strata on the pseudo
experiment, plants an effect whose true cost per incremental sale is set in
the analysis plan, and runs the same two-stage rule as the case: the slope
test, the interval for the shape it selects, and the cap. The pseudo window
has no weeks after it, so the check covers the test window only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from budget_rule.cost import cost_against_cap

from geo_holdout.design import PseudoExperiment, pseudo_experiment
from geo_holdout.estimators import EFFECT_SHAPES, cuped_total, effect_profile, picks_to_masks, stratified_picks
from geo_holdout.rule import decision_interval

SHAPES = EFFECT_SHAPES
SLOPE_SEED_OFFSET = 500_000


def mc_se(share: float, n_draws: int) -> float:
    """Monte Carlo standard error of a share over `n_draws` independent draws."""
    if n_draws < 1:
        raise ValueError("A share needs at least one draw.")
    return math.sqrt(share * (1 - share) / n_draws)


@dataclass(frozen=True)
class ValidationCell:
    shape: str
    true_cost: float | None
    daily_effect: float
    n_draws: int
    coverage: float
    mean_bias: float
    weighted_mean_bias: float
    scale_share: float
    correct_share: float
    proportional_share: float

    @property
    def scale_se(self) -> float:
        return mc_se(self.scale_share, self.n_draws)

    def as_dict(self) -> dict[str, str | float | int | None]:
        return {**self.__dict__, "scale_se": self.scale_se}


def planted_effect(pre: np.ndarray, treated: np.ndarray, shape: str, daily_effect: float) -> np.ndarray:
    """Per-geo daily effect whose mean over the treated geos is `daily_effect`.

    Additive gives every geo the same effect. Proportional scales it with pre-period sales.
    """
    profile = effect_profile(pre, shape)
    return daily_effect * profile / float(profile[treated].mean())


def daily_effect_for_cost(test_cost: float, true_cost: float, treated_location_days: float) -> float:
    """Mean daily effect per treated geo-day at which test cost over true incremental sales is `true_cost`."""
    if treated_location_days <= 0 or true_cost <= 0:
        raise ValueError("Treated days and the true cost must be positive.")
    return test_cost / (true_cost * treated_location_days)


def cap_call(low: float, truth: float, *, test_cost: float, cap: float) -> tuple[bool, bool]:
    """Whether the rule scales at this lower bound, and whether that matches the truth."""
    fund = cost_against_cap(low, test_cost, cap).clears
    return fund, fund == cost_against_cap(truth, test_cost, cap).clears


def validate_cell(
    pseudo: PseudoExperiment,
    *,
    shape: str,
    true_cost: float | None,
    treated_per_stratum: int,
    test_cost: float,
    max_cost_per_incremental: float,
    shape_alpha: float,
    n_slope_reassignments: int,
    n_draws: int,
    n_reassignments: int,
    seed: int,
) -> ValidationCell:
    """Coverage, bias and decision accuracy of the case rule at one true cost. None plants no effect."""
    if n_draws < 1:
        raise ValueError("Validation needs at least one draw.")
    rng = np.random.default_rng(seed)
    masks = picks_to_masks(
        stratified_picks(pseudo.stratum_ids, treated_per_stratum, n_draws, rng),
        len(pseudo.pre),
    )
    covered = scaled = correct = proportional = 0
    bias: list[float] = []
    weighted_bias: list[float] = []
    daily = 0.0
    for index, treated in enumerate(masks):
        treated_days = float(pseudo.days[treated].sum())
        daily = 0.0 if true_cost is None else daily_effect_for_cost(test_cost, true_cost, treated_days)
        effect = planted_effect(pseudo.pre, treated, shape, daily)
        injected = pseudo.delta.copy()
        injected[treated] += effect[treated]
        truth = float(np.sum(effect[treated] * pseudo.days[treated]))
        chosen = decision_interval(
            pseudo.pre,
            injected,
            pseudo.days,
            treated,
            pseudo.stratum_ids,
            treated_per_stratum=treated_per_stratum,
            alpha=shape_alpha,
            n_slope_reassignments=n_slope_reassignments,
            n_interval_reassignments=n_reassignments,
            slope_seed=seed + SLOPE_SEED_OFFSET + index,
            interval_seed=seed + 1 + index,
        )
        low, high = chosen.bounds
        proportional += int(chosen.shape == "proportional")
        covered += int(low <= truth <= high)
        bias.append(cuped_total(pseudo.pre, injected, pseudo.days, treated) - truth)
        weighted_bias.append(
            cuped_total(pseudo.pre, injected, pseudo.days, treated, weights=pseudo.pre) - truth
        )
        fund, right = cap_call(low, truth, test_cost=test_cost, cap=max_cost_per_incremental)
        scaled += int(fund)
        correct += int(right)
    return ValidationCell(
        shape=shape,
        true_cost=true_cost,
        daily_effect=daily,
        n_draws=n_draws,
        coverage=covered / n_draws,
        mean_bias=float(np.mean(bias)),
        weighted_mean_bias=float(np.mean(weighted_bias)),
        scale_share=scaled / n_draws,
        correct_share=correct / n_draws,
        proportional_share=proportional / n_draws,
    )


def run_validation(
    pretest: pd.DataFrame,
    *,
    stratum_size: int,
    treated_per_stratum: int,
    duration_days: int,
    true_costs: list[float],
    test_cost: float,
    max_cost_per_incremental: float,
    shape_alpha: float,
    n_slope_reassignments: int,
    n_draws: int,
    n_reassignments: int,
    seed: int,
) -> list[ValidationCell]:
    """One null cell and one cell per true cost, for each effect shape."""
    pseudo = pseudo_experiment(pretest, stratum_size=stratum_size, duration_days=duration_days)
    grid: list[float | None] = [None, *true_costs]
    return [
        validate_cell(
            pseudo,
            shape=shape,
            true_cost=cost,
            treated_per_stratum=treated_per_stratum,
            test_cost=test_cost,
            max_cost_per_incremental=max_cost_per_incremental,
            shape_alpha=shape_alpha,
            n_slope_reassignments=n_slope_reassignments,
            n_draws=n_draws,
            n_reassignments=n_reassignments,
            seed=seed + 1000 * position,
        )
        for position, (shape, cost) in enumerate((shape, cost) for shape in SHAPES for cost in grid)
    ]
