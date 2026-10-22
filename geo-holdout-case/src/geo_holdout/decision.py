"""Budget rule for the geo holdout."""

from __future__ import annotations

import math
from dataclasses import dataclass

from budget_rule.cost import cost_against_cap

from geo_holdout.estimators import EffectEstimate


@dataclass(frozen=True)
class BudgetDecision:
    fund: bool
    recommendation: str
    cost_per_incremental: float | None
    reason: str
    clears_cap: bool
    confirmatory: bool

    def as_dict(self) -> dict[str, bool | str | float | None]:
        return {
            "fund": self.fund,
            "recommendation": self.recommendation,
            "cost_per_incremental": self.cost_per_incremental,
            "reason": self.reason,
            "clears_cap": self.clears_cap,
            "confirmatory": self.confirmatory,
        }


def decide(
    cumulative: EffectEstimate,
    *,
    test_cost: float,
    max_cost_per_incremental: float,
    confirmatory: bool = True,
) -> BudgetDecision:
    """Fund only when the cap clears at the randomization lower bound and the plan is confirmatory.

    The cumulative window runs from the first test day through the last day
    after the test, so sales pulled forward from the later weeks lower the
    bound that decides. Power is a design check and does not decide.
    A post-exploration plan can clear the cap and still not fund.
    """
    if test_cost < 0 or max_cost_per_incremental <= 0:
        raise ValueError("Cost assumptions must be positive.")
    if not math.isfinite(cumulative.incremental_low):
        raise ValueError("The decision bound must be finite.")
    cost = cost_against_cap(cumulative.incremental_low, test_cost, max_cost_per_incremental)
    if cost.clears and confirmatory:
        return BudgetDecision(
            fund=True,
            recommendation="Scale the tactic.",
            cost_per_incremental=cost.cost_per_incremental,
            reason=(
                "The cumulative incremental interval is positive and the cost per incremental sale "
                "at its lower bound is within the cap."
            ),
            clears_cap=True,
            confirmatory=True,
        )
    if cost.clears:
        return BudgetDecision(
            fund=False,
            recommendation="Hold. Confirm on a new panel.",
            cost_per_incremental=cost.cost_per_incremental,
            reason=(
                "The cumulative incremental interval is positive and the cost per incremental sale "
                "at its lower bound is within the cap. This plan was written after the slope on this "
                "panel, so that is not a confirmatory scale."
            ),
            clears_cap=True,
            confirmatory=False,
        )
    reason = (
        "The cumulative incremental interval includes zero or a loss."
        if cost.reason == "interval_includes_zero"
        else "The cost per incremental sale at the cumulative lower bound is above the cap."
    )
    return BudgetDecision(
        fund=False,
        recommendation="Do not scale.",
        cost_per_incremental=cost.cost_per_incremental,
        reason=reason,
        clears_cap=False,
        confirmatory=confirmatory,
    )
