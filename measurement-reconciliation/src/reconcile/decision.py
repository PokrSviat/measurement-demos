"""The experiment's lower bound decides the budget. Models do not."""

from __future__ import annotations

import math
from dataclasses import dataclass

from budget_rule.cost import cost_against_cap

from reconcile.methods import MethodEstimate


def overlaps(left: MethodEstimate, right: MethodEstimate) -> bool:
    """Intervals overlap only when both have width.

    A point mass lies inside another interval by arithmetic and is not agreement.
    """
    if left.low >= left.high or right.low >= right.high:
        return False
    return left.low <= right.high and right.low <= left.high


@dataclass(frozen=True)
class Reconciliation:
    agree: bool
    decision: str
    driver: str
    test_cost: float
    max_cost_per_incremental: float
    cost_per_incremental: float | None

    def as_dict(self) -> dict[str, str | bool | float | None]:
        return {
            "agree": self.agree,
            "decision": self.decision,
            "driver": self.driver,
            "test_cost": self.test_cost,
            "max_cost_per_incremental": self.max_cost_per_incremental,
            "cost_per_incremental": self.cost_per_incremental,
        }


def reconcile(
    geo: MethodEstimate,
    mmm: MethodEstimate,
    mta: MethodEstimate,
    *,
    test_cost: float,
    max_cost_per_incremental: float,
) -> Reconciliation:
    """Scale only when the experiment lower bound is positive and clears the cap.

    Overlap of the model intervals does not change the cost test. A zero-width
    interval does not count as agreement.
    """
    estimates = (geo, mmm, mta)
    if any(item.estimand != "incremental_conversions" for item in estimates):
        raise ValueError("Methods can be reconciled only on incremental_conversions.")
    if test_cost < 0 or max_cost_per_incremental <= 0:
        raise ValueError("Cost assumptions must be positive.")
    if not math.isfinite(geo.low):
        raise ValueError("The experiment lower bound must be finite.")
    agree = overlaps(geo, mmm) and overlaps(geo, mta)
    cost_test = cost_against_cap(geo.low, test_cost, max_cost_per_incremental)
    cost = cost_test.cost_per_incremental
    if cost_test.reason == "interval_includes_zero":
        action = "Do not scale."
        detail = (
            "The experiment lower bound is not strictly positive, "
            "so cost per incremental conversion is not defined."
        )
    elif cost_test.clears:
        action = "Scale."
        detail = (
            f"Cost per incremental conversion at the experiment lower bound is {cost:.4f}, "
            f"at or under the cap {max_cost_per_incremental:.4f}."
        )
    else:
        action = "Do not scale."
        detail = (
            f"Cost per incremental conversion at the experiment lower bound is {cost:.4f}, "
            f"above the cap {max_cost_per_incremental:.4f}."
        )
    if agree:
        text = f"The intervals agree. {action} {detail}"
    else:
        text = (
            "The intervals disagree. The experiment drives the go or no-go. "
            f"{action} {detail} The models stay diagnostic."
        )
    return Reconciliation(
        agree=agree,
        decision=text,
        driver="geo_did",
        test_cost=test_cost,
        max_cost_per_incremental=max_cost_per_incremental,
        cost_per_incremental=cost,
    )
