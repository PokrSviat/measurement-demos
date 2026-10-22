"""The cost test both budget decisions share.

A positive lower bound clears the cap when test cost divided by that bound
is at or under the cap. Equality with the cap passes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CapTest:
    clears: bool
    cost_per_incremental: float | None
    reason: str


def cost_against_cap(low: float, test_cost: float, cap: float) -> CapTest:
    if test_cost < 0 or cap <= 0:
        raise ValueError("Cost assumptions must be positive.")
    if not math.isfinite(low):
        raise ValueError("The lower bound must be finite.")
    if low <= 0:
        return CapTest(False, None, "interval_includes_zero")
    cost = test_cost / low
    if cost > cap:
        return CapTest(False, cost, "above_cap")
    return CapTest(True, cost, "within_cap")
