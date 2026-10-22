"""Which channel the next lift test should go to, and how long it has to run.

Priority is spend share times the width of the ROI interval, over channels
whose interval contains payback. It is a value-of-information heuristic: a
wide interval on a large budget is where a wrong call costs most. It is not a
full loss function.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

PAYBACK = 1.0
MAX_NATIONAL_TEST_WEEKS = 52
POWER_TARGET = 0.8


def experiment_priority(
    spend_share: Mapping[str, float],
    low: Mapping[str, float],
    high: Mapping[str, float],
) -> dict[str, float]:
    """Spend share times interval width for every channel whose interval contains payback."""
    if set(low) != set(high) or not set(low) <= set(spend_share):
        raise ValueError("Every channel needs a spend share and both interval ends.")
    priority: dict[str, float] = {}
    for channel in sorted(low):
        if high[channel] < low[channel]:
            raise ValueError(f"The interval for {channel} is reversed.")
        if spend_share[channel] < 0:
            raise ValueError("Spend shares must be non-negative.")
        if low[channel] <= PAYBACK <= high[channel]:
            priority[channel] = spend_share[channel] * (high[channel] - low[channel])
    return priority


def next_experiment(
    spend_share: Mapping[str, float],
    low: Mapping[str, float],
    high: Mapping[str, float],
) -> str | None:
    """The channel with the highest priority, or None when every interval settles payback."""
    priority = experiment_priority(spend_share, low, high)
    if not priority:
        return None
    return max(priority, key=lambda channel: (priority[channel], channel))


def lift_test_weeks(
    *,
    sigma: float,
    rho: float,
    weekly_increase: float,
    expected_roi: float,
    z: float = 1.96,
) -> int | None:
    """Weeks per arm so a test of raised spend against baseline weeks excludes payback.

    `sigma` is the weekly innovation standard deviation and `rho` the AR(1)
    coefficient, so the long-run standard deviation of a weekly mean is
    sigma / (1 - rho). The half-width of the ROI interval after W weeks per arm
    is z * sigma / (1 - rho) * sqrt(2 / W) / weekly_increase. Carryover through
    adstock and saturation are not modelled. None means payback cannot be
    excluded because the expected ROI is payback itself.
    """
    if sigma <= 0 or weekly_increase <= 0 or z <= 0:
        raise ValueError("sigma, the weekly increase and z must be positive.")
    if not -1 < rho < 1:
        raise ValueError("rho must be inside (-1, 1).")
    gap = abs(expected_roi - PAYBACK)
    if gap == 0:
        return None
    return max(1, math.ceil(2 * (z * sigma / ((1 - rho) * weekly_increase * gap)) ** 2))


def national_test_feasible(
    weeks: Mapping[float, int | None],
    *,
    max_weeks: int = MAX_NATIONAL_TEST_WEEKS,
) -> bool:
    """True when at least one national arm length is finite and within `max_weeks`."""
    sized = [value for value in weeks.values() if value is not None]
    return bool(sized) and min(sized) <= max_weeks


def lift_design_powered(payload: Mapping[str, object] | None, *, target: float = POWER_TARGET) -> bool:
    """True when a lift-design JSON has a cell at or above the power target."""
    if not payload:
        return False
    cells = payload.get("cells")
    if not isinstance(cells, list):
        return False
    return any(float(cell.get("power_at_point_roi", 0)) >= target for cell in cells if isinstance(cell, dict))


def publish_lift_test(*, national_feasible: bool, geo_powered: bool) -> bool:
    """A published test exists only if a national sizing or a powered geo cell exists."""
    return national_feasible or geo_powered
