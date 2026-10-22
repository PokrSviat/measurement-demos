"""Three small estimators on the synthetic world. None of them is a second production pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from reconcile.dgp import SyntheticWorld


@dataclass(frozen=True)
class MethodEstimate:
    method: str
    estimand: str
    point: float
    low: float
    high: float

    def as_dict(self) -> dict[str, str | float]:
        return {
            "method": self.method,
            "estimand": self.estimand,
            "point": self.point,
            "low": self.low,
            "high": self.high,
        }


def geo_incremental(world: SyntheticWorld) -> MethodEstimate:
    """Unweighted two-period DiD. The interval enumerates every treated assignment.

    The point is the DiD total. A hypothesized total is removed from the geos
    that were actually treated, and the interval keeps the totals whose observed
    statistic sits inside the central 95% of every reassignment. It is not a
    normal approximation.
    """
    pre = world.panel.loc[world.panel["week"] < world.post_start_week]
    post = world.panel.loc[world.panel["week"] >= world.post_start_week]
    delta_series = (post.groupby("location")["y"].mean() - pre.groupby("location")["y"].mean()).sort_index()
    locations = list(delta_series.index)
    missing = [name for name in world.treated_locations if name not in delta_series.index]
    if missing:
        raise ValueError(f"Treated geos are missing from the panel: {missing}.")
    delta = delta_series.to_numpy(dtype=float)
    treated_index = np.array([locations.index(name) for name in world.treated_locations], dtype=int)
    n_treated = len(treated_index)
    n_locations = len(locations)
    if n_treated < 2 or n_treated > n_locations - 2:
        raise ValueError("Need at least two treated geos and two control geos.")
    n_post = int(post["week"].nunique())
    if n_post <= 0:
        raise ValueError("The test window has no weeks.")
    scale = float(n_treated * n_post)
    treated_mask = np.zeros(n_locations, dtype=bool)
    treated_mask[treated_index] = True
    point = _did_total(delta, treated_mask, scale)
    picks = np.array(list(combinations(range(n_locations), n_treated)), dtype=int)
    low, high = _randomization_bounds(delta, treated_mask, picks, scale, point)
    return MethodEstimate("geo_did", "incremental_conversions", point, low, high)


def _did_total(delta: np.ndarray, treated: np.ndarray, scale: float) -> float:
    weekly = float(delta[treated].mean() - delta[~treated].mean())
    return weekly * scale


def _did_total_many(delta: np.ndarray, picks: np.ndarray, scale: float) -> np.ndarray:
    n_locations = len(delta)
    n_treated = picks.shape[1]
    treated = np.zeros((len(picks), n_locations), dtype=bool)
    treated[np.arange(len(picks))[:, None], picks] = True
    mean_treated = (treated * delta).sum(axis=1) / n_treated
    mean_control = ((~treated) * delta).sum(axis=1) / (n_locations - n_treated)
    return (mean_treated - mean_control) * scale


def _randomization_bounds(
    delta: np.ndarray,
    treated_mask: np.ndarray,
    picks: np.ndarray,
    scale: float,
    point: float,
) -> tuple[float, float]:
    def covers(tau: float) -> bool:
        adjusted = delta.copy()
        adjusted[treated_mask] -= tau / scale
        observed = _did_total(adjusted, treated_mask, scale)
        samples = _did_total_many(adjusted, picks, scale)
        low, high = np.quantile(samples, [0.025, 0.975])
        return bool(low <= observed <= high)

    null_samples = _did_total_many(delta, picks, scale)
    span = float(np.quantile(null_samples, 0.975) - np.quantile(null_samples, 0.025))
    span = max(span, abs(point), scale)
    bracket_low = point - span
    bracket_high = point + span
    for _ in range(20):
        if not covers(bracket_low):
            break
        bracket_low -= span
    else:
        raise ValueError("The randomization search did not find a lower edge.")
    for _ in range(20):
        if not covers(bracket_high):
            break
        bracket_high += span
    else:
        raise ValueError("The randomization search did not find an upper edge.")
    grid = np.linspace(bracket_low, bracket_high, 41)
    inside = np.array([covers(float(tau)) for tau in grid])
    if not inside.any():
        raise ValueError("The randomization interval is empty on the search grid.")
    first = int(np.argmax(inside))
    last = int(len(inside) - 1 - np.argmax(inside[::-1]))
    tolerance = max(1.0, scale * 1e-4)
    low = _transition(covers, float(grid[max(first - 1, 0)]), float(grid[first]), True, tolerance)
    high = _transition(covers, float(grid[last]), float(grid[min(last + 1, len(grid) - 1)]), False, tolerance)
    return low, high


def _transition(covers, left: float, right: float, inside_on_the_right: bool, tolerance: float) -> float:
    for _ in range(40):
        if right - left <= tolerance:
            break
        mid = (left + right) / 2
        if covers(mid) == inside_on_the_right:
            right = mid
        else:
            left = mid
    return float(right if inside_on_the_right else left)


def ols_incremental(world: SyntheticWorld) -> MethodEstimate:
    """OLS of weekly revenue on tactic spend.

    The interval plugs the residual variance in as if sigma were known. It is not a posterior.
    Contribution during the test weeks is the incremental-conversion estimand.
    """
    spend = world.weekly["tactic_spend"].to_numpy(dtype=float)
    y = world.weekly["revenue"].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(spend)), spend])
    coef, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ coef
    residual_var = float(np.sum((y - fitted) ** 2) / (len(y) - 2))
    xtx_inv = np.linalg.inv(x.T @ x)
    se_beta = float(np.sqrt(residual_var * xtx_inv[1, 1]))
    test_spend = float(spend[world.post_start_week :].sum())
    point = float(coef[1] * test_spend)
    half = 1.96 * se_beta * test_spend
    return MethodEstimate(
        "ols",
        "incremental_conversions",
        point,
        point - half,
        point + half,
    )


def _transition_counts(paths: list[list[str]], removed: str | None) -> dict[tuple[str, str], int]:
    """Count transitions. A removed channel sends incoming flow to null.

    The journey is not reassigned to another channel. That is the translation
    used by ``markov_incremental``.
    """
    counts: dict[tuple[str, str], int] = {}
    for path in paths:
        for origin, destination in zip(path, path[1:]):
            if origin == removed:
                continue
            if destination == removed:
                destination = "null"
            counts[(origin, destination)] = counts.get((origin, destination), 0) + 1
    return counts


def _absorbing_conversion_probability(paths: list[list[str]], removed: str | None) -> float:
    counts = _transition_counts(paths, removed)
    origins = sorted({origin for origin, _ in counts} - {"conversion", "null"})
    if "start" not in origins:
        return 0.0
    index = {state: position for position, state in enumerate(origins)}
    size = len(origins)
    system = np.eye(size)
    rhs = np.zeros(size)
    for origin in origins:
        total = sum(value for (start, _), value in counts.items() if start == origin)
        row = index[origin]
        for (start, destination), value in counts.items():
            if start != origin or total == 0:
                continue
            probability = value / total
            if destination == "conversion":
                rhs[row] += probability
            elif destination in index:
                system[row, index[destination]] -= probability
    solution = np.linalg.solve(system, rhs)
    return float(solution[index["start"]])


def markov_incremental(paths: list[list[str]]) -> MethodEstimate:
    """Removal effect translated to incremental conversions.

    incremental = (p_with - p_without) * n_journeys.

    A removal share is a different estimand and is not returned here.
    """
    n_journeys = len(paths)
    with_channel = _absorbing_conversion_probability(paths, removed=None)
    without_channel = _absorbing_conversion_probability(paths, removed="tactic")
    point = (with_channel - without_channel) * n_journeys
    return MethodEstimate("markov_mta", "incremental_conversions", point, point, point)


def removal_share(paths: list[list[str]]) -> MethodEstimate:
    with_channel = _absorbing_conversion_probability(paths, removed=None)
    without_channel = _absorbing_conversion_probability(paths, removed="tactic")
    share = 0.0 if with_channel == 0 else (with_channel - without_channel) / with_channel
    return MethodEstimate("markov_share", "removal_share", share, share, share)
