"""Difference-in-differences and CUPED on a geo-day panel.

The decision estimand is the CUPED total with one vote per geo: the mean
adjusted change of the test geos minus that of the controls, times the observed
test location-days. A missing geo-day is not imputed, and one geo's hole does
not delete that date for everyone else. The decision interval reassigns the
treated slots inside pretest-volume strata.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

Z_95 = 1.96


@dataclass(frozen=True)
class EffectEstimate:
    name: str
    att_daily: float
    se_daily: float
    ci_low: float
    ci_high: float
    incremental: float
    incremental_low: float
    incremental_high: float
    n_treated: int
    n_control: int
    n_post_days: int | None
    theta: float | None = None
    treated_location_days: int | None = None

    def as_dict(self) -> dict[str, float | int | str | None]:
        return {
            "name": self.name,
            "att_daily": self.att_daily,
            "se_daily": self.se_daily,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "incremental": self.incremental,
            "incremental_low": self.incremental_low,
            "incremental_high": self.incremental_high,
            "n_treated": self.n_treated,
            "n_control": self.n_control,
            "n_post_days": self.n_post_days,
            "theta": self.theta,
            "treated_location_days": self.treated_location_days,
        }


def location_window_means(
    panel: pd.DataFrame,
    *,
    pre_end: str,
    post_start: str,
    post_end: str,
) -> pd.DataFrame:
    """One row per location with pre-period and post-period daily means."""
    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    pre = frame.loc[frame["date"] <= pd.Timestamp(pre_end)]
    post = frame.loc[
        (frame["date"] >= pd.Timestamp(post_start))
        & (frame["date"] <= pd.Timestamp(post_end))
    ]
    if pre.empty or post.empty:
        raise ValueError("Pre or post window has no rows.")
    pre_days = int(pre["date"].nunique())
    calendar_post_days = int(post["date"].nunique())
    pre_means = pre.groupby("location", as_index=False)["Y"].mean().rename(columns={"Y": "pre"})
    post_summary = post.groupby("location", as_index=False).agg(
        post=("Y", "mean"),
        n_post_days=("date", "nunique"),
    )
    summary = pre_means.merge(post_summary, on="location", how="inner")
    if summary.empty:
        raise ValueError("No location has both a pre and a post window.")
    summary["delta"] = summary["post"] - summary["pre"]
    summary["n_pre_days"] = pre_days
    summary["n_post_days"] = summary["n_post_days"].astype(int)
    observed = int(summary["n_post_days"].sum())
    summary["missing_post_location_days"] = calendar_post_days * len(summary) - observed
    return summary


def _treated_day_scale(location_means: pd.DataFrame, treated_mask: pd.Series) -> tuple[float, int | None, int]:
    """Scale the daily effect by observed treated location-days. Missing days are not imputed."""
    days = location_means.loc[treated_mask, "n_post_days"].astype(int)
    total = int(days.sum())
    per_geo = int(days.iloc[0]) if int(days.nunique()) == 1 else None
    return float(total), per_geo, total


def _two_sample_effect(
    values: pd.Series,
    treated_mask: pd.Series,
    *,
    name: str,
    location_means: pd.DataFrame,
    theta: float | None,
) -> EffectEstimate:
    treated = values.loc[treated_mask]
    control = values.loc[~treated_mask]
    n_treated = int(treated.shape[0])
    n_control = int(control.shape[0])
    if n_treated < 2 or n_control < 2:
        raise ValueError("Need at least two treated locations and two control locations.")
    att = float(treated.mean() - control.mean())
    se = float(
        np.sqrt(treated.var(ddof=1) / n_treated + control.var(ddof=1) / n_control)
    )
    scale, per_geo, treated_location_days = _treated_day_scale(location_means, treated_mask)
    return EffectEstimate(
        name=name,
        att_daily=att,
        se_daily=se,
        ci_low=att - Z_95 * se,
        ci_high=att + Z_95 * se,
        incremental=att * scale,
        incremental_low=(att - Z_95 * se) * scale,
        incremental_high=(att + Z_95 * se) * scale,
        n_treated=n_treated,
        n_control=n_control,
        n_post_days=per_geo,
        theta=theta,
        treated_location_days=treated_location_days,
    )


def _require_treated(location_means: pd.DataFrame, treated_locations: list[str]) -> pd.Series:
    treated_mask = location_means["location"].isin(treated_locations)
    if int(treated_mask.sum()) != len(set(treated_locations)):
        missing = sorted(set(treated_locations) - set(location_means["location"]))
        raise ValueError(f"Treated locations missing from the panel: {missing}")
    return treated_mask


def difference_in_differences(
    location_means: pd.DataFrame,
    treated_locations: list[str],
) -> EffectEstimate:
    """Classic two-period DiD on geo-level pre/post changes."""
    treated_mask = _require_treated(location_means, treated_locations)
    return _two_sample_effect(
        location_means["delta"],
        treated_mask,
        name="did",
        location_means=location_means,
        theta=None,
    )


def cuped_difference_in_differences(
    location_means: pd.DataFrame,
    treated_locations: list[str],
) -> EffectEstimate:
    """DiD after CUPED, with the pre-period mean as the covariate.

    Theta is fit on control geos only, then applied to every geo.
    """
    treated_mask = _require_treated(location_means, treated_locations)
    control = location_means.loc[~treated_mask]
    if control.shape[0] < 2:
        raise ValueError("CUPED needs at least two control locations.")
    pre = control["pre"].to_numpy(dtype=float)
    delta = control["delta"].to_numpy(dtype=float)
    var_pre = float(np.var(pre, ddof=1))
    theta = 0.0 if var_pre == 0 else float(np.cov(delta, pre, ddof=1)[0, 1] / var_pre)
    center = float(pre.mean())
    adjusted = location_means["delta"] - theta * (location_means["pre"] - center)
    return _two_sample_effect(
        adjusted,
        treated_mask,
        name="cuped",
        location_means=location_means,
        theta=theta,
    )


def pretest_weighted_total(location_means: pd.DataFrame, treated_locations: list[str]) -> float:
    """CUPED total with each geo weighted by its pre-period mean. Descriptive only.

    It equals total incremental sales only when every geo has the same additive
    daily effect. When the effect grows with geo size, the largest geos count
    twice, once in the weight and once in the effect, and the total is inflated.
    """
    treated_mask = _require_treated(location_means, treated_locations).to_numpy()
    pre = location_means["pre"].to_numpy(dtype=float)
    delta = location_means["delta"].to_numpy(dtype=float)
    days = location_means["n_post_days"].to_numpy(dtype=float)
    if np.any(pre <= 0):
        raise ValueError("Pre-test sales must be positive to weight the estimand.")
    return cuped_total(pre, delta, days, treated_mask, weights=pre)


def picks_to_masks(picks: np.ndarray, n_locations: int) -> np.ndarray:
    masks = np.zeros((len(picks), n_locations), dtype=bool)
    masks[np.arange(len(picks))[:, None], picks] = True
    return masks


def cuped_totals(
    pre: np.ndarray,
    delta: np.ndarray,
    days: np.ndarray,
    treated: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """CUPED total for each row of a treated mask. Theta is refit on that row's controls.

    Without weights every geo has one vote. The daily effect is then scaled by the
    observed treated location-days of that row.
    """
    masks = np.atleast_2d(np.asarray(treated, dtype=bool))
    control = ~masks
    n_control = control.sum(axis=1).astype(float)
    if np.any(n_control < 2) or np.any(masks.sum(axis=1) < 1):
        raise ValueError("Each assignment needs a treated geo and two control geos.")
    pre_c = (control * pre).sum(axis=1) / n_control
    delta_c = (control * delta).sum(axis=1) / n_control
    var_pre = ((control * pre**2).sum(axis=1) - n_control * pre_c**2) / (n_control - 1)
    cov = ((control * pre * delta).sum(axis=1) - n_control * pre_c * delta_c) / (n_control - 1)
    theta = np.divide(cov, var_pre, out=np.zeros(len(masks)), where=var_pre > 0)
    adjusted = delta - theta[:, None] * (pre - pre_c[:, None])
    weight = np.ones_like(pre, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    daily = (masks * weight * adjusted).sum(axis=1) / (masks * weight).sum(axis=1) - (
        control * weight * adjusted
    ).sum(axis=1) / (control * weight).sum(axis=1)
    return daily * (masks * days).sum(axis=1)


def cuped_total(
    pre: np.ndarray,
    delta: np.ndarray,
    days: np.ndarray,
    treated: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    return float(cuped_totals(pre, delta, days, np.asarray(treated, dtype=bool)[None, :], weights)[0])


def stratified_picks(
    stratum_ids: np.ndarray,
    treated_per_stratum: int | Mapping[int, int],
    n_simulations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw treated slots inside each stratum. Draws do not cross strata.

    A mapping gives the treated count of each stratum, and a stratum with no
    treated geo stays all control. An int gives every stratum the same count.
    """
    if n_simulations < 1:
        raise ValueError("Need at least one reassignment.")
    if isinstance(treated_per_stratum, int) and treated_per_stratum < 1:
        raise ValueError("Each stratum needs at least one treated geo.")
    groups: list[tuple[np.ndarray, int]] = []
    for stratum in pd.unique(stratum_ids):
        members = np.flatnonzero(np.asarray(stratum_ids) == stratum)
        count = (
            treated_per_stratum
            if isinstance(treated_per_stratum, int)
            else int(treated_per_stratum[int(stratum)])
        )
        if count < 0:
            raise ValueError("A stratum cannot have a negative treated count.")
        if len(members) < count + 2:
            raise ValueError("Each stratum needs two control geos after the treated draw.")
        groups.append((members, count))
    chosen = [
        members[np.argsort(rng.random((n_simulations, len(members))), axis=1)[:, :count]]
        for members, count in groups
    ]
    return np.concatenate(chosen, axis=1)


EFFECT_SHAPES = ("additive", "proportional")


def effect_profile(pre: np.ndarray, effect_shape: str) -> np.ndarray:
    """Relative daily effect of each geo under a hypothesis: equal, or proportional to pre-period sales."""
    if effect_shape == "additive":
        return np.ones(len(pre), dtype=float)
    if effect_shape == "proportional":
        if np.any(np.asarray(pre) <= 0):
            raise ValueError("A proportional effect needs positive pre-period sales.")
        return np.asarray(pre, dtype=float)
    raise ValueError(f"Unknown effect shape: {effect_shape}.")


def control_theta(pre: np.ndarray, delta: np.ndarray, treated: np.ndarray) -> float:
    """CUPED coefficient of the change on the pre-period mean, fit on control geos only."""
    control = ~np.asarray(treated, dtype=bool)
    if control.sum() < 2:
        raise ValueError("Theta needs two control geos.")
    variance = float(np.var(pre[control], ddof=1))
    if variance <= 0:
        return 0.0
    return float(np.cov(pre[control], delta[control], ddof=1)[0, 1] / variance)


def randomization_interval(
    location_means: pd.DataFrame,
    treated_locations: list[str],
    stratum_ids: np.ndarray,
    *,
    treated_per_stratum: int | None = None,
    weights: np.ndarray | None = None,
    effect_shape: str = "additive",
    n_simulations: int = 2000,
    seed: int,
) -> tuple[float, float]:
    """95% interval from reassigning the treated slots inside volume strata.

    Each stratum keeps the treated count it actually has. Under a hypothesized
    total, that total is removed from the treated geos, spread equally per
    treated location-day (additive) or in proportion to pre-period sales
    (proportional), and the labels are redrawn. The interval keeps every total
    whose observed statistic sits inside the central 95% of the redraws.
    """
    return randomization_bounds(
        location_means["pre"].to_numpy(dtype=float),
        location_means["delta"].to_numpy(dtype=float),
        location_means["n_post_days"].to_numpy(dtype=float),
        _require_treated(location_means, treated_locations).to_numpy(),
        stratum_ids,
        treated_per_stratum=treated_per_stratum,
        weights=weights,
        effect_shape=effect_shape,
        n_simulations=n_simulations,
        seed=seed,
    )


def randomization_bounds(
    pre: np.ndarray,
    delta: np.ndarray,
    days: np.ndarray,
    treated_mask: np.ndarray,
    stratum_ids: np.ndarray,
    *,
    treated_per_stratum: int | None = None,
    weights: np.ndarray | None = None,
    effect_shape: str = "additive",
    n_simulations: int = 2000,
    seed: int,
) -> tuple[float, float]:
    """Array form of `randomization_interval`."""
    if n_simulations < 100:
        raise ValueError("The randomization interval needs at least 100 reassignments.")
    treated_mask = np.asarray(treated_mask, dtype=bool)
    stratum_ids = np.asarray(stratum_ids)
    if len(stratum_ids) != len(pre):
        raise ValueError("Strata must align with the geo table.")
    if weights is not None and np.any(np.asarray(weights) <= 0):
        raise ValueError("Weights must be positive.")
    counts = {
        int(stratum): int(treated_mask[stratum_ids == stratum].sum())
        for stratum in pd.unique(stratum_ids)
    }
    if treated_per_stratum is not None and set(counts.values()) != {treated_per_stratum}:
        raise ValueError(
            "The treated geos do not match the stratified design: "
            f"stratum counts are {sorted(counts.values())}."
        )
    n_treated = int(treated_mask.sum())
    if n_treated < 2 or n_treated > len(pre) - 2:
        raise ValueError("Need at least two treated locations and two control locations.")
    treated_days = float(days[treated_mask].sum())
    if treated_days <= 0:
        raise ValueError("Treated locations have no observed post days.")
    point = cuped_total(pre, delta, days, treated_mask, weights)
    rng = np.random.default_rng(seed)
    masks = picks_to_masks(stratified_picks(stratum_ids, counts, n_simulations, rng), len(pre))
    profile = effect_profile(pre, effect_shape)
    per_unit = profile[treated_mask] / float(np.sum(profile[treated_mask] * days[treated_mask]))

    def covers(tau: float) -> bool:
        adjusted = delta.copy()
        adjusted[treated_mask] -= tau * per_unit
        observed = cuped_total(pre, adjusted, days, treated_mask, weights)
        samples = cuped_totals(pre, adjusted, days, masks, weights)
        low, high = np.quantile(samples, [0.025, 0.975])
        return bool(low <= observed <= high)

    null_samples = cuped_totals(pre, delta, days, masks, weights)
    span = float(np.quantile(null_samples, 0.975) - np.quantile(null_samples, 0.025))
    span = max(span, abs(point), treated_days)
    return invert_test(covers, point, span, tolerance=max(1.0, treated_days * 1e-4))


def invert_test(
    covers: Callable[[float], bool],
    point: float,
    span: float,
    *,
    tolerance: float,
) -> tuple[float, float]:
    """Edges of the set of effects that the test does not reject, by bracket, grid and bisection."""
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
    low = _transition(covers, float(grid[max(first - 1, 0)]), float(grid[first]), True, tolerance)
    high = _transition(covers, float(grid[last]), float(grid[min(last + 1, len(grid) - 1)]), False, tolerance)
    return low, high


def _transition(
    covers: Callable[[float], bool],
    left: float,
    right: float,
    inside_on_the_right: bool,
    tolerance: float,
) -> float:
    for _ in range(40):
        if right - left <= tolerance:
            break
        mid = (left + right) / 2
        if covers(mid) == inside_on_the_right:
            right = mid
        else:
            left = mid
    return float(right if inside_on_the_right else left)
