"""Sharp regression discontinuity with a local linear fit and the usual checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

Z_95 = float(norm.ppf(0.975))


@dataclass(frozen=True)
class LocalLinear:
    tau: float
    se: float
    ci_low: float
    ci_high: float
    n_left: int
    n_right: int
    h: float


def _local_wls(
    df: pd.DataFrame,
    cutoff: float,
    h: float,
    x: str,
    y: str,
):
    """Weighted least squares with a triangular kernel. Returns the fit and the side counts."""
    if h <= 0:
        raise ValueError("h must be positive.")
    running = df[x].to_numpy(dtype=float)
    outcome = df[y].to_numpy(dtype=float)
    distance = running - cutoff
    weights = np.clip(1.0 - np.abs(distance) / h, 0.0, None)
    inside = weights > 0.0
    n_left = int(np.sum(inside & (running < cutoff)))
    n_right = int(np.sum(inside & (running >= cutoff)))
    if n_left < 2 or n_right < 2:
        raise ValueError("Each side of the cutoff needs at least two observations.")
    treated = (running[inside] >= cutoff).astype(float)
    dx = distance[inside]
    design = pd.DataFrame(
        {
            "intercept": 1.0,
            "treatment": treated,
            "dx": dx,
            "interaction": treated * dx,
        }
    )
    result = sm.WLS(outcome[inside], design, weights=weights[inside]).fit(cov_type="HC1")
    return result, n_left, n_right


def local_linear(
    df: pd.DataFrame,
    cutoff: float,
    h: float,
    x: str = "spend_prior",
    y: str = "spend_next",
) -> LocalLinear:
    """One weighted least squares fit with its own intercept and slope on each side.

    Weights are the triangular kernel on (x - cutoff) / h. The running
    variable enters as x minus the cutoff, so the intercept is the level at
    the cutoff and the treatment coefficient is the jump. Standard errors are
    HC1. They do not include the bias from curvature inside a wide window.
    """
    result, n_left, n_right = _local_wls(df, cutoff, h, x, y)
    tau = float(result.params["treatment"])
    se = float(result.bse["treatment"])
    half = Z_95 * se
    return LocalLinear(tau, se, tau - half, tau + half, n_left, n_right, float(h))


def local_linear_segments(
    df: pd.DataFrame,
    cutoff: float,
    h: float,
    x: str = "spend_prior",
    y: str = "spend_next",
    grid: int = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fitted lines on each side of the cutoff, from the same weighted least squares."""
    result, _, _ = _local_wls(df, cutoff, h, x, y)
    intercept = float(result.params["intercept"])
    treatment = float(result.params["treatment"])
    slope = float(result.params["dx"])
    interaction = float(result.params["interaction"])
    left_x = np.linspace(cutoff - h, cutoff, grid)
    right_x = np.linspace(cutoff, cutoff + h, grid)
    left_y = intercept + slope * (left_x - cutoff)
    right_y = intercept + treatment + (slope + interaction) * (right_x - cutoff)
    return left_x, left_y, right_x, right_y


@dataclass(frozen=True)
class DensityCheck:
    estimate: float
    se: float
    p_value: float


def _histogram_sides(
    x: np.ndarray, cutoff: float, bin_width: float, h: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_bins = int(np.floor((h - bin_width / 2.0) / bin_width)) + 1
    if n_bins < 2:
        raise ValueError("h must cover at least two bins on each side.")
    edges = cutoff + np.arange(-n_bins, n_bins + 1) * bin_width
    counts, edges = np.histogram(x, bins=edges)
    mids = (edges[:-1] + edges[1:]) / 2.0
    return mids[:n_bins], counts[:n_bins], mids[n_bins:], counts[n_bins:]


def _heaping_grain(values: np.ndarray) -> float | None:
    """Smallest gap when the values sit on a lattice, else None."""
    uniq = np.unique(values)
    if len(uniq) < 3:
        return None
    gaps = np.diff(uniq)
    grain = float(gaps.min())
    if not np.isfinite(grain) or grain <= 0:
        return None
    steps = gaps / grain
    on_lattice = np.abs(steps - np.round(steps)) <= 1e-6
    if float(on_lattice.mean()) < 0.95:
        return None
    return grain


def _log_count_level(mids: np.ndarray, counts: np.ndarray, cutoff: float, h: float) -> tuple[float, float]:
    """Triangular local linear of log bin counts. The intercept is the level at the cutoff."""
    if np.any(counts <= 0):
        raise ValueError("A bin inside the window is empty, so log counts are undefined.")
    z = mids - cutoff
    weights = np.clip(1.0 - np.abs(z) / h, 0.0, None)
    keep = weights > 0.0
    if int(np.sum(keep)) < 3:
        raise ValueError("Each side needs at least three bins with a positive kernel weight.")
    design = pd.DataFrame({"intercept": 1.0, "z": z[keep]})
    result = sm.WLS(np.log(counts[keep].astype(float)), design, weights=weights[keep]).fit(cov_type="HC1")
    level = float(result.params["intercept"])
    se = float(result.bse["intercept"])
    if not np.isfinite(level) or not np.isfinite(se) or se <= 0:
        raise ValueError("The log-count fit did not return a finite standard error.")
    return level, se


def density_check(x: np.ndarray, cutoff: float, bin_width: float = 0.5, h: float = 5.0) -> DensityCheck:
    """Log-density jump at the cutoff.

    This is a simplified version of McCrary (2008), not the rddensity package.
    Bins do not cross the cutoff. On each side, log counts
    within h are fit by a local line. The estimate is the right intercept
    minus the left one. Heaped data, such as sums rounded to 1000, need a bin
    width that is a multiple of that rounding. A finer or offset bin can
    report a jump that is only the heap.
    """
    if bin_width <= 0 or h <= 0:
        raise ValueError("bin_width and h must be positive.")
    values = np.asarray(x, dtype=float)
    grain = _heaping_grain(values)
    if grain is not None:
        steps = bin_width / grain
        if steps < 1 - 1e-8 or abs(steps - round(steps)) > 1e-6:
            raise ValueError(
                "bin_width must be a positive multiple of the rounding unit "
                f"{grain:g}. A finer or offset bin creates a false jump."
            )
    left_mids, left_counts, right_mids, right_counts = _histogram_sides(values, cutoff, bin_width, h)
    left_level, left_se = _log_count_level(left_mids, left_counts, cutoff, h)
    right_level, right_se = _log_count_level(right_mids, right_counts, cutoff, h)
    estimate = right_level - left_level
    se = float(np.sqrt(left_se**2 + right_se**2))
    p_value = float(2 * norm.sf(abs(estimate) / se))
    return DensityCheck(estimate, se, p_value)


def placebo_cutoffs(df: pd.DataFrame, cutoff: float, points: list[float], h: float = 5.0) -> pd.DataFrame:
    """Local linear jumps at fake cutoffs, each using only its own side of the real cutoff."""
    rows = []
    for point in points:
        point = float(point)
        if point == cutoff:
            raise ValueError("A placebo cutoff has to differ from the real one.")
        if point < cutoff:
            sample = df.loc[df["spend_prior"] < cutoff]
        else:
            sample = df.loc[df["spend_prior"] >= cutoff]
        fit = local_linear(sample, point, h)
        rows.append(
            {
                "cutoff": point,
                "tau": fit.tau,
                "se": fit.se,
                "ci_low": fit.ci_low,
                "ci_high": fit.ci_high,
                "n_left": fit.n_left,
                "n_right": fit.n_right,
                "h": fit.h,
            }
        )
    return pd.DataFrame(rows)


def covariate_jump(df: pd.DataFrame, cutoff: float, h: float, covariate: str = "tenure_years") -> LocalLinear:
    """The local linear jump in a covariate. Tenure should not jump at the cutoff."""
    return local_linear(df, cutoff, h, y=covariate)


@dataclass(frozen=True)
class RobustCheck:
    tau: float
    ci_low: float
    ci_high: float
    h_left: float
    h_right: float
    density_p_value: float


def rdd_robust_check(
    df: pd.DataFrame,
    cutoff: float,
    x: str = "spend_prior",
    y: str = "spend_next",
) -> RobustCheck:
    """Bias-corrected jump from rdrobust, with its robust interval and an rddensity p-value.

    Needs the rd extra. tau is the bias-corrected estimate. The interval is the
    robust one, not the conventional interval. h_left and h_right are the
    estimation bandwidths. The density p-value is the jackknife p-value, or
    the asymptotic one when the jackknife value is missing.
    """
    try:
        from rdrobust import rdrobust
    except ImportError as exc:
        raise ImportError("rdd_robust_check needs the rd extra (rdrobust).") from exc
    try:
        from rddensity import rddensity
    except ImportError as exc:
        raise ImportError("rdd_robust_check needs the rd extra (rddensity).") from exc

    estimate = rdrobust(y=df[y], x=df[x], c=float(cutoff))
    density = rddensity(X=df[x].to_numpy(dtype=float), c=float(cutoff))
    jackknife = float(density.test["p_jk"])
    asymptotic = float(density.test["p_asy"])
    p_value = jackknife if np.isfinite(jackknife) else asymptotic
    if not np.isfinite(p_value):
        raise RuntimeError("rddensity did not return a p-value.")
    return RobustCheck(
        tau=float(estimate.coef.loc["Bias-Corrected", "Coeff"]),
        ci_low=float(estimate.ci.loc["Robust", "CI Lower"]),
        ci_high=float(estimate.ci.loc["Robust", "CI Upper"]),
        h_left=float(estimate.bws.loc["h", "left"]),
        h_right=float(estimate.bws.loc["h", "right"]),
        density_p_value=p_value,
    )


def bandwidth_sensitivity(df: pd.DataFrame, cutoff: float, hs: list[float]) -> pd.DataFrame:
    """The same local linear fit at each bandwidth."""
    fits = [local_linear(df, cutoff, h) for h in hs]
    return pd.DataFrame(
        {
            "h": [fit.h for fit in fits],
            "tau": [fit.tau for fit in fits],
            "se": [fit.se for fit in fits],
            "ci_low": [fit.ci_low for fit in fits],
            "ci_high": [fit.ci_high for fit in fits],
            "n_left": [fit.n_left for fit in fits],
            "n_right": [fit.n_right for fit in fits],
        }
    )
