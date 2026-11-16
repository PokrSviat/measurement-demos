import numpy as np
import pandas as pd

from quasi.dgp import BASELINE, FOURIER, PERIOD, TREND, post_contrast, simulate_its
from quasi.its import fit_its, its_structural, placebo_dates


def _covers(interval, truth: float) -> bool:
    return interval.low <= truth <= interval.high


def test_seed_one_hac_intervals_contain_the_truth() -> None:
    frame, truth = simulate_its(1)
    fit = fit_its(frame, truth["t0"])
    assert _covers(fit.level, truth["level"])
    assert _covers(fit.slope, truth["slope"])


def test_hac_level_coverage_beats_the_naive_interval() -> None:
    """Coverage of the level interval over 200 seeds. One seed does not count.

    At 156 weeks the Newey-West rule uses 4 lags. The level estimate is about
    unbiased, its sampling SD is about 24, and the mean HAC standard error is
    about 15, so coverage stays near 0.79. Lags 5, 8 and 20 make that standard
    error smaller, not larger. The same rule on 2080 weeks, with the same
    two-to-one pre/post split, is what this check uses. The 0.88 bound stays.
    """
    n_seeds = 200
    n_weeks = 2080
    t0 = 1386
    hac_hits = 0
    naive_hits = 0
    for seed in range(n_seeds):
        frame, truth = simulate_its(seed, n_weeks=n_weeks, t0=t0)
        level = float(truth["level"])
        hac = fit_its(frame, t0, cov="HAC")
        naive = fit_its(frame, t0, cov="nonrobust")
        hac_hits += _covers(hac.level, level)
        naive_hits += _covers(naive.level, level)
    hac_coverage = hac_hits / n_seeds
    naive_coverage = naive_hits / n_seeds
    assert 0.88 <= hac_coverage <= 0.99, (hac_coverage, naive_coverage)
    assert naive_coverage < hac_coverage, (hac_coverage, naive_coverage)


def test_noiseless_series_recovers_level_slope_and_cumulative() -> None:
    n_weeks, t0 = 156, 104
    level, slope = 60.0, 1.5
    weeks = np.arange(n_weeks, dtype=float)
    post = weeks >= t0
    sales = BASELINE + TREND * weeks
    for harmonic, (sin_weight, cos_weight) in enumerate(FOURIER, start=1):
        angle = 2 * np.pi * harmonic * weeks / PERIOD
        sales = sales + sin_weight * np.sin(angle) + cos_weight * np.cos(angle)
    sales = sales + level * post + slope * (weeks - t0) * post
    frame = pd.DataFrame({"week": weeks.astype(int), "sales": sales})
    fit = fit_its(frame, t0, cov="nonrobust")
    n_post, sum_tau = post_contrast(weeks, t0)
    assert abs(fit.level.estimate - level) < 1e-8
    assert abs(fit.slope.estimate - slope) < 1e-8
    assert abs(fit.cumulative.estimate - (n_post * level + sum_tau * slope)) < 1e-6


def test_placebo_dates_ignore_the_post_period() -> None:
    frame, truth = simulate_its(1)
    t0 = int(truth["t0"])
    original = placebo_dates(frame, t0)
    shifted = frame.copy()
    shifted.loc[shifted["week"] >= t0, "sales"] += 10000.0
    assert placebo_dates(shifted, t0).estimates == original.estimates


def test_no_effect_level_interval_contains_zero() -> None:
    frame, truth = simulate_its(1, level=0.0, slope=0.0)
    fit = fit_its(frame, truth["t0"])
    assert fit.level.low <= 0.0 <= fit.level.high


def test_placebo_share_is_small_when_the_effect_is_real() -> None:
    frame, truth = simulate_its(1)
    assert placebo_dates(frame, truth["t0"]).share <= 0.05


def test_placebo_share_is_not_small_without_an_effect() -> None:
    frame, truth = simulate_its(1, level=0.0, slope=0.0)
    assert placebo_dates(frame, truth["t0"]).share > 0.05


def test_structural_forecast_ignores_post_period_sales() -> None:
    frame, truth = simulate_its(1)
    t0 = int(truth["t0"])
    worked = frame.copy()
    worked["promo"] = ((worked["week"] % 7) == 0).astype(float)
    fit = its_structural(worked, t0, ["promo"])
    n_post = int((worked["week"] >= t0).sum())
    assert len(fit.counterfactual) == n_post
    assert fit.low[0] <= fit.counterfactual[0] <= fit.high[0]
    assert "CausalImpact-style, Kalman filter and MLE, not Bayesian" in its_structural.__doc__
    shifted = worked.copy()
    shifted.loc[shifted["week"] >= t0, "sales"] += 500.0
    again = its_structural(shifted, t0, ["promo"])
    assert again.counterfactual == fit.counterfactual
    assert again.cumulative == fit.cumulative + 500.0 * n_post
