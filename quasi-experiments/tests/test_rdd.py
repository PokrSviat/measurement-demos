import math

import numpy as np
import pandas as pd
import pytest

from quasi.dgp import simulate_rdd
from quasi.rdd import covariate_jump, density_check, local_linear, placebo_cutoffs, rdd_robust_check


def test_seed_one_interval_contains_tau() -> None:
    frame, truth = simulate_rdd(1)
    fit = local_linear(frame, truth["cutoff"], 5)
    assert fit.ci_low <= truth["tau"] <= fit.ci_high


def test_h5_mean_bias_and_coverage() -> None:
    n_seeds = 200
    bias_sum = 0.0
    hits = 0
    for seed in range(n_seeds):
        frame, truth = simulate_rdd(seed)
        fit = local_linear(frame, truth["cutoff"], 5)
        bias_sum += fit.tau - truth["tau"]
        hits += fit.ci_low <= truth["tau"] <= fit.ci_high
    mean_bias = bias_sum / n_seeds
    coverage = hits / n_seeds
    assert abs(mean_bias) <= 0.1
    assert 0.88 <= coverage <= 0.99


def test_wide_window_bias_is_reported() -> None:
    """A wide window is biased because each side is one straight line on a curve.

    Customers far from the cutoff pull that line, so its intercept at 50 picks
    up curvature as well as the jump. The narrow window stays where the curve
    is close to linear. HC1 describes the variance of that fit and does not
    subtract the curvature bias, so coverage falls below the nominal rate.
    The bandwidth table is there for that reason. This test reports the
    wide-window bias and does not check coverage.
    """
    n_seeds = 200
    bias_sum = 0.0
    for seed in range(n_seeds):
        frame, truth = simulate_rdd(seed)
        fit = local_linear(frame, truth["cutoff"], 25)
        bias_sum += fit.tau - truth["tau"]
    mean_bias = bias_sum / n_seeds
    print(f"wide-window mean bias at h=25: {mean_bias:.4f}")
    assert math.isfinite(mean_bias)


def test_density_detects_bunching_and_ignores_a_smooth_running_variable() -> None:
    frame, truth = simulate_rdd(1)
    smooth = density_check(frame["spend_prior"], truth["cutoff"])
    assert smooth.p_value > 0.05
    bunched, bunched_truth = simulate_rdd(1, bunching_share=0.3)
    jump = density_check(bunched["spend_prior"], bunched_truth["cutoff"])
    assert jump.p_value < 0.001


def test_placebo_cutoffs_away_from_the_threshold_cover_zero() -> None:
    frame, truth = simulate_rdd(1)
    points = list(range(35, 46)) + list(range(55, 66))
    table = placebo_cutoffs(frame, truth["cutoff"], points)
    covered = (table["ci_low"] <= 0) & (0 <= table["ci_high"])
    assert covered.mean() >= 0.90


def test_local_linear_recovers_a_noiseless_jump() -> None:
    cutoff = 50.0
    tau = 2.0
    spend = np.linspace(40.0, 60.0, 4001)
    distance = spend - cutoff
    outcome = 10.0 + 0.4 * distance + tau * (spend >= cutoff) - 0.2 * distance * (spend >= cutoff)
    frame = pd.DataFrame({"spend_prior": spend, "spend_next": outcome})
    fit = local_linear(frame, cutoff, 5)
    assert abs(fit.tau - tau) < 1e-8


def test_rounded_amounts_need_a_bin_multiple_of_the_rounding() -> None:
    rng = np.random.default_rng(1)
    latent = rng.lognormal(math.log(48000.0), 0.22, size=80000)
    rounded = np.round(latent / 1000.0) * 1000.0
    aligned = density_check(rounded, 50000.0, bin_width=1000.0, h=10000.0)
    assert aligned.p_value > 0.05
    with pytest.raises(ValueError, match="rounding unit"):
        density_check(rounded, 50000.0, bin_width=1200.0, h=4800.0)


def test_tenure_interval_contains_zero() -> None:
    frame, truth = simulate_rdd(1)
    fit = covariate_jump(frame, truth["cutoff"], 5)
    assert fit.ci_low <= 0.0 <= fit.ci_high


def test_rdrobust_interval_contains_tau_and_density_p_is_a_probability() -> None:
    pytest.importorskip("rdrobust")
    frame, truth = simulate_rdd(1)
    check = rdd_robust_check(frame, truth["cutoff"])
    assert check.ci_low <= truth["tau"] <= check.ci_high
    assert check.h_left > 0 and check.h_right > 0
    assert 0.0 <= check.density_p_value <= 1.0
    assert check.density_p_value > 0.05
