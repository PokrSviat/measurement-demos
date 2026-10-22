import numpy as np
import pandas as pd
import pytest

from geo_holdout.estimators import (
    cuped_difference_in_differences,
    cuped_total,
    difference_in_differences,
    location_window_means,
    pretest_weighted_total,
    randomization_interval,
    stratified_picks,
)


def _panel(att: float, seed: int, noise: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    treated = {f"t{i}" for i in range(4)}
    locations = [f"t{i}" for i in range(4)] + [f"c{i}" for i in range(20)]
    for location in locations:
        level = float(rng.normal(100, 10))
        shock = float(rng.normal(0, noise))
        for day in range(40):
            date = pd.Timestamp("2021-01-01") + pd.Timedelta(days=day)
            post = day >= 30
            value = level + shock + (att if post and location in treated else 0.0)
            if noise:
                value += float(rng.normal(0, noise))
            rows.append({"location": location, "date": date, "Y": value})
    return pd.DataFrame(rows)


def test_did_recovers_a_noiseless_effect() -> None:
    means = location_window_means(
        _panel(att=5.0, seed=1, noise=0.0),
        pre_end="2021-01-30",
        post_start="2021-01-31",
        post_end="2021-02-09",
    )
    estimate = difference_in_differences(means, [f"t{i}" for i in range(4)])
    assert estimate.att_daily == pytest.approx(5.0)
    assert estimate.n_post_days == 10
    assert estimate.incremental == pytest.approx(200.0)


def test_cuped_stays_unbiased_and_shrinks_the_interval() -> None:
    rng = np.random.default_rng(7)
    rows: list[dict[str, object]] = []
    att = 8.0
    for name, treated in [("t", True), ("c", False)]:
        count = 8 if treated else 50
        for index in range(count):
            noise = float(rng.normal(0, 10))
            pre = 0.95 * noise
            delta = (att if treated else 0.0) + noise
            rows.append(
                {
                    "location": f"{name}{index}",
                    "pre": pre,
                    "post": pre + delta,
                    "delta": delta,
                    "n_pre_days": 20,
                    "n_post_days": 10,
                }
            )
    means = pd.DataFrame(rows)
    names = [f"t{i}" for i in range(8)]
    did = difference_in_differences(means, names)
    cuped = cuped_difference_in_differences(means, names)
    assert abs(did.att_daily - att) < 2
    assert abs(cuped.att_daily - att) < 2
    assert cuped.se_daily < did.se_daily


def test_cuped_rejects_a_missing_treated_location() -> None:
    means = location_window_means(
        _panel(att=5.0, seed=1, noise=0.0),
        pre_end="2021-01-30",
        post_start="2021-01-31",
        post_end="2021-02-09",
    )
    with pytest.raises(ValueError, match="missing"):
        cuped_difference_in_differences(means, ["t0", "missing"])


def test_one_treated_location_is_rejected() -> None:
    means = location_window_means(
        _panel(att=5.0, seed=1, noise=0.0),
        pre_end="2021-01-30",
        post_start="2021-01-31",
        post_end="2021-02-09",
    )
    with pytest.raises(ValueError, match="two treated"):
        difference_in_differences(means, ["t0"])


def test_a_missing_geo_day_is_not_imputed_and_does_not_delete_the_date() -> None:
    panel = _panel(att=5.0, seed=1, noise=0.0)
    hole = pd.Timestamp("2021-02-09")
    panel = panel.loc[~((panel["location"] == "t0") & (panel["date"] == hole))]
    means = location_window_means(
        panel,
        pre_end="2021-01-30",
        post_start="2021-01-31",
        post_end="2021-02-09",
    )
    others = means.loc[means["location"] != "t0", "n_post_days"]
    assert int(means.loc[means["location"] == "t0", "n_post_days"].iloc[0]) == 9
    assert set(others) == {10}
    estimate = difference_in_differences(means, [f"t{i}" for i in range(4)])
    assert estimate.treated_location_days == 39
    assert estimate.incremental == pytest.approx(5.0 * 39)


def test_weights_follow_pretest_sales() -> None:
    means = pd.DataFrame(
        {
            "location": ["big", "small", "c1", "c2"],
            "pre": [1000.0, 1.0, 100.0, 100.0],
            "post": [1010.0, 51.0, 100.0, 100.0],
            "delta": [10.0, 50.0, 0.0, 0.0],
            "n_post_days": [10, 10, 10, 10],
        }
    )
    weighted = pretest_weighted_total(means, ["big", "small"])
    unweighted = difference_in_differences(means, ["big", "small"]).incremental
    assert weighted == pytest.approx((1000.0 * 10.0 + 1.0 * 50.0) / 1001.0 * 20.0)
    assert unweighted == pytest.approx(600.0)
    assert weighted < unweighted


def test_randomization_interval_covers_a_constant_effect() -> None:
    means = location_window_means(
        _panel(att=5.0, seed=1, noise=0.0),
        pre_end="2021-01-30",
        post_start="2021-01-31",
        post_end="2021-02-09",
    )
    treated = [f"t{i}" for i in range(4)]
    low, high = randomization_interval(
        means,
        treated,
        np.zeros(len(means), dtype=int),
        treated_per_stratum=4,
        n_simulations=200,
        seed=1,
    )
    assert low > 0.0
    assert low <= 200.0 + 1.0
    assert high >= 200.0 - 1.0


def test_the_unweighted_total_is_the_cuped_point() -> None:
    means = location_window_means(
        _panel(att=5.0, seed=3, noise=2.0),
        pre_end="2021-01-30",
        post_start="2021-01-31",
        post_end="2021-02-09",
    )
    treated = [f"t{i}" for i in range(4)]
    mask = means["location"].isin(treated).to_numpy()
    total = cuped_total(
        means["pre"].to_numpy(dtype=float),
        means["delta"].to_numpy(dtype=float),
        means["n_post_days"].to_numpy(dtype=float),
        mask,
    )
    assert total == pytest.approx(cuped_difference_in_differences(means, treated).incremental)


def test_the_weighted_total_overstates_an_effect_that_grows_with_size() -> None:
    pre = np.array([1000.0, 10.0, 1000.0, 10.0, 500.0, 500.0])
    effect = 0.1 * pre
    treated = np.array([True, True, False, False, False, False])
    delta = np.where(treated, effect, 0.0)
    days = np.full(len(pre), 10.0)
    truth = float(np.sum(effect[treated] * days[treated]))
    unweighted = cuped_total(pre, delta, days, treated)
    weighted = cuped_total(pre, delta, days, treated, weights=pre)
    assert abs(unweighted - truth) < abs(weighted - truth)
    assert weighted > truth


def test_strata_keep_their_own_treated_counts_and_an_all_control_stratum_stays_control() -> None:
    strata = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
    picks = stratified_picks(strata, {0: 2, 1: 0}, 50, np.random.default_rng(1))
    assert picks.shape == (50, 2)
    assert set(np.unique(picks)) <= {0, 1, 2, 3}
    with pytest.raises(ValueError, match="two control"):
        stratified_picks(strata, {0: 3, 1: 0}, 5, np.random.default_rng(1))
