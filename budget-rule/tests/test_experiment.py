import math

import pytest

from budget_rule.experiment import (
    experiment_priority,
    lift_design_powered,
    lift_test_weeks,
    national_test_feasible,
    next_experiment,
    publish_lift_test,
)


def test_priority_skips_settled_channels_and_weights_by_spend() -> None:
    share = {"tv": 0.5, "search": 0.1, "ooh": 0.4}
    low = {"tv": 2.0, "search": -5.0, "ooh": 0.2}
    high = {"tv": 5.0, "search": 15.0, "ooh": 1.5}
    priority = experiment_priority(share, low, high)
    assert set(priority) == {"search", "ooh"}
    assert priority["search"] == pytest.approx(2.0)
    assert priority["ooh"] == pytest.approx(0.52)
    assert next_experiment(share, low, high) == "search"
    assert next_experiment(share, {"tv": 2.0}, {"tv": 3.0}) is None


def test_priority_rejects_a_reversed_interval() -> None:
    with pytest.raises(ValueError, match="reversed"):
        experiment_priority({"tv": 1.0}, {"tv": 2.0}, {"tv": 1.0})


def test_lift_test_weeks_solves_the_half_width() -> None:
    weeks = lift_test_weeks(sigma=100.0, rho=0.0, weekly_increase=50.0, expected_roi=3.0)
    assert weeks == math.ceil(2 * (1.96 * 100 / (50 * 2)) ** 2)
    half_width = 1.96 * 100 * math.sqrt(2 / weeks) / 50
    assert half_width <= 2.0
    autocorrelated = lift_test_weeks(sigma=100.0, rho=0.5, weekly_increase=50.0, expected_roi=3.0)
    assert autocorrelated is not None and autocorrelated > weeks
    assert lift_test_weeks(sigma=100.0, rho=0.0, weekly_increase=50.0, expected_roi=1.0) is None
    with pytest.raises(ValueError):
        lift_test_weeks(sigma=100.0, rho=1.0, weekly_increase=50.0, expected_roi=3.0)
    assert national_test_feasible({0.1: 40, 0.5: None})
    assert not national_test_feasible({0.1: 9000, 0.5: 400})
    assert lift_design_powered({"cells": [{"power_at_point_roi": 0.9}]})
    assert not lift_design_powered({"cells": [{"power_at_point_roi": 0.2}]})
    assert publish_lift_test(national_feasible=False, geo_powered=False) is False
    assert publish_lift_test(national_feasible=True, geo_powered=False) is True
