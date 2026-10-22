import pandas as pd

from geo_holdout.diagnostics import (
    control_pre_delta_correlation,
    cooldown_window,
    highest_cost_geos,
)


def test_control_correlation_is_finite() -> None:
    means = pd.DataFrame(
        {
            "location": ["t", "c1", "c2", "c3"],
            "pre": [10.0, 1.0, 2.0, 3.0],
            "delta": [4.0, 1.5, 2.5, 3.5],
        }
    )
    correlation = control_pre_delta_correlation(means, ["t"])
    assert correlation == correlation
    assert -1 <= correlation <= 1


def test_cooldown_window_starts_after_the_test() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2015-03-15", "2015-03-16", "2015-04-07"]),
            "location": ["1", "1", "1"],
            "Y": [1.0, 1.0, 1.0],
        }
    )
    start, end = cooldown_window(panel, "2015-03-15")
    assert start > "2015-03-15"
    assert start == "2015-03-16"
    assert end == "2015-04-07"


def test_highest_cost_geos_are_the_top_spend() -> None:
    rows = [
        {"location": name, "date": "2015-02-16", "cost": cost, "Y": 1.0}
        for name, cost in (
            ("a", 1),
            ("b", 50),
            ("c", 40),
            ("d", 30),
            ("e", 20),
            ("f", 10),
            ("g", 5),
            ("outside", 999),
        )
    ]
    panel = pd.DataFrame(rows)
    panel["date"] = pd.to_datetime(panel["date"])
    dropped = highest_cost_geos(
        panel,
        ["a", "b", "c", "d", "e", "f", "g"],
        start="2015-02-16",
        end="2015-03-15",
    )
    assert dropped == ["b", "c", "d", "e", "f"]
