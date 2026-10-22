import math

import pytest

from budget_rule.cost import cost_against_cap


def test_the_cap_itself_passes() -> None:
    result = cost_against_cap(100, 1000, 10)
    assert result.clears
    assert result.cost_per_incremental == 10
    assert result.reason == "within_cap"


def test_a_bound_at_or_below_zero_has_no_cost() -> None:
    result = cost_against_cap(0, 100, 10)
    assert not result.clears
    assert result.cost_per_incremental is None
    assert result.reason == "interval_includes_zero"


def test_a_cost_above_the_cap_keeps_the_ratio() -> None:
    result = cost_against_cap(10, 500, 10)
    assert not result.clears
    assert result.cost_per_incremental == 50
    assert result.reason == "above_cap"


def test_a_non_finite_bound_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        cost_against_cap(math.nan, 1, 1)
