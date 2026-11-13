"""Deterministic memo built from the facts alone. Used when no LLM is configured or the check fails."""
from __future__ import annotations

from results_memo.facts import Facts


def _r(x: float) -> str:
    return f"{x:,.0f}"


def template_memo(f: Facts) -> str:
    v = f.values
    return "\n".join([
        f.question,
        "",
        f"A geo test ran from {f.dates['window_start']} to {f.dates['test_end']}, read through {f.dates['window_end']}, "
        f"with {int(v['n_treated'])} test and {int(v['n_control'])} control geos and a test cost of {_r(v['test_cost'])}.",
        f"Cumulative incremental sales (CUPED): {_r(v['cuped_incremental'])}; "
        f"randomization interval {_r(v['randomization_low'])} to {_r(v['randomization_high'])}.",
        f"After the test the effect was {_r(v['after_test_incremental'])} "
        f"(interval {_r(v['after_test_randomization_low'])} to {_r(v['after_test_randomization_high'])}).",
        f"Cost per incremental sale at the lower bound: {v['cost_per_incremental_at_lower_bound']:.2f}.",
        "",
        f"Recommendation: {f.decision_text} {f.reason}",
    ])
