"""One seeded world with a known incremental effect for a single tactic."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SEED = 20241006
N_GEO = 16
N_WEEKS = 40
N_TEST_GEO = 4
TEST_WEEKS = 3
INCREMENT_PER_GEO_WEEK = 20.0
# Geo-week noise is the same size as the planted increment. The experiment
# interval is then wide enough that the planning cap can refuse.
WEEK_NOISE_SD = 20.0
# Share of baseline journeys that also touch the tactic. Those journeys are not
# incremental sales, so a Markov removal effect is larger than the planted effect.
BASELINE_TACTIC_SHARE = 0.02
# Planning cap for this synthetic case, in spend units per incremental conversion.
# It is not fit to a draw.
MAX_COST_PER_INCREMENTAL = 0.10


@dataclass(frozen=True)
class SyntheticWorld:
    panel: pd.DataFrame
    weekly: pd.DataFrame
    paths: list[list[str]]
    true_incremental: float
    treated_locations: list[str]
    post_start_week: int


def generate(
    seed: int = SEED,
    *,
    noise_sd: float = WEEK_NOISE_SD,
    increment: float = INCREMENT_PER_GEO_WEEK,
    include_paths: bool = True,
) -> SyntheticWorld:
    """One seeded world. The path log is drawn after the panel, so skipping it leaves the panel unchanged."""
    if increment < 0:
        raise ValueError("The planted increment cannot be negative.")
    rng = np.random.default_rng(seed)
    locations = [f"g{i:02d}" for i in range(N_GEO)]
    # The treated geos are drawn before the noise, on this same stream.
    drawn = rng.choice(N_GEO, N_TEST_GEO, replace=False)
    treated = [locations[index] for index in sorted(int(index) for index in drawn)]
    post_start = N_WEEKS - TEST_WEEKS
    rows: list[dict[str, object]] = []
    for week in range(N_WEEKS):
        for location in locations:
            treated_now = location in treated and week >= post_start
            baseline = 80.0 + rng.normal(0, noise_sd)
            always_on = 15.0
            other = 10.0
            incremental = increment if treated_now else 0.0
            rows.append(
                {
                    "location": location,
                    "week": week,
                    "y": baseline + always_on + other + incremental,
                    "treated": int(treated_now),
                    "always_on_spend": 1.0,
                    "other_spend": 1.0,
                    "tactic_spend": 1.0 if treated_now else 0.0,
                }
            )
    panel = pd.DataFrame(rows)
    weekly = (
        panel.groupby("week", as_index=False)
        .agg(
            revenue=("y", "sum"),
            always_on_spend=("always_on_spend", "sum"),
            other_spend=("other_spend", "sum"),
            tactic_spend=("tactic_spend", "sum"),
        )
        .sort_values("week")
    )
    paths: list[list[str]] = []
    for row in rows if include_paths else []:
        baseline_conversions = int(round(row["y"] - (increment if row["treated"] else 0.0)))
        for _ in range(baseline_conversions):
            channel = "always_on" if rng.random() < 0.6 else "other"
            if rng.random() < BASELINE_TACTIC_SHARE:
                paths.append(["start", channel, "tactic", "conversion"])
            else:
                paths.append(["start", channel, "conversion"])
        if row["treated"]:
            for _ in range(int(increment)):
                paths.append(["start", "always_on", "tactic", "conversion"])
    true_incremental = increment * N_TEST_GEO * TEST_WEEKS
    return SyntheticWorld(panel, weekly, paths, true_incremental, treated, post_start)


def experiment_cost(world: SyntheticWorld) -> float:
    """Tactic spend on treated geo-weeks. That is the cost the budget rule uses."""
    return float(world.panel.loc[world.panel["treated"] == 1, "tactic_spend"].sum())


def omit_tactic(paths: list[list[str]]) -> list[list[str]]:
    """Same journeys with the tactic step removed from the path log.

    The panel and the weekly series are unchanged. Only the Markov read moves.
    """
    return [[step for step in path if step != "tactic"] for path in paths]
