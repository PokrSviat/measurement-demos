"""Checks that sit beside the scale decision and do not enter it."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def highest_cost_geos(
    panel: pd.DataFrame,
    locations: list[str],
    *,
    start: str,
    end: str,
    n: int = 5,
) -> list[str]:
    """Treated geos with the highest advertising cost in the window, largest first."""
    if n < 1:
        raise ValueError("n must be positive.")
    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    window = frame.loc[
        frame["location"].isin(locations)
        & (frame["date"] >= pd.Timestamp(start))
        & (frame["date"] <= pd.Timestamp(end))
    ]
    totals = window.groupby("location")["cost"].sum().sort_values(ascending=False)
    if int(totals.shape[0]) <= n:
        raise ValueError("Not enough geos to drop the highest-cost ones.")
    return [str(location) for location in totals.head(n).index]


def cooldown_window(panel: pd.DataFrame, post_end: str) -> tuple[str, str]:
    """Day after the test through the last date in the panel."""
    start = pd.Timestamp(post_end) + pd.Timedelta(days=1)
    end = pd.Timestamp(panel["date"].max())
    if end < start:
        raise ValueError("Panel ends before the post-test window.")
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def control_pre_delta_correlation(
    location_means: pd.DataFrame,
    treated_locations: list[str],
) -> float:
    """Correlation of the pre-period mean and the change, on control geos only."""
    control = location_means.loc[~location_means["location"].isin(treated_locations)]
    if control.shape[0] < 2:
        raise ValueError("Need at least two control geos.")
    return float(control["pre"].corr(control["delta"]))


def daily_group_means(
    panel: pd.DataFrame,
    treated_locations: list[str],
    control_locations: list[str],
) -> pd.DataFrame:
    """One row per day with the mean outcome in each group."""
    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    treated = (
        frame.loc[frame["location"].isin(treated_locations)]
        .groupby("date")["Y"]
        .mean()
        .rename("treated")
    )
    control = (
        frame.loc[frame["location"].isin(control_locations)]
        .groupby("date")["Y"]
        .mean()
        .rename("control")
    )
    means = pd.concat([treated, control], axis=1).sort_index()
    if means.empty or means.isna().any().any():
        raise ValueError("Daily group means are incomplete.")
    return means


def write_path_chart(
    means: pd.DataFrame,
    *,
    post_start: str,
    post_end: str,
    path: Path,
) -> None:
    """Mean daily sales for the test and control groups, with the test window marked."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.axvspan(
        pd.Timestamp(post_start),
        pd.Timestamp(post_end),
        color="0.85",
        zorder=0,
    )
    axis.plot(means.index, means["treated"], label="Test geos")
    axis.plot(means.index, means["control"], label="Control geos")
    axis.set_ylabel("Mean daily sales")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
