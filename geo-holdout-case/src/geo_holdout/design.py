"""Stratified design and the pseudo experiment that the design checks run on.

Decision power, the share of draws in which the rule says scale at a known
true cost, lives in `validation.py`. It is reported and does not decide.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from geo_holdout.estimators import location_window_means


def volume_strata(pretest: pd.DataFrame, stratum_size: int) -> pd.DataFrame:
    """Rank geos by pre-period sales and cut them into contiguous strata.

    Ties break on the geo id. The assignment file has to put the declared
    number of treated geos in each of these strata, or the case stops.
    """
    if stratum_size < 4:
        raise ValueError("A stratum has to leave room for treated and control geos.")
    frame = pretest.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    totals = frame.groupby("location", as_index=False)["Y"].sum()
    totals = totals.sort_values(["Y", "location"], kind="mergesort").reset_index(drop=True)
    if len(totals) == 0 or len(totals) % stratum_size != 0:
        raise ValueError("The number of geos is not divisible by the stratum size.")
    totals["stratum"] = np.arange(len(totals)) // stratum_size
    return totals.rename(columns={"Y": "volume"})


def assert_stratified_assignment(
    assignment: pd.DataFrame,
    strata: pd.DataFrame,
    *,
    treated_group: int,
    treated_per_stratum: int,
) -> None:
    """The file is the design only when every volume stratum has the declared treated count."""
    merged = strata.merge(assignment, on="location", how="left")
    if merged["group"].isna().any():
        raise ValueError("Every geo in the volume ranking needs a group.")
    counts = merged.groupby("stratum")["group"].apply(lambda group: int((group == treated_group).sum()))
    if not bool((counts == treated_per_stratum).all()):
        raise ValueError(
            "The assignment file does not put "
            f"{treated_per_stratum} treated geos in each pretest-volume stratum."
        )


@dataclass(frozen=True)
class PseudoExperiment:
    """Pre-test split into a pseudo pre-period and a pseudo test window. No geo is treated."""

    pre: np.ndarray
    delta: np.ndarray
    days: np.ndarray
    stratum_ids: np.ndarray
    n_pre_days: int


def pseudo_experiment(
    pretest: pd.DataFrame,
    *,
    stratum_size: int,
    duration_days: int,
) -> PseudoExperiment:
    """The last `duration_days` of the pre-test are the pseudo test window.

    Strata are ranked on the pseudo pre-period only.
    """
    frame = pretest.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    dates = np.array(sorted(frame["date"].unique()))
    if len(dates) <= duration_days + 5:
        raise ValueError("Pre-test is too short for this duration.")
    split = pd.Timestamp(dates[-duration_days])
    earlier = frame.loc[frame["date"] < split]
    strata = volume_strata(earlier, stratum_size)
    means = location_window_means(
        frame,
        pre_end=(split - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        post_start=split.strftime("%Y-%m-%d"),
        post_end=pd.Timestamp(dates[-1]).strftime("%Y-%m-%d"),
    )
    stratum_ids = means["location"].map(strata.set_index("location")["stratum"]).to_numpy()
    if np.isnan(stratum_ids.astype(float)).any():
        raise ValueError("A geo in the pseudo-experiment has no volume stratum.")
    return PseudoExperiment(
        pre=means["pre"].to_numpy(dtype=float),
        delta=means["delta"].to_numpy(dtype=float),
        days=means["n_post_days"].to_numpy(dtype=float),
        stratum_ids=stratum_ids.astype(int),
        n_pre_days=int(earlier["date"].nunique()),
    )
