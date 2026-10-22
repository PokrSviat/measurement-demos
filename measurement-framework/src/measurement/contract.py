"""Channel taxonomy and the identification rule used before a model is fit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

Role = Literal["paid", "organic", "context"]
ExposureKind = Literal["impressions", "clicks"]
SPLIT_CORR_MAX = 0.90
CALENDAR_TERMS = ("trend", "sin_1", "cos_1", "sin_2", "cos_2")


@dataclass(frozen=True)
class Channel:
    channel_id: str
    role: Role
    spend_column: str | None = None
    exposure_column: str | None = None
    exposure_kind: ExposureKind | None = None


TAXONOMY: tuple[Channel, ...] = (
    Channel("tv", "paid", spend_column="tv_S"),
    Channel("ooh", "paid", spend_column="ooh_S"),
    Channel("print", "paid", spend_column="print_S"),
    Channel("facebook", "paid", spend_column="facebook_S", exposure_column="facebook_I", exposure_kind="impressions"),
    Channel("search", "paid", spend_column="search_S", exposure_column="search_clicks_P", exposure_kind="clicks"),
    Channel("newsletter", "organic", exposure_column="newsletter"),
    Channel("competitor_sales", "context", exposure_column="competitor_sales_B"),
    Channel("events", "context", exposure_column="events"),
)

_BY_ID = {channel.channel_id: channel for channel in TAXONOMY}


def media_regressor(channel: Channel) -> str | None:
    """One regressor per paid channel. Exposure wins over spend when both exist."""
    if channel.role != "paid":
        return None
    return channel.exposure_column or channel.spend_column


def paid_channels() -> tuple[Channel, ...]:
    return tuple(channel for channel in TAXONOMY if channel.role == "paid")


def column_owner(column: str) -> str | None:
    for channel in TAXONOMY:
        if column in {channel.spend_column, channel.exposure_column}:
            return channel.channel_id
    return None


def correlation(left: pd.Series, right: pd.Series) -> float:
    paired = pd.concat([left, right], axis=1).dropna()
    if paired.shape[0] < 3:
        return 1.0
    value = float(paired.iloc[:, 0].corr(paired.iloc[:, 1]))
    if np.isnan(value):
        return 1.0
    return abs(value)


def split_is_identified(left: pd.Series, right: pd.Series, threshold: float = SPLIT_CORR_MAX) -> bool:
    """A finer split is allowed only when the two series are not the same signal."""
    return correlation(left, right) < threshold


def validate_regressors(frame: pd.DataFrame, regressors: list[str]) -> None:
    """Reject two regressors from one channel, and reject an unidentified split."""
    owners: dict[str, str] = {}
    for column in regressors:
        owner = column_owner(column)
        if owner is None:
            raise ValueError(f"Column {column} is not in the taxonomy.")
        if owner in owners:
            raise ValueError(
                f"{column} and {owners[owner]} belong to channel {owner}. Keep one regressor."
            )
        owners[owner] = column
    for index, left in enumerate(regressors):
        for right in regressors[index + 1 :]:
            if not split_is_identified(frame[left], frame[right]):
                raise ValueError(
                    f"{left} and {right} are not identified as separate regressors."
                )


def model_regressors() -> list[str]:
    return [column for channel in paid_channels() if (column := media_regressor(channel))]


def spend_column_for(regressor: str) -> str:
    owner = column_owner(regressor)
    if owner is None:
        raise ValueError(f"Unknown regressor {regressor}.")
    channel = _BY_ID[owner]
    if channel.spend_column is None:
        raise ValueError(f"{regressor} has no spend column for ROI.")
    return channel.spend_column
