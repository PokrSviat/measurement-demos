"""DuckDB metric dictionary and the weekly grain used by the model."""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from measurement.contract import CALENDAR_TERMS, TAXONOMY, media_regressor

WEEKS_PER_YEAR = 52.18

METRIC_SQL = """
SELECT * FROM (
    VALUES
        ('tv', 'paid', 'spend', 'tv_S', 'tv_S'),
        ('ooh', 'paid', 'spend', 'ooh_S', 'ooh_S'),
        ('print', 'paid', 'spend', 'print_S', 'print_S'),
        ('facebook', 'paid', 'impressions', 'facebook_I', 'facebook_S'),
        ('search', 'paid', 'clicks', 'search_clicks_P', 'search_S'),
        ('newsletter', 'organic', 'activity', 'newsletter', NULL),
        ('competitor_sales', 'context', 'level', 'competitor_sales_B', NULL),
        ('events', 'context', 'flag', 'events', NULL)
) AS metric(channel_id, role, metric_kind, regressor_column, spend_column)
"""

GRAIN_SQL = """
SELECT
    CAST(DATE AS DATE) AS date,
    revenue,
    tv_S,
    ooh_S,
    print_S,
    facebook_I,
    search_clicks_P,
    facebook_S,
    search_S,
    tv_S AS tv_spend,
    ooh_S AS ooh_spend,
    print_S AS print_spend,
    facebook_S AS facebook_spend,
    search_S AS search_spend,
    newsletter,
    competitor_sales_B,
    CASE WHEN lower(CAST(events AS VARCHAR)) IN ('na', 'none', '') THEN 0 ELSE 1 END AS event_flag
FROM weekly
ORDER BY 1
"""


def connect(csv_path: Path) -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect()
    connection.execute(
        "CREATE TABLE weekly AS SELECT * FROM read_csv_auto(?)",
        [str(csv_path)],
    )
    return connection


def metric_dictionary(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return connection.execute(METRIC_SQL).fetchdf()


def add_holiday_flag(grain: pd.DataFrame, holidays: pd.DataFrame) -> pd.DataFrame:
    """Flag weeks that contain a holiday from the public demo calendar.

    The Robyn demo tags that calendar as DE. This is a property of the public
    series, not a market claim.
    """
    frame = grain.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    holiday_dates = pd.to_datetime(holidays["ds"])
    flags = []
    for date in frame["date"]:
        in_week = (holiday_dates >= date) & (holiday_dates < date + pd.Timedelta(days=7))
        flags.append(int(in_week.any()))
    frame["holiday_flag"] = flags
    return frame


def add_calendar_terms(grain: pd.DataFrame, origin: str | pd.Timestamp | None = None) -> pd.DataFrame:
    """Linear trend in years and two yearly harmonics, counted in weeks from `origin`.

    The origin defaults to the first date of the frame. A saved week keeps its
    terms because they are columns of the frame, not recomputed from a window.
    """
    frame = grain.copy()
    dates = pd.to_datetime(frame["date"])
    if dates.empty:
        raise ValueError("The grain has no dates.")
    start = dates.min() if origin is None else pd.Timestamp(origin)
    weeks = (dates - start).dt.days.to_numpy(dtype=float) / 7.0
    frame["trend"] = weeks / WEEKS_PER_YEAR
    for harmonic in (1, 2):
        angle = 2 * np.pi * harmonic * weeks / WEEKS_PER_YEAR
        frame[f"sin_{harmonic}"] = np.sin(angle)
        frame[f"cos_{harmonic}"] = np.cos(angle)
    missing = [term for term in CALENDAR_TERMS if term not in frame.columns]
    if missing:
        raise ValueError(f"Calendar terms are missing: {missing}")
    return frame


def model_grain(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    grain = connection.execute(GRAIN_SQL).fetchdf()
    expected = [media_regressor(channel) for channel in TAXONOMY if channel.role == "paid"]
    missing = [column for column in expected if column not in grain.columns]
    if missing:
        raise ValueError(f"Grain is missing paid regressors: {missing}")
    return grain
