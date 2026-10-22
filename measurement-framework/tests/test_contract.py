from pathlib import Path

import pandas as pd
import pytest

from measurement.contract import (
    TAXONOMY,
    media_regressor,
    model_regressors,
    split_is_identified,
    validate_regressors,
)
from measurement.sql_layer import connect, metric_dictionary, model_grain

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "dt_simulated_weekly.csv"


def test_one_regressor_per_paid_channel_prefers_exposure() -> None:
    assert model_regressors() == ["tv_S", "ooh_S", "print_S", "facebook_I", "search_clicks_P"]
    facebook = next(channel for channel in TAXONOMY if channel.channel_id == "facebook")
    assert media_regressor(facebook) == "facebook_I"


def test_same_channel_metrics_are_rejected() -> None:
    frame = pd.read_csv(CSV)
    with pytest.raises(ValueError, match="channel facebook"):
        validate_regressors(frame, ["facebook_S", "facebook_I"])


def test_unidentified_split_is_rejected_and_a_distinct_split_is_kept() -> None:
    frame = pd.DataFrame(
        {
            "tv_S": [1.0, 2.0, 3.0, 4.0, 5.0],
            "ooh_S": [1.0, 2.0, 3.0, 4.0, 5.1],
            "print_S": [5.0, 1.0, 4.0, 2.0, 3.0],
        }
    )
    assert not split_is_identified(frame["tv_S"], frame["ooh_S"])
    assert split_is_identified(frame["tv_S"], frame["print_S"])
    with pytest.raises(ValueError, match="not identified"):
        validate_regressors(frame, ["tv_S", "ooh_S"])
    validate_regressors(frame, ["tv_S", "print_S"])


def test_sql_grain_matches_the_contract() -> None:
    connection = connect(CSV)
    dictionary = metric_dictionary(connection)
    grain = model_grain(connection)
    assert set(dictionary["channel_id"]) == {
        "tv",
        "ooh",
        "print",
        "facebook",
        "search",
        "newsletter",
        "competitor_sales",
        "events",
    }
    for column in model_regressors():
        assert column in grain.columns
    assert "facebook_S" in grain.columns
    assert "facebook_I" in grain.columns
    facebook = dictionary.loc[dictionary["channel_id"] == "facebook"].iloc[0]
    assert facebook["regressor_column"] == "facebook_I"
    assert facebook["spend_column"] == "facebook_S"
