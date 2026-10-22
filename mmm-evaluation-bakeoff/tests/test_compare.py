import importlib.util
import json
from pathlib import Path

import pytest

from bakeoff.compare import (
    comparison_table,
    earning_channels,
    experiment_by_model,
    experiment_channel,
    experiment_priorities,
    holdout_mape,
    load_export,
    load_seed_check,
    memo_text,
    published_exports,
    seed_sentence,
    split_exports,
    unsettled_channels,
)

ACTUAL = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0]


def _export(model: str, digest: str = "abc", r_hat: float | None = None, error: float = 0.1) -> dict:
    return {
        "model": model,
        "manifest": {
            "dataset_sha256": digest,
            "target": "revenue",
            "date_min": "2015-11-23",
            "date_max": "2019-11-11",
            "regressors": ["tv_S", "facebook_I"],
            "holdout_weeks": 8,
        },
        "holdout_mape": 0.999,
        "holdout_actual": ACTUAL,
        "holdout_pred": [value * (1 + error) for value in ACTUAL],
        "estimand": "marginal_roi_next_ruble",
        "channel_roi": {"tv": 0.5, "facebook": 1.5},
        "channel_roi_low": {"tv": 0.2, "facebook": 0.5},
        "channel_roi_high": {"tv": 0.8, "facebook": 3.0},
        "interval": "test interval",
        "interval_kind": "model_spread" if model == "robyn" else "posterior",
        "channel_spend_share": {"tv": 0.8, "facebook": 0.2},
        "r_hat_max": r_hat,
        "prior_sensitivity_relative": 0.1 if model == "meridian" else None,
    }


def test_matching_manifests_build_one_table(tmp_path: Path) -> None:
    exports = []
    for model, r_hat in (("pymc", None), ("robyn", None), ("meridian", 1.004)):
        path = tmp_path / f"{model}.json"
        path.write_text(json.dumps(_export(model, r_hat=r_hat)), encoding="utf-8")
        exports.append(load_export(path))
    table = comparison_table(exports)
    assert list(table["model"]) == ["pymc", "robyn", "meridian"]
    assert table.loc[table["model"] == "meridian", "r_hat_max"].iloc[0] == pytest.approx(1.004)
    assert "comparable" not in table.columns
    assert json.loads(table["posterior_roi"].iloc[0])["tv"] == [0.2, 0.5, 0.8]
    robyn = table.loc[table["model"] == "robyn"].iloc[0]
    assert robyn["interval_kind"] == "model_spread"
    assert robyn.isna()["posterior_roi"] and isinstance(robyn["model_spread_roi"], str)


def test_an_unknown_interval_kind_is_rejected(tmp_path: Path) -> None:
    export = _export("pymc")
    export["interval_kind"] = "bootstrap"
    path = tmp_path / "pymc.json"
    path.write_text(json.dumps(export), encoding="utf-8")
    with pytest.raises(ValueError, match="interval_kind"):
        load_export(path)


def test_the_harness_computes_mape_and_ignores_the_stored_value() -> None:
    table = comparison_table([_export("pymc", error=0.1), _export("robyn", error=0.2)])
    assert list(table["holdout_mape"]) == pytest.approx([0.1, 0.2])
    assert holdout_mape([0.0, 10.0], [1.0, 12.0]) == pytest.approx((1.0 + 0.2) / 2)
    with pytest.raises(ValueError, match="matching"):
        holdout_mape([1.0], [1.0, 2.0])


def test_an_export_without_a_holdout_path_is_rejected(tmp_path: Path) -> None:
    export = _export("pymc")
    del export["holdout_pred"]
    path = tmp_path / "pymc.json"
    path.write_text(json.dumps(export), encoding="utf-8")
    with pytest.raises(ValueError, match="holdout_pred"):
        load_export(path)


def test_different_holdout_weeks_are_rejected() -> None:
    other = _export("robyn")
    other["holdout_actual"] = [value + 1 for value in ACTUAL]
    with pytest.raises(ValueError, match="different weeks"):
        comparison_table([_export("pymc"), other])


def test_a_different_dataset_hash_is_rejected() -> None:
    exports = [
        _export("pymc"),
        _export("robyn", digest="other"),
        _export("meridian", r_hat=1.0),
    ]
    with pytest.raises(ValueError, match="dataset_sha256"):
        comparison_table(exports)


def test_meridian_without_rhat_is_rejected() -> None:
    with pytest.raises(ValueError, match="r_hat_max"):
        comparison_table([_export("pymc"), _export("robyn"), _export("meridian")])


def test_an_export_on_another_estimand_is_rejected() -> None:
    other = _export("robyn")
    other["estimand"] = "roi_total"
    with pytest.raises(ValueError, match="marginal_roi_next_ruble"):
        comparison_table([_export("pymc"), other])


def test_the_experiment_is_the_majority_of_per_model_picks() -> None:
    left = _export("pymc")
    right = _export("robyn")
    right["channel_roi_high"] = {"tv": 1.4, "facebook": 3.0}
    picks = experiment_by_model([left, right])
    assert picks["pymc"] == "facebook"
    assert picks["robyn"] == "tv"
    assert experiment_channel([left, right]) == "tv"
    assert experiment_priorities([left, right])["robyn"]["tv"] == pytest.approx(0.8 * 1.2)
    assert "A lift test on tv" not in memo_text([left, right])
    assert "No lift test is recommended" in memo_text([left, right])
    assert "pymc (posterior) facebook" in memo_text([left, right])
    third = _export("meridian", r_hat=1.0)
    third["channel_roi_high"] = {"tv": 0.8, "facebook": 3.0}
    assert experiment_channel([left, right, third]) == "facebook"
    assert unsettled_channels([left, right]) == ["facebook", "tv"]
    right["channel_roi_low"] = {"tv": 1.2, "facebook": 0.5}
    right["channel_roi_high"] = {"tv": 1.8, "facebook": 3.0}
    left["channel_roi_low"] = {"tv": 1.1, "facebook": 0.5}
    left["channel_roi_high"] = {"tv": 1.6, "facebook": 3.0}
    assert earning_channels([left, right]) == ["tv"]
    assert "Every model puts the interval for tv above 1." in memo_text([left, right])


def test_the_experiment_needs_spend_shares_and_can_be_none() -> None:
    left = _export("pymc")
    right = _export("robyn")
    for export in (left, right):
        export["channel_roi_low"] = {"tv": 1.1, "facebook": 1.2}
        export["channel_roi_high"] = {"tv": 1.5, "facebook": 3.0}
    assert experiment_channel([left, right]) is None
    assert "No lift test is recommended" in memo_text([left, right])
    del left["channel_spend_share"]
    with pytest.raises(ValueError, match="channel_spend_share"):
        experiment_channel([left, right])


def test_seed_sentence_names_the_largest_move() -> None:
    assert seed_sentence(None) == ""
    text = seed_sentence({"seeds": [1, 2, 3], "point_roi_spread": {"tv": 0.1, "search": 0.4}})
    assert "seeds 1, 2, 3" in text and "0.40, on search" in text


def test_published_exports_do_not_reallocate_and_name_an_experiment() -> None:
    root = Path(__file__).resolve().parents[1]
    exports, excluded = split_exports(root)
    assert published_exports(root) == exports
    assert [export["model"] for export in exports] == ["pymc", "robyn", "meridian"]
    assert excluded == []
    meridian = next(export for export in exports if export["model"] == "meridian")
    assert meridian["r_hat_max"] is not None
    assert meridian["prior_sensitivity_relative"] is not None
    for export in exports:
        assert export["estimand"] == "marginal_roi_next_ruble"
        for channel in export["channel_roi"]:
            assert export["channel_roi_low"][channel] <= export["channel_roi_high"][channel]
    kinds = {export["model"]: export["interval_kind"] for export in exports}
    assert kinds == {"pymc": "posterior", "robyn": "model_spread", "meridian": "posterior"}
    assert set(meridian["channel_contraction"]) == set(meridian["channel_roi"])
    seed_check = load_seed_check(root)
    assert seed_check is not None and len(seed_check["seeds"]) == 3
    text = memo_text(exports, excluded=excluded, seed_check=seed_check)
    assert (root / "results" / "memo.txt").read_text(encoding="utf-8") == text + "\n"
    pdf = root / "results" / "memo.pdf"
    assert pdf.is_file() and pdf.read_bytes().startswith(b"%PDF")
    lowered = text.lower()
    for forbidden in ("m.video", "yandex", "hoff", "okkam", "apple", "synthetic control"):
        assert forbidden not in lowered
    assert "Do not reallocate." in text
    assert "do not identify the channel returns" in text
    assert "A lift test on ooh" not in text
    assert "A lift test on search" not in text
    assert "No lift test is recommended" in text
    assert "do not share an interval kind" in text
    assert "one formula" in text
    assert "not a posterior" in text
    table = comparison_table(exports)
    stored = {export["model"]: float(export["holdout_mape"]) for export in exports}
    for row in table.itertuples(index=False):
        if row.model == "pymc":
            assert row.holdout_mape == pytest.approx(stored["pymc"])


def test_prior_sensitivity_is_the_relative_roi_change() -> None:
    path = Path(__file__).resolve().parents[1] / "meridian" / "run_meridian.py"
    spec = importlib.util.spec_from_file_location("run_meridian", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.prior_sensitivity(2.0, 3.0) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="non-zero"):
        module.prior_sensitivity(0.0, 1.0)
    summary = module.summarize_draws({"tv": __import__("numpy").arange(1, 101, dtype=float)})
    assert summary["tv"]["mean"] == pytest.approx(50.5)
    assert summary["tv"]["low"] < summary["tv"]["mean"] < summary["tv"]["high"]
    numpy = __import__("numpy")
    rng = numpy.random.default_rng(0)
    prior = rng.normal(0, 2.0, (1, 4000, 2))
    posterior = numpy.stack([rng.normal(0, 1.0, (4, 1000)), rng.normal(0, 2.0, (4, 1000))], axis=-1)
    contraction = module.channel_contraction(prior, posterior, ["tv", "search"])
    assert contraction["tv"] == pytest.approx(0.75, abs=0.03)
    assert contraction["search"] == pytest.approx(0.0, abs=0.06)
    with pytest.raises(ValueError, match="positive variance"):
        module.draw_contraction(numpy.ones(10), numpy.ones(10))
    spread = module.seed_spread([{"channel_roi": {"tv": 1.0}}, {"channel_roi": {"tv": 1.3}}])
    assert spread["tv"] == pytest.approx(0.3)


def test_matching_roi_order_still_does_not_reallocate() -> None:
    left = _export("pymc")
    right = _export("robyn", error=0.2)
    text = memo_text([left, right])
    assert "Do not reallocate." in text
    assert "The ROI order is the same" in text
    other = _export("meridian", r_hat=1.02)
    other["channel_roi"] = {"tv": 2.0, "facebook": 0.5}
    different = memo_text([left, other])
    assert "The ROI order differs" in different
    assert "r_hat_max 1.020, at or above 1.01" in different
