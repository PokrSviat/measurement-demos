import ast
import json
import sys
from pathlib import Path

import pytest

from bakeoff.compare import holdout_mape
from bakeoff.track import main
from bakeoff.tracking import log_bakeoff, metrics_from_exports

ACTUAL = [100.0, 200.0]


def _export(model: str, roi: float, *, stored_mape: float = 0.999) -> dict:
    return {
        "model": model,
        "manifest": {
            "dataset_sha256": "abc123def456",
            "target": "revenue",
            "date_min": "2015-11-23",
            "date_max": "2019-11-11",
            "regressors": ["tv_S"],
            "holdout_weeks": 2,
            "seed": 7,
        },
        "holdout_mape": stored_mape,
        "holdout_actual": ACTUAL,
        "holdout_pred": [value * (1 + roi / 10) for value in ACTUAL],
        "estimand": "marginal_roi_next_ruble",
        "channel_roi": {"tv": roi},
        "channel_roi_low": {"tv": roi - 0.1},
        "channel_roi_high": {"tv": roi + 0.1},
        "interval_kind": "model_spread" if model == "robyn" else "posterior",
    }


def _write_case(root: Path) -> dict[str, dict]:
    exports = {"pymc": _export("pymc", 1.5), "robyn": _export("robyn", 0.4)}
    artifacts = root / "artifacts"
    results = root / "results"
    artifacts.mkdir()
    results.mkdir()
    for model, export in exports.items():
        (artifacts / f"{model}_export.json").write_text(json.dumps(export), encoding="utf-8")
    (results / "memo.txt").write_text("Do not reallocate.\n", encoding="utf-8")
    return exports


def test_tracking_module_does_not_import_mlflow() -> None:
    source = Path(__file__).resolve().parents[1] / "src" / "bakeoff" / "tracking.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] != "mlflow" for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] != "mlflow"


def test_tracking_is_off_by_default(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING", raising=False)

    def refuse(*_args, **_kwargs):
        raise AssertionError("tracking ran")

    monkeypatch.setattr("bakeoff.track.log_bakeoff", refuse)
    assert main([]) == 0
    assert "Tracking is off" in capsys.readouterr().out


def test_missing_mlflow_is_a_message_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    real_import = __import__

    def blocked(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "mlflow" or name.startswith("mlflow."):
            raise ImportError("mlflow is not installed")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", blocked)
    monkeypatch.delitem(sys.modules, "mlflow", raising=False)
    assert main(["--track"]) == 1
    captured = capsys.readouterr()
    assert "not installed" in captured.err
    assert "Traceback" not in captured.err
    assert "Traceback" not in captured.out


def test_log_bakeoff_matches_the_export_json(tmp_path: Path) -> None:
    mlflow = pytest.importorskip("mlflow")
    exports = _write_case(tmp_path)
    store = tmp_path / "mlruns"
    run_id = log_bakeoff(tmp_path, store)
    client = mlflow.tracking.MlflowClient(tracking_uri=store.resolve().as_uri())
    run = client.get_run(run_id)
    expected = metrics_from_exports(list(exports.values()))
    for name, value in expected.items():
        assert run.data.metrics[name] == pytest.approx(value)
    for model, export in exports.items():
        assert run.data.metrics[f"{model}_roi_tv"] == pytest.approx(export["channel_roi"]["tv"])
        formula = holdout_mape(export["holdout_actual"], export["holdout_pred"])
        assert run.data.metrics[f"{model}_holdout_mape"] == pytest.approx(formula)
        assert run.data.metrics[f"{model}_holdout_mape"] != export["holdout_mape"]
    assert run.data.params["target"] == "revenue"
    assert run.data.params["seed"] == "7"
    assert "memo.txt" in {item.path for item in client.list_artifacts(run_id)}
    memo = (tmp_path / "results" / "memo.txt").read_text(encoding="utf-8")
    assert memo == "Do not reallocate.\n"
