"""The bake-off next-ruble operator is one finite difference, with the hill frozen."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from bakeoff.next_ruble import next_ruble_roi

ROOT = Path(__file__).resolve().parents[1]
MEDIA = np.array([2.0, 2.0, 2.0, 2.0, 10.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
SPEND = np.array([2.0, 0.0, 2.0, 2.0, 10.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
THETA = 0.8
ALPHA = 1.5
GAMMA = 0.5
COEF = 2.0
HORIZON = 8


def _geometric_decay(media: np.ndarray, theta: float) -> np.ndarray:
    decayed = np.empty(len(media), dtype=float)
    decayed[0] = media[0]
    for index in range(1, len(media)):
        decayed[index] = media[index] + theta * decayed[index - 1]
    return decayed


def _explicit_finite_difference() -> float:
    """The same ruble, written out here so the helper cannot drift into another sum."""
    decayed = _geometric_decay(MEDIA, THETA)
    inflexion = float(np.max(decayed) * GAMMA)
    base = COEF * (decayed**ALPHA) / (decayed**ALPHA + inflexion**ALPHA)
    last = len(MEDIA) - HORIZON
    deltas = []
    weights = []
    for week in range(last + 1):
        if SPEND[week] <= 0:
            continue
        bumped = MEDIA.copy()
        bumped[week] = bumped[week] + MEDIA[week] / SPEND[week]
        changed_decayed = _geometric_decay(bumped, THETA)
        changed = COEF * (changed_decayed**ALPHA) / (changed_decayed**ALPHA + inflexion**ALPHA)
        deltas.append(float(np.sum(changed[week : week + HORIZON] - base[week : week + HORIZON])))
        weights.append(float(SPEND[week]))
    return float(np.average(deltas, weights=weights))


def _live_hill_max() -> float:
    """Recomputing inflexion from the bumped series is the operator this column rejects."""
    deltas = []
    weights = []
    last = len(MEDIA) - HORIZON
    for week in range(last + 1):
        if SPEND[week] <= 0:
            continue
        base_decayed = _geometric_decay(MEDIA, THETA)
        base_inflexion = float(np.max(base_decayed) * GAMMA)
        base = COEF * (base_decayed**ALPHA) / (base_decayed**ALPHA + base_inflexion**ALPHA)
        bumped = MEDIA.copy()
        bumped[week] = bumped[week] + MEDIA[week] / SPEND[week]
        changed_decayed = _geometric_decay(bumped, THETA)
        live_inflexion = float(np.max(changed_decayed) * GAMMA)
        changed = COEF * (changed_decayed**ALPHA) / (changed_decayed**ALPHA + live_inflexion**ALPHA)
        deltas.append(float(np.sum(changed - base)))
        weights.append(float(SPEND[week]))
    return float(np.average(deltas, weights=weights))


def test_the_helper_matches_the_explicit_finite_difference() -> None:
    assert next_ruble_roi(MEDIA, SPEND, THETA, ALPHA, GAMMA, COEF, HORIZON) == pytest.approx(
        _explicit_finite_difference()
    )


def test_recomputing_the_hill_max_is_a_different_number() -> None:
    frozen = next_ruble_roi(MEDIA, SPEND, THETA, ALPHA, GAMMA, COEF, HORIZON)
    assert frozen != pytest.approx(_live_hill_max())
    assert frozen > 0


def test_the_r_publisher_uses_the_frozen_operator() -> None:
    publisher = (ROOT / "robyn" / "run_robyn.R").read_text(encoding="utf-8")
    assert 'source(file.path(script_dir, "next_ruble.R"))' in publisher
    assert "saturation_hill" not in publisher
    assert "marginal_roi(" not in (ROOT / "meridian" / "run_meridian.py").read_text(encoding="utf-8")


def test_the_r_operator_matches_the_python_finite_difference() -> None:
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript is not installed, so the frozen-hill operator is not checked.")
    script = ROOT / "robyn" / "next_ruble.R"
    media = ",".join(str(value) for value in MEDIA)
    spend = ",".join(str(value) for value in SPEND)
    program = f"""
source("{script.as_posix()}")
media <- c({media})
spend <- c({spend})
value <- next_ruble_roi(media, spend, {THETA}, {ALPHA}, {GAMMA}, {COEF}, {HORIZON}L)
cat(sprintf("%.16g", value))
"""
    completed = subprocess.run(
        ["Rscript", "-e", program],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise AssertionError(completed.stderr)
    assert float(completed.stdout.strip()) == pytest.approx(
        next_ruble_roi(MEDIA, SPEND, THETA, ALPHA, GAMMA, COEF, HORIZON)
    )
