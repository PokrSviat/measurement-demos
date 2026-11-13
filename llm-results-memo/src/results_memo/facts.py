"""Facts the memo may use, read from a published result file.

Only the fields listed here go into the prompt and into the grounding check.
A number that is in the source file but not in this list is not allowed in
the text, so the check stays tied to what the prompt showed the model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Facts:
    source: str
    question: str
    decision_label: str
    decision_text: str
    reason: str
    values: dict[str, float]
    dates: dict[str, str] = field(default_factory=dict)


DECISION_WORDS = ("Hold", "Scale", "Stop")


def decision_label(recommendation: str, fund: bool | None = None) -> str:
    """First decision word of the recommendation; fund=True without a word means Scale."""
    head = recommendation.strip().split(".")[0].split()[0].capitalize() if recommendation.strip() else ""
    if head in DECISION_WORDS:
        return head
    if fund is True:
        return "Scale"
    raise ValueError(f"No decision word in recommendation: {recommendation!r}")


def geo_holdout_facts(path: str | Path) -> Facts:
    data = json.loads(Path(path).read_text())
    d = data["decision"]
    values = {
        "n_treated": data["n_treated"],
        "n_control": data["n_control"],
        "test_cost": data["test_cost"],
        "cuped_incremental": data["cuped"]["incremental"],
        "randomization_low": data["randomization_interval"]["low"],
        "randomization_high": data["randomization_interval"]["high"],
        "did_incremental": data["did"]["incremental"],
        "test_window_incremental": data["test_window"]["cuped"]["incremental"],
        "after_test_incremental": data["after_test"]["cuped"]["incremental"],
        "after_test_randomization_low": data["after_test"]["randomization_interval"]["low"],
        "after_test_randomization_high": data["after_test"]["randomization_interval"]["high"],
        "cost_per_incremental_at_lower_bound": d["cost_per_incremental"],
        "top5_pretest_share": data["top5_pretest_share"],
    }
    dates = {
        "window_start": data["window"]["start"],
        "test_end": data["window"]["test_end"],
        "window_end": data["window"]["end"],
    }
    return Facts(
        source=str(path),
        question=data["question"],
        decision_label=decision_label(d["recommendation"], d.get("fund")),
        decision_text=d["recommendation"],
        reason=d["reason"],
        values={k: float(v) for k, v in values.items()},
        dates=dates,
    )
