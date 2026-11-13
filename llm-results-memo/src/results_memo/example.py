"""Reproduce results/fake_llm_examples.json: two scripted drafts (not model output) run through the check."""
from __future__ import annotations

import json
from pathlib import Path

from results_memo.facts import geo_holdout_facts
from results_memo.llm import FakeLLM
from results_memo.run import produce

GROUNDED = (
    "The geo test with 50 test and 50 control geos added about 151k incremental sales "
    "(randomization interval 115,768 to 198,149). After the test the effect was close to zero (1,126).\n"
    "Recommendation: Hold. Confirm on a new panel before any scale decision."
)
HALLUCINATED = (
    "The geo test lifted sales by 18% and returned 3.2x ROI, about 151k incremental sales.\n"
    "Recommendation: Scale the tactic to all regions."
)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    facts = geo_holdout_facts(root.parent / "geo-holdout-case/results/case_result.json")
    from dataclasses import replace
    facts = replace(facts, source="geo-holdout-case/results/case_result.json")
    runs = []
    for name, text in [("scripted-grounded", GROUNDED), ("scripted-hallucinated", HALLUCINATED)]:
        out = produce(facts, FakeLLM(text, name=f"fake-llm:{name}"))
        runs.append(out)
    note = ("Scripted drafts from FakeLLM, written by hand to exercise the check. "
            "They are not language-model output.")
    (root / "results/fake_llm_examples.json").write_text(json.dumps({"note": note, "runs": runs}, indent=2) + "\n")


if __name__ == "__main__":
    main()
