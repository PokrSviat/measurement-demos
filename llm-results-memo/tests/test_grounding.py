from pathlib import Path

import pytest

from results_memo.facts import Facts, decision_label, geo_holdout_facts
from results_memo.grounding import check, extract_numbers
from results_memo.llm import FakeLLM
from results_memo.run import produce
from results_memo.template import template_memo

SOURCE = Path(__file__).resolve().parents[2] / "geo-holdout-case/results/case_result.json"


def toy(**over) -> Facts:
    base = dict(
        source="toy",
        question="Q?",
        decision_label="Hold",
        decision_text="Hold. Confirm on a new panel.",
        reason="Not a confirmatory scale.",
        values={"incremental": 151035.46, "low": 115768.35, "high": 198149.23,
                "share": 0.4018, "cost": 0.4319, "n": 50.0, "after": -19182.38},
        dates={"start": "2015-02-16"},
    )
    base.update(over)
    return Facts(**base)


FAITHFUL = (
    "The test on 50 geos starting 2015-02-16 added about 151,035 incremental sales "
    "(interval 115,768 to 198,149). The five largest geos are 40% of pre-test sales.\n"
    "Recommendation: Hold. The plan was not confirmatory, so it is not a scale decision yet."
)


def test_faithful_summary_passes():
    res = check(FAITHFUL, toy())
    assert res.passed, res.problems
    assert res.decision_found == "Hold"


def test_hallucinated_number_fails():
    res = check(FAITHFUL.replace("151,035", "171,035"), toy())
    assert not res.passed
    assert "171,035" in res.unsupported


def test_invented_percent_fails():
    res = check(FAITHFUL + "\nROI improved by 12%.", toy())
    assert not res.passed and "12%" in res.unsupported


def test_wrong_decision_fails():
    res = check(FAITHFUL.replace("Recommendation: Hold.", "Recommendation: Scale now."), toy())
    assert not res.passed and res.decision_found == "Scale"


def test_missing_recommendation_line_fails():
    res = check(FAITHFUL.split("\nRecommendation")[0], toy())
    assert not res.passed


def test_wrong_date_fails():
    res = check(FAITHFUL.replace("2015-02-16", "2015-02-17"), toy())
    assert not res.passed and res.bad_dates == ["2015-02-17"]


@pytest.mark.parametrize("shown,ok", [
    ("151k", True),          # 151035 rounds to 151k
    ("151.0k", True),
    ("152k", False),         # off by more than half a step
    ("0.15 million", True),
    ("0.2 million", True),   # coarse but a valid rounding of 151035
    ("0.3 million", False),
    ("151,035", True),
    ("151,036", False),
    ("151,035.46", True),
    ("40%", True),           # share 0.4018 shown as a percent
    ("40.2%", True),
    ("40.5%", False),
    ("0.43", True),          # cost 0.4319
    ("0.44", False),
    ("43%", True),           # a fraction may be written as a percent
    ("-19,182", True),
    ("19,182", False),       # dropping the sign changes the number
    ("$50", True),
])
def test_rounding_and_percent_rules(shown, ok):
    res = check(f"Value {shown}.\nRecommendation: Hold.", toy())
    assert res.passed is ok, (shown, res.unsupported)


def test_extract_skips_dates_and_reads_suffixes():
    mentions, dates = extract_numbers("From 2015-02-16: 1.5 million and 12 percent, then 3K.")
    assert dates == ["2015-02-16"]
    assert [m.value for m in mentions] == [1.5e6, 12.0, 3000.0]
    assert [m.percent for m in mentions] == [False, True, False]


def test_decision_label_reads_first_word():
    assert decision_label("Hold. Confirm on a new panel.", False) == "Hold"
    assert decision_label("Scale the tactic.", True) == "Scale"
    with pytest.raises(ValueError):
        decision_label("Maybe later.", None)


def test_real_source_template_passes_its_own_check():
    facts = geo_holdout_facts(SOURCE)
    assert facts.decision_label == "Hold"
    assert check(template_memo(facts), facts).passed


def test_produce_falls_back_when_draft_fails():
    facts = geo_holdout_facts(SOURCE)
    bad = FakeLLM("Incremental sales were 250,000.\nRecommendation: Scale.")
    out = produce(facts, bad)
    assert out["used"] == "template (llm draft rejected)"
    assert out["memo"] == template_memo(facts)
    assert not out["check"]["passed"]


def test_produce_keeps_a_grounded_draft():
    facts = geo_holdout_facts(SOURCE)
    good = FakeLLM(
        "The geo test with 50 test geos added about 151k incremental sales "
        "(interval 115,768 to 198,149).\nRecommendation: Hold. Confirm on a new panel."
    )
    out = produce(facts, good)
    assert out["used"] == "llm"
    assert out["memo"] == good.text


def test_no_client_uses_template():
    facts = geo_holdout_facts(SOURCE)
    assert produce(facts, None)["used"] == "template"
