"""Number-grounding and decision check.

Every number in the draft has to match a fact value after rounding to the
precision the draft shows. Percents match a fraction times 100. ISO dates
must equal a source date. The 'Recommendation:' line must start with the source
decision word (the first decision word on that line decides).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from results_memo.facts import DECISION_WORDS, Facts

DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
NUM_RE = re.compile(
    r"(?<![\w.])(?P<sign>[-\u2212])?\$?(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"\s?(?P<suffix>%|percent\b|k\b|K\b|thousand\b|m\b|M\b|million\b)?"
)
SCALE = {"k": 1e3, "K": 1e3, "thousand": 1e3, "m": 1e6, "M": 1e6, "million": 1e6}


@dataclass(frozen=True)
class Mention:
    text: str
    value: float
    step: float  # half of this is the rounding tolerance
    percent: bool


@dataclass
class CheckResult:
    passed: bool
    unsupported: list[str] = field(default_factory=list)
    bad_dates: list[str] = field(default_factory=list)
    decision_found: str | None = None
    decision_ok: bool = False
    problems: list[str] = field(default_factory=list)


def extract_numbers(text: str) -> tuple[list[Mention], list[str]]:
    dates = DATE_RE.findall(text)
    stripped = DATE_RE.sub(" ", text)
    out: list[Mention] = []
    for m in NUM_RE.finditer(stripped):
        raw = m.group("num")
        digits = raw.replace(",", "")
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        value = float(digits)
        step = 10.0 ** (-decimals)
        suffix = m.group("suffix")
        percent = suffix in ("%", "percent")
        if suffix in SCALE:
            value *= SCALE[suffix]
            step *= SCALE[suffix]
        if m.group("sign"):
            value = -value
        out.append(Mention(m.group(0).strip(), value, step, percent))
    return out, dates


def _close(shown: float, step: float, source: float) -> bool:
    # Allowed: any rounding of the source to the shown precision (half step, plus float slack).
    return abs(shown - source) <= step / 2 + 1e-9 * max(1.0, abs(source))


def supported(m: Mention, facts: Facts) -> bool:
    for v in facts.values.values():
        if m.percent:
            if _close(m.value, m.step, v * 100) or _close(m.value, m.step, v):
                return True
        elif _close(m.value, m.step, v):
            return True
    return False


def recommendation_line(text: str) -> str | None:
    for line in text.splitlines():
        if line.strip().lower().startswith("recommendation:"):
            return line.strip()[len("recommendation:"):].strip()
    return None


def check(text: str, facts: Facts) -> CheckResult:
    mentions, dates = extract_numbers(text)
    res = CheckResult(passed=False)
    res.unsupported = [m.text for m in mentions if not supported(m, facts)]
    res.bad_dates = [d for d in dates if d not in facts.dates.values()]
    line = recommendation_line(text)
    if line is None:
        res.problems.append("no 'Recommendation:' line")
    else:
        hits = sorted((m.start(), w) for w in DECISION_WORDS
                      for m in re.finditer(rf"\b{w}\b", line, re.IGNORECASE))
        # The decision is the first decision word on the line; later words may appear in the reason.
        res.decision_found = hits[0][1] if hits else None
        res.decision_ok = res.decision_found == facts.decision_label
        if not res.decision_ok:
            res.problems.append(f"recommendation says {res.decision_found!r}, source decision is {facts.decision_label!r}")
    if res.unsupported:
        res.problems.append(f"numbers not in source: {res.unsupported}")
    if res.bad_dates:
        res.problems.append(f"dates not in source: {res.bad_dates}")
    res.passed = not res.problems
    return res
