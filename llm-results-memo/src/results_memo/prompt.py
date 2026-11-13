"""Prompt for the marketing-facing summary."""
from __future__ import annotations

import json

from results_memo.facts import Facts

SYSTEM = (
    "You write short result summaries for a marketing audience. "
    "Use only the numbers in FACTS. You may round them, but do not compute new numbers, "
    "do not add counts, years, or percentages that are not in FACTS, and spell out small "
    "counts in words only if they are in FACTS. Shares in FACTS are fractions; you may write them as percents. "
    "Dates must be written exactly as given. "
    "End with one line that starts with 'Recommendation: ' followed by the DECISION word, "
    "then one sentence that explains it using the REASON."
)


def build_messages(facts: Facts, max_words: int = 150) -> list[dict[str, str]]:
    payload = {
        "QUESTION": facts.question,
        "DECISION": facts.decision_label,
        "DECISION_TEXT": facts.decision_text,
        "REASON": facts.reason,
        "FACTS": facts.values,
        "DATES": facts.dates,
    }
    user = (
        f"Write at most {max_words} words for a marketing lead.\n"
        "Say what was tested, what the decision interval shows, and what to do next.\n\n"
        + json.dumps(payload, indent=2)
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
