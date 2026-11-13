"""Draft a memo with an LLM, check it, and fall back to the template when the check fails."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from results_memo.facts import Facts, geo_holdout_facts
from results_memo.grounding import check
from results_memo.llm import LLMClient, OpenAICompatibleClient
from results_memo.prompt import build_messages
from results_memo.template import template_memo


def produce(facts: Facts, client: LLMClient | None) -> dict:
    template = template_memo(facts)
    template_check = check(template, facts)
    if not template_check.passed:  # the template is held to the same rule
        raise RuntimeError(f"template failed its own check: {template_check.problems}")
    if client is None:
        return {"source": facts.source, "used": "template", "client": None, "draft": None,
                "check": None, "memo": template}
    draft = client.complete(build_messages(facts))
    res = check(draft, facts)
    return {
        "source": facts.source,
        "used": "llm" if res.passed else "template (llm draft rejected)",
        "client": client.name,
        "draft": draft,
        "check": asdict(res),
        "memo": draft if res.passed else template,
    }


def main(argv: list[str] | None = None) -> None:
    here = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=str(here.parent / "geo-holdout-case/results/case_result.json"))
    p.add_argument("--out", default=str(here / "results"))
    a = p.parse_args(argv)
    facts = geo_holdout_facts(a.source)
    try:
        facts = replace(facts, source=str(Path(a.source).resolve().relative_to(here.parent)))
    except ValueError:
        pass
    out = produce(facts, OpenAICompatibleClient.from_env())
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "memo.txt").write_text(out["memo"] + "\n")
    (out_dir / "run.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"used: {out['used']}")


if __name__ == "__main__":
    main()
