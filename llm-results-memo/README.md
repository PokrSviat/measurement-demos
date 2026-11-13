# LLM results memo

Author: Sviatoslav Pokrovskii

A language model drafts a short marketing-facing summary of the geo-holdout result. The draft is used only if every number in it and its recommendation match the source file. Otherwise the memo is a deterministic template built from the same facts.

This is a guard around a text generator. It does not make the model more accurate, and a draft that passes can still be badly worded or leave things out.

## What it does

1. `facts.py` reads `geo-holdout-case/results/case_result.json` and keeps a fixed list of fields: test and control geo counts, test cost, the CUPED total and its randomization interval, the after-test total and interval, DiD for reference, the cost per incremental sale at the lower bound, the top-five pre-test share, the window dates, and the decision. Only these values go into the prompt and only these values can support a number in the text.
2. `prompt.py` builds the messages: use only the given numbers, rounding allowed, dates as given, last line `Recommendation: <decision word>`.
3. `llm.py` has an OpenAI-compatible client (stdlib `urllib`, no SDK) configured by `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, and a `FakeLLM` that returns a scripted text.
4. `grounding.py` checks the draft:
   - Every number is extracted (thousands separators, `k`/`thousand`/`m`/`million`, signs, `%`/`percent`, a leading `$`). ISO dates are pulled out first and must equal a source date.
   - A number is supported when some fact value rounds to it at the precision the text shows: `151k` covers 151,035, `152k` does not. A percent is matched against a fraction times 100 or against the value itself, so `40%` covers a share of 0.4018. A dropped minus sign is a different number.
   - The `Recommendation:` line must exist and its first decision word (Hold, Scale, Stop) must equal the source decision. Later words on the line, such as "not a confirmatory scale" in the reason, do not decide.
5. `run.py` fails closed: no client, or any problem in the check, and the memo is the template. The template is held to the same check and the run stops if it fails.

Rounding tolerance is generous on purpose: `0.2 million` passes for 151,035 because it is a valid rounding. The check catches invented and altered numbers, not vague ones. Counts written as words ("five") are not checked.

## Results

- `results/memo.txt` and `results/run.json` — the run in this repository. No API key was set, so no model was called and the memo is the template.
- `results/fake_llm_examples.json` — two hand-written drafts passed through `FakeLLM` to show the check. They are not model output. The first is accepted. The second claims an 18% lift, a 3.2x ROI and a Scale, and it is rejected for both, so the template is used.

## Run

```bash
pip install -e "./llm-results-memo[dev]"
cd llm-results-memo && python -m pytest -q
results-memo                       # template unless LLM_API_KEY is set
LLM_API_KEY=... LLM_MODEL=... LLM_BASE_URL=... results-memo
python -m results_memo.example     # rewrites results/fake_llm_examples.json
```

Tests and CI use `FakeLLM` only and need no key or network.
