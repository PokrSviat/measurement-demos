# Measurement demos

Author: Sviatoslav Pokrovskii

How to make a budget decision when the data cannot identify the effect — and when to refuse to change the budget.

The answers in the files are Hold, Do not scale, and Do not reallocate. The geo plan was written after exploring that panel, the reconciliation lower bound misses its cap, and the weekly series does not identify channel returns. The numbers and the memos are in [DECISIONS.md](DECISIONS.md).

These demos are public method work. No employer series, budget, or KPI is in the files.

| Module | Question | Decision | Data | Key file |
| --- | --- | --- | --- | --- |
| `geo-holdout-case` | Scale the tested tactic? | Hold | simulated Google geo panel | [results/memo.txt](geo-holdout-case/results/memo.txt) |
| `measurement-reconciliation` | Which number decides when geo, OLS, and Markov disagree? | Do not scale | synthetic generator in this repo | [results/memo.txt](measurement-reconciliation/results/memo.txt) |
| `measurement-framework` | Which paid channel earns back the next ruble? | No channel change; no feasible lift test | public simulated Robyn weekly series | [results/memo.txt](measurement-framework/results/memo.txt) |
| `mmm-evaluation-bakeoff` | Should three models move the mix? | Do not reallocate | same public simulated weekly series | [results/memo.txt](mmm-evaluation-bakeoff/results/memo.txt) |
| `llm-results-memo` | Can a draft keep the geo decision and its numbers? | Hold, or the template if the draft fails | the geo result file; tests use a scripted fake model | [results/memo.txt](llm-results-memo/results/memo.txt) |
| `budget-rule` | Does cost at the lower bound clear the cap? | No budget of its own; the demos call it | no series | [src/budget_rule/cost.py](budget-rule/src/budget_rule/cost.py) |

Real-time bidding and a user-level experiment are not in this set.

## How to run

```bash
bash scripts/test.sh
```

That command installs the pinned test dependencies and runs pytest in `budget-rule`, `geo-holdout-case`, `measurement-framework`, `measurement-reconciliation`, `mmm-evaluation-bakeoff`, and `llm-results-memo`. It does not run `python -m geo_holdout.run_case`, `python -m measurement.fit`, or `bash scripts/refresh_bakeoff.sh`. Those refit the geo case, the PyMC model, and Robyn and Meridian.
