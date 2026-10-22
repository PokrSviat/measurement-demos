# Reconciliation of a geo experiment, an MMM, and Markov attribution

Author: Sviatoslav Pokrovskii

One seeded synthetic dataset. A known tactic effect is estimated three ways: a geo difference-in-differences, an OLS of weekly revenue on tactic spend, and a Markov removal effect. The comparison uses one estimand, incremental conversions, on that generator. The OLS interval plugs in the residual variance. It is not a posterior.

The treated geos are a seeded draw of four out of sixteen, taken before the noise. The experiment interval is the exact randomization interval of that assignment. Geo-week noise has standard deviation 20, the same size as the planted weekly increment, so the experiment interval is wide enough for the planning cap to refuse. The budget rule uses the experiment lower bound and the tactic spend on treated geo-weeks. Scale only when that bound is positive and spend divided by the bound is within the planning cap of 0.10. Overlap does not change the cost test. A point with no width does not count as overlap. The generator places the tactic on 2% of baseline journeys as well as on the incremental journeys, so the Markov removal effect is larger than the planted effect and is a point. The run does not say the intervals agree. The experiment still drives the go or no-go, and the sentence states the cost at the lower bound. The memo also drops the tactic from the path log: Markov moves to zero, and the experiment still decides. That read does not replace the recommendation. A Markov removal share is a different estimand and the comparison rejects it until it is translated with the formula in `methods.py`: `(conversion probability with the channel - probability without it) * number of journeys`. Removing the channel sends that flow to a non-conversion, rather than handing the journey to another channel.

One seed shows one call. The memo also reports the operating characteristics of the rule: for true costs from 0.03 to 0.15 per incremental conversion, 300 seeds each, the share of runs that say scale and the coverage of the experiment interval, with Monte Carlo standard errors. At a true cost of 0.03 the rule says scale in 86% of runs; at 0.05 in 29%; at the cap of 0.10 in 3%. The interval covers the planted effect in 93% of runs at every point. The rule is conservative by design: it scales only when the lower bound clears the cap.

This dataset is generated here. It is not the geo-holdout panel and not the weekly revenue series. The Markov model is a method demo. It is not a production attribution system.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
python -m reconcile.run
```

`results/reconciliation.json`, `results/memo.txt`, and `results/memo.pdf` are the files to check before publishing. The memo text is also stored in the JSON.

## CV line

Reconciled a geo experiment, an OLS revenue regression, and a Markov attribution model on one seeded synthetic dataset, with an explicit rule for which estimate drives the decision when they disagree. The OLS interval is not a posterior.
