# Reading guide

Author: Sviatoslav Pokrovskii

These demos are public method work. They are not client work. No employer series, budget, or KPI is in the files.

Read in this order:

1. `geo-holdout-case` — a published geo assignment read under an analysis plan written after earlier reads of the same panel: a cumulative CUPED total with one vote per geo through the weeks after the test, a two-stage randomization interval that switches to a proportional-effect hypothesis when a slope test rejects a constant effect, the rule's decision power at known true costs, a blind check on 200 synthetic panels, and a Hold because that plan is not confirmatory.
2. `measurement-framework` — SQL metric dictionary, channel grain, a PyMC revenue model with trend, seasonality and AR(1) errors, a rolling-origin backtest against naive forecasts, residual autocorrelation, prior contraction on the marginal ROI, prior, control and misspecified-recovery checks as diagnostics, and a finding that no feasible lift test exists on the available units. The mix decision is the memo. A local forecast page is not that decision.
3. `measurement-reconciliation` — geo, an OLS revenue regression, and Markov attribution on one seeded synthetic dataset, with a rule for which number decides when they disagree and that rule's operating characteristics over 300 seeds. The OLS interval is not a posterior. The Markov model is a method demo, not a production attribution system.
4. `mmm-evaluation-bakeoff` — PyMC, Robyn, and Meridian on one public weekly series. The harness computes holdout MAPE with one formula and shows each ROI with its interval. Posterior intervals and model spread sit in separate columns. Each model reports a value-of-information channel on its own interval; the models are not voted and no lift test is recommended. The channels are not identified on this series, so the memo does not reallocate. Robyn and Meridian are participants, not a production stack.

Real-time bidding and a user-level experiment are not in this set.
