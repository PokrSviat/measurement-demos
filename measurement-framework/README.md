# Measurement framework and revenue model

Author: Sviatoslav Pokrovskii

A reproducible weekly revenue model on Meta Robyn's public simulated series `dt_simulated_weekly`. The repository fixes the channel taxonomy and grain before the fit, checks that grain in SQL, fits a Bayesian model in PyMC, backtests a revenue forecast on three holdouts, checks prior sensitivity, a control ablation and parameter recovery, and serves a local point forecast from the saved curves. The mix decision is the memo.

This is a method demo on a public simulated series. It is not a client model and not a production warehouse.

## Data

- `data/dt_simulated_weekly.csv`: Robyn demo series, 208 weeks from 2015-11-23 to 2019-11-11. Target `revenue`. Paid inputs `tv_S`, `ooh_S`, `print_S`, `facebook_S`, `search_S`, plus exposures `facebook_I` and `search_clicks_P`. Organic `newsletter`. Context `competitor_sales_B` and `events`.
- `data/dt_prophet_holidays_de.csv`: the DE slice of Robyn's `dt_prophet_holidays`. The public demo tags its holiday calendar as DE. That tag is not a market claim.

Source: [facebookexperimental/Robyn](https://github.com/facebookexperimental/Robyn).

## Grain

One paid regressor per channel. Facebook uses impressions and search uses clicks. Spend stays in the table as the ROI denominator. Two columns from the same channel are rejected. Two different channels are rejected as a split when their absolute correlation is 0.90 or higher.

## Result

tv's interval is above 1 on this fit: marginal ROI 3.73, 95% interval 2.36 to 5.51. Recovery coverage is 95% over 99 passed refits of 100, mean bias −0.05 under a Hill truth. That is an interval fact, not a mix move: this national weekly series does not identify the mix, so no channel is scaled or cut. print, facebook, search and ooh have intervals that contain 1. No feasible lift test exists on the available units: a national test cannot settle search, and no planning-proxy geo design reached 80% power. AR(1) rho is 0.00 (−0.17 to 0.18). Under a misspecified Hill and delayed-adstock truth, coverage is 95% to 99% per channel. `results/memo.txt` and `results/memo.pdf` have every number.

Two runs of `python -m measurement.fit` with the same seed wrote byte-identical `qa.json`, `baseline.json`, `curves.csv` and `holdout.json` on this machine.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
pip install -e ".[fit,app]"
python -m measurement.fit
streamlit run app/calculator.py
```

The mix decision is the memo. The page is a local forecast, not that decision. It reads `artifacts/qa.json` and the saved curves. When `artifacts/posterior_draws.npz` is present the number is the mean across draws; otherwise it is the curve forecast. The page still opens when `artifacts/prior_sensitivity.json` is absent. The opening budget is the spend on the last saved week.

The full fit writes `artifacts/qa.json` from four chains of 1000 draws with target_accept 0.95. A fit with r_hat at or above 1.01 does not publish curves, draws, or the reference history, and it does not mark a channel as earning. The decision ROI is the marginal return of the next ruble at observed spend, with a 95% interval from the posterior draws. The media coefficient prior is normal and centered at zero, so a channel can take a negative contribution. Controls are newsletter, `competitor_sales_B`, events, holidays, a linear trend and two yearly harmonics. Errors are AR(1) with rho uniform on (-0.9, 0.9), so autocorrelated weeks do not narrow the intervals; the memo reports rho and a Ljung-Box test on the residuals before and after the AR term. Each channel also reports prior contraction, one minus posterior over prior variance of the marginal ROI, so a channel whose interval only restates the prior is visible. An interval entirely above or below 1 is an interval fact, not a mix move. Average contribution over total spend is not the decision.

The forecast check is a rolling-origin backtest: three non-overlapping eight-week holdouts, each fit only on the weeks before it, against a training-mean and a lag-52 forecast. The 0.5 MAPE gate is a sanity check that catches a broken fit; the comparison that matters is whether the model beats lag-52 in each window. The backtest does not award a label.

The checks after the primary fit, then the memo:

```bash
python -m measurement.fit --wide       # every media prior twice as wide
python -m measurement.fit --ablation   # refit without competitor_sales_B
python -m measurement.fit --recovery   # 100 seeds of revenue from a Hill, delayed-adstock truth with AR(1) noise
python -m measurement.fit --memo
```

The recovery check plants a response the model does not assume, Hill saturation with delayed adstock, and reports per-channel coverage of the true marginal ROI with a binomial standard error and mean bias. Those numbers are diagnostics; they do not license a budget label. Prior contraction is computed on the marginal ROI from prior predictive draws, not on the media coefficient. The memo reports the prior move per channel, the ROI with and without `competitor_sales_B`, that coverage, and whether a feasible lift test exists. Value of information still ranks unsettled channels by spend share times interval width. A published test exists only if a national sizing is at most 52 weeks per arm or a planning-proxy geo cell reaches 80% power. The full fit script writes `artifacts/manifest.json`. The bake-off repository must use that manifest.

## CV line

Defined a tested SQL measurement framework and fit a Bayesian revenue model in PyMC with AR(1) errors on a public weekly series, treated recovery coverage as a diagnostic rather than a budget license, and showed that no feasible lift test exists on the available units. The mix decision is the memo.
