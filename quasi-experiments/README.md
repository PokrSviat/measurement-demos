# Quasi-experiments

Author: Sviatoslav Pokrovskii

The data are simulated with known effects.

## Method

Interrupted time series. Weekly sales, 156 weeks, baseline 1000, trend 0.5, AR(1) correlation 0.6, innovation standard deviation 25. At week 104 the level shifts by 60 and the slope by 1.5. The fit is OLS with that level, that slope change, the trend, and Fourier seasonality. Intervals use Newey-West HAC. The cumulative effect adds the level over the post weeks to the slope change over those weeks. Placebo dates are the same fit on the pre-period only. The share is how often a fake date moves at least as much as the real date.

Sharp regression discontinuity. 20000 customers, cutoff 50, noise standard deviation 8, true jump 2 in next-year spend. One triangular-kernel local linear, HC1, with its own slope on each side. The reported fit uses bandwidth 5. The same fit is stored at bandwidths 2, 5, 10, and 25. Tenure has truth 0. Placebo cutoffs stay on the same side of the real cutoff. The density check is a local linear on log bin counts on each side; the estimate is the difference of the two intercepts. A second series sets the bunching share to 0.3 and moves that share of customers from just below the cutoff to just above it.

## Result

From `results/its.json`. Level truth 60, estimate 72.18 (51.89 to 92.47). Slope truth 1.5, estimate 1.17 (0.66 to 1.68). Cumulative truth 5109, estimate 5299.26 (4363.90 to 6234.62). Placebo share 0.00.

From `results/rdd.json`. Tau truth 2, bandwidth 5, estimate 2.416 (1.619 to 3.212), 3729 on the left and 3173 on the right. Bandwidth 2 gives 2.153, bandwidth 10 gives 2.293, bandwidth 25 gives 2.295. Tenure truth 0, estimate -0.024 (-0.064 to 0.017). Density jump 0.011 (se 0.033, p 0.752). With bunching share 0.3 the density jump is 1.109 (se 0.098, p 1.641e-29). Placebo cutoffs containing 0: 1.00.

`results/its.png` is actual sales against the counterfactual, with week 104 marked. `results/rdd.png` is binned means with both local linear fits.

## What this is not

There is no real panel. The density check is a simplified McCrary test, not rddensity. The intervals are conventional Newey-West and HC1 intervals, not bias-corrected ones, unless the `rd` extra is used. This run does not use it.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
python -m quasi.run
```

That writes `results/its.json`, `results/rdd.json`, `results/memo.txt`, `results/its.png`, and `results/rdd.png`.

## CV line

Fit an interrupted time series and a sharp regression discontinuity on simulated series with known effects, with Newey-West and conventional local-linear intervals, placebo dates, and a simplified density check.
