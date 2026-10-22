# PyMC, Robyn, and Meridian on one weekly series

Author: Sviatoslav Pokrovskii

The published comparison is PyMC, Robyn, and Meridian on one manifest. The series is Meta Robyn's public `dt_simulated_weekly`. PyMC numbers come from the measurement-framework export. Robyn and Meridian are fit here on that same file, window, target, regressors, and holdout. Meridian's own demo dataset is not part of the score.

Robyn and Meridian are bake-off participants. Neither is a production stack in this repository.

## Compare

The question is whether holdout error or channel ROI from three models should move the mix. Each export is a JSON file with `model`, `manifest`, `estimand`, the eight holdout weeks as `holdout_actual` and `holdout_pred`, `channel_roi` with `channel_roi_low` and `channel_roi_high`, a note on what the interval is, `interval_kind` (`posterior` or `model_spread`), and, for Meridian, `r_hat_max`. The harness computes every holdout MAPE from those weeks with one formula; a stored MAPE is not read. The PyMC and Meridian intervals are posterior percentiles. The Robyn interval is the spread across the first Pareto front, which is not a posterior, and the table puts the two kinds in separate columns. PyMC and Meridian also export prior contraction per channel. The estimand on every included export is `marginal_roi_next_ruble`: one extra ruble on a training week with spend, at that week's media per ruble, summed over that week and its adstock horizon only when the horizon sits inside training, then averaged with spend as the weight. Robyn freezes the Hill inflexion on the unbumped series. Meridian does not use a one-percent bump of total spend. The comparison refuses to build a table when the manifest hash, target, dates, regressors, holdout, or estimand differ. An export that is not that estimand stays out of the ROI column.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
python -m bakeoff.compare
```

That command writes `artifacts/comparison.csv`, `results/memo.txt`, and `results/memo.pdf` from the PyMC, Robyn, and Meridian exports. A Meridian row needs `r_hat_max`. An export without the holdout path or the ROI interval is rejected, and so is one that holds out different weeks.

The memo does not reallocate. It states whether the ROI order agrees, which channels have an interval containing 1 in at least one model, and the prior sensitivity. Each model reports a value-of-information channel with the shared rule on its own interval. The models are not voted: posterior intervals and model spread are not the same kind. No lift test is recommended. Spend shares come from the PyMC export.

## Fits you run

`bash ../scripts/refresh_bakeoff.sh` runs every step below in order. Versions are in `meridian/requirements.lock` and `robyn/versions.txt`.

PyMC: in `measurement-framework`, `python -m measurement.fit`, the checks, and `python -m measurement.fit --memo`. The memo step writes `artifacts/pymc_export.json` there; copy it and `artifacts/manifest.json` into this repository's `artifacts/`.

Robyn, from this directory, after `install.packages` / `remotes` for Robyn and Nevergrad:

```bash
Rscript robyn/run_robyn.R ../measurement-framework/data/dt_simulated_weekly.csv ../measurement-framework/data/dt_prophet_holidays_de.csv artifacts/manifest.json
```

The R script writes the next-ruble marginal ROI and the 8-week holdout path for the cluster winner with the lowest test NRMSE, and the 2.5th to 97.5th percentile of the same ROI across the first Pareto front. The ROI is the frozen-hill finite difference in `robyn/next_ruble.R`, not a Hill inflexion recomputed from the bumped series. The holiday file is the DE slice because that is the public demo's calendar tag. The script adds `country` and `year` before the fit.

Meridian:

```bash
pip install -e ".[meridian]"
python meridian/run_meridian.py --weekly ../measurement-framework/data/dt_simulated_weekly.csv --manifest artifacts/manifest.json
python meridian/run_meridian.py --weekly ../measurement-framework/data/dt_simulated_weekly.csv --manifest artifacts/manifest.json --sensitivity
python meridian/run_meridian.py --weekly ../measurement-framework/data/dt_simulated_weekly.csv --manifest artifacts/manifest.json --seed-check /tmp/meridian-seeds
```

Both commands write under `artifacts/` of this repository, whichever directory you start them from. The first writes `meridian_export.json` with the holdout path, channel ROI with posterior percentiles, and `r_hat_max`. The ROI prior is Meridian's LogNormal (`loc` 0.2, scale 0.9). The sensitivity run widens the facebook scale from 0.9 to 1.8, writes `meridian_sensitivity.json`, and stores the facebook relative ROI change on the main export. Run the main fit first. Prior contraction is computed on log `roi_m` from 1000 prior draws. `--seed` overrides the manifest seed, and `--seed-check` refits on three consecutive seeds and writes `meridian_seed_check.json` with the point ROI spread.

## CV line

Compared PyMC, Robyn, and Meridian on one public weekly series with one holdout MAPE formula and ROI intervals on one estimand, the marginal ROI of the next ruble. The memo concludes the channels are not identified on this series, does not move budget, and does not recommend a lift test.
