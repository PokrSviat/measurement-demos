# Geo holdout decision memo

Author: Sviatoslav Pokrovskii

A read of a public simulated geo panel. The question is whether to scale the tested tactic. The estimand is cumulative incremental sales from the first test day through the last day after the test, with one vote per geo, so sales pulled forward from the weeks after the test lower the number that decides. Version 5 of `config/analysis_plan.yaml` was written after earlier reads of this panel, including the slope test, had been seen. It is a plan fixed after exploration, not before any data were seen. A stratified slope test chooses the interval shape: if the effect per geo-day grows with log pre-test sales, hypothesized totals are spread in proportion to pre-period sales; otherwise they are spread equally per location-day. The statistic is the same unweighted CUPED total either way. The plan sha256 is written to `results/case_result.json`.

This repository is a method demo on a public simulated panel. It is not a client study.

## Data

`data/sales_and_cost.csv` and `data/geo_assignment.csv` are the `salesandcost` and `geoassignment` frames from [GeoexperimentsResearch](https://github.com/google/GeoexperimentsResearch). The package describes the series as simulated. Columns in the panel are `location`, `date`, `Y`, and `cost`. The assignment file maps each geo to a group. One hundred geos, daily, from 2015-01-05 through 2015-04-07.

Group 2 is the test group: 50 geos with advertising cost from 2015-02-16 through 2015-03-15. Group 1 is the control group, with no advertising cost in that window. The assignment file is accepted only when each pretest-sales stratum of 10 geos contains 5 treated geos.

## Method

- Estimand: cumulative incremental sales from 2015-02-16 through 2015-04-07. The CUPED difference gives each geo one vote and is scaled by the observed treated location-days. The test window and the weeks after it are reported separately for reading. DiD is reported and does not decide.
- Interval: the randomization interval reassigns the treated slots inside the strata. Version 5 inverts the shape the slope test selects. The additive interval is reported and does not decide when the slope test rejects a constant effect.
- Decision: the cap of 0.5 requires incremental sales of at least twice the test spend. A confirmatory plan would scale only if observed test cost divided by the cumulative randomization lower bound is within that cap. Version 5 has a history field, so a clear cap is a Hold, not a scale. The cap is a planning assumption for this case.
- Decision power and the blind check run the whole two-stage rule, the slope test and then the chosen interval, on the same true-cost grid as version 4.
- Tail checks, for reading only: CUPED without the five highest-cost test geos; the daily effect per stratum with an exact interval over the 252 assignments inside each stratum; and the slope test itself.
- `config/search_lift.yaml` and `results/search_lift_design.json` size a planning-proxy geo lift for the weekly-series search channel on these control geos. That file does not decide the geo case and no cell is a test to run.

## Result

CUPED cumulative incremental sales 151035. The slope test rejects a constant effect (p below 0.001), so the proportional interval decides: 115768 to 198149. Cost at that lower bound is 0.43 against a cap of 0.5. The recommendation is Hold. Confirm on a new panel: version 5 was written after the slope on this panel. The additive interval, for reading, is 88404 to 226169, which would have stayed above the cap.

Under a proportional effect the two-stage rule scales a tactic with true cost 0.40 in 87% of pre-test draws and 82% of blind panels. Under an additive effect those shares are 87% and 82%. `results/memo.txt` and `results/memo.pdf` have the full grid and the change log.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
python -m geo_holdout.run_case
python -m geo_holdout.lift_design
```

Outputs land in `results/case_result.json`, `results/validation.json`, `results/blind_check.json`, `results/search_lift_design.json`, `results/memo.txt`, `results/memo.pdf`, and `results/path.png`. The case run takes about seven minutes on a laptop. Re-run before publishing and check that the memo numbers match the JSON.

## CV line

Estimated a published geo assignment on a public geo panel with CUPED and a two-stage randomization interval that switches to a proportional-effect hypothesis when a slope test rejects a constant effect, under an analysis plan written after earlier reads of the same panel, and held the scale call until a confirmatory panel.
