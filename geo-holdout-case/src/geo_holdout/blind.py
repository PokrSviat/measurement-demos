"""Blind check of the plan's two-stage rule on synthetic geo panels drawn from a seed in the plan.

Each panel draws geo sizes from a log-normal fitted to pre-test sales, so the
tail is as heavy as in the public panel, and draws the change into the window
with noise that grows with geo size as it does on the pseudo experiment. The
rule, the interval and the cap are the ones in the analysis plan.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geo_holdout.design import PseudoExperiment
from geo_holdout.estimators import picks_to_masks, stratified_picks
from geo_holdout.rule import decision_interval
from geo_holdout.validation import SHAPES, SLOPE_SEED_OFFSET, cap_call, daily_effect_for_cost, mc_se, planted_effect


@dataclass(frozen=True)
class PanelModel:
    log_pre_mean: float
    log_pre_sd: float
    intercept: float
    slope: float
    noise_log_scale: float
    noise_power: float
    noise_unit_sd: float
    n_geos: int
    stratum_size: int
    window_days: float


def calibrate(pseudo: PseudoExperiment, *, stratum_size: int) -> PanelModel:
    """Fit geo sizes, the drift of the change with size, and noise that scales as a power of size."""
    pre = pseudo.pre
    if np.any(pre <= 0):
        raise ValueError("Calibration needs positive pre-period sales.")
    log_pre = np.log(pre)
    slope, intercept = np.polyfit(pre, pseudo.delta, 1)
    residual = pseudo.delta - (intercept + slope * pre)
    magnitude = np.log(np.maximum(np.abs(residual), 1e-9))
    power, log_scale = np.polyfit(log_pre, magnitude, 1)
    standardized = residual / np.exp(log_scale + power * log_pre)
    return PanelModel(
        log_pre_mean=float(log_pre.mean()),
        log_pre_sd=float(log_pre.std(ddof=1)),
        intercept=float(intercept),
        slope=float(slope),
        noise_log_scale=float(log_scale),
        noise_power=float(power),
        noise_unit_sd=float(standardized.std(ddof=1)),
        n_geos=len(pre),
        stratum_size=stratum_size,
        window_days=float(np.median(pseudo.days)),
    )


def synthetic_panel(model: PanelModel, rng: np.random.Generator) -> PseudoExperiment:
    """One untreated panel: sizes, strata ranked on size, and the change into the window."""
    pre = np.exp(rng.normal(model.log_pre_mean, model.log_pre_sd, model.n_geos))
    noise_sd = model.noise_unit_sd * np.exp(model.noise_log_scale + model.noise_power * np.log(pre))
    delta = model.intercept + model.slope * pre + rng.normal(0.0, 1.0, model.n_geos) * noise_sd
    ranks = np.argsort(np.argsort(pre, kind="mergesort"), kind="mergesort")
    return PseudoExperiment(
        pre=pre,
        delta=delta,
        days=np.full(model.n_geos, model.window_days),
        stratum_ids=ranks // model.stratum_size,
        n_pre_days=0,
    )


@dataclass(frozen=True)
class BlindCell:
    shape: str
    true_cost: float
    n_panels: int
    scale_share: float
    correct_share: float
    proportional_share: float

    @property
    def scale_se(self) -> float:
        return mc_se(self.scale_share, self.n_panels)

    def as_dict(self) -> dict[str, str | float | int]:
        return {**self.__dict__, "scale_se": self.scale_se}


def blind_cell(
    model: PanelModel,
    *,
    shape: str,
    true_cost: float,
    treated_per_stratum: int,
    test_cost: float,
    max_cost_per_incremental: float,
    shape_alpha: float,
    n_slope_reassignments: int,
    n_panels: int,
    n_reassignments: int,
    seed: int,
) -> BlindCell:
    """Share of fresh panels in which the frozen rule scales, and in which that call is right."""
    if n_panels < 1:
        raise ValueError("The blind check needs at least one panel.")
    rng = np.random.default_rng(seed)
    scaled = correct = proportional = 0
    for index in range(n_panels):
        panel = synthetic_panel(model, rng)
        treated = picks_to_masks(stratified_picks(panel.stratum_ids, treated_per_stratum, 1, rng), model.n_geos)[0]
        daily = daily_effect_for_cost(test_cost, true_cost, float(panel.days[treated].sum()))
        effect = planted_effect(panel.pre, treated, shape, daily)
        injected = panel.delta.copy()
        injected[treated] += effect[treated]
        truth = float(np.sum(effect[treated] * panel.days[treated]))
        chosen = decision_interval(
            panel.pre,
            injected,
            panel.days,
            treated,
            panel.stratum_ids,
            treated_per_stratum=treated_per_stratum,
            alpha=shape_alpha,
            n_slope_reassignments=n_slope_reassignments,
            n_interval_reassignments=n_reassignments,
            slope_seed=seed + SLOPE_SEED_OFFSET + index,
            interval_seed=seed + 1 + index,
        )
        low = chosen.low
        proportional += int(chosen.shape == "proportional")
        fund, right = cap_call(low, truth, test_cost=test_cost, cap=max_cost_per_incremental)
        scaled += int(fund)
        correct += int(right)
    return BlindCell(
        shape=shape,
        true_cost=true_cost,
        n_panels=n_panels,
        scale_share=scaled / n_panels,
        correct_share=correct / n_panels,
        proportional_share=proportional / n_panels,
    )


def run_blind_check(
    pseudo: PseudoExperiment,
    *,
    stratum_size: int,
    treated_per_stratum: int,
    true_costs: list[float],
    test_cost: float,
    max_cost_per_incremental: float,
    shape_alpha: float,
    n_slope_reassignments: int,
    n_panels: int,
    n_reassignments: int,
    seed: int,
) -> tuple[PanelModel, list[BlindCell]]:
    model = calibrate(pseudo, stratum_size=stratum_size)
    cells = [
        blind_cell(
            model,
            shape=shape,
            true_cost=cost,
            treated_per_stratum=treated_per_stratum,
            test_cost=test_cost,
            max_cost_per_incremental=max_cost_per_incremental,
            shape_alpha=shape_alpha,
            n_slope_reassignments=n_slope_reassignments,
            n_panels=n_panels,
            n_reassignments=n_reassignments,
            seed=seed + 1000 * position,
        )
        for position, (shape, cost) in enumerate((shape, cost) for shape in SHAPES for cost in true_costs)
    ]
    return model, cells
