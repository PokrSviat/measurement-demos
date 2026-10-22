"""One-page decision memo rendered from the case result."""

from __future__ import annotations

from dataclasses import dataclass

from budget_rule.memo_pdf import write_memo_pdf
from geo_holdout.blind import BlindCell
from geo_holdout.decision import BudgetDecision
from geo_holdout.estimators import EffectEstimate
from geo_holdout.tail import SlopeTest, StratumEffect
from geo_holdout.validation import ValidationCell

CHANGE_LOG = "\n".join(
    [
        "What changed after the first run",
        "Commit 2af3759 decided on the unweighted CUPED point estimate against the cap, with a normal interval, and said scale.",
        "Commit 436088c moved the decision to a pretest-weighted CUPED total with a stratified randomization interval and its lower bound, added a power gate from 80 draws and a cooldown lower-bound gate, and said do not scale.",
        "Commit 3831ec4 returned to one vote per geo, made power a design check, and made the cooldown test refuse only on sales pulled forward. It said do not scale.",
        "Version 4 of the analysis plan makes the cumulative total through the weeks after the test the estimand, so pull-forward lowers the bound that decides and the separate cooldown test is gone. Power against zero is replaced by decision power at known true costs.",
        "Version 5 keeps that estimand and the cap and changes the interval: a stratified slope test chooses between hypotheses spread equally per location-day and hypotheses proportional to pre-period sales, because the version 4 read showed an effect that grows with geo size and a rule that could not fund a tactic under the cap with that shape. Version 5 was written after that read, so it is a plan fixed after exploration on this panel, not one fixed before any data were seen; the decision power and blind check run the whole two-stage rule.",
        "The plan file was committed before this run, its hash is in the case result, and the blind check draws its panels from a seed fixed in that plan. The cap did not change in any version. The recommendation speech was later tightened to Hold on this post-exploration plan. The interval, the cap and the plan hash did not change.",
    ]
)

POWER_FLOOR = 0.8


@dataclass(frozen=True)
class Window:
    start: str
    end: str
    estimate: EffectEstimate
    interval: tuple[float, float]


def _interval_text(bounds: tuple[float, float]) -> str:
    return f"{bounds[0]:.0f} to {bounds[1]:.0f}"


def _window_line(label: str, window: Window) -> str:
    return (
        f"- {label}, {window.start} to {window.end}: CUPED {window.estimate.incremental:.0f} "
        f"(randomization interval {_interval_text(window.interval)})."
    )


def _cost_label(true_cost: float | None) -> str:
    return "no true effect" if true_cost is None else f"true cost {true_cost:.2f}"


def _call_text(true_cost: float | None, correct_share: float, cap: float) -> str:
    if true_cost is not None and abs(true_cost - cap) < 1e-12:
        return "at the cap either call is defensible"
    return f"correct call in {correct_share:.0%}"


def _validation_line(cell: ValidationCell, cap: float) -> str:
    return (
        f"- {cell.shape}, {_cost_label(cell.true_cost)}: scale in {cell.scale_share:.0%} "
        f"(Monte Carlo standard error {cell.scale_se:.1%}), {_call_text(cell.true_cost, cell.correct_share, cap)}, "
        f"coverage {cell.coverage:.0%}, proportional interval chosen in {cell.proportional_share:.0%}, "
        f"mean bias {cell.mean_bias:.0f}, weighted-total mean bias {cell.weighted_mean_bias:.0f}."
    )


def _blind_line(cell: BlindCell, cap: float) -> str:
    return (
        f"- {cell.shape}, {_cost_label(cell.true_cost)}: scale in {cell.scale_share:.0%} "
        f"(Monte Carlo standard error {cell.scale_se:.1%}), {_call_text(cell.true_cost, cell.correct_share, cap)}, "
        f"proportional interval chosen in {cell.proportional_share:.0%}."
    )


def _shape_text(shape: str) -> str:
    return "proportional to pre-period sales" if shape == "proportional" else "equal per treated location-day"


def _whole(value: float) -> str:
    """Rounded to a whole number without a negative zero."""
    return f"{round(value) + 0:.0f}"


def _p_text(p_value: float) -> str:
    return "below 0.001" if p_value < 0.001 else f"{p_value:.3f}"


def _strata_lines(effects: list[StratumEffect]) -> list[str]:
    count = effects[0].n_assignments if effects else 0
    lines = [
        "Daily effect per geo-day by pretest-volume stratum, smallest first, "
        f"exact interval over the {count} assignments inside each stratum:"
    ]
    lines.extend(
        f"- stratum {item.stratum + 1} (median {item.median_pre:.0f}): {_whole(item.daily_effect)} "
        f"({_whole(item.low)} to {_whole(item.high)})"
        for item in effects
    )
    return lines


def _overturn(decision: BudgetDecision) -> str:
    if decision.fund:
        return "A cumulative randomization interval that includes zero, or a cost at its lower bound above the cap. "
    if decision.clears_cap:
        return (
            "A confirmatory read on a new panel under a plan fixed before that assignment. "
            "The interval and the cap on this panel did not change. "
        )
    return "A cumulative randomization lower bound high enough that the cost at it is at or under the cap. "


def _slope_sentence(test: SlopeTest) -> str:
    return (
        f"The effect per geo-day changes by {test.slope:.0f} per unit of log pre-test sales "
        f"(randomization p-value {_p_text(test.p_value)} against a constant additive effect, "
        f"{test.n_reassignments} stratified reassignments)."
    )


def shape_warning(
    slope: SlopeTest,
    validation: list[ValidationCell],
    blind: list[BlindCell],
    cap: float,
    *,
    alpha: float = 0.05,
) -> str:
    """When the effect grows with geo size and the rule still misses tactics under the cap with that shape."""
    if slope.p_value >= alpha or slope.slope <= 0:
        return ""
    below = sorted(
        {cell.true_cost for cell in validation if cell.shape == "proportional" and cell.true_cost and cell.true_cost < cap}
    )
    if not below:
        return ""
    worst = below[-1]
    on_pretest = next(cell for cell in validation if cell.shape == "proportional" and cell.true_cost == worst)
    if on_pretest.scale_share >= POWER_FLOOR:
        return ""
    on_blind = next((cell for cell in blind if cell.shape == "proportional" and cell.true_cost == worst), None)
    blind_text = "" if on_blind is None else f" and {on_blind.scale_share:.0%} of blind panels"
    return (
        " The effect grows with geo size on this panel. Under a proportional effect the rule funded a tactic "
        f"with true cost {worst:.2f}, under the cap, in {on_pretest.scale_share:.0%} of pre-test draws{blind_text}. "
        "So this do-not-scale does not rule out a tactic within the cap."
    )


def memo_text(
    *,
    question: str,
    estimand: str,
    plan_version: int,
    plan_sha256: str,
    n_treated: int,
    stratum_size: int,
    treated_per_stratum: int,
    cumulative: Window,
    decision_shape: str,
    shape_alpha: float,
    additive_interval: tuple[float, float],
    test_window: Window,
    after_test: Window,
    did: EffectEstimate,
    control_correlation: float,
    missing_treated_location_days: int,
    weighted_total: float,
    median_pre: float,
    max_pre: float,
    top5_pretest_share: float,
    without_largest: EffectEstimate,
    without_largest_interval: tuple[float, float],
    strata: list[StratumEffect],
    slope: SlopeTest,
    validation: list[ValidationCell],
    validation_reassignments: int,
    pseudo_window_days: int,
    pseudo_pre_days: int,
    blind: list[BlindCell],
    blind_reassignments: int,
    decision: BudgetDecision,
    test_cost: float,
    max_cost: float,
    assumption_note: str,
) -> str:
    cuped = cumulative.estimate
    cost_text = (
        "not defined, because the cumulative interval is not strictly positive"
        if decision.cost_per_incremental is None
        else f"{decision.cost_per_incremental:.2f}"
    )
    ratio = weighted_total / cuped.incremental if cuped.incremental else float("nan")
    n_draws = validation[0].n_draws if validation else 0
    n_panels = blind[0].n_panels if blind else 0
    warning = "" if decision.clears_cap else shape_warning(slope, validation, blind, max_cost)
    return "\n".join(
        [
            question,
            "",
            "Estimand",
            estimand,
            "",
            "Analysis plan",
            f"Version {plan_version}, sha256 {plan_sha256[:12]}, committed before this run. "
            "It was written after earlier reads of this panel had been seen, so it is fixed after exploration, "
            "not before any data were seen.",
            "",
            "Read of the public panel",
            f"The test group is {n_treated} geos, assigned {treated_per_stratum} per stratum of {stratum_size} geos "
            f"ranked by pre-test sales. The cumulative window runs from {cumulative.start} to {cumulative.end} and "
            f"uses {cuped.treated_location_days} observed treated location-days. "
            f"{missing_treated_location_days} treated location-days inside it are missing and are not imputed.",
            f"CUPED cumulative incremental sales: {cuped.incremental:.0f}.",
            f"Randomization interval: {_interval_text(cumulative.interval)}, with each hypothesized total spread "
            f"{_shape_text(decision_shape)}. That interval reassigns the treated slots inside the strata, refits "
            "theta on every redraw, and it decides. The plan picks the spread with the slope test below at alpha "
            f"{shape_alpha:.2f}: p-value {_p_text(slope.p_value)}, so the "
            f"{'proportional' if decision_shape == 'proportional' else 'additive'} interval decides.",
            f"Interval with an equal spread per location-day, for reading: {_interval_text(additive_interval)}.",
            f"Normal approximation, for reference: {cuped.incremental_low:.0f} to {cuped.incremental_high:.0f}.",
            f"DiD, for reference: {did.incremental:.0f} "
            f"(normal approximation {did.incremental_low:.0f} to {did.incremental_high:.0f}).",
            f"On control geos the pre-period mean and the change into the window have correlation "
            f"{control_correlation:.3f}, so the daily standard error falls from {did.se_daily:.0f} to {cuped.se_daily:.0f}.",
            "",
            "The two parts of that window, for reading only:",
            _window_line("test window", test_window),
            _window_line("after the test", after_test),
            "",
            "Pretest-weighted CUPED total",
            f"Descriptive only: {weighted_total:.0f}, {ratio:.1f} times the decision total. "
            "Under a homogeneous additive effect both totals estimate the same number. "
            "When the effect grows with geo size the weighted total counts the large geos twice. "
            f"Median pre-test daily sales are {median_pre:.0f}, the largest geo is {max_pre:.0f}, "
            f"and the five largest geos are {top5_pretest_share:.0%} of pre-test sales.",
            "",
            "Tail, for reading only",
            "Without the five test geos that have the highest advertising cost, "
            f"CUPED cumulative incremental sales: {without_largest.incremental:.0f} "
            f"(randomization interval {_interval_text(without_largest_interval)}).",
            *_strata_lines(strata),
            _slope_sentence(slope),
            "",
            "Decision power on the pre-test",
            "It does not decide. "
            f"Each of {n_draws} draws per row assigns test geos inside strata on a {pseudo_window_days}-day pseudo window, "
            "plants an effect whose true cost per incremental sale is fixed, and runs the same two-stage rule, "
            f"the slope test and then the chosen interval, with {validation_reassignments} reassignments and the same cap. "
            "Scale is right below the cap and wrong above it. The pseudo window has no weeks after it, "
            f"so this covers the test window only, and {pseudo_pre_days} pseudo pre-period days make the covariate "
            "noisier than in the read.",
            *(_validation_line(cell, max_cost) for cell in validation),
            "",
            "Blind check",
            f"{n_panels} synthetic panels per row, drawn after the plan was fixed from a seed in it, "
            "with log-normal geo sizes fitted to pre-test sales and noise that grows with geo size. "
            "The sizes and noise come from the same pre-test, so this checks the rule under that family of panels, "
            f"not under an unknown one. The two-stage rule runs with {blind_reassignments} reassignments.",
            *(_blind_line(cell, max_cost) for cell in blind),
            "",
            "Recommendation",
            f"{decision.recommendation} {decision.reason}",
            f"Cost per incremental sale at the decision bound: {cost_text}.",
            f"Observed test cost {test_cost:.0f}. Cap {max_cost:.1f}.",
            "The decision uses the randomization lower bound of the cumulative total, not the point estimate "
            f"and not the normal approximation. {assumption_note}{warning}",
            "",
            CHANGE_LOG,
            "",
            "What would overturn this",
            _overturn(decision) + "The numbers are from this public panel and the assumptions in the analysis plan.",
        ]
    )


