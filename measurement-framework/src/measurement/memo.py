"""One-page mix recommendation from the fitted revenue model."""

from __future__ import annotations

from dataclasses import dataclass

from budget_rule.experiment import (
    experiment_priority,
    lift_test_weeks,
    national_test_feasible,
    next_experiment,
    publish_lift_test,
)
from budget_rule.memo_pdf import write_memo_pdf

from measurement.qa import PAYBACK, RECOVERY_LABEL_FLOOR, channel_payback, gated_payback, sensitivity_is_material

LIFT_INCREASES = (0.10, 0.25, 0.50)
WEAK_CONTRACTION = 0.5
SATURATED_CONTRACTION = 0.99
MAX_NATIONAL_TEST_WEEKS = 52


@dataclass(frozen=True)
class BacktestOrigin:
    offset_weeks: int
    holdout_start: str
    holdout_mape: float
    naive_mape: dict[str, float]
    beats_lag_52: bool
    failures: list[str]


@dataclass(frozen=True)
class PriorCheck:
    """Every media prior widened by `scale`. A failed widened fit publishes no move."""

    scale: float
    failures: list[str]
    moves: dict[str, float]


@dataclass(frozen=True)
class AblationCheck:
    dropped_control: str
    failures: list[str]
    channel_roi: dict[str, float]


@dataclass(frozen=True)
class ResidualCheck:
    rho: float
    rho_interval: tuple[float, float]
    structural_acf: list[float]
    structural_p_value: float
    innovation_acf: list[float]
    innovation_p_value: float


@dataclass(frozen=True)
class RecoveryCheck:
    n_fits: int
    n_passed: int
    coverage: dict[str, float]
    coverage_se: dict[str, float]
    mean_bias: dict[str, float]
    true_roi: dict[str, float]


@dataclass(frozen=True)
class LiftPlan:
    channel: str
    priority: float
    spend_share: float
    weekly_spend: float
    expected_roi: float
    weeks: dict[float, int | None]


def lift_plan(
    *,
    spend_share: dict[str, float],
    weekly_spend: dict[str, float],
    channel_roi: dict[str, float],
    channel_roi_low: dict[str, float],
    channel_roi_high: dict[str, float],
    sigma: float,
    rho: float,
) -> LiftPlan | None:
    """The shared lift-test rule, with weeks per arm for each spend increase."""
    channel = next_experiment(spend_share, channel_roi_low, channel_roi_high)
    if channel is None:
        return None
    priority = experiment_priority(spend_share, channel_roi_low, channel_roi_high)[channel]
    weeks = {
        increase: lift_test_weeks(
            sigma=sigma,
            rho=rho,
            weekly_increase=increase * weekly_spend[channel],
            expected_roi=channel_roi[channel],
        )
        for increase in LIFT_INCREASES
    }
    return LiftPlan(channel, priority, spend_share[channel], weekly_spend[channel], channel_roi[channel], weeks)


def memo_text(
    *,
    question: str,
    estimand: str,
    holdout_mape: float,
    naive_mape: dict[str, float],
    backtest: list[BacktestOrigin],
    channel_roi: dict[str, float],
    channel_roi_low: dict[str, float],
    channel_roi_high: dict[str, float],
    channel_contraction: dict[str, float],
    residuals: ResidualCheck,
    prior: PriorCheck,
    ablation: AblationCheck,
    recovery: RecoveryCheck,
    lift: LiftPlan | None,
    geo_lift: str = "",
    geo_powered: bool = False,
) -> str:
    ranked = sorted(channel_roi, key=channel_roi.get, reverse=True)
    coverage = recovery.coverage if recovery.n_passed else {}
    reads = [
        f"- {channel}: ROI {channel_roi[channel]:.2f} "
        f"(95% interval {channel_roi_low[channel]:.2f} to {channel_roi_high[channel]:.2f}), "
        f"prior contraction {channel_contraction[channel]:.2f}. "
        + _gated_status_sentence(
            channel_payback(channel_roi[channel], channel_roi_low[channel], channel_roi_high[channel]),
            coverage.get(channel),
            mean_bias=recovery.mean_bias.get(channel) if recovery.n_passed else None,
            true_roi=recovery.true_roi.get(channel) if recovery.n_passed else None,
        )
        for channel in ranked
    ]
    weak = [channel for channel in ranked if channel_contraction[channel] < WEAK_CONTRACTION]
    beaten = sum(origin.beats_lag_52 for origin in backtest)
    return "\n".join(
        [
            question,
            "",
            "Estimand",
            estimand,
            "",
            "Holdout",
            f"Model MAPE {holdout_mape:.3f}. "
            f"Training-mean MAPE {naive_mape['training_mean']:.3f}. "
            f"Lag-52 MAPE {naive_mape['lag_52']:.3f}. "
            "The 0.5 MAPE gate is a sanity check that this series passes easily. "
            "Forecast accuracy does not validate a channel ROI.",
            "Backtest on non-overlapping eight-week windows, each fit only on the weeks before it. "
            f"The model beats the lag-52 forecast in {beaten} of {len(backtest)} windows. "
            "It checks the forecast and does not award a label.",
            *(_backtest_line(origin) for origin in backtest),
            "",
            "Errors",
            _residual_sentence(residuals),
            "",
            "Channels",
            f"An interval entirely below {PAYBACK:.0f} or entirely above {PAYBACK:.0f} is an interval fact. "
            "A point below that line whose interval includes it is not a cut. "
            f"Recovery coverage of at least {RECOVERY_LABEL_FLOOR:.0%} and the mean bias are diagnostics on a "
            "misspecified response; they do not license a budget label. "
            "This national weekly series does not identify the mix, so no channel is scaled or cut. "
            "Prior contraction is one minus the posterior over the prior variance of the marginal ROI itself, "
            "with the prior from prior predictive draws; near zero means the data barely moved the prior. "
            "Channels are reported at 95% each, without a family-wise correction. "
            "This is one fit on one public series, not a budget optimizer.",
            *reads,
            _contraction_sentence(weak, channel_contraction),
            "",
            "Prior check",
            _prior_sentence(prior),
            "",
            "Control check",
            _ablation_sentence(ablation, channel_roi),
            "",
            "Recovery check.",
            _recovery_sentence(recovery),
            "",
            _lift_sentence(lift, channel_roi_low, channel_roi_high, geo_powered=geo_powered),
            geo_lift.strip(),
            "",
            "What would overturn this",
            "A series that identifies the channel returns, or a feasible lift test that "
            "settles payback. The numbers are from this public weekly series.",
        ]
    )


def _acf_text(values: list[float]) -> str:
    return ", ".join(f"{value:+.2f}" for value in values)


def _residual_sentence(residuals: ResidualCheck) -> str:
    low, high = residuals.rho_interval
    return (
        f"The model carries AR(1) errors with rho {residuals.rho:.2f} (95% interval {low:.2f} to {high:.2f}). "
        f"Autocorrelation of the structural errors at lags 1 to {len(residuals.structural_acf)}: "
        f"{_acf_text(residuals.structural_acf)} (Ljung-Box p {residuals.structural_p_value:.3f}). "
        f"After the AR(1) term: {_acf_text(residuals.innovation_acf)} (p {residuals.innovation_p_value:.3f}). "
        + (
            "The rho interval includes 0 and the residuals show no autocorrelation, so the intervals are not "
            "narrowed by correlated weeks."
            if low <= 0 <= high
            else "Without the AR term these correlated weeks would make the ROI intervals too narrow."
        )
    )


def _backtest_line(origin: BacktestOrigin) -> str:
    gate = " The fit at this origin failed the QA gate." if origin.failures else ""
    return (
        f"- holdout from {origin.holdout_start}: model MAPE {origin.holdout_mape:.3f}, "
        f"training-mean {origin.naive_mape['training_mean']:.3f}, lag-52 {origin.naive_mape['lag_52']:.3f}.{gate}"
    )


def _prior_sentence(prior: PriorCheck) -> str:
    if prior.failures:
        return (
            f"The fit with every media prior {prior.scale:.0f} times as wide did not clear the QA gate: "
            + " ".join(prior.failures)
            + " No prior move is published, so prior sensitivity is not established."
        )
    material = [channel for channel, move in prior.moves.items() if sensitivity_is_material(move)]
    moves = ", ".join(f"{channel} {move:+.0%}" for channel, move in prior.moves.items())
    tail = (
        f" {', '.join(material)} moved by 25% or more, so the prior shapes that read."
        if material
        else " Every move is inside the 25% check."
    )
    return f"Every media prior {prior.scale:.0f} times as wide moves ROI by {moves}.{tail}"


def _ablation_sentence(ablation: AblationCheck, channel_roi: dict[str, float]) -> str:
    if ablation.failures:
        return f"The fit without {ablation.dropped_control} did not clear the QA gate: " + " ".join(ablation.failures)
    moves = ", ".join(
        f"{channel} {channel_roi[channel]:.2f} to {ablation.channel_roi[channel]:.2f}"
        for channel in sorted(channel_roi, key=channel_roi.get, reverse=True)
    )
    return (
        f"Without {ablation.dropped_control} as a control, ROI moves {moves}. "
        "That control is market demand, so part of what it absorbs may be media. This check does not decide."
    )


def _recovery_sentence(recovery: RecoveryCheck) -> str:
    head = (
        f"Revenue simulated {recovery.n_fits} times from a delayed adstock and a Hill saturation that the model "
        "cannot represent, with AR(1) noise, and refit. "
    )
    if not recovery.n_passed:
        return head + "No refit cleared the QA gate, so coverage is not established."
    parts = ", ".join(
        f"{channel} {share:.0%} (Monte Carlo standard error {recovery.coverage_se[channel]:.0%}, "
        f"true {recovery.true_roi[channel]:.2f}, mean bias {recovery.mean_bias[channel]:+.2f})"
        for channel, share in recovery.coverage.items()
    )
    short = [channel for channel, share in recovery.coverage.items() if share + 2 * recovery.coverage_se[channel] < 0.95]
    verdict = (
        f"For {', '.join(short)} coverage is more than two standard errors under 95%, so those intervals "
        "understate uncertainty when the response shape is wrong."
        if short
        else "No channel is more than two standard errors under 95%."
    )
    return (
        head
        + f"{recovery.n_passed} refits cleared the QA gate. The 95% interval held the true marginal ROI in: {parts}. "
        + verdict
    )


def geo_lift_sentence(payload: dict, *, target: float = 0.8) -> str:
    """One sentence from the planning-proxy geo design JSON."""
    closer = "That panel is simulated and no cell is a test to run."
    powered = [cell for cell in payload.get("cells", []) if cell["power_at_point_roi"] >= target]
    channel = payload.get("channel", "search")
    point = float(payload.get("point_roi", 0.0))
    if not powered:
        return (
            f"On the planning-proxy geo panel no design reached {target:.0%} power at the current "
            f"{channel} point ROI {point:.2f}. {closer}"
        )
    chosen = min(powered, key=lambda cell: (cell["duration_days"], cell["increase"]))
    mde = chosen.get("mde_roi_80")
    mde_text = "none of the searched ROIs" if mde is None else f"{mde:.2f}"
    return (
        f"The shortest planning-proxy geo design with at least {target:.0%} power at the current "
        f"{channel} point ROI {point:.2f} is {payload['n_treated']} treated geos of {payload['n_geos']}, "
        f"{chosen['duration_days']} days, and a {chosen['increase']:.0%} heavy-up "
        f"(false-settle at payback {chosen['false_settle_at_payback']:.0%}, MDE ROI {mde_text} at 80% power). "
        "A test that returned a 95% ROI interval [L, H] would replace the current search coefficient prior "
        "with a normal whose mean is (L+H)/2 and whose standard deviation is (H-L)/3.92, mapped through the "
        f"current saturation slope at observed search spend. {payload.get('note', '')} {closer}"
    )


def _lift_sentence(
    lift: LiftPlan | None,
    low: dict[str, float],
    high: dict[str, float],
    *,
    geo_powered: bool = False,
) -> str:
    if lift is None:
        return (
            "No feasible lift test on the available units.\n"
            "Every channel interval clears or misses 1, so no lift test is needed to settle payback."
        )
    durations = "; ".join(
        f"+{increase:.0%} spend: "
        + ("cannot exclude payback at the current point" if weeks is None else f"{weeks} weeks per arm")
        for increase, weeks in lift.weeks.items()
    )
    national = national_test_feasible(lift.weeks)
    feasible = publish_lift_test(national_feasible=national, geo_powered=geo_powered)
    feasibility = (
        f" Even the shortest is more than {MAX_NATIONAL_TEST_WEEKS} weeks, so a national test over time cannot "
        f"settle {lift.channel}."
        if not national
        else ""
    )
    body = (
        f"{lift.channel}: {lift.spend_share:.0%} of spend and a 95% interval "
        f"from {low[lift.channel]:.2f} to {high[lift.channel]:.2f}, the highest spend share times interval width "
        "among channels whose interval contains 1. "
        f"To exclude payback if the true ROI is the current point {lift.expected_roi:.2f}, raised-spend weeks "
        f"against baseline weeks need: {durations}. That ignores carryover and saturation, so it is a lower bound "
        f"on duration.{feasibility} The geo case in this package tests a different tactic and does not calibrate "
        "this model."
    )
    if feasible:
        return f"Next experiment.\nA lift test on {body}"
    return f"No feasible lift test on the available units.\nThe highest-value unsettled channel is {body}"


def forecast_notice(
    *,
    channel_roi: dict[str, float],
    channel_payback: dict[str, str],
    prior_moves: dict[str, float] | None = None,
) -> str:
    """Point-forecast caption. Payback labels come from the published QA file.

    A missing prior check is a line on the page. It does not hide the channels.
    """
    lines = ["This prediction is the point fit. It is not a budget optimizer."]
    if prior_moves is None:
        lines.append("The prior move is not published.")
    for channel in sorted(channel_roi, key=channel_roi.get, reverse=True):
        sentence = _status_sentence(channel_payback[channel])
        if prior_moves is not None and channel in prior_moves:
            move = prior_moves[channel]
            sentence += (
                f" The ROI moves by {abs(move):.0%} when the priors are wider"
                + (", so the prior shapes this read." if sensitivity_is_material(move) else ", inside the 25% check.")
            )
        lines.append(f"- {channel}: {sentence}")
    return "\n".join(lines)


def _contraction_sentence(weak: list[str], channel_contraction: dict[str, float]) -> str:
    if weak:
        return (
            f"Contraction is below {WEAK_CONTRACTION:.1f} for {', '.join(weak)}, so the data do not identify "
            "those returns on their own."
        )
    if min(channel_contraction.values()) >= SATURATED_CONTRACTION:
        return (
            f"Contraction is at least {SATURATED_CONTRACTION:.2f} for every channel because the ROI prior implied by "
            "the coefficient priors is very wide. It does not separate the channels here; the interval widths and "
            "the wide-prior check do."
        )
    return f"Contraction is at least {WEAK_CONTRACTION:.1f} for every channel."


def _gated_status_sentence(
    status: str,
    recovery_coverage: float | None,
    *,
    mean_bias: float | None = None,
    true_roi: float | None = None,
) -> str:
    if gated_payback(status, recovery_coverage) == "recovery_gate":
        side = "above" if status == "earns" else "below"
        covered = "is not established" if recovery_coverage is None else f"is {recovery_coverage:.0%}"
        return (
            f"The interval is {side} 1, but recovery coverage for this channel {covered}, under the "
            f"{RECOVERY_LABEL_FLOOR:.0%} floor, so the channel is not labelled."
        )
    if status in ("earns", "refuse") and recovery_coverage is not None:
        side = "above" if status == "earns" else "below"
        bias_text = ""
        if mean_bias is not None and true_roi is not None:
            bias_text = f", mean bias {mean_bias:.2f} under the Hill truth (true {true_roi:.2f})"
        return (
            f"The interval is {side} 1 on this fit; recovery coverage is {recovery_coverage:.0%}"
            f"{bias_text}. That is not a mix move."
        )
    return _status_sentence(status)


def _status_sentence(status: str) -> str:
    if status == "refuse":
        return "The interval is below 1 on this fit. That is not a mix move."
    if status == "earns":
        return "The interval is above 1 on this fit. That is not a mix move."
    if status == "withheld":
        return "The fit did not clear the QA gate, so this channel is not marked as earning or refused."
    return "Payback is not settled and the channel is not cut."


