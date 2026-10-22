"""Parameter recovery when the model is wrong about the response.

The true revenue uses a delayed adstock that peaks after the first week and a
Hill saturation, neither of which the fitted model can represent. Each channel's
true response is scaled so its training contribution matches the published fit.
Controls and the baseline are the published posterior means, and the noise is
AR(1) with the published rho and sigma. The true marginal ROI of the next ruble
is a finite difference of that true response, spend-weighted as in the estimand.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from measurement.fit import ARTIFACTS, _fit_origin, _origin, _write_json, load_grain
from measurement.model import L_MAX
from measurement.transforms import ChannelCurve, channel_contribution, predict_revenue

RECOVERY_SEEDS = tuple(range(101, 201))
HILL_SHAPE = 2.0


def delayed_adstock(values: np.ndarray, decay: float, l_max: int = L_MAX) -> np.ndarray:
    """Carryover with weights proportional to (lag + 1) * decay**lag, so the peak is after the spend week."""
    if not 0 < decay < 1:
        raise ValueError("decay must be in (0, 1).")
    weights = np.array([(lag + 1) * decay**lag for lag in range(l_max)], dtype=float)
    weights /= weights.sum()
    output = np.zeros(len(values), dtype=float)
    for lag, weight in enumerate(weights):
        output[lag:] += weight * values[: len(values) - lag]
    return output


@dataclass(frozen=True)
class TrueResponse:
    channel: str
    regressor: str
    spend_column: str
    decay: float
    half_saturation: float
    scale: float

    def contribution(self, media: np.ndarray) -> np.ndarray:
        adstocked = np.maximum(delayed_adstock(media, self.decay), 0.0)
        powered = adstocked**HILL_SHAPE
        return self.scale * powered / (powered + self.half_saturation**HILL_SHAPE)


def true_response(train: pd.DataFrame, curve: ChannelCurve) -> TrueResponse:
    """Hill response whose training contribution equals the published fit's contribution."""
    media = train[curve.regressor].to_numpy(dtype=float)
    decay = min(max(curve.alpha, 0.2), 0.8)
    adstocked = delayed_adstock(media, decay)
    positive = adstocked[adstocked > 0]
    if positive.size == 0:
        raise ValueError(f"{curve.channel} has no exposure to calibrate.")
    unit = TrueResponse(curve.channel, curve.regressor, curve.spend_column, decay, float(np.median(positive)), 1.0)
    target = abs(float(channel_contribution(media, curve).sum()))
    return TrueResponse(
        curve.channel,
        curve.regressor,
        curve.spend_column,
        decay,
        unit.half_saturation,
        target / float(unit.contribution(media).sum()),
    )


def true_marginal_roi(train: pd.DataFrame, response: TrueResponse, l_max: int = L_MAX) -> float:
    """Spend-weighted revenue from one more ruble in each week whose horizon sits inside training."""
    media = train[response.regressor].to_numpy(dtype=float)
    spend = train[response.spend_column].to_numpy(dtype=float)
    last = len(media) - l_max
    base = response.contribution(media)
    weeks = [week for week in range(last + 1) if spend[week] > 0]
    if not weeks:
        raise ValueError("No positive-spend week has a complete adstock horizon.")
    returns = []
    for week in weeks:
        rubles = max(1.0, 1e-3 * spend[week])
        bumped = media.copy()
        bumped[week] += media[week] / spend[week] * rubles
        returns.append(float((response.contribution(bumped) - base).sum()) / rubles)
    return float(np.average(returns, weights=spend[weeks]))


def ar_noise(n_weeks: int, *, sigma: float, rho: float, rng: np.random.Generator) -> np.ndarray:
    if sigma <= 0 or not -1 < rho < 1:
        raise ValueError("sigma must be positive and rho inside (-1, 1).")
    noise = np.empty(n_weeks, dtype=float)
    noise[0] = rng.normal(0.0, sigma / math.sqrt(1 - rho**2))
    for week in range(1, n_weeks):
        noise[week] = rho * noise[week - 1] + rng.normal(0.0, sigma)
    return noise


def coverage_share(fits: list[dict[str, bool]]) -> dict[str, float]:
    if not fits:
        raise ValueError("Coverage needs at least one fit.")
    return {channel: float(np.mean([fit[channel] for fit in fits])) for channel in sorted(fits[0])}


def _load_published() -> tuple[list[ChannelCurve], dict]:
    frame = pd.read_csv(ARTIFACTS / "curves.csv")
    curves = [
        ChannelCurve(row.channel, row.regressor, row.spend_column, float(row.alpha), float(row.lam), float(row.beta))
        for row in frame.itertuples(index=False)
    ]
    return curves, json.loads((ARTIFACTS / "baseline.json").read_text(encoding="utf-8"))


def run_recovery() -> None:
    grain = load_grain()
    train, _ = _origin(grain, 0)
    curves, published = _load_published()
    responses = [true_response(train, curve) for curve in curves]
    roi_true = {response.channel: true_marginal_roi(train, response) for response in responses}
    mean = predict_revenue(train, [], baseline=float(published["baseline"]), control_coefs=published["control_coefs"])
    for response in responses:
        mean = mean + response.contribution(train[response.regressor].to_numpy(dtype=float))
    fits: list[dict] = []
    for seed in RECOVERY_SEEDS:
        noise = ar_noise(len(mean), sigma=float(published["sigma"]), rho=float(published["rho"]), rng=np.random.default_rng(seed))
        origin = _fit_origin(grain, 0, revenue=mean + noise, seed=seed)
        bounds = origin["roi_bounds"]
        fits.append(
            {
                "seed": seed,
                "failures": origin["failures"],
                "covered": {
                    channel: bool(bounds[channel]["low"] <= roi_true[channel] <= bounds[channel]["high"])
                    for channel in roi_true
                },
                "roi_mean": {channel: bounds[channel]["mean"] for channel in roi_true},
                "roi_low": {channel: bounds[channel]["low"] for channel in roi_true},
                "roi_high": {channel: bounds[channel]["high"] for channel in roi_true},
            }
        )
    passed = [fit for fit in fits if not fit["failures"]]
    coverage = coverage_share([fit["covered"] for fit in passed]) if passed else {}
    _write_json(
        ARTIFACTS / "recovery.json",
        {
            "design": "delayed adstock and Hill saturation, AR(1) noise; the fitted model is geometric and exponential",
            "true_roi": roi_true,
            "n_fits": len(fits),
            "n_passed": len(passed),
            "fits": fits,
            "coverage": coverage,
            "coverage_se": {
                channel: math.sqrt(share * (1 - share) / len(passed)) for channel, share in coverage.items()
            },
            "mean_bias": {
                channel: float(np.mean([fit["roi_mean"][channel] for fit in passed]) - roi_true[channel])
                for channel in roi_true
            }
            if passed
            else {},
        },
    )
