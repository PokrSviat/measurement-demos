"""Local forecast page. It reads saved curves and does not refit. The mix decision is the memo."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from measurement.memo import forecast_notice
from measurement.model import L_MAX
from measurement.transforms import ChannelCurve, load_posterior, mix_prediction

ROOT = Path(__file__).resolve().parents[1]
CURVES = ROOT / "artifacts" / "curves.csv"
BASELINE = ROOT / "artifacts" / "baseline.json"
GRAIN = ROOT / "artifacts" / "reference_week.csv"
QA = ROOT / "artifacts" / "qa.json"
PRIOR = ROOT / "artifacts" / "prior_sensitivity.json"
DRAWS = ROOT / "artifacts" / "posterior_draws.npz"
SPEND_COLUMNS = ("tv_spend", "ooh_spend", "print_spend", "facebook_spend", "search_spend")


def load_curves() -> list[ChannelCurve]:
    frame = pd.read_csv(CURVES)
    return [
        ChannelCurve(
            channel=row.channel,
            regressor=row.regressor,
            spend_column=row.spend_column,
            alpha=float(row.alpha),
            lam=float(row.lam),
            beta=float(row.beta),
        )
        for row in frame.itertuples(index=False)
    ]


def opening_budget(history: pd.DataFrame) -> float:
    present = [column for column in SPEND_COLUMNS if column in history.columns]
    if not present or history.empty:
        return 0.0
    return float(history.iloc[-1][present].astype(float).sum())


def main() -> None:
    st.title("Mix and revenue forecast")
    st.caption("Public weekly series. This page does not refit the model. The mix decision is the memo.")
    if not CURVES.exists() or not BASELINE.exists() or not GRAIN.exists():
        st.error("Fit the model first: python -m measurement.fit")
        return
    if not QA.exists():
        st.error("Write the memo first: python -m measurement.fit --memo")
        return
    qa = json.loads(QA.read_text(encoding="utf-8"))
    prior = json.loads(PRIOR.read_text(encoding="utf-8")) if PRIOR.exists() else None
    moves = (
        {channel: item["relative_change"] for channel, item in prior["channels"].items()}
        if prior and prior.get("published")
        else None
    )
    st.info(
        forecast_notice(
            channel_roi=qa["channel_roi"],
            channel_payback=qa["channel_payback"],
            prior_moves=moves,
        )
    )
    curves = load_curves()
    baseline_payload = json.loads(BASELINE.read_text(encoding="utf-8"))
    history = pd.read_csv(GRAIN)
    if len(history) < L_MAX:
        st.error(
            "The saved history is shorter than the adstock window. "
            "Refit the model before using the calculator."
        )
        return
    budget = st.number_input("Weekly budget", min_value=0.0, value=opening_budget(history), step=1000.0)
    shares: dict[str, float] = {}
    for curve in curves:
        shares[curve.channel] = st.slider(curve.channel, min_value=0.0, max_value=1.0, value=1.0 / len(curves))
    share_sum = sum(shares.values())
    st.write(f"Share sum: {share_sum:.3f}")
    if abs(share_sum - 1.0) > 1e-6:
        st.warning("Shares must sum to 1 before a prediction is shown.")
        return
    prediction = mix_prediction(
        history,
        curves,
        baseline=float(baseline_payload["baseline"]),
        control_coefs=baseline_payload["control_coefs"],
        budget=float(budget),
        shares=shares,
        posterior=load_posterior(DRAWS) if DRAWS.exists() else None,
    )
    st.metric("Predicted weekly revenue", f"{prediction:,.0f}")


if __name__ == "__main__":
    main()
