"""
Arthagati — Signal Validation view.

Answers one question with one number: does the Mood Score carry
out-of-sample predictive power, and is it distinguishable from chance?

Nothing here is fitted. The view reports a measurement made on a holdout the
engine had no part in shaping, against a permutation null, alongside the
null model the engine has to beat — the negated PE ratio, "cheap is good"
with no engine at all.

This replaced the Intelligence Center, which reported diagnostics for a
fitted ensemble that reduced the signal's out-of-sample power on every real
configuration tested. See validation.py for the measurements.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.components import (
    render_section_header,
    render_metric_card,
    render_interpretation_card,
    section_divider,
    section_gap,
)
from ui.theme import C_EMERALD, C_ROSE, C_MUTED
from ui.charts import PLOT_CONFIG, chart_height
import validation as val


def _verdict_class(v: str) -> str:
    return {
        val.VERDICT_EDGE: "success",
        val.VERDICT_NO_EDGE: "danger",
        val.VERDICT_INSUFFICIENT: "neutral",
    }.get(v, "neutral")


@st.cache_data(max_entries=3, show_spinner=False)
def _run_validation(mood_df: pd.DataFrame, pe: np.ndarray) -> dict:
    return val.validate(mood_df, baseline=-pe).to_dict()


def render(mood_df: pd.DataFrame, raw_df: pd.DataFrame) -> None:
    render_section_header(
        title="Signal Validation",
        description="Out-of-sample Spearman rho on a held-out window, against a permutation null",
        icon="shield",
        accent="emerald",
    )

    if mood_df is None or mood_df.empty or "NIFTY" not in mood_df.columns:
        st.caption("Run an analysis first.")
        return

    pe = (raw_df["NIFTY50_PE"].to_numpy(dtype=float)
          if "NIFTY50_PE" in raw_df.columns else np.full(len(mood_df), np.nan))
    with st.spinner("Scoring the holdout and running the permutation null…"):
        r = _run_validation(mood_df, pe)

    section_gap()
    c1, c2, c3, c4 = st.columns(4, gap="small")
    with c1:
        render_metric_card(
            "Holdout rho", f"{r['holdout_rho']:+.3f}",
            f"{r['n_holdout']:,} rows from {r['holdout_start']}",
            "success" if r["holdout_rho"] > 0 else "danger",
            icon="target",
        )
    with c2:
        render_metric_card(
            "Significance", f"p = {r['p_value']:.3f}",
            f"vs {r['n_permutations']} circular shifts",
            "success" if r["p_value"] <= val.GATE_MAX_P_VALUE else "danger",
            icon="shield",
        )
    with c3:
        margin = r["holdout_rho"] - r["baseline_rho"] if np.isfinite(r["baseline_rho"]) else float("nan")
        render_metric_card(
            "vs −PE baseline", f"{margin:+.3f}" if np.isfinite(margin) else "—",
            f"baseline {r['baseline_rho']:+.3f} · no engine",
            "success" if np.isfinite(margin) and margin > 0.02 else "warning",
            icon="zap",
        )
    with c4:
        render_metric_card(
            "Verdict", r["verdict"],
            f"{r['independent_windows']:.1f} independent windows",
            _verdict_class(r["verdict"]),
            icon="check-circle",
        )

    _v = ", ".join(f"+{h}D" for h in r["validated_horizons"]) or "none"
    _d = ", ".join(f"+{h}D" for h in r["descriptive_horizons"])
    st.caption(
        f"Development rho {r['dev_rho']:+.3f} · holdout {r['holdout_rho']:+.3f}. "
        f"The verdict is computed on **{_v}** — the horizons this holdout can support. "
        + (f"**{_d}** are shown below but not validated: forward windows overlap, so a "
           f"{r['n_holdout']:,}-row holdout carries too few independent windows at that "
           "length to separate them from chance. " if _d else "")
        + "Nothing is fitted to the holdout; it is scored once, after the fact."
    )

    section_divider()
    render_section_header(
        title="Rho by Horizon",
        description="Where the signal lives — held-out window · * = descriptive, not validated",
        icon="bar-chart",
        accent="cyan",
    )

    per = {int(k): float(v) for k, v in r["per_horizon"].items()}
    hs = sorted(per)
    validated = set(r["validated_horizons"])
    # Validated horizons solid; descriptive ones dimmed, so the chart cannot
    # be read as claiming more than the test established.
    colors = [
        (C_EMERALD if per[h] > 0 else C_ROSE) if h in validated
        else ("rgba(45,212,168,0.35)" if per[h] > 0 else "rgba(232,85,90,0.35)")
        for h in hs
    ]
    fig = go.Figure(go.Bar(
        x=[f"+{h}D{'' if h in validated else ' *'}" for h in hs], y=[per[h] for h in hs],
        marker_color=colors,
        text=[f"{per[h]:+.3f}" for h in hs], textposition="outside",
        hovertemplate="%{x}<br>rho %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="rgba(148,163,184,0.4)", line_width=1)
    # Font, margin, grid, hover, legend and crosshair all come from the
    # registered template. Only the structural bits are declared here.
    fig.update_layout(
        height=chart_height("sm"),
        showlegend=False,
        hovermode="closest",          # categorical bars, not a time series
        xaxis=dict(showgrid=False),
        yaxis=dict(title=dict(text="Spearman rho")),
    )
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)
    st.markdown("</div>", unsafe_allow_html=True)

    section_divider()

    if r["verdict"] == val.VERDICT_EDGE:
        body = (
            f"Mood Score ranks forward NIFTY returns on held-out data with mean Spearman "
            f"<strong>rho {r['holdout_rho']:+.3f}</strong> across {len(hs)} horizons "
            f"(<strong>p = {r['p_value']:.3f}</strong> against {r['n_permutations']} "
            "circularly shifted copies of itself). The relationship strengthens with "
            "horizon — this is a positioning signal measured in months, not days.<br><br>"
            f"<strong>What it is not.</strong> The negated PE ratio alone — no engine, no "
            f"percentiles, no correlations — scores <strong>{r['baseline_rho']:+.3f}</strong> "
            "on the same window. Most of the edge belongs to the valuation anchor, not to "
            "the five-layer pipeline. The engine's contribution is a bounded, comparable "
            "score and its diagnostics, not additional rank information."
        )
        render_interpretation_card("Edge Confirmed — with a caveat worth reading", body, color="success")
    elif r["verdict"] == val.VERDICT_INSUFFICIENT:
        body = (
            f"The holdout spans {r['independent_windows']:.1f} independent "
            f"{max(r['horizons'])}-day windows; at least {val.MIN_INDEPENDENT_WINDOWS:.0f} are "
            "needed before a verdict means anything. Forward windows overlap, so a short "
            "holdout carries far fewer independent observations than it has rows. "
            "No verdict is issued rather than a guess."
        )
        render_interpretation_card("Insufficient Data to Validate", body, color="warning")
    else:
        body = (
            f"Holdout rho {r['holdout_rho']:+.3f} at p = {r['p_value']:.3f} — not "
            "distinguishable from a circularly shifted copy of the same signal. On this "
            "data and this predictor set the score carries no demonstrable forward "
            "information. Try a longer history or a different predictor set."
        )
        render_interpretation_card("No Demonstrable Edge", body, color="danger")
