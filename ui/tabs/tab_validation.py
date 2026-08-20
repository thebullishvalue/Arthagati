"""
Arthagati — Validation: is any of this distinguishable from chance?

Nothing here is fitted. The view reports a measurement made on a holdout the
engine had no part in shaping, against a permutation null, alongside the null
model the engine has to beat — the negated PE ratio, "cheap is good" with no
engine at all. See validation.py for the measurements.

The result is also written to session state, because the conviction chain on
the Mood Engine reads two of these numbers as gates: a reading the holdout cannot
support must not carry conviction anywhere in the app.

Reading order:

  1 TRUST    can this be believed?      Verdict and its gates
  2 ANCHOR   what was measured?         Holdout, null and baseline
  3 DETAIL   where does it live?        Rho by horizon
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import validation as val
from ui import format as fmt
from ui.components import (
    render_chart_panel,
    render_empty_state,
    render_interpretation_card,
    render_kpi_strip,
    render_note,
    render_section_header,
    render_table_panel,
)
from ui.theme import chart_color, chart_layout, chart_rgba, grid_rgba, style_axes


def _verdict_tone(v: str) -> str:
    return {val.VERDICT_EDGE: "success",
            val.VERDICT_NO_EDGE: "danger",
            val.VERDICT_INSUFFICIENT: "warning"}.get(v, "neutral")


@st.cache_data(max_entries=3, show_spinner=False)
def _run_validation(mood_df: pd.DataFrame, pe: np.ndarray) -> dict:
    return val.validate(mood_df, baseline=-pe).to_dict()


def render(mood_df: pd.DataFrame, raw_df: pd.DataFrame) -> None:
    if mood_df is None or mood_df.empty or "NIFTY" not in mood_df.columns:
        render_empty_state(
            "No engine output to validate",
            "The holdout is cut from the engine's own frame, so there is nothing to "
            "score until a run completes.",
            eyebrow="Not scored")
        return

    pe = (raw_df["NIFTY50_PE"].to_numpy(dtype=float)
          if "NIFTY50_PE" in raw_df.columns else np.full(len(mood_df), np.nan))
    with st.spinner("Scoring the holdout and running the permutation null…"):
        r = _run_validation(mood_df, pe)

    # Publish the two figures the conviction chain gates on. Written here
    # rather than recomputed there so the card and this page cannot disagree
    # about whether an edge was measured.
    st.session_state["_validation_summary"] = {
        "holdout_rho": r["holdout_rho"], "p_value": r["p_value"],
        "baseline_rho": r["baseline_rho"], "n_holdout": r["n_holdout"],
        "verdict": r["verdict"],
    }

    margin = (r["holdout_rho"] - r["baseline_rho"]
              if np.isfinite(r["baseline_rho"]) else float("nan"))
    p_ok = r["p_value"] <= val.GATE_MAX_P_VALUE
    enough = r["independent_windows"] >= val.MIN_INDEPENDENT_WINDOWS

    # ── 1 · TRUST ─────────────────────────────────────────────────────────
    render_section_header(
        "Out-of-Sample Verdict",
        "Spearman rho on a window the engine had no part in shaping, scored once "
        "after the fact against a permutation null and against the no-engine "
        "baseline.",
        icon="shield",
        accent="emerald",
    )
    render_kpi_strip([
        {"label": "Holdout rho", "value": fmt.rho(r["holdout_rho"]),
         "subtext": f"{r['n_holdout']:,} rows from {r['holdout_start']}",
         "color_class": "success" if r["holdout_rho"] > 0 else "danger", "icon": "target"},
        {"label": "Significance", "value": f"p {fmt.pvalue(r['p_value'])}",
         "subtext": f"vs {r['n_permutations']} circular shifts",
         "color_class": "success" if p_ok else "danger", "icon": "shield"},
        {"label": "vs −PE baseline",
         "value": fmt.rho(margin) if np.isfinite(margin) else "—",
         "subtext": f"baseline {fmt.rho(r['baseline_rho'])} · no engine",
         "color_class": "success" if np.isfinite(margin) and margin > 0.02 else "warning",
         "icon": "zap"},
        {"label": "Development rho", "value": fmt.rho(r["dev_rho"]),
         "subtext": "in-sample, for contrast", "color_class": "neutral", "icon": "chart"},
        {"label": "Indep. windows", "value": fmt.num(r["independent_windows"], 1),
         "subtext": f"gate at {val.MIN_INDEPENDENT_WINDOWS:.0f}",
         "color_class": "success" if enough else "warning", "icon": "layers"},
        {"label": "Verdict", "value": r["verdict"],
         "subtext": "on validated horizons",
         "color_class": _verdict_tone(r["verdict"]), "icon": "check-circle"},
    ], max_cols=6)

    # ── 2 · ANCHOR ────────────────────────────────────────────────────────
    validated = ", ".join(f"+{h}D" for h in r["validated_horizons"]) or "none"
    descriptive = ", ".join(f"+{h}D" for h in r["descriptive_horizons"])

    if r["verdict"] == val.VERDICT_EDGE:
        body = (
            f"Mood Score ranks forward NIFTY returns on held-out data with mean "
            f"Spearman <strong>rho {r['holdout_rho']:+.3f}</strong> across "
            f"{len(r['per_horizon'])} horizons, at <strong>p "
            f"{fmt.pvalue(r['p_value'])}</strong> against {r['n_permutations']} "
            "circularly shifted copies of itself. The relationship strengthens with "
            "horizon: this is a positioning signal measured in months, not days."
            "<br><br><strong>What it is not.</strong> The negated PE ratio alone — no "
            "engine, no percentiles, no correlations — scores "
            f"<strong>{r['baseline_rho']:+.3f}</strong> on the same window. Most of "
            "the edge belongs to the valuation anchor, not to the five-layer "
            "pipeline. The engine's contribution is a bounded, comparable score and "
            "its diagnostics, not additional rank information — which is why the "
            "conviction chain on the Mood Engine carries the margin over this baseline "
            "as a gate of its own."
        )
        render_interpretation_card("Edge confirmed — with a caveat worth reading",
                                   body, color="success")
    elif r["verdict"] == val.VERDICT_INSUFFICIENT:
        body = (
            f"The holdout spans {r['independent_windows']:.1f} independent "
            f"{max(r['horizons'])}-day windows; at least "
            f"{val.MIN_INDEPENDENT_WINDOWS:.0f} are needed before a verdict means "
            "anything. Forward windows overlap, so a short holdout carries far fewer "
            "independent observations than it has rows. No verdict is issued rather "
            "than a guess."
        )
        render_interpretation_card("Insufficient data to validate", body, color="warning")
    else:
        body = (
            f"Holdout rho {r['holdout_rho']:+.3f} at p {fmt.pvalue(r['p_value'])} — not "
            "distinguishable from a circularly shifted copy of the same signal. On "
            "this data and this predictor set the score carries no demonstrable "
            "forward information. Drivers ranks the universe by the same quality "
            "shape the engine weights with; a different predictor set is the next "
            "thing to try."
        )
        render_interpretation_card("No demonstrable edge", body, color="danger")

    render_note(
        f"The verdict is computed on <strong>{validated}</strong> — the horizons this "
        f"holdout can support."
        + (f" <strong>{descriptive}</strong> are shown below but not validated: a "
           f"{r['n_holdout']:,}-row holdout carries too few independent windows at "
           "that length to separate them from chance." if descriptive else "")
        + " Nothing is fitted to the holdout; it is scored once, after the fact.")

    # ── 3 · DETAIL ────────────────────────────────────────────────────────
    render_section_header(
        "Rho by Horizon",
        "Where the signal lives. Bars for horizons the holdout cannot validate are "
        "dimmed, so the chart cannot be read as claiming more than the test "
        "established.",
        icon="bar-chart",
        accent="cyan",
    )
    per = {int(k): float(v) for k, v in r["per_horizon"].items()}
    hs = sorted(per)
    validated_set = set(r["validated_horizons"])
    colours = [
        (chart_color("emerald") if per[h] > 0 else chart_color("rose"))
        if h in validated_set
        else (chart_rgba("emerald", 0.30) if per[h] > 0 else chart_rgba("rose", 0.30))
        for h in hs
    ]
    fig = go.Figure(go.Bar(
        x=[f"+{h}D{'' if h in validated_set else ' *'}" for h in hs],
        y=[per[h] for h in hs], marker_color=colours, width=0.55,
        text=[f"{per[h]:+.3f}" for h in hs], textposition="outside",
        hovertemplate="%{x}  rho %{y:.3f}<extra></extra>"))
    fig.add_hline(y=0, line_color=grid_rgba(0.11), line_width=1)
    fig.update_layout(**chart_layout(height=320, show_legend=False))
    fig.update_layout(hovermode="closest")
    style_axes(fig, y_title="Spearman rho")

    render_chart_panel(
        fig, key="val-horizons",
        context=f"Held-out window · validated {validated}",
        chip=(r["verdict"].upper(), _verdict_tone(r["verdict"])),
        footer="Asterisk marks a descriptive, unvalidated horizon.",
    )
    render_table_panel(
        pd.DataFrame([{"Horizon": f"+{h}D", "Holdout rho": per[h],
                       "Status": "Validated" if h in validated_set else "Descriptive"}
                      for h in hs]),
        key="val-horizon-table", label_col="Horizon",
        context="One row per evaluated forward window",
        sign_color_cols={"Holdout rho"}, precision=3, max_height=240,
    )
