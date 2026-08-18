"""
Arthagati — Mood Engine: how the score got where it is.

Three stacked panes on one date axis, inside one panel:

  Row 1 · Mood Score   Kalman-smoothed sentiment, 95% band, OU forward projection
  Row 2 · MSF Spread   Four-component oscillator, OB/OS zones, divergence marks
  Row 3 · WaveTrend    LazyBear oscillator on the Mood Score itself, crossovers

Mood Score and WaveTrend y-axes are REVERSED — bearish above the axis. That is
the convention the three signal panes share, so a reader tracking a single
vertical does not have to flip their reading between rows.

Reading order:

  1 ANCHOR   the instrument itself         The three-pane stack
  2 STATE    what the window contains      Window statistics
  3 DETAIL   what the oscillator is made of MSF decomposition
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from config import (
    MSF_OB_LEVEL_1, MSF_OB_LEVEL_2, MSF_OS_LEVEL_1, MSF_OS_LEVEL_2,
    MSF_SIGNAL_Y, WT_OB_LEVEL_1, WT_OB_LEVEL_2,
)
from ui import format as fmt
from ui import signals as sig
from ui.components import (
    render_chart_panel,
    render_note,
    render_section_header,
    render_table_panel,
)
from ui.theme import chart_color, chart_layout, chart_rgba, grid_rgba, style_axes

_TRI = 9  # marker size, shared by divergence and crossover glyphs
_TONE = {"pos": "success", "neg": "danger", "warn": "warning",
         "info": "info", "neutral": "neutral"}


def _window_stats(df: pd.DataFrame, msf: pd.DataFrame) -> pd.DataFrame:
    """Window statistics as a table, not four metric cards.

    Four cards for four related statistics of one series is a KPI strip that
    should have been a table: the reader wants to compare high against low
    against mean, and cards put a border through every comparison.
    """
    dates = df["DATE"].to_numpy()
    rows = []
    for label, arr in (("Mood Score", df["Mood_Score"].to_numpy(dtype=float)),
                       ("MSF Spread", msf["msf_spread"].to_numpy(dtype=float))):
        finite = np.isfinite(arr)
        if not finite.any() or len(arr) != len(dates):
            continue
        vals = np.where(finite, arr, np.nan)
        hi, lo = int(np.nanargmax(vals)), int(np.nanargmin(vals))
        rows.append({
            "Series": label,
            "Last": float(arr[int(np.max(np.flatnonzero(finite)))]),
            "High": float(vals[hi]), "High on": pd.Timestamp(dates[hi]).date(),
            "Low": float(vals[lo]), "Low on": pd.Timestamp(dates[lo]).date(),
            "Mean": float(np.nanmean(vals)), "Sigma": float(np.nanstd(vals)),
        })
    return pd.DataFrame(rows)


def _msf_components(msf: pd.DataFrame, idx: int) -> pd.DataFrame:
    rows = []
    for name, label in (("momentum", "Momentum"), ("structure", "Structure"),
                        ("regime", "Regime"), ("flow", "Flow")):
        if name not in msf.columns:
            continue
        rows.append({
            "Component": label,
            "Now": float(msf[name].iloc[idx]),
            "Window mean": float(msf[name].mean()),
            "Window sigma": float(msf[name].std()),
        })
    return pd.DataFrame(rows)


def render(mood_df, msf_df, *, timeframes, mood_scale, ou_proj_days) -> None:
    tf = st.session_state.get("tf_selected", "1Y")
    mask = sig.window(mood_df, timeframes, tf)
    df = mood_df.loc[mask].copy()
    msf_filtered = msf_df.loc[mask.to_numpy()].copy()

    # A very short window — a fresh sheet, or a window over a holiday break —
    # still needs enough points to draw.
    if len(df) < 2:
        df = mood_df.tail(min(len(mood_df), 30)).copy()
        msf_filtered = msf_df.tail(len(df)).copy()

    render_section_header(
        "Mood · MSF · WaveTrend",
        "The Kalman-smoothed score with its 95% band and Ornstein-Uhlenbeck forward "
        "projection, the oscillator that confirms or contradicts it, and a WaveTrend "
        "overlay on the score itself. Bearish is plotted ABOVE the axis on both "
        "signal panes.",
        icon="activity",
    )

    if df.empty:
        render_note("No observations in this window. Select a longer one on the panel "
                    "header, or check that the sheet carries rows for this period.")
        return

    show_wt = "WT1" in df.columns and "WT2" in df.columns
    accent, slate, brass = chart_color("accent"), chart_color("slate"), chart_color("amber")
    long_c, short_c = chart_color("emerald"), chart_color("rose")

    if show_wt:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            vertical_spacing=0.045, row_heights=[0.50, 0.25, 0.25])
    else:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.06, row_heights=[0.65, 0.35])

    # ── Row 1 · Mood Score ────────────────────────────────────────────────
    if "Confidence_Upper" in df.columns and "Confidence_Lower" in df.columns:
        fig.add_trace(go.Scatter(x=df["DATE"], y=df["Confidence_Upper"], mode="lines",
                                 line=dict(width=0), showlegend=False, hoverinfo="skip"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=df["DATE"], y=df["Confidence_Lower"], mode="lines",
                                 line=dict(width=0), showlegend=False, hoverinfo="skip",
                                 fill="tonexty", fillcolor=chart_rgba("accent", 0.12),
                                 name="95% band"),
                      row=1, col=1)
    fig.add_trace(go.Scattergl(x=df["DATE"], y=df["Mood_Score"], mode="lines",
                               name="Mood Score", line=dict(color=accent, width=1.6)),
                  row=1, col=1)
    fig.add_hline(y=0, line_color=grid_rgba(0.11), line_width=1, row=1, col=1)

    last = df.iloc[-1]

    # ── OU forward projection (maths unchanged) ───────────────────────────
    ou_theta = float(last.get("OU_Theta", 0.05))
    ou_mu = float(last.get("OU_Mu", 0.0))
    ou_sigma = float(last.get("OU_Sigma", 1.0))
    ou_std = ou_sigma / np.sqrt(2.0 * max(ou_theta, 1e-4))
    proj_dates = pd.date_range(start=last["DATE"], periods=ou_proj_days + 1, freq="D")[1:]
    proj_n = np.arange(1, ou_proj_days + 1, dtype=np.float64)
    x_now = last["Mood_Score"] / mood_scale * max(ou_std, 1e-6) + ou_mu
    proj_ou = ou_mu + (x_now - ou_mu) * np.exp(-ou_theta * proj_n)
    proj_mood = np.clip((proj_ou - ou_mu) / max(ou_std, 1e-6) * mood_scale, -100, 100)

    fig.add_trace(go.Scatter(x=proj_dates, y=proj_mood, mode="lines",
                             name="OU projection",
                             line=dict(color=accent, width=1.2, dash="dot"), opacity=0.6),
                  row=1, col=1)

    _yc = [df["Mood_Score"].to_numpy()]
    for c in ("Confidence_Upper", "Confidence_Lower"):
        if c in df.columns:
            _yc.append(df[c].to_numpy())
    _yc.append(proj_mood)
    _all = np.concatenate([c[np.isfinite(c)] for c in _yc])
    _lo, _hi = (float(_all.min()), float(_all.max())) if len(_all) else (-100.0, 100.0)
    _pad = max((_hi - _lo) * 0.08, 2.0)
    mood_lo, mood_hi = _lo - _pad, _hi + _pad

    # ── Row 2 · MSF Spread ────────────────────────────────────────────────
    msf_values = msf_filtered["msf_spread"].to_numpy()
    fig.add_trace(go.Scattergl(x=df["DATE"], y=msf_values, mode="lines",
                               name="MSF Spread", line=dict(color=slate, width=1.4)),
                  row=2, col=1)
    fig.add_hline(y=0, line_color=grid_rgba(0.11), line_width=1, row=2, col=1)
    for lvl, colour, dash in (
        (MSF_OB_LEVEL_1, chart_rgba("rose", 0.40), "solid"),
        (MSF_OS_LEVEL_1, chart_rgba("emerald", 0.40), "solid"),
        (MSF_OB_LEVEL_2, chart_rgba("rose", 0.22), "dot"),
        (MSF_OS_LEVEL_2, chart_rgba("emerald", 0.22), "dot"),
    ):
        fig.add_hline(y=lvl, line_color=colour, line_width=1, line_dash=dash, row=2, col=1)

    bull_idx, bear_idx = sig.msf_divergences(
        df["Mood_Score"], pd.Series(msf_values, index=df.index))
    for idxs, y, symb, colour, name in (
        (bear_idx, +MSF_SIGNAL_Y, "triangle-down", short_c, "Bearish divergence"),
        (bull_idx, -MSF_SIGNAL_Y, "triangle-up", long_c, "Bullish divergence"),
    ):
        if len(idxs):
            fig.add_trace(go.Scatter(
                x=df["DATE"].iloc[idxs], y=[y] * len(idxs), mode="markers", name=name,
                marker=dict(symbol=symb, size=_TRI, color=colour),
                hoverinfo="skip", showlegend=False), row=2, col=1)

    _fin = msf_values[np.isfinite(msf_values)]
    _mlo = min(float(_fin.min()), MSF_OS_LEVEL_1 - 0.5) if len(_fin) else MSF_OS_LEVEL_1 - 0.5
    _mhi = max(float(_fin.max()), MSF_OB_LEVEL_1 + 0.5) if len(_fin) else MSF_OB_LEVEL_1 + 0.5
    _mpad = max((_mhi - _mlo) * 0.05, 0.5)

    # ── Row 3 · WaveTrend ─────────────────────────────────────────────────
    wt_lo = wt_hi = 0.0
    if show_wt:
        fig.add_trace(go.Scatter(x=df["DATE"], y=df["WT1"] - df["WT2"], mode="lines",
                                 line=dict(width=0), fill="tozeroy",
                                 fillcolor=chart_rgba("slate", 0.16),
                                 name="WT1 − WT2", hoverinfo="skip", showlegend=False),
                      row=3, col=1)
        fig.add_trace(go.Scattergl(x=df["DATE"], y=df["WT2"], mode="lines",
                                   name="WT2 signal",
                                   line=dict(color=brass, width=1.1, dash="dash")),
                      row=3, col=1)
        fig.add_trace(go.Scattergl(x=df["DATE"], y=df["WT1"], mode="lines",
                                   name="WT1 wave", line=dict(color=accent, width=1.5)),
                      row=3, col=1)

        # Bands are calibrated from the full history of |WT1|: LazyBear's
        # hlc3-derived ±80 is unreachable when the source is the Mood Score.
        bands = st.session_state.get("_wt_bands")
        ob1, ob2 = ((float(bands[0]), float(bands[1])) if bands
                    else (float(WT_OB_LEVEL_1), float(WT_OB_LEVEL_2)))
        fig.add_hline(y=0, line_color=grid_rgba(0.11), line_width=1, row=3, col=1)
        for lvl, colour, dash in (
            (ob1, chart_rgba("emerald", 0.40), "solid"),
            (-ob1, chart_rgba("rose", 0.40), "solid"),
            (ob2, chart_rgba("emerald", 0.22), "dot"),
            (-ob2, chart_rgba("rose", 0.22), "dot"),
        ):
            fig.add_hline(y=lvl, line_color=colour, line_width=1, line_dash=dash, row=3, col=1)

        g_idx, r_idx = sig.wt_crossovers(df["WT1"].to_numpy(dtype=np.float64),
                                         df["WT2"].to_numpy(dtype=np.float64))
        my = max(ob1 * 0.85, 8.0)
        for idxs, y, symb, colour in ((r_idx, -my, "triangle-down", short_c),
                                      (g_idx, +my, "triangle-up", long_c)):
            if len(idxs):
                fig.add_trace(go.Scatter(
                    x=df["DATE"].iloc[idxs], y=[y] * len(idxs), mode="markers",
                    marker=dict(symbol=symb, size=_TRI, color=colour),
                    hoverinfo="skip", showlegend=False), row=3, col=1)

        w = np.concatenate([df["WT1"].to_numpy(dtype=np.float64),
                            df["WT2"].to_numpy(dtype=np.float64),
                            np.array([ob1 + 8, -ob1 - 8])])
        w = w[np.isfinite(w)]
        wmin, wmax = (float(w.min()), float(w.max())) if len(w) else (-100.0, 100.0)
        wpad = max((wmax - wmin) * 0.05, 4.0)
        wt_lo, wt_hi = wmin - wpad, wmax + wpad

    panes = 3 if show_wt else 2
    fig.update_layout(**chart_layout(height=760 if panes == 3 else 560, show_legend=True))
    style_axes(fig, y_title="Mood", y_range=[mood_hi, mood_lo], row=1, col=1)
    style_axes(fig, y_title="MSF", y_range=[_mlo - _mpad, _mhi + _mpad], row=2, col=1)
    if show_wt:
        style_axes(fig, y_title="WaveTrend", y_range=[wt_hi, wt_lo], row=3, col=1)

    # Hairlines on the row boundaries, computed from row_heights so they sit
    # exactly on them.
    cum = 1.0
    for h in ([0.50, 0.25] if panes == 3 else [0.65]):
        cum -= h
        fig.add_shape(type="line", xref="paper", yref="paper", x0=0, y0=cum, x1=1, y1=cum,
                      line=dict(color=grid_rgba(0.07), width=1))

    mood_label, mood_tone = sig.mood_state(float(last["Mood_Score"]))
    render_chart_panel(
        fig, key="mood-stack",
        units=f"OU projection {ou_proj_days}D",
        chip=(mood_label, _TONE[mood_tone]),
        window=True,
        footer=f"{fmt.when(df['DATE'].iloc[0])} — {fmt.when(df['DATE'].iloc[-1])} · "
               f"{len(df):,} observations · MSF bands fixed at "
               f"±{MSF_OB_LEVEL_1:.0f} and ±{MSF_OB_LEVEL_2:.0f}.",
    )

    # ── 2 · STATE ─────────────────────────────────────────────────────────
    render_section_header(
        "Window Statistics",
        "Both series over the selected window, with the date each extreme was set.",
        icon="bar-chart",
        accent="cyan",
    )
    render_table_panel(
        _window_stats(df, msf_filtered), key="mood-stats",
        label_col="Series", sign_color_cols={"Last", "High", "Low", "Mean"},
        col_precision={"Last": 2, "High": 2, "Low": 2, "Mean": 2, "Sigma": 2},
        max_height=160,
    )

    # ── 3 · DETAIL ────────────────────────────────────────────────────────
    render_section_header(
        "MSF Decomposition",
        "What the oscillator is currently made of. Weights are inverse-variance and "
        "auto-calibrated, so a component with no variance is excluded rather than "
        "capturing all of the weight.",
        icon="layers",
        accent="violet",
    )
    idx = min(len(msf_filtered) - 1, len(df) - 1)
    render_table_panel(
        _msf_components(msf_filtered, idx), key="mood-msf-parts",
        label_col="Component", sign_color_cols={"Now", "Window mean"},
        precision=2, max_height=180,
    )
