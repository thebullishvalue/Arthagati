"""
Arthagati — Historical Mood view (TradingView-style 3-pane chart).

Panes:
    Row 1 · Mood Score       — Kalman-smoothed sentiment + 95% band + OU projection
    Row 2 · MSF Spread       — 4-component oscillator + ±5 OB/OS bands + divergence ▲▼
    Row 3 · WaveTrend        — LazyBear oscillator on Mood Score with ±80 bands
                                and WT1/WT2 crossover ▲▼

All oscillator panes share the date axis and use the Obsidian Quant
chrome: glass surfaces, JetBrains Mono ticks, dashed spike crosshairs,
transparent plot/paper backgrounds. Mood Score and WaveTrend y-axes are
reversed (negative on top, positive on bottom) — bearish-on-top
convention shared across the three signal panes.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ui.components import (
    render_section_header,
    render_metric_card,
    section_divider,
)
from config import (
    MSF_OB_LEVEL_1,
    MSF_OB_LEVEL_2,
    MSF_OS_LEVEL_1,
    MSF_OS_LEVEL_2,
    MSF_SIGNAL_Y,
    WT_OB_LEVEL_1,
    WT_OB_LEVEL_2,
)
from ui.theme import (
    C_AMBER,
    C_AMBER_BRIGHT,
    C_CYAN,
    C_EMERALD,
    C_ROSE,
    C_MUTED,
    PLOTLY_BASE,
    PLOTLY_GRID,
    PLOTLY_GRID_ZERO,
    PLOTLY_HOVERLABEL,
    PLOTLY_LEGEND,
    PLOTLY_SPIKE_X,
    PLOTLY_SPIKE_Y,
)


def render(mood_df, msf_df, *, timeframes, mood_scale, ou_proj_days) -> None:
    """TradingView-style Mood + MSF + WaveTrend chart, period summary, MSF breakdown."""

    render_section_header(
        title="Market Mood Terminal",
        description="TradingView-style analysis · Mood Score + MSF Spread indicator",
        icon="activity",
    )

    # ── Timeframe selector (Google-Finance style row) ─────────────────────
    if "tf_selected" not in st.session_state:
        st.session_state.tf_selected = "1Y"

    tf_cols = st.columns(len(timeframes))
    for i, tf in enumerate(timeframes.keys()):
        with tf_cols[i]:
            btn_type = "primary" if st.session_state.tf_selected == tf else "secondary"
            if st.button(tf, key=f"tf_{tf}", use_container_width=True, type=btn_type):
                st.session_state.tf_selected = tf
                st.rerun()

    selected_tf = st.session_state.tf_selected

    # Windows are selected by DATE, not by row count.
    #
    # `timeframes` holds calendar-day spans, and these used to be passed
    # straight to `.tail(n)` — a row count. On a trading-day series that
    # stretched every window by ~1.4x: "1Y" returned 365 rows spanning 510
    # calendar days, and "5Y" spanned just over seven years.
    last_date = mood_df["DATE"].max()
    if selected_tf == "YTD":
        cutoff = pd.Timestamp(year=last_date.year, month=1, day=1)
    elif timeframes.get(selected_tf):
        cutoff = last_date - pd.Timedelta(days=timeframes[selected_tf])
    else:
        cutoff = None

    if cutoff is not None:
        window = mood_df["DATE"] >= cutoff
        df = mood_df.loc[window].copy()
        msf_filtered = msf_df.loc[window.to_numpy()].copy()
    else:
        df = mood_df.copy()
        msf_filtered = msf_df.copy()

    # A very short window (a fresh sheet, or 1W over a holiday break) still
    # needs enough points to draw.
    if len(df) < 2:
        df = mood_df.tail(min(len(mood_df), 30)).copy()
        msf_filtered = msf_df.tail(len(df)).copy()

    if df.empty:
        st.warning("No data available for selected timeframe.")
        return

    # ═══════════════════════════════════════════════════════════════════════
    # CHART layout:
    #   Row 1: Mood Score   (always)
    #   Row 2: MSF Spread   (always)
    #   Row 3: WaveTrend    (always — LazyBear · Mood-driven)
    #
    # Calibrated Conviction is no longer drawn on this chart. The signal
    # is still surfaced in the top-of-page metric strip and in the
    # Intelligence Center dashboard — it just doesn't compete with the
    # raw Mood Score on the historical pane any more.
    # ═══════════════════════════════════════════════════════════════════════
    show_wt_pane = "WT1" in df.columns and "WT2" in df.columns

    if show_wt_pane:
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            vertical_spacing=0.06,
            row_heights=[0.50, 0.25, 0.25],
        )
    else:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.65, 0.35],
        )

    # Kalman confidence band
    if "Confidence_Upper" in df.columns and "Confidence_Lower" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["DATE"], y=df["Confidence_Upper"],
            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df["DATE"], y=df["Confidence_Lower"],
            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
            fill="tonexty", fillcolor="rgba(212,168,83,0.16)",
            name="95% Confidence",
        ), row=1, col=1)

    # Mood Score line (raw engine output)
    fig.add_trace(go.Scattergl(
        x=df["DATE"], y=df["Mood_Score"],
        mode="lines", name="Mood Score",
        line=dict(color=C_AMBER, width=2),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Mood: %{y:.2f}<extra></extra>",
    ), row=1, col=1)

    fig.add_hline(y=0, line_color="rgba(155,170,191,0.55)", line_width=1, line_dash="dash", row=1, col=1)

    last_point = df.iloc[-1]
    fig.add_annotation(
        x=last_point["DATE"], y=last_point["Mood_Score"],
        text=f"<b>{last_point['Mood_Score']:.1f}</b>",
        showarrow=True, arrowhead=2, arrowcolor=C_AMBER,
        ax=40, ay=0,
        bgcolor="rgba(14,19,31,0.92)", bordercolor=C_AMBER, borderwidth=1,
        font=dict(color=C_AMBER_BRIGHT, size=11, family="JetBrains Mono, monospace"),
        row=1, col=1,
    )

    # ── OU forward projection ────────────────────────────────────────────
    ou_theta = float(last_point.get("OU_Theta", 0.05))
    ou_mu    = float(last_point.get("OU_Mu",    0.0))
    ou_sigma = float(last_point.get("OU_Sigma", 1.0))
    ou_std   = ou_sigma / np.sqrt(2.0 * max(ou_theta, 1e-4))

    last_date  = last_point["DATE"]
    proj_dates = pd.date_range(start=last_date, periods=ou_proj_days + 1, freq="D")[1:]
    proj_n     = np.arange(1, ou_proj_days + 1, dtype=np.float64)

    x_current_ou = last_point["Mood_Score"] / mood_scale * max(ou_std, 1e-6) + ou_mu
    proj_ou   = ou_mu + (x_current_ou - ou_mu) * np.exp(-ou_theta * proj_n)
    proj_mood = np.clip((proj_ou - ou_mu) / max(ou_std, 1e-6) * mood_scale, -100, 100)

    fig.add_trace(go.Scatter(
        x=proj_dates, y=proj_mood,
        mode="lines", name="OU Projection",
        line=dict(color=C_AMBER, width=1.5, dash="dot"),
        opacity=0.55,
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Projected: %{y:.1f}<extra></extra>",
    ), row=1, col=1)

    fig.add_annotation(
        x=proj_dates[-1], y=0.0,
        text=f"EQ ({last_point.get('OU_Half_Life', 0):.0f}d t½)",
        showarrow=False,
        font=dict(color="#9BAABF", size=10, family="JetBrains Mono, monospace"),
        # Nudged off the zero line, which it previously sat on top of.
        xanchor="left", xshift=6, yshift=-12, row=1, col=1,
    )

    # Dynamic y-bounds
    _yc = [df["Mood_Score"].values]
    if "Confidence_Upper" in df.columns:
        _yc.append(df["Confidence_Upper"].values)
    if "Confidence_Lower" in df.columns:
        _yc.append(df["Confidence_Lower"].values)
    _yc.append(proj_mood)
    _y_all = np.concatenate([c[np.isfinite(c)] for c in _yc])
    _y_min, _y_max = float(_y_all.min()) if len(_y_all) else -100, float(_y_all.max()) if len(_y_all) else 100
    _y_pad = max((_y_max - _y_min) * 0.08, 2.0)
    mood_y_lo, mood_y_hi = _y_min - _y_pad, _y_max + _y_pad

    # Regime transition markers were removed for visual clarity — the regime
    # state is still surfaced in the top metric strip card and used elsewhere
    # by the engine; we just don't draw vertical dotted lines on the chart.

    # Degraded-input banner — an MSF component with no variance is excluded
    # from the composite rather than silently capturing all of the weight.
    _degenerate = st.session_state.get("_msf_degenerate") or []
    if _degenerate:
        st.warning(
            f"MSF Spread is running on {4 - len(_degenerate)} of 4 components. "
            f"No signal in: {', '.join(_degenerate)}. "
            "Check that the source columns (NIFTY, AD_RATIO) are populated in the sheet.",
            icon="⚠️",
        )

    # ── Row 2: MSF Spread ─────────────────────────────────────────────────
    msf_values = msf_filtered["msf_spread"].values
    fig.add_trace(go.Scattergl(
        x=df["DATE"], y=msf_values,
        mode="lines", name="MSF Spread",
        line=dict(color=C_CYAN, width=2),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>MSF: %{y:.2f}<extra></extra>",
    ), row=2, col=1)
    fig.add_hline(y=0, line_color="rgba(155,170,191,0.55)", line_width=1, row=2, col=1)
    # MSF Spread OB/OS reference bands — ±4 primary (solid), ±3 secondary (dotted)
    fig.add_hline(y=MSF_OB_LEVEL_1, line_color="rgba(232,85,90,0.42)",
                  line_width=1, line_dash="solid", row=2, col=1)
    fig.add_hline(y=MSF_OS_LEVEL_1, line_color="rgba(45,212,168,0.42)",
                  line_width=1, line_dash="solid", row=2, col=1)
    fig.add_hline(y=MSF_OB_LEVEL_2, line_color="rgba(232,85,90,0.26)",
                  line_width=1, line_dash="dot", row=2, col=1)
    fig.add_hline(y=MSF_OS_LEVEL_2, line_color="rgba(45,212,168,0.26)",
                  line_width=1, line_dash="dot", row=2, col=1)

    # Divergence triangles
    lookback = 10
    mood_series = df["Mood_Score"]
    msf_series  = pd.Series(msf_values, index=df.index)
    rmm_min = mood_series.rolling(lookback + 1, min_periods=1).min()
    rmm_max = mood_series.rolling(lookback + 1, min_periods=1).max()
    rms_min = msf_series.rolling(lookback + 1, min_periods=1).min()
    rms_max = msf_series.rolling(lookback + 1, min_periods=1).max()
    p_mood_min, p_msf_min = rmm_min.shift(lookback), rms_min.shift(lookback)
    p_mood_max, p_msf_max = rmm_max.shift(lookback), rms_max.shift(lookback)
    bear_mask = (mood_series == rmm_min) & (mood_series < p_mood_min) & (rms_min > p_msf_min)
    bull_mask = (mood_series == rmm_max) & (mood_series > p_mood_max) & (rms_max < p_msf_max)
    valid = np.zeros(len(df), dtype=bool)
    valid[lookback * 2 : len(df) - 1] = True
    red_idx   = np.where(bear_mask & valid)[0]
    green_idx = np.where(bull_mask & valid)[0]

    # Divergence triangles sit just inside the ±5 OB/OS bands (at ±4)
    # so the marker and the level line don't visually overlap.
    _TRI_SIZE = 9   # shared marker pixel-size (must match WT triangles below)

    if len(red_idx):
        fig.add_trace(go.Scatter(
            x=[df["DATE"].iloc[i] for i in red_idx],
            y=[+MSF_SIGNAL_Y] * len(red_idx),
            mode="markers", name="Bearish Signal",
            marker=dict(symbol="triangle-down", size=_TRI_SIZE, color=C_ROSE,
                        line=dict(color=C_ROSE, width=1)),
            hoverinfo="skip", showlegend=False,
        ), row=2, col=1)
    if len(green_idx):
        fig.add_trace(go.Scatter(
            x=[df["DATE"].iloc[i] for i in green_idx],
            y=[-MSF_SIGNAL_Y] * len(green_idx),
            mode="markers", name="Bullish Signal",
            marker=dict(symbol="triangle-up", size=_TRI_SIZE, color=C_EMERALD,
                        line=dict(color=C_EMERALD, width=1)),
            hoverinfo="skip", showlegend=False,
        ), row=2, col=1)

    # ── Row 3: WaveTrend Oscillator (LazyBear · Mood-driven) ────────────
    # Trace order matters for the area fill — plot the difference area
    # first (zero-baselined), then the WT2 signal line, then WT1 wave on
    # top so the lines aren't hidden by the fill.
    wt_row = 3 if show_wt_pane else None
    if show_wt_pane:
        # WT1 − WT2 area (cyan, transparent, fills to y=0)
        fig.add_trace(go.Scatter(
            x=df["DATE"], y=df["WT1"] - df["WT2"],
            mode="lines",
            line=dict(color="rgba(6,182,212,0.0)", width=0),
            fill="tozeroy",
            fillcolor="rgba(6,182,212,0.16)",
            name="WT1 − WT2",
            hoverinfo="skip",
            showlegend=False,
        ), row=wt_row, col=1)

        # WT2 (signal line — rose, dashed)
        fig.add_trace(go.Scattergl(
            x=df["DATE"], y=df["WT2"],
            mode="lines", name="WT2 (signal)",
            line=dict(color=C_ROSE, width=1.4, dash="dash"),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>WT2: %{y:.2f}<extra></extra>",
        ), row=wt_row, col=1)

        # WT1 (wave line — emerald, solid)
        fig.add_trace(go.Scattergl(
            x=df["DATE"], y=df["WT1"],
            mode="lines", name="WT1 (wave)",
            line=dict(color=C_EMERALD, width=1.8),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>WT1: %{y:.2f}<extra></extra>",
        ), row=wt_row, col=1)

        # Reference levels: 0 baseline + OB/OS bands.
        # Axis is reversed (negative on top, positive on bottom), so colour
        # coding follows the user's preference: emerald on positive levels,
        # rose on negative — independent of the visual position.
        # Bands are calibrated from the full history of |wt1| (see
        # arthagati.wavetrend_bands) because LazyBear's hlc3-derived +/-80
        # is unreachable when the source is Mood_Score. Fall back to the
        # config constants if the engine did not publish them.
        _bands = st.session_state.get("_wt_bands")
        if _bands:
            WT_OB_1, WT_OB_2 = float(_bands[0]), float(_bands[1])
        else:
            WT_OB_1, WT_OB_2 = float(WT_OB_LEVEL_1), float(WT_OB_LEVEL_2)
        WT_OS_1, WT_OS_2 = -WT_OB_1, -WT_OB_2

        fig.add_hline(y=0, line_color="rgba(155,170,191,0.55)",
                      line_width=1, line_dash="dash", row=wt_row, col=1)
        fig.add_hline(y=WT_OB_1, line_color="rgba(45,212,168,0.42)",
                      line_width=1, line_dash="solid", row=wt_row, col=1)
        fig.add_hline(y=WT_OS_1, line_color="rgba(232,85,90,0.42)",
                      line_width=1, line_dash="solid", row=wt_row, col=1)
        fig.add_hline(y=WT_OB_2, line_color="rgba(45,212,168,0.26)",
                      line_width=1, line_dash="dot", row=wt_row, col=1)
        fig.add_hline(y=WT_OS_2, line_color="rgba(232,85,90,0.26)",
                      line_width=1, line_dash="dot", row=wt_row, col=1)

        # ── WT crossover markers ────────────────────────────────────────
        # Green triangle: WT1 crosses above WT2 (bullish momentum)
        # Red triangle:   WT2 crosses above WT1 (bearish momentum)
        # With the y-axis flipped (negative up, positive down), bullish
        # markers sit at the BOTTOM of the pane visually, which means a
        # POSITIVE y-coordinate. Bearish markers go at the top → negative.
        wt1_arr = df["WT1"].to_numpy(dtype=np.float64)
        wt2_arr = df["WT2"].to_numpy(dtype=np.float64)
        prev_wt1 = np.concatenate([[np.nan], wt1_arr[:-1]])
        prev_wt2 = np.concatenate([[np.nan], wt2_arr[:-1]])
        wt_valid = (
            np.isfinite(wt1_arr) & np.isfinite(wt2_arr)
            & np.isfinite(prev_wt1) & np.isfinite(prev_wt2)
        )
        red_cross = wt_valid & (wt1_arr > wt2_arr) & (prev_wt1 <= prev_wt2) & (wt1_arr < 0)
        green_cross   = wt_valid & (wt2_arr > wt1_arr) & (prev_wt2 <= prev_wt1) & (wt1_arr > 0)
        # Suppress markers in the very first lookback to avoid noise
        warmup = 32
        if len(green_cross) > warmup:
            green_cross[:warmup] = False
            red_cross[:warmup]   = False
        wt_green_idx = np.where(green_cross)[0]
        wt_red_idx   = np.where(red_cross)[0]

        # Marker y placement — just inside the OB/OS bands so triangles
        # are visible without clashing with the reference lines.
        _marker_y = max(WT_OB_1 * 0.85, 8.0)   # just inside the primary band

        if len(wt_red_idx):
            fig.add_trace(go.Scatter(
                x=[df["DATE"].iloc[i] for i in wt_red_idx],
                y=[-_marker_y] * len(wt_red_idx),
                mode="markers", name="WT Bearish Cross",
                marker=dict(symbol="triangle-down", size=_TRI_SIZE, color=C_ROSE,
                            line=dict(color=C_ROSE, width=1)),
                hoverinfo="skip", showlegend=False,
            ), row=wt_row, col=1)

        if len(wt_green_idx):
            fig.add_trace(go.Scatter(
                x=[df["DATE"].iloc[i] for i in wt_green_idx],
                y=[+_marker_y] * len(wt_green_idx),
                mode="markers", name="WT Bullish Cross",
                marker=dict(symbol="triangle-up", size=_TRI_SIZE, color=C_EMERALD,
                            line=dict(color=C_EMERALD, width=1)),
                hoverinfo="skip", showlegend=False,
            ), row=wt_row, col=1)

        # Dynamic y-bounds — include the OB/OS bands so they're always
        # visible, then REVERSE the range so negative is up / positive is
        # down (matches mood + calibrated panes).
        _wt_finite = np.concatenate([
            wt1_arr[np.isfinite(wt1_arr)],
            wt2_arr[np.isfinite(wt2_arr)],
            np.array([WT_OB_1 + 8, WT_OS_1 - 8], dtype=np.float64),
        ])
        if len(_wt_finite) > 0:
            _w_min = float(_wt_finite.min())
            _w_max = float(_wt_finite.max())
        else:
            _w_min, _w_max = -100.0, 100.0
        _w_pad = max((_w_max - _w_min) * 0.05, 4.0)
        wt_y_lo, wt_y_hi = _w_min - _w_pad, _w_max + _w_pad

    # Calibrated Conviction pane removed — see header comment at top of render().

    # ── Layout — Obsidian Quant ───────────────────────────────────────────
    _shared_tick = dict(size=10, family="JetBrains Mono, monospace", color="#9BAABF")
    # Vertical line from the x-axis spike, horizontal from the y-axis spike.
    # Both are required for a full crosshair; the chart previously set only
    # the x-axis, which is why the horizontal line was missing.
    _shared_spike = PLOTLY_SPIKE_X

    # MSF y-range — guarantee the ±4 OB/OS bands are always visible
    _msf_finite = msf_values[np.isfinite(msf_values)] if msf_values is not None else np.array([])
    if len(_msf_finite) > 0:
        _msf_min = min(float(_msf_finite.min()), MSF_OS_LEVEL_1 - 0.5)
        _msf_max = max(float(_msf_finite.max()), MSF_OB_LEVEL_1 + 0.5)
    else:
        _msf_min, _msf_max = MSF_OS_LEVEL_1 - 0.5, MSF_OB_LEVEL_1 + 0.5
    _msf_pad = max((_msf_max - _msf_min) * 0.05, 0.5)
    _msf_range = [_msf_min - _msf_pad, _msf_max + _msf_pad]

    # Chart height grows with pane count: 2 panes = 750, 3 = 880
    _pane_count = 2 + int(show_wt_pane)
    _heights = {2: 750, 3: 880}
    layout_kwargs = dict(
        **PLOTLY_BASE,
        height=_heights[_pane_count],
        hovermode="x unified",
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1,
            font=dict(size=10, family="JetBrains Mono, monospace"),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=60, r=20, t=60, b=40),
        # Keep the crosshair alive anywhere in the plot area, not just near a trace.
        spikedistance=-1,
        hoverdistance=-1,
        yaxis=dict(
            title=dict(text="Mood Score", font=dict(size=11, color=C_MUTED, family="JetBrains Mono, monospace")),
            showgrid=True, gridcolor=PLOTLY_GRID, gridwidth=0.5,
            zeroline=True, zerolinecolor=PLOTLY_GRID_ZERO, zerolinewidth=0.5,
            linecolor="rgba(255,255,255,0.04)",
            tickfont=_shared_tick,
            range=[mood_y_hi, mood_y_lo],
            **PLOTLY_SPIKE_Y,
        ),
        yaxis2=dict(
            title=dict(text="MSF Spread", font=dict(size=11, color=C_MUTED, family="JetBrains Mono, monospace")),
            showgrid=True, gridcolor=PLOTLY_GRID, gridwidth=0.5,
            zeroline=True, zerolinecolor=PLOTLY_GRID_ZERO, zerolinewidth=0.5,
            linecolor="rgba(255,255,255,0.04)",
            tickfont=_shared_tick,
            # Lock the y-range wide enough to always show the OB/OS bands
            range=_msf_range,
            **PLOTLY_SPIKE_Y,
        ),
        xaxis=dict(
            showgrid=False, linecolor="rgba(255,255,255,0.04)",
            **_shared_spike,
        ),
    )

    # WaveTrend pane (row 3 when present) — y-axis reversed: negative
    # on top, positive on bottom (matches mood pane convention).
    if show_wt_pane:
        layout_kwargs["yaxis3"] = dict(
            title=dict(
                text="WaveTrend",
                font=dict(size=11, color=C_MUTED, family="JetBrains Mono, monospace"),
            ),
            showgrid=True, gridcolor=PLOTLY_GRID, gridwidth=0.5,
            zeroline=False,
            linecolor="rgba(255,255,255,0.04)",
            tickfont=_shared_tick,
            range=[wt_y_hi, wt_y_lo],  # reversed
            **PLOTLY_SPIKE_Y,
        )

    # X-axes: only the bottom-most row carries date ticks.
    bottom_row_axis = f"xaxis{_pane_count}" if _pane_count > 1 else "xaxis"
    for i in range(2, _pane_count + 1):
        key = f"xaxis{i}"
        if key == bottom_row_axis:
            layout_kwargs[key] = dict(
                showgrid=True, gridcolor=PLOTLY_GRID, gridwidth=0.5, type="date",
                linecolor="rgba(255,255,255,0.04)",
                tickfont=_shared_tick,
                **_shared_spike,
            )
        else:
            layout_kwargs[key] = dict(
                showgrid=False, linecolor="rgba(255,255,255,0.04)",
                **_shared_spike,
            )

    fig.update_layout(**layout_kwargs)

    # Thin separator lines between panes. Positions are computed from
    # row_heights so they always sit exactly on the row boundaries.
    if _pane_count == 3:
        heights = [0.50, 0.25, 0.25]
    else:
        heights = [0.65, 0.35]
    cum = 1.0
    for h in heights[:-1]:
        cum -= h
        fig.add_shape(
            type="line", xref="paper", yref="paper",
            x0=0, y0=cum, x1=1, y1=cum,
            line=dict(color="rgba(255,255,255,0.06)", width=1),
        )

    st.markdown('<div class="chart-container mood">', unsafe_allow_html=True)
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": True,
            "scrollZoom": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════
    # PERIOD SUMMARY METRICS
    # ═══════════════════════════════════════════════════════════════════════
    section_divider()
    render_section_header(
        title="Period Summary",
        description=f"Mood & MSF statistics across the {selected_tf} window",
        icon="bar-chart",
        accent="cyan",
    )

    period_high = df["Mood_Score"].max()
    period_low  = df["Mood_Score"].min()
    period_avg  = df["Mood_Score"].mean()
    msf_avg     = msf_filtered["msf_spread"].mean()

    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        render_metric_card("Period High", f"{period_high:.1f}", "Most bullish", color_class="success", icon="arrow-up")
    with sc2:
        render_metric_card("Period Low", f"{period_low:.1f}", "Most bearish", color_class="danger", icon="arrow-down")
    with sc3:
        avg_cls = "success" if period_avg > 0 else "danger" if period_avg < 0 else "neutral"
        render_metric_card("Average Mood", f"{period_avg:.1f}", f"{selected_tf} period", color_class=avg_cls)
    with sc4:
        # Lower MSF is "more oversold" → success; higher → danger
        msf_cls = "success" if msf_avg < 0 else "danger" if msf_avg > 0 else "neutral"
        render_metric_card("Avg MSF Spread", f"{msf_avg:+.2f}", f"{selected_tf} period", color_class=msf_cls)

    # ═══════════════════════════════════════════════════════════════════════
    # MSF COMPONENT DECOMPOSITION
    # ═══════════════════════════════════════════════════════════════════════
    section_divider()
    render_section_header(
        title="MSF Component Breakdown",
        description="Current contribution of each component · weights = inverse-variance (auto-calibrated)",
        icon="layers",
        accent="violet",
    )

    msf_idx = min(len(msf_filtered) - 1, len(df) - 1)
    if msf_idx >= 0 and not msf_filtered.empty:
        comps = [
            ("momentum",  "Momentum",  "var(--amber)"),
            ("structure", "Structure", "var(--violet)"),
            ("regime",    "Regime",    "var(--emerald)"),
            ("flow",      "Flow",      "var(--cyan)"),
        ]
        c_cols = st.columns(4, gap="small")
        for j, (name, label, color) in enumerate(comps):
            val = msf_filtered[name].iloc[msf_idx] if name in msf_filtered.columns else 0
            period_val = msf_filtered[name].mean() if name in msf_filtered.columns else 0
            bar_pct = max(0, min(100, (val + 10) / 20 * 100))
            with c_cols[j]:
                st.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(145deg, rgba(17,24,39,0.45) 0%, rgba(17,24,39,0.4) 100%);
                        border: 1px solid var(--border);
                        border-radius: var(--r-md);
                        padding: var(--sp-4) var(--sp-4);
                        backdrop-filter: blur(8px);
                    ">
                        <div style="display:flex; justify-content:space-between;
                                    align-items:center; margin-bottom:0.6rem;">
                            <span style="font-family:var(--data); font-size:0.62rem; color:var(--ink-tertiary);
                                         font-weight:600; text-transform:uppercase; letter-spacing:0.1em;">
                                {label}
                            </span>
                            <span style="font-family:var(--display); font-size:1.1rem; font-weight:700;
                                         color:{color}; font-variant-numeric:tabular-nums;">
                                {val:+.1f}
                            </span>
                        </div>
                        <div style="height:4px; background:rgba(255,255,255,0.04);
                                    border-radius:2px; position:relative;">
                            <div style="position:absolute; left:50%; top:0; width:1px; height:4px;
                                        background:rgba(255,255,255,0.12);"></div>
                            <div style="width:{bar_pct:.0f}%; height:100%; background:{color};
                                        border-radius:2px; opacity:0.85; box-shadow:0 0 8px {color};"></div>
                        </div>
                        <div style="font-family:var(--data); font-size:0.6rem; color:var(--ink-tertiary);
                                    margin-top:0.4rem; letter-spacing:0.04em;">
                            Period avg: <span style="color:var(--ink-secondary);">{period_val:+.1f}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
