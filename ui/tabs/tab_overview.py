"""
Arthagati — Overview: the reading, and whether it can be believed.

Reading order — the house convention every page follows:

  1 SCAN      what changed since I last looked?   KPI strip
  2 CLAIM     what does the engine say?           The conviction chain
  3 EVIDENCE  what is it saying it about?         Mood against the index
  4 STATE     how does that sit historically?     Regime & diagnostics
  5 DETAIL    the events behind it                Signal log

The KPI strip leads and the conviction chain follows it. It was the other way
round in an earlier build: a tall verdict card, then a section header, then the
six numbers that summarise it — so the one row a returning user actually needs
sat below the fold, under prose they had already read. Six numbers across the
top answers "what changed" in one saccade; the chain below answers "why", for
the reader who wants it. The numbers are the same objects the card is built
from, so the two cannot disagree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from config import MOOD_BAND_INNER, MOOD_BAND_OUTER
from ui import format as fmt
from ui import signals as sig
from ui.components import (
    render_chart_panel,
    render_hero_card,
    render_kpi_strip,
    render_note,
    render_section_header,
    render_table_panel,
)
from ui.theme import chart_color, chart_layout, chart_rgba, grid_rgba, style_axes


def _mood_vs_index(df: pd.DataFrame):
    """Mood against the index it is scored on, one pane, two axes.

    These belong together and were never plotted together. The score is a
    valuation reading anchored to PE, so it moves AGAINST price by design
    (rho about -0.54 against the trailing 60-day return). Showing them apart
    hides the single most important thing about the signal — and a reader who
    has not been told will read a falling score during a rally as the engine
    being wrong.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scattergl(
        x=df["DATE"], y=df["NIFTY"], mode="lines", name="NIFTY 50",
        line=dict(color=chart_color("slate"), width=1.1),
    ), secondary_y=True)
    fig.add_trace(go.Scattergl(
        x=df["DATE"], y=df["Mood_Score"], mode="lines", name="Mood Score",
        line=dict(color=chart_color("accent"), width=1.6),
    ), secondary_y=False)

    # The classification bands, drawn as the faintest thing on the plot. They
    # are fixed (VISION §6), so "Bullish" means the same thing on every date —
    # which is only useful if the reader can see where the line is.
    for lvl in (MOOD_BAND_INNER, -MOOD_BAND_INNER, MOOD_BAND_OUTER, -MOOD_BAND_OUTER):
        fig.add_hline(y=lvl, line_color=grid_rgba(0.07), line_width=1,
                      line_dash="dot", secondary_y=False)

    fig.update_layout(**chart_layout(height=380, show_legend=True))
    style_axes(fig, y_title="Mood Score")
    fig.update_yaxes(showgrid=False, title_text="NIFTY 50", tickformat=",.0f",
                     secondary_y=True)
    return fig


def _diagnostics_frame(last) -> pd.DataFrame:
    hurst = float(last.get("Hurst", np.nan))
    entropy = float(last.get("Market_Entropy", np.nan))
    h_label, _ = sig.hurst_state(hurst)
    e_label, _ = sig.entropy_state(entropy)
    return pd.DataFrame([
        {"Diagnostic": "Regime", "Value": str(last.get("Regime", "Unknown")),
         "Reads": "Hurst x entropy over 90 days"},
        {"Diagnostic": "Hurst exponent", "Value": fmt.num(hurst, 2),
         "Reads": f"{h_label.title()} — above 0.55 trends, below 0.45 reverts"},
        {"Diagnostic": "Market entropy", "Value": fmt.num(entropy, 2),
         "Reads": f"{e_label.title()} — higher is less structured"},
        {"Diagnostic": "OU half-life", "Value": fmt.days(last.get("OU_Half_Life")),
         "Reads": "Expected time to revert halfway to equilibrium"},
        {"Diagnostic": "OU equilibrium", "Value": fmt.num(last.get("OU_Mu"), 3, signed=True),
         "Reads": "Long-run mean, in the engine's own units"},
    ])


def _signal_log(df: pd.DataFrame, msf: pd.DataFrame, limit: int = 14) -> pd.DataFrame:
    """Divergences and crossovers as a dated list, newest first.

    The chart draws these as triangles the reader has to hunt for across an
    840px stack. The log is the same events with the state at the time, which
    is how a desk consumes them.
    """
    events: list[tuple[int, str, str]] = []
    spread = pd.Series(msf["msf_spread"].to_numpy(), index=df.index)
    bull, bear = sig.msf_divergences(df["Mood_Score"], spread)
    events += [(i, "MSF divergence", "Bullish") for i in bull]
    events += [(i, "MSF divergence", "Bearish") for i in bear]
    if "WT1" in df.columns and "WT2" in df.columns:
        g, r = sig.wt_crossovers(df["WT1"].to_numpy(dtype=float),
                                 df["WT2"].to_numpy(dtype=float))
        events += [(i, "WaveTrend cross", "Bullish") for i in g]
        events += [(i, "WaveTrend cross", "Bearish") for i in r]
    if not events:
        return pd.DataFrame()
    events.sort(key=lambda e: e[0], reverse=True)
    rows = []
    for i, kind, direction in events[:limit]:
        rows.append({
            "Date": df["DATE"].iloc[i].date(),
            "Event": kind,
            "Direction": direction,
            "Mood": float(df["Mood_Score"].iloc[i]),
            "NIFTY": float(df["NIFTY"].iloc[i]),
        })
    return pd.DataFrame(rows)


def render(mood_df, msf_df, *, verdict, timeframes, tf, periods, data_age) -> None:
    last = mood_df.iloc[-1]
    mood = float(last["Mood_Score"])
    spread = float(last["MSF_Spread"])
    mood_label, mood_tone = sig.mood_state(mood)
    msf_label, msf_tone = sig.msf_state(spread)
    _TONE = {"pos": "success", "neg": "danger", "warn": "warning",
             "info": "info", "neutral": "neutral"}

    # ── 1 · SCAN ──────────────────────────────────────────────────────────
    render_section_header(
        "Current Reading",
        "Where valuation sits against its own recent history, and the state of "
        "the instruments that qualify it.",
        icon="activity",
    )
    render_kpi_strip([
        {"label": "Mood Score", "value": fmt.num(mood, 1, signed=True),
         "subtext": mood_label.title(), "color_class": _TONE[mood_tone], "icon": "activity",
         "tooltip": "Anchored to PE and Earnings Yield: cheap scores high, expensive "
                    "scores low. It moves against recent price action by design "
                    "(rho -0.54 vs the trailing 60d return) and is not a momentum "
                    "indicator."},
        {"label": "MSF Spread", "value": fmt.num(spread, 2, signed=True),
         "subtext": msf_label.title(), "color_class": _TONE[msf_tone], "icon": "chart",
         "tooltip": "Momentum, structure, regime and flow, blended by inverse "
                    "variance. Built to be independent of the mood score, so a "
                    "disagreement between the two is information."},
        {"label": "NIFTY 50", "value": fmt.price(last["NIFTY"]),
         "subtext": "Index level", "color_class": "neutral", "icon": "trending-up"},
        {"label": "Regime", "value": str(last.get("Regime", "Unknown")),
         "subtext": "Hurst x entropy", "color_class": "info", "icon": "compass"},
        {"label": "Conviction", "value": f"{verdict['conviction']:.2f}",
         "subtext": verdict["action"]["label"].title(), "color_class": "accent",
         "icon": "target",
         "tooltip": "The product of six gates, each of which can independently "
                    "invalidate the reading. The smallest gate is the binding "
                    "constraint and is named on the card below."},
        {"label": "As of", "value": fmt.when(last["DATE"], "%d %b %y"),
         "subtext": f"{data_age}d old",
         "color_class": "neutral" if data_age <= 4 else "warning", "icon": "globe"},
    ], max_cols=6)

    # ── 2 · CLAIM ─────────────────────────────────────────────────────────
    render_section_header(
        "The Reading",
        "One claim, and every condition attached to it. Conviction is the product "
        "of the gates — the weakest caps it, and is named as the constraint.",
        icon="target",
        accent="cyan",
    )
    render_hero_card(verdict)

    # ── 3 · EVIDENCE ──────────────────────────────────────────────────────
    render_section_header(
        "Valuation Against Price",
        "The score and the index it is scored on, together. The two are expected "
        "to diverge: a falling score during a rally is the instrument working, "
        "not failing.",
        icon="chart",
    )
    mask = sig.window(mood_df, timeframes, tf)
    win = mood_df.loc[mask].copy()
    win_msf = msf_df.loc[mask.to_numpy()].copy()
    if len(win) < 2:
        win, win_msf = mood_df.tail(60).copy(), msf_df.tail(60).copy()

    render_chart_panel(
        _mood_vs_index(win), key="ov-mood-price",
        units="Mood -100 to +100",
        chip=(mood_label, _TONE[mood_tone]),
        window=True,
        footer=f"Dotted rules mark the fixed classification bands at "
               f"±{MOOD_BAND_INNER:.0f} and ±{MOOD_BAND_OUTER:.0f}.",
    )

    # ── 4 · STATE ─────────────────────────────────────────────────────────
    render_section_header(
        "Regime & Diagnostics",
        "The classifiers behind the regime label. Both axes split at their own "
        "expanding median, not at a fixed 0.5.",
        icon="cpu",
        accent="violet",
    )
    render_table_panel(
        _diagnostics_frame(last), key="ov-diagnostics",
        context="Latest session", label_col="Diagnostic", max_height=240,
    )

    # ── 5 · DETAIL ────────────────────────────────────────────────────────
    render_section_header(
        "Signal Log",
        "Divergence and crossover events in the selected window, newest first, "
        "with the state at the time.",
        icon="layers",
        accent="emerald",
    )
    log = _signal_log(win, win_msf)
    if log.empty:
        render_table_panel(
            pd.DataFrame(), key="ov-signals",
            context=f"{tf} window",
        )
        render_note("No divergences or crossovers in this window. Both are sparse "
                    "by construction — widen the window on the chart above to look "
                    "further back.")
    else:
        render_table_panel(
            log, key="ov-signals",
            context=f"{tf} window · {len(log)} shown",
            sign_color_cols={"Mood"},
            col_precision={"Mood": 1, "NIFTY": 0},
            label_col="Date", max_height=340,
        )
