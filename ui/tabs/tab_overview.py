"""
Arthagati — Overview: the reading, and whether it can be believed.

Reading order — the house convention every page follows:

  1 SCAN   what changed since I last looked?   KPI strip
  2 CLAIM  what does the engine say?           The conviction chain
  3 STATE  how does that sit historically?     Regime & diagnostics

The KPI strip leads and the conviction chain follows it. It was the other way
round in an earlier build: a tall verdict card, then a section header, then the
six numbers that summarise it — so the one row a returning user actually needs
sat below the fold, under prose they had already read. Six numbers across the
top answers "what changed" in one saccade; the chain below answers "why", for
the reader who wants it. The numbers are the same objects the card is built
from, so the two cannot disagree.

The SERIES are not here, and neither is the signal log. Overview answers "what
is the reading and should I trust it"; how the score got there is the Mood
Engine's question. A second chart of the same data under a second heading only
gave the two pages a way to disagree about which window they were showing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ui import format as fmt
from ui import signals as sig
from ui.components import (
    render_hero_card,
    render_kpi_strip,
    render_section_header,
    render_table_panel,
)


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


def render(mood_df, *, verdict, data_age) -> None:
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

    # ── 3 · STATE ─────────────────────────────────────────────────────────
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
