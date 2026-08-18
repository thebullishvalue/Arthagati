"""
Arthagati landing page — three system cards + methodology + awaiting-data prompt.
Mirrors Nishkarsh's tab_landing structure & visual fidelity.
"""

from __future__ import annotations

import streamlit as st

from ui.components import (
    render_header,
    render_section_header,
    render_system_card,
    render_metric_card,
    render_landing_prompt,
    render_interpretation_card,
    section_gap,
)


def render_landing_page(version: str, n_predictors: int) -> None:
    """Informational landing page shown before analysis starts."""

    # ── Masthead ────────────────────────────────────────────────────
    render_header(
        title="Arthagati",
        tagline="Ornstein-Uhlenbeck  ·  Kalman  ·  Decay-Spearman  ·  Adaptive Percentiles  |  Valuation-Anchored Market Positioning",
    )

    section_gap()

    # ── Three system feature cards ──────────────────────────────────
    col1, col2, col3 = st.columns(3, gap="small")
    with col1:
        render_system_card(
            title="Historical Mood",
            description=(
                "Full sentiment timeline with OU forward projection, Kalman confidence "
                "bands, and a WaveTrend oscillator on a TradingView-style chart."
            ),
            specs=[
                ("Range:", "Mood Score −100 → +100"),
                ("Confirmation:", "MSF Spread oscillator"),
                ("Projection:", "90-day OU mean-reversion"),
            ],
            card_class="mood",
            icon="chart",
        )
    with col2:
        render_system_card(
            title="Similar Periods",
            description=(
                "Historical analog matching against the full dataset with forward-return "
                "outcomes, aggregate win-rates, and a backtest scatter."
            ),
            specs=[
                ("Distance:", "Mahalanobis (55%)"),
                ("Shape:", "Trajectory cosine (35%)"),
                ("Separation:", "20 trading days minimum"),
            ],
            card_class="similar",
            icon="search",
        )
    with col3:
        render_system_card(
            title="Correlation Analysis",
            description=(
                "Full transparency into which variables drive the score and which are "
                "noise, ranked by the engine's own quality formula."
            ),
            specs=[
                ("Anchors:", "PE  &  Earnings Yield"),
                ("Method:", "Decay-Spearman + Entropy"),
                ("Output:", "Keep / Useful / Weak"),
            ],
            card_class="corr",
            icon="file-text",
        )

    section_gap()

    # ── Methodology — three coloured interpretation cards ──────────
    render_section_header(
        title="Analysis Methodology",
        description="Physics-informed scoring pipeline · confirmation · regime detection",
        icon="cpu",
    )

    m1, m2, m3 = st.columns(3, gap="small")
    with m1:
        render_interpretation_card(
            title="Mood Engine — 5 Layers",
            body=(
                "<ul style='margin:0; padding-left:1.1rem; line-height:1.8;'>"
                "<li><strong>Decay-Spearman</strong> correlations, walk-forward (504d half-life)</li>"
                "<li><strong>Entropy weighting</strong> — noisy variables suppressed</li>"
                "<li><strong>Adaptive percentiles</strong> — decay-weighted CDF</li>"
                "<li><strong>OU normalisation</strong> → [−100, +100]</li>"
                "<li><strong>Kalman smoothing</strong> + ±1.96σ band</li>"
                "</ul>"
            ),
            color="success",
        )
    with m2:
        render_interpretation_card(
            title="MSF Spread — Confirmation",
            body=(
                "<ul style='margin:0; padding-left:1.1rem; line-height:1.8;'>"
                "<li><strong>Momentum</strong> — NIFTY ROC z-score (14d)</li>"
                "<li><strong>Structure</strong> — mood trend divergence</li>"
                "<li><strong>Flow</strong> — breadth participation</li>"
                "<li><strong>Regime</strong> — adaptive directional count</li>"
                "<li><strong>Weights</strong> — inverse-variance (Markowitz)</li>"
                "</ul>"
            ),
            color="info",
        )
    with m3:
        render_interpretation_card(
            title="Regime Detection",
            body=(
                "<ul style='margin:0; padding-left:1.1rem; line-height:1.8;'>"
                "<li><strong>Trending</strong> — momentum strategies favoured</li>"
                "<li><strong>Volatile Trend</strong> — directional with swings</li>"
                "<li><strong>Mean-Reverting</strong> — contrarian strategies</li>"
                "<li><strong>Choppy</strong> — reduce size, avoid</li>"
                "<li><strong>Output</strong> — diagnostic only; never feeds the score</li>"
                "</ul>"
            ),
            color="warning",
        )

    section_gap()

    # ── Mood score interpretation zones ─────────────────────────────
    render_section_header(
        title="Mood Score Interpretation",
        description="What the score measures, and which way to read it",
        icon="target",
        accent="cyan",
    )

    render_interpretation_card(
        title="Read this first — the score is a valuation gauge, not a sentiment gauge",
        body=(
            "The Mood Score is anchored to <strong>PE and Earnings Yield</strong>. A cheap "
            "market scores <strong>high</strong>; an expensive one scores <strong>low</strong>. "
            "It therefore moves <em>against</em> recent price action — measured on NIFTY "
            "2006–2026, the score correlates <strong>−0.54</strong> with the trailing 60-day "
            "return.<br><br>"
            "That is the intended behaviour, and it is where the score's value lies: over the "
            "same twenty years the mean forward 250-day NIFTY return was "
            "<strong>+19.7%</strong> following readings above +20, against <strong>+5.9%</strong> "
            "following readings below −20 (Spearman +0.22).<br><br>"
            "The practical consequence: a high score during a sell-off is the signal working, "
            "not a contradiction. In October 2008 the score read <strong>+21 to +39</strong> "
            "while NIFTY fell 25%; through the 2020–21 melt-up it read <strong>−36 to −15</strong>. "
            "Do not read it as a momentum or trend-following indicator."
        ),
        color="warning",
    )

    section_gap()

    z1, z2, z3 = st.columns(3, gap="small")
    with z1:
        render_interpretation_card(
            title="Constructive Zone (> +20)",
            body=(
                "Valuation is cheap against its own recent history — forward returns have "
                "historically been strongest here. Favours <strong>accumulation</strong>. "
                "At extremes (&gt; +45, <strong>Very Bullish</strong>) the market is usually "
                "in or just past a drawdown."
            ),
            color="success",
        )
    with z2:
        render_interpretation_card(
            title="Neutral Zone (−20 to +20)",
            body=(
                "Valuation is near its recent norm and carries little directional "
                "information. Use the MSF Spread, WaveTrend and Similar Periods for "
                "context rather than leaning on the score alone."
            ),
            color="info",
        )
    with z3:
        render_interpretation_card(
            title="Expensive Zone (&lt; −20)",
            body=(
                "Valuation is stretched against its own recent history — forward returns "
                "have historically been weakest here. Favours <strong>defensive</strong> "
                "positioning. At extremes (&lt; −45, <strong>Very Bearish</strong>) the "
                "market has usually just run hard."
            ),
            color="danger",
        )

    section_gap()

    # ── System coverage strip ───────────────────────────────────────
    render_section_header(
        title="System Coverage",
        description="Anchors · predictors · mathematical primitives",
        icon="layers",
        accent="violet",
    )

    c1, c2, c3, c4, c5 = st.columns(5, gap="small")
    with c1:
        render_metric_card("Score Anchors", "2", "PE · Earnings Yield", color_class="neutral")
    with c2:
        render_metric_card("Predictors", f"{n_predictors}", "Macro + breadth vars", color_class="neutral")
    with c3:
        render_metric_card("Math Primitives", "11", "Pure NumPy functions", color_class="neutral")
    with c4:
        render_metric_card("OU Projection", "90d", "Forward reversion path", color_class="neutral")
    with c5:
        render_metric_card("Analog Returns", "4", "5 · 20 · 60 · 90 day", color_class="neutral")

    section_gap()

    # ── Awaiting-data prompt ────────────────────────────────────────
    render_interpretation_card(
        title="What has been measured",
        body=(
            "On this sheet — NIFTY, 2006–2026 — the Mood Score ranks forward returns on a "
            "held-out window (2021–2026, never used to build or select anything) with mean "
            "Spearman <strong>rho +0.54</strong>, <strong>p = 0.005</strong> against 200 "
            "circularly shifted copies of itself. The edge is real and it is measured, not "
            "asserted.<br><br>"
            "<strong>Where it comes from.</strong> The negated PE ratio alone — no engine at "
            "all — scores <strong>+0.53</strong> on the same window. Most of the edge belongs "
            "to the valuation anchor. The five-layer pipeline contributes a bounded, "
            "comparable score and its diagnostics rather than additional forecasting power. "
            "Read the Signal Validation view for the full measurement."
        ),
        color="info",
    )

    section_gap()

    render_landing_prompt(
        title="Awaiting Run",
        body_html=(
            "Click <strong>Run Analysis</strong> in the sidebar to fetch live data from Google Sheets "
            "and execute the full 5-layer sentiment pipeline. Once loaded, switch between "
            "<strong>Historical Mood</strong>, <strong>Similar Periods</strong>, "
            "<strong>Correlation Analysis</strong>, and <strong>Signal Validation</strong> "
            "views — or tune the active predictor set in <strong>Model Configuration</strong>."
        ),
    )
