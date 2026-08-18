"""
Arthagati — Similar Periods view (analog matching + forward returns + backtest).
"""

from __future__ import annotations

import html as html_mod

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ui.components import (
    render_section_header,
    render_metric_card,
    render_interpretation_card,
    section_divider,
)
from ui.theme import (
    C_AMBER,
    C_CYAN,
    C_EMERALD,
    C_ROSE,
    C_MUTED,
    PLOTLY_BASE,
    PLOTLY_GRID,
    PLOTLY_GRID_ZERO,
)


def _classify_mood(mood_val: float) -> tuple[str, str, str, str]:
    """Map a mood score to (tier_class, badge_class, badge_label, bar_fill_class)."""
    if mood_val >= 40:
        return "tier-strong-buy", "badge-strong-buy", "Strong Bull", "fill-strong-buy"
    if mood_val >= 15:
        return "tier-buy",        "badge-buy",        "Bullish",     "fill-buy"
    if mood_val <= -40:
        return "tier-caution",    "badge-caution",    "Strong Bear", "fill-caution"
    if mood_val <= -15:
        return "tier-caution",    "badge-caution",    "Bearish",     "fill-caution"
    return     "tier-hold",       "badge-hold",       "Neutral",     "fill-hold"


def _render_fwd_tile(horizon: int, val: float | None) -> str:
    """Render one forward-return tile in the analog card's footer grid."""
    if val is None:
        return (
            f'<div class="analog-fwd-tile neutral">'
            f'<span class="analog-fwd-label">+{horizon}D</span>'
            f'<span class="analog-fwd-value">—</span>'
            f"</div>"
        )
    cls = "pos" if val > 0 else "neg"
    return (
        f'<div class="analog-fwd-tile {cls}">'
        f'<span class="analog-fwd-label">+{horizon}D</span>'
        f'<span class="analog-fwd-value">{val:+.1f}%</span>'
        f"</div>"
    )


def _render_period_card(period: dict) -> None:
    """Render one analog-period card — Obsidian Quant fidelity.

    Anatomy:
      • Eyebrow + date (symbol) + tier badge
      • Hero stat row: Similarity · Mood · NIFTY (3-col grid)
      • Forward NIFTY Return tile group (3-col, signed-coloured)
      • Footer: similarity progress bar
    """
    mood_val       = period["mood_score"]
    similarity_pct = period["similarity"] * 100
    nifty_val      = period["nifty"]
    tier_cls, badge_cls, badge_label, bar_cls = _classify_mood(mood_val)

    fwd_tiles = "".join(
        _render_fwd_tile(h, period.get(k))
        for h, k in [(5, "fwd_5d"), (20, "fwd_20d"), (60, "fwd_60d"), (90, "fwd_90d")]
    )

    mood_color = "pos" if mood_val > 0 else "neg" if mood_val < 0 else "neutral"

    # NOTE on layout: the multi-line f-string below has NO blank lines and
    # starts at column 0 (`f"""\`). Both rules are load-bearing:
    #
    #   • Streamlit feeds the string to a CommonMark parser. A blank line
    #     inside an HTML block CLOSES the block; subsequent indented HTML
    #     then renders as raw text (indented code block).
    #   • The opening tag must have ≤3 spaces of leading indent to be
    #     recognised as an HTML block at all.
    #
    # If you reformat this, keep the first line flush-left and avoid blank
    # lines between sub-elements.
    st.markdown(
        f"""\
<div class="position-card analog-card {tier_cls}">
  <div class="analog-card-head">
    <div class="analog-card-id">
      <div class="analog-eyebrow">Analog · Historical Match</div>
      <div class="analog-symbol">{html_mod.escape(period['date'])}</div>
    </div>
    <span class="position-card-badge {badge_cls}">{badge_label}</span>
  </div>
  <div class="analog-stat-row">
    <div class="analog-stat">
      <span class="analog-stat-label">Similarity</span>
      <span class="analog-stat-value amber">{similarity_pct:.1f}%</span>
    </div>
    <div class="analog-stat">
      <span class="analog-stat-label">Mood at T</span>
      <span class="analog-stat-value {mood_color}">{mood_val:+.1f}</span>
    </div>
    <div class="analog-stat">
      <span class="analog-stat-label">NIFTY at T</span>
      <span class="analog-stat-value">{nifty_val:,.0f}</span>
    </div>
  </div>
  <div class="analog-fwd-block">
    <div class="analog-fwd-block-label">Forward NIFTY Return</div>
    <div class="analog-fwd-grid">{fwd_tiles}</div>
  </div>
  <div class="analog-card-foot">
    <span class="analog-foot-label">Similarity</span>
    <div class="conviction-bar">
      <div class="conviction-bar-fill {bar_cls}" style="width:{similarity_pct:.0f}%;"></div>
    </div>
    <span class="analog-foot-pct">{similarity_pct:.0f}%</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render(mood_df, *, find_similar_periods, backtest_horizon) -> None:
    """Render Similar Periods view — analog cards + forward-return summary + backtest."""

    render_section_header(
        title="Similar Historical Periods",
        description="Mahalanobis + trajectory matching · forward NIFTY returns from each analog",
        icon="search",
        accent="emerald",
    )

    similar_periods = find_similar_periods(mood_df)
    if not similar_periods:
        st.warning("Not enough historical data to find similar periods.")
        return

    # ── Forward return summary cards ─────────────────────────────────────
    fwd_5  = [p["fwd_5d"]  for p in similar_periods if p["fwd_5d"]  is not None]
    fwd_20 = [p["fwd_20d"] for p in similar_periods if p["fwd_20d"] is not None]
    fwd_60 = [p["fwd_60d"] for p in similar_periods if p["fwd_60d"] is not None]
    fwd_90 = [p["fwd_90d"] for p in similar_periods if p["fwd_90d"] is not None]

    if fwd_5 or fwd_20 or fwd_60 or fwd_90:
        cols = st.columns(4, gap="small")
        for col, horizon, values in zip(
            cols, [5, 20, 60, 90], [fwd_5, fwd_20, fwd_60, fwd_90],
        ):
            if not values:
                continue
            median_ret = np.median(values)
            positive_pct = sum(1 for v in values if v > 0) / len(values) * 100
            with col:
                render_metric_card(
                    label=f"+{horizon}D Median Return",
                    value=f"{median_ret:+.1f}%",
                    subtext=f"{positive_pct:.0f}% positive · {len(values)} separated analogs",
                    color_class="success" if median_ret > 0 else "danger",
                    icon="trending-up" if median_ret > 0 else "trending-down",
                )

        sep = similar_periods[0].get("separation_days", 0)
        st.caption(
            f"Analogs are selected with a minimum separation of {sep} trading days, so "
            "each one represents a distinct episode rather than a run of consecutive "
            f"days from the same event. Even so, {len(similar_periods)} episodes is a small "
            "sample and the forward windows overlap the rest of the history — read these "
            "as context, not as a probability."
        )

    section_divider()

    # ── Analog period cards (2-column grid) ──────────────────────────────
    render_section_header(
        title="Top Analog Periods",
        description="Top 10 historical matches by similarity score",
        icon="layers",
    )

    analog_cols = st.columns(2, gap="medium")
    for i, period in enumerate(similar_periods[:10]):
        with analog_cols[i % 2]:
            _render_period_card(period)
            # Extra inter-row breathing room — the position-card animation has
            # a translateX(-12) entry; pair it with vertical rhythm.
            st.markdown('<div style="height: var(--sp-3);"></div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════
    # BACKTEST SANITY CHECK
    # ═══════════════════════════════════════════════════════════════════════
    section_divider()
    render_section_header(
        title="Backtest · Mood Score vs Forward NIFTY Return",
        description="Each dot = one historical day · pattern indicates predictive relationship",
        icon="chart",
        accent="rose",
    )

    render_interpretation_card(
        title="How to read this scatter",
        body=(
            "Each dot is one trading day, and consecutive days share almost all of their "
            f"forward window — a {backtest_horizon}-day return overlaps the next "
            f"{backtest_horizon - 1} days' returns. The effective number of independent "
            "observations is therefore far smaller than the dot count, and the correlation "
            "coefficients below are correspondingly <strong>more certain-looking than they "
            "are</strong>. The train/test split is chronological with a one-horizon embargo, "
            "but no correction is applied for the overlap. Treat this as "
            "<strong>descriptive</strong>."
        ),
        color="warning",
    )

    n = len(mood_df)
    horizon = backtest_horizon
    if n <= horizon + 10:
        st.caption("Insufficient data points for backtest.")
        return

    bt_mood  = mood_df["Mood_Score"].values[: n - horizon]
    bt_nifty = mood_df["NIFTY"].values
    bt_fwd   = (bt_nifty[horizon:] / bt_nifty[: n - horizon] - 1) * 100

    valid = np.isfinite(bt_mood) & np.isfinite(bt_fwd)
    # Warm-up rows carry borrowed correlation statistics — exclude them.
    if "Is_Warmup" in mood_df.columns:
        valid &= ~mood_df["Is_Warmup"].to_numpy(dtype=bool)[: n - horizon]
    bt_mood_clean = bt_mood[valid]
    bt_fwd_clean  = bt_fwd[valid]

    if len(bt_mood_clean) <= 20:
        st.caption("Insufficient data points for backtest.")
        return

    from scipy.stats import spearmanr as _spearmanr

    # 70/30 chronological split with an embargo of one full horizon between
    # the halves. Without it the last `horizon` training points draw their
    # labels from inside the test window, so the "out-of-sample" figure was
    # partly in-sample.
    split_idx = int(len(bt_mood_clean) * 0.7)
    train_m, train_r = bt_mood_clean[:split_idx], bt_fwd_clean[:split_idx]
    test_start = min(split_idx + horizon, len(bt_mood_clean))
    test_m,  test_r  = bt_mood_clean[test_start:], bt_fwd_clean[test_start:]

    if len(test_m) < 20:
        st.caption("Insufficient out-of-sample points after the embargo.")
        return

    bt_pearson  = np.corrcoef(train_m, train_r)[0, 1] if len(train_m) > 2 else 0
    bt_spearman, _ = _spearmanr(train_m, train_r)
    if not np.isfinite(bt_spearman):
        bt_spearman = 0.0
    oos_pearson = np.corrcoef(test_m, test_r)[0, 1] if len(test_m) > 2 else 0
    oos_spearman, _ = _spearmanr(test_m, test_r) if len(test_m) > 2 else (0.0, 1.0)
    if not np.isfinite(oos_spearman):
        oos_spearman = 0.0

    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scattergl(
        x=train_m, y=train_r, mode="markers",
        marker=dict(size=4, color=np.where(train_m > 0, C_EMERALD, C_ROSE), opacity=0.4),
        hovertemplate="Mood: %{x:.1f}<br>Forward return: %{y:.1f}%<extra></extra>",
        name=f"Train (70%, n={len(train_m)})",
    ))
    fig_bt.add_trace(go.Scattergl(
        x=test_m, y=test_r, mode="markers",
        marker=dict(size=6, color=np.where(test_m > 0, C_EMERALD, C_ROSE),
                    opacity=0.85, symbol="diamond"),
        hovertemplate="Mood: %{x:.1f}<br>Forward return: %{y:.1f}%<extra></extra>",
        name=f"Test (30%, n={len(test_m)})",
    ))

    if len(train_m) > 10:
        x_range = np.linspace(bt_mood_clean.min(), bt_mood_clean.max(), 50)
        z1 = np.polyfit(train_m, train_r, 1)
        fig_bt.add_trace(go.Scatter(
            x=x_range, y=z1[0] * x_range + z1[1],
            mode="lines", line=dict(color=C_AMBER, width=2, dash="dash"),
            name=f"Linear (train ρ={bt_pearson:.2f}, test ρ={oos_pearson:.2f})",
        ))
        z2 = np.polyfit(train_m, train_r, 2)
        fig_bt.add_trace(go.Scatter(
            x=x_range, y=z2[0] * x_range ** 2 + z2[1] * x_range + z2[2],
            mode="lines", line=dict(color=C_CYAN, width=2, dash="dot"),
            name=f"Quadratic (train ρ_s={bt_spearman:.2f}, test ρ_s={oos_spearman:.2f})",
        ))

    fig_bt.add_hline(y=0, line_color="rgba(148,163,184,0.35)", line_width=1, line_dash="dot")
    fig_bt.add_vline(x=0, line_color="rgba(148,163,184,0.35)", line_width=1, line_dash="dot")

    fig_bt.update_layout(
        **PLOTLY_BASE,
        height=420,
        hovermode="closest",
        showlegend=True,
        margin=dict(l=60, r=20, t=20, b=50),
        xaxis=dict(
            title=dict(text="Mood Score at T",
                       font=dict(size=11, color=C_MUTED, family="JetBrains Mono, monospace")),
            showgrid=True, gridcolor=PLOTLY_GRID, gridwidth=0.5,
            zeroline=True, zerolinecolor=PLOTLY_GRID_ZERO,
            tickfont=dict(size=9, family="JetBrains Mono, monospace", color="#64748B"),
        ),
        yaxis=dict(
            title=dict(text=f"NIFTY Return T+{horizon}d (%)",
                       font=dict(size=11, color=C_MUTED, family="JetBrains Mono, monospace")),
            showgrid=True, gridcolor=PLOTLY_GRID, gridwidth=0.5,
            zeroline=True, zerolinecolor=PLOTLY_GRID_ZERO,
            tickfont=dict(size=9, family="JetBrains Mono, monospace", color="#64748B"),
        ),
        legend=dict(
            x=0.02, y=0.98,
            bgcolor="rgba(10,14,23,0.85)",
            bordercolor="rgba(255,255,255,0.06)", borderwidth=1,
            font=dict(size=10, family="JetBrains Mono, monospace"),
        ),
    )

    st.markdown('<div class="chart-container similar">', unsafe_allow_html=True)
    st.plotly_chart(fig_bt, use_container_width=True, config={"displayModeBar": False, "displaylogo": False})
    st.markdown("</div>", unsafe_allow_html=True)

    # Interpretation — driven by OOS results
    oos_stronger = oos_spearman if abs(oos_spearman) > abs(oos_pearson) else oos_pearson
    # The threshold is deliberately high. With overlapping forward windows the
    # effective sample is a small multiple of n/horizon, not n, so a coefficient
    # that would be decisive on independent data is not decisive here.
    if abs(oos_stronger) > 0.3:
        strength  = "strong" if abs(oos_stronger) > 0.5 else "moderate"
        direction = "positive" if oos_stronger > 0 else "negative"
        body = (
            f"<strong>Out-of-sample (30%):</strong> Pearson {oos_pearson:.2f} · Spearman {oos_spearman:.2f} — "
            f"{strength} {direction} association on the held-out window.<br>"
            f"<span style='color:var(--ink-tertiary);'>In-sample (70%): Pearson {bt_pearson:.2f} · "
            f"Spearman {bt_spearman:.2f}</span><br><br>"
            + (
                "Higher mood scores have historically been followed by positive NIFTY returns."
                if oos_stronger > 0 else
                "Higher mood scores have historically been followed by negative NIFTY returns "
                "(contrarian signal)."
            )
        )
        render_interpretation_card("Association Persists Out-of-Sample", body, color="success")
    else:
        body = (
            f"<strong>Out-of-sample (30%):</strong> Pearson {oos_pearson:.2f} · Spearman {oos_spearman:.2f} — "
            f"weak out-of-sample relationship at the {horizon}-day horizon.<br>"
            f"<span style='color:var(--ink-tertiary);'>In-sample (70%): Pearson {bt_pearson:.2f} · "
            f"Spearman {bt_spearman:.2f}</span><br><br>"
            "The mood score's predictive power may be non-linear (check the quadratic curve) "
            "or work better at different horizons."
        )
        render_interpretation_card("Weak Out-of-Sample Fit", body, color="warning")
