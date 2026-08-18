"""
Arthagati — Analogs: what happened last time the state looked like this.

Mahalanobis distance over the state the engine measures, plus trajectory shape,
under a minimum-separation window so the returned analogs are distinct episodes
rather than adjacent days of one episode. The output is an empirical base rate
that does not depend on the engine being right — it is the one read on the app
that is independent of the pipeline.

Reading order:

  1 ANCHOR   what is the base rate?        Forward-return distribution
  2 STATE    which episodes produced it?   The analog cards
  3 DETAIL   does it hold on everything?   Mood vs forward return, split
"""

from __future__ import annotations

import html as html_mod

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui import format as fmt
from ui.components import (
    MIN_PRECEDENT_N,
    render_chart_panel,
    render_empty_state,
    render_interpretation_card,
    render_kpi_strip,
    render_note,
    render_section_header,
    render_table_panel,
)
from ui.theme import chart_color, chart_layout, chart_rgba, grid_rgba, style_axes

#: Analog cards rendered. The base rate is computed over EVERY returned
#: episode, not just these — see the section description.
MAX_ANALOG_CARDS = 10

_HORIZONS = ((5, "fwd_5d"), (20, "fwd_20d"), (60, "fwd_60d"), (90, "fwd_90d"))


def _classify(mood: float) -> tuple[str, str, str, str]:
    """Map the mood at T to (tier class, chip tone, label, bar fill class).

    Describes WHERE valuation sat, not what happened next; the forward tiles
    carry the realised outcome.
    """
    if mood >= 45:
        return "tier-strong-buy", "success", "Deep Value", "fill-strong-buy"
    if mood >= 20:
        return "tier-buy", "success", "Cheap", "fill-buy"
    if mood <= -45:
        return "tier-caution", "danger", "Very Rich", "fill-caution"
    if mood <= -20:
        return "tier-caution", "danger", "Rich", "fill-caution"
    return "tier-hold", "neutral", "Fair", "fill-hold"


def _fwd_tile(horizon: int, val: float | None) -> str:
    if val is None:
        return (f'<div class="analog-fwd-tile neutral">'
                f'<span class="analog-fwd-label">+{horizon}D</span>'
                f'<span class="analog-fwd-value">—</span></div>')
    cls = "pos" if val > 0 else "neg" if val < 0 else "flat"
    return (f'<div class="analog-fwd-tile {cls}">'
            f'<span class="analog-fwd-label">+{horizon}D</span>'
            f'<span class="analog-fwd-value">{val:+.1f}%</span></div>')


def _render_analog_card(period: dict) -> None:
    """One analog episode.

    NOTE on layout: the f-string below has NO blank lines and starts at column
    0. Both rules are load-bearing — Streamlit feeds the string to a CommonMark
    parser, a blank line inside an HTML block CLOSES it, and the opening tag
    needs at most three spaces of indent to be recognised as an HTML block at
    all. If you reformat this, keep the first line flush-left.
    """
    mood = float(period["mood_score"])
    sim = float(period["similarity"]) * 100.0
    tier, tone, label, fill = _classify(mood)
    tiles = "".join(_fwd_tile(h, period.get(k)) for h, k in _HORIZONS)
    mood_cls = "pos" if mood > 0 else "neg" if mood < 0 else ""
    st.markdown(
        f"""\
<div class="analog-card {tier}">
  <div class="analog-card-head">
    <div class="analog-card-id">
      <div class="analog-eyebrow">Analog · historical match</div>
      <div class="analog-symbol">{html_mod.escape(str(period['date']))}</div>
    </div>
    <span class="chip chip-{tone}">{label}</span>
  </div>
  <div class="analog-stat-row">
    <div class="analog-stat">
      <span class="analog-stat-label">Similarity</span>
      <span class="analog-stat-value accent">{sim:.1f}%</span>
    </div>
    <div class="analog-stat">
      <span class="analog-stat-label">Mood at T</span>
      <span class="analog-stat-value {mood_cls}">{mood:+.1f}</span>
    </div>
    <div class="analog-stat">
      <span class="analog-stat-label">NIFTY at T</span>
      <span class="analog-stat-value">{period['nifty']:,.0f}</span>
    </div>
  </div>
  <div class="analog-fwd-block">
    <div class="analog-fwd-block-label">Forward NIFTY return</div>
    <div class="analog-fwd-grid">{tiles}</div>
  </div>
  <div class="analog-card-foot">
    <span class="analog-foot-label">Similarity</span>
    <div class="conviction-bar"><div class="conviction-bar-fill {fill}" style="width:{sim:.0f}%;"></div></div>
    <span class="analog-foot-pct">{sim:.0f}%</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def _distribution(periods: list[dict]) -> pd.DataFrame:
    """The sample's actual shape, per horizon.

    A median alone hides whether the analogs agree. Worst, best and the hit
    rate are what tell the reader that a +4% median across ten overlapping
    windows is a wide, weak distribution rather than a forecast.
    """
    rows = []
    for h, key in _HORIZONS:
        vals = [p[key] for p in periods if p.get(key) is not None]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        rows.append({
            "Horizon": f"+{h}D",
            "Median": float(np.median(arr)),
            "Mean": float(arr.mean()),
            "Worst": float(arr.min()),
            "Best": float(arr.max()),
            "Sigma": float(arr.std()),
            "Hit rate": float((arr > 0).mean() * 100),
            "N": int(len(arr)),
        })
    return pd.DataFrame(rows)


def _backtest(mood_df, horizon: int):
    """Chronological 70/30 split with a one-horizon embargo.

    The maths is unchanged from the view this replaces: warm-up rows are
    excluded because they carry borrowed correlation statistics, and the
    embargo stops the last `horizon` training points drawing their labels from
    inside the test window. Only the presentation changed — the four
    coefficients moved out of the legend strings, where they could not be
    aligned or compared, into a table beside the chart.
    """
    from scipy.stats import spearmanr

    n = len(mood_df)
    if n <= horizon + 10:
        return None
    mood = mood_df["Mood_Score"].to_numpy()[: n - horizon]
    nifty = mood_df["NIFTY"].to_numpy()
    fwd = (nifty[horizon:] / nifty[: n - horizon] - 1) * 100
    valid = np.isfinite(mood) & np.isfinite(fwd)
    if "Is_Warmup" in mood_df.columns:
        valid &= ~mood_df["Is_Warmup"].to_numpy(dtype=bool)[: n - horizon]
    m, r = mood[valid], fwd[valid]
    if len(m) <= 20:
        return None

    split = int(len(m) * 0.7)
    train_m, train_r = m[:split], r[:split]
    test_start = min(split + horizon, len(m))
    test_m, test_r = m[test_start:], r[test_start:]
    if len(test_m) < 20:
        return None

    def _p(a, b):
        return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else float("nan")

    def _s(a, b):
        if len(a) <= 2:
            return float("nan")
        v = spearmanr(a, b)[0]
        return float(v) if np.isfinite(v) else float("nan")

    stats = pd.DataFrame([
        {"Split": "Train", "Rows": len(train_m),
         "Pearson r": _p(train_m, train_r), "Spearman rho": _s(train_m, train_r)},
        {"Split": "Test", "Rows": len(test_m),
         "Pearson r": _p(test_m, test_r), "Spearman rho": _s(test_m, test_r)},
    ])

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=train_m, y=train_r, mode="markers", name="Train",
        marker=dict(size=3.5, color=chart_rgba("slate", 0.35), line=dict(width=0)),
        hovertemplate="Mood %{x:.1f} → %{y:+.1f}%<extra>Train</extra>"))
    fig.add_trace(go.Scattergl(
        x=test_m, y=test_r, mode="markers", name="Test (post-embargo)",
        marker=dict(size=5, color=chart_rgba("accent", 0.80), symbol="diamond",
                    line=dict(width=0)),
        hovertemplate="Mood %{x:.1f} → %{y:+.1f}%<extra>Test</extra>"))
    if len(train_m) > 10:
        xs = np.linspace(m.min(), m.max(), 60)
        z1 = np.polyfit(train_m, train_r, 1)
        fig.add_trace(go.Scatter(x=xs, y=z1[0] * xs + z1[1], mode="lines",
                                 name="Linear fit", hoverinfo="skip",
                                 line=dict(color=chart_color("accent"), width=1.5)))
        z2 = np.polyfit(train_m, train_r, 2)
        fig.add_trace(go.Scatter(x=xs, y=z2[0] * xs ** 2 + z2[1] * xs + z2[2],
                                 mode="lines", name="Quadratic fit", hoverinfo="skip",
                                 line=dict(color=chart_color("amber"), width=1.2, dash="dot")))
    fig.add_hline(y=0, line_color=grid_rgba(0.11), line_width=1)
    fig.add_vline(x=0, line_color=grid_rgba(0.11), line_width=1)
    fig.update_layout(**chart_layout(height=420, show_legend=True))
    fig.update_layout(hovermode="closest")
    style_axes(fig, y_title=f"NIFTY return T+{horizon}d (%)", x_title="Mood Score at T")
    return fig, stats


def render(mood_df, *, periods, backtest_horizon) -> None:
    if not periods:
        render_empty_state(
            "Not enough history to match against",
            "The matcher needs a full covariance estimate plus the trajectory window "
            "before it can return distinct episodes. Load a longer series in the "
            "source sheet.",
            eyebrow="No analogs",
        )
        return

    sep = periods[0].get("separation_days", 0)
    dist = _distribution(periods)

    # ── 1 · ANCHOR ────────────────────────────────────────────────────────
    render_section_header(
        "Forward-Return Base Rate",
        f"Across every one of the {len(periods)} separated episodes the matcher "
        f"returned — not just the cards below. Minimum separation {sep} trading days, "
        "so each is a distinct event rather than a run of consecutive days.",
        icon="target",
    )
    if not dist.empty:
        render_kpi_strip([
            {"label": f"+{h}D median",
             "value": fmt.pct(row["Median"]),
             "subtext": f"{row['Hit rate']:.0f}% positive · n={row['N']}",
             "color_class": "success" if row["Median"] > 0 else "danger",
             "icon": "trending-up" if row["Median"] > 0 else "trending-down"}
            for (h, _), (_, row) in zip(_HORIZONS, dist.iterrows())
        ], max_cols=4, key="kpi-strip")
        render_table_panel(
            dist, key="an-distribution", label_col="Horizon",
            context=f"{len(periods)} separated analogs",
            sign_color_cols={"Median", "Mean", "Worst", "Best"},
            col_precision={"Median": 1, "Mean": 1, "Worst": 1, "Best": 1,
                           "Sigma": 1, "Hit rate": 0, "N": 0},
            max_height=200,
        )

    render_interpretation_card(
        "Context, not probability",
        f"The matcher returns <strong>{len(periods)}</strong> episodes separated by at "
        f"least <strong>{sep}</strong> trading days, so each one is a distinct event. "
        "Even so, the forward windows overlap the rest of the history: a +90D return "
        "measured from ten different starting points inside one bull market is close "
        "to one observation, not ten. Read the table as a set of precedents to open "
        f"and examine, not as a distribution to draw a probability from. Below "
        f"<strong>{MIN_PRECEDENT_N}</strong> distinct episodes the conviction chain "
        "treats this base rate as unusable for exactly that reason.",
        color="warning",
    )

    # ── 2 · STATE ─────────────────────────────────────────────────────────
    render_section_header(
        "Matched Episodes",
        f"The {min(MAX_ANALOG_CARDS, len(periods))} closest matches by composite "
        "similarity, each with the NIFTY return that followed it.",
        icon="layers",
        accent="emerald",
    )
    cols = st.columns(2, gap="small")
    for i, period in enumerate(periods[:MAX_ANALOG_CARDS]):
        with cols[i % 2]:
            _render_analog_card(period)
    render_note("Similarity is Mahalanobis distance (55%), trajectory cosine (35%) "
                "and recency (10%). Forward returns are NIFTY close-to-close and are "
                "not adjusted for dividends.")

    # ── 3 · DETAIL ────────────────────────────────────────────────────────
    render_section_header(
        "State vs Forward Return",
        "Every observation in the history, split chronologically with a one-horizon "
        "embargo. This is a shape check on the whole sample, not a significance "
        "test — Validation carries the held-out measurement.",
        icon="chart",
        accent="rose",
    )
    result = _backtest(mood_df, backtest_horizon)
    if result is None:
        render_empty_state(
            "Not enough post-embargo observations",
            f"The split needs more than {backtest_horizon + 30} non-warm-up rows, with "
            "at least 20 remaining in the test half after the embargo.",
            eyebrow="Backtest unavailable",
        )
        return
    fig, stats = result
    render_chart_panel(
        fig, key="an-backtest",
        units=f"+{backtest_horizon}D forward",
        context=f"Full history · train 70% · embargo {backtest_horizon}d · test 30%",
        footer=f"Consecutive days share almost all of their forward window: a "
               f"{backtest_horizon}-day return overlaps the next {backtest_horizon - 1} "
               "days'. The effective number of independent observations is far smaller "
               "than the dot count, so the coefficients look more certain than they are.",
    )
    render_table_panel(
        stats, key="an-backtest-stats", label_col="Split",
        context="Fits are estimated on train only and drawn across the full range",
        sign_color_cols={"Pearson r", "Spearman rho"},
        col_precision={"Pearson r": 3, "Spearman rho": 3, "Rows": 0},
        max_height=140,
    )
