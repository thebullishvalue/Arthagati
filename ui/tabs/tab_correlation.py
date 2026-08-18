"""
Arthagati — Correlation & Predictor Analysis view.

Decay-weighted Spearman vs PE/EY anchors + entropy quality ranking.
Cards use the Obsidian Quant ``position-card`` system with tier accents.
"""

from __future__ import annotations

import html as html_mod

import pandas as pd
import streamlit as st

from ui.components import (
    render_section_header,
    render_warning_box,
    render_interpretation_card,
    section_divider,
    section_gap,
    vertical_divider,
)


# ── Correlation card ────────────────────────────────────────────────────────

def _corr_tier(corr_val: float) -> tuple[str, str, str]:
    """Map an absolute-correlation magnitude to (tier_cls, fill_cls, label)."""
    abs_v = abs(corr_val)
    direction = "Positive" if corr_val > 0 else "Negative"
    if abs_v >= 0.5:
        if corr_val > 0:
            return "tier-strong-buy", "fill-strong-buy", f"Strong {direction}"
        return "tier-caution",    "fill-caution",    f"Strong {direction}"
    if abs_v >= 0.3:
        if corr_val > 0:
            return "tier-buy",     "fill-buy",        f"Moderate {direction}"
        return "tier-caution",    "fill-caution",    f"Moderate {direction}"
    return "tier-hold",            "fill-hold",       f"Weak {direction}"


def _render_corr_card(variable: str, corr_val: float) -> None:
    """One correlation entry rendered as a proper Obsidian Quant card."""
    tier_cls, fill_cls, tier_label = _corr_tier(corr_val)
    bar_pct = min(abs(corr_val) * 100, 100)
    value_cls = "pos" if corr_val > 0 else "neg"

    # See _render_period_card for the rationale behind flush-left + no
    # blank lines in this f-string.
    st.markdown(
        f"""\
<div class="position-card corr-card {tier_cls}">
  <div class="corr-card-row">
    <div class="corr-card-var">{html_mod.escape(variable)}</div>
    <div class="corr-card-val {value_cls}">{corr_val:+.2f}</div>
  </div>
  <div class="conviction-bar corr-card-bar">
    <div class="conviction-bar-fill {fill_cls}" style="width:{bar_pct:.0f}%;"></div>
  </div>
  <div class="corr-card-tier">{tier_label}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def _render_corr_grid(corrs: pd.DataFrame) -> None:
    """Render a list of correlations as a 2-column card grid."""
    if corrs is None or corrs.empty:
        st.caption("No correlations computed. Check data source.")
        return
    corrs_display = corrs.sort_values("correlation", key=abs, ascending=False)
    rows = list(corrs_display.iterrows())
    cols = st.columns(2, gap="medium")
    for i, (_, r) in enumerate(rows):
        with cols[i % 2]:
            _render_corr_card(r["variable"], r["correlation"])
            st.markdown('<div style="height: var(--sp-2);"></div>', unsafe_allow_html=True)


# ── Quality card ────────────────────────────────────────────────────────────

def _qual_tier(row: dict, max_quality: float) -> tuple[str, str, str, str]:
    """Map a quality row → (tier_cls, fill_cls, badge_cls, label)."""
    if row["quality"] >= max_quality * 0.5 and row["coverage"] > 50:
        return "tier-strong-buy", "fill-strong-buy", "badge-strong-buy", "Keep"
    if row["quality"] >= max_quality * 0.2 and row["coverage"] > 30:
        return "tier-hold",        "fill-hold",        "badge-hold",      "Useful"
    if row["coverage"] < 10:
        return "tier-caution",     "fill-caution",     "badge-caution",   "No Data"
    return "tier-caution",         "fill-caution",     "badge-caution",   "Weak"


def _render_qual_card(rank: int, row: dict, max_quality: float) -> None:
    """One predictor-quality entry as a card."""
    tier_cls, fill_cls, badge_cls, badge_label = _qual_tier(row, max_quality)
    bar_pct = (row["quality"] / max_quality) * 100 if max_quality else 0
    active_state = "active" if row["active"] else "inactive"
    active_dot = "●" if row["active"] else "○"
    active_label = "Active" if row["active"] else "Inactive"

    # Flush-left + no blank lines — see _render_period_card.
    st.markdown(
        f"""\
<div class="position-card qual-card {tier_cls}">
  <div class="qual-card-head">
    <div class="qual-card-id">
      <span class="qual-card-rank">{rank:02d}</span>
      <span class="qual-card-var">{html_mod.escape(row['variable'])}</span>
    </div>
    <span class="position-card-badge {badge_cls}">{badge_label}</span>
  </div>
  <div class="conviction-bar qual-card-bar">
    <div class="conviction-bar-fill {fill_cls}" style="width:{bar_pct:.0f}%;"></div>
  </div>
  <div class="qual-card-stats">
    <div class="qual-stat">
      <span class="qual-stat-label">|ρ|</span>
      <span class="qual-stat-value">{row['avg_corr']:.2f}</span>
    </div>
    <div class="qual-stat">
      <span class="qual-stat-label">H</span>
      <span class="qual-stat-value">{row['entropy']:.2f}</span>
    </div>
    <div class="qual-stat">
      <span class="qual-stat-label">Coverage</span>
      <span class="qual-stat-value">{row['coverage']:.0f}%</span>
    </div>
    <div class="qual-card-active {active_state}">{active_dot} {active_label}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


# ── Main view ───────────────────────────────────────────────────────────────

def render(
    raw_df,
    *,
    active_preds,
    non_predictor_cols,
    calculate_anchor_correlations,
    shannon_entropy,
) -> None:
    """Render the Correlation & Predictor Analysis view."""

    render_section_header(
        title="Correlation & Predictor Analysis",
        description="Decay-weighted Spearman correlations vs PE & EY anchors · entropy-weighted predictor quality",
        icon="file-text",
        accent="cyan",
    )

    # ── Anchor health diagnostic ─────────────────────────────────────────
    anchors = {"NIFTY50_PE": "PE Ratio", "NIFTY50_EY": "Earnings Yield"}
    anchor_health: dict[str, dict] = {}
    for col, label in anchors.items():
        if col in raw_df.columns:
            nunique = raw_df[col].nunique()
            has_variance = nunique > 3 and raw_df[col].std() > 1e-6
            anchor_health[col] = {"label": label, "ok": has_variance, "nunique": nunique}
        else:
            anchor_health[col] = {"label": label, "ok": False, "nunique": 0}

    bad_anchors = [v["label"] for v in anchor_health.values() if not v["ok"]]
    if bad_anchors:
        render_warning_box(
            title="Data Quality Issue",
            content=(
                f"{', '.join(bad_anchors)} has insufficient variance in the source data. "
                "If Earnings Yield is empty in the sheet, it is auto-derived from PE (1/PE × 100). "
                "Check that your Google Sheet has valid data for these columns."
            ),
        )

    # ── PE & EY Correlations — side-by-side sections, each with a 2-col
    #    nested card grid (cards = ~half-of-half width). A thin amber-tinted
    #    vertical divider sits between them to break up the dense grid.
    section_divider()
    pe_col, mid_col, ey_col = st.columns([10, 1, 10], gap="small")

    with mid_col:
        vertical_divider()

    with pe_col:
        render_section_header(
            title="PE Ratio Correlations",
            description="Variables ranked by |ρ| with NIFTY50_PE",
            icon="chart",
            accent="cyan",
        )
        if not anchor_health.get("NIFTY50_PE", {}).get("ok", False):
            st.caption("NIFTY50_PE has insufficient data variance — correlations may be unreliable.")
        pe_corrs = calculate_anchor_correlations(raw_df, "NIFTY50_PE", active_preds)
        _render_corr_grid(pe_corrs)

    with ey_col:
        render_section_header(
            title="Earnings Yield Correlations",
            description="Variables ranked by |ρ| with NIFTY50_EY",
            icon="bar-chart",
            accent="emerald",
        )
        if not anchor_health.get("NIFTY50_EY", {}).get("ok", False):
            st.caption("NIFTY50_EY has insufficient data variance — correlations may be unreliable.")
        ey_corrs = calculate_anchor_correlations(raw_df, "NIFTY50_EY", active_preds)
        _render_corr_grid(ey_corrs)

    # ── Predictor Quality Assessment (full width, 2-col card grid) ───────
    section_divider()
    render_section_header(
        title="Predictor Quality Assessment",
        description="Quality = |ρ| × (1 − entropy) — the same shape as the engine's weighting, measured full-sample",
        icon="target",
        accent="violet",
    )

    # Quality is computed over EVERY numeric column, not just the active set.
    #
    # The correlation frames above are built from `active_preds`, and this
    # loop used to read its |rho| out of them — so an inactive predictor
    # always scored 0.00 and was badged "Weak". A panel whose stated job is
    # to guide predictor selection could therefore never recommend anything
    # the user had not already selected. Correlations for the full column set
    # are computed once here and reused.
    all_vars = [
        c for c in raw_df.columns
        if c not in non_predictor_cols and pd.api.types.is_numeric_dtype(raw_df[c])
    ]
    pe_all = calculate_anchor_correlations(raw_df, "NIFTY50_PE", all_vars)
    ey_all = calculate_anchor_correlations(raw_df, "NIFTY50_EY", all_vars)
    pe_lookup = dict(zip(pe_all["variable"], pe_all["correlation"])) if not pe_all.empty else {}
    ey_lookup = dict(zip(ey_all["variable"], ey_all["correlation"])) if not ey_all.empty else {}

    quality_rows = []
    for var in all_vars:
        pe_corr = abs(float(pe_lookup.get(var, 0.0)))
        ey_corr = abs(float(ey_lookup.get(var, 0.0)))
        avg_corr = (pe_corr + ey_corr) / 2

        # First differences, not pct_change. Several predictors cross zero
        # (both term spreads by construction, plus PE_DEV / EY_DEV), and a
        # denominator near zero produced changes of several hundred x, which
        # then dominated the histogram bin width and corrupted the entropy.
        # Entropy is estimated from an adaptive-width histogram, so it is
        # invariant to the units of the differenced series.
        var_returns = raw_df[var].diff().dropna().values
        entropy = shannon_entropy(var_returns) if len(var_returns) > 10 else 0.5
        info_quality = 1.0 - entropy
        quality_score = avg_corr * max(info_quality, 0.1)
        # Coverage is the share of rows carrying a real, non-zero value.
        # `raw_df[var] != 0` counts NaN as True in pandas, so an all-NaN
        # column used to report 100% coverage.
        col = raw_df[var]
        non_zero_pct = float((col.notna() & (col != 0)).mean() * 100)
        quality_rows.append({
            "variable": var,
            "pe_corr": pe_corr,
            "ey_corr": ey_corr,
            "avg_corr": avg_corr,
            "entropy": entropy,
            "quality": quality_score,
            "coverage": non_zero_pct,
            "active": var in active_preds,
        })

    quality_rows.sort(key=lambda x: x["quality"], reverse=True)
    if not quality_rows:
        return

    max_quality = max(r["quality"] for r in quality_rows) or 1.0
    qcols = st.columns(2, gap="medium")
    for i, row in enumerate(quality_rows):
        with qcols[i % 2]:
            _render_qual_card(i + 1, row, max_quality)
            st.markdown('<div style="height: var(--sp-2);"></div>', unsafe_allow_html=True)

    # ── Summary interpretation ───────────────────────────────────────────
    keep_count = sum(1 for r in quality_rows if r["quality"] >= max_quality * 0.5 and r["coverage"] > 50)
    useful_count = sum(
        1 for r in quality_rows
        if max_quality * 0.2 <= r["quality"] < max_quality * 0.5 and r["coverage"] > 30
    )
    weak_count = len(quality_rows) - keep_count - useful_count

    section_gap()
    summary_body = (
        f"<strong style='color:var(--emerald);'>{keep_count} strong</strong> predictors "
        f"(high correlation × low entropy) · "
        f"<strong style='color:var(--amber);'>{useful_count} useful</strong> (moderate signal) · "
        f"<strong style='color:var(--ink-tertiary);'>{weak_count} weak</strong> (low signal or noisy).<br><br>"
        "<span style='font-size:0.72rem; color:var(--ink-tertiary);'>"
        "|ρ| = average |correlation| with PE &amp; EY anchors · "
        "H = Shannon entropy of returns (lower = more structured) · "
        "Quality = |ρ| × (1−H). The engine applies this same shape per anchor, but on "
        "<em>walk-forward</em> correlations blended across quarterly checkpoints, so its "
        "internal weights differ in magnitude from the full-sample figures shown here. "
        "Read this as a ranking, not as the engine's live coefficients."
        "</span>"
    )
    render_interpretation_card(
        title="Recommendation Summary",
        body=summary_body,
        color="info",
    )
