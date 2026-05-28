"""
Arthagati — Intelligence Center (read-only dashboard).

Five sections, all using the Obsidian Quant card system:
  1. Dataset strip (5 metric cards)
  2. Calibration Diagnostics (4 metric cards: Train IR · Val IR · Stability · Quality)
  3. Calibration Impact (NEW — shows what changes when Intelligence is ON):
     raw Mood vs Calibrated Conviction, per-horizon IR lift, top drivers
  4. Ensemble Weights — 2-column card grid (per-feature weight cards)
  5. Parameter Importance — 2-column card grid (fANOVA / weight-share)
  6. Profile metadata — stat-card grid

Import / Export / Reset controls live in the sidebar passport — this
view is intentionally read-only.
"""

from __future__ import annotations

import html as html_mod

import numpy as np
import pandas as pd
import streamlit as st

from ui.components import (
    render_section_header,
    render_metric_card,
    render_interpretation_card,
    section_divider,
    section_gap,
    vertical_divider,
)

import intelligence as intel


# ════════════════════════════════════════════════════════════════════════════
# Severity helpers
# ════════════════════════════════════════════════════════════════════════════

def _quality_severity(label: str) -> str:
    return {
        "Quality OK": "success",
        "Overfit":    "warning",
        "No Edge":    "danger",
    }.get(label, "neutral")


def _stability_severity(stability: float, train_ir: float) -> str:
    if train_ir <= 0:
        return "neutral"
    if 0.30 <= stability <= 1.30:
        return "info"
    return "warning"


def _val_ir_severity(val_ir: float) -> str:
    if val_ir > 0.05:
        return "success"
    if val_ir > 0:
        return "warning"
    return "danger"


# ════════════════════════════════════════════════════════════════════════════
# Feature Analysis — consolidated weight + importance per feature
# (Replaces the old "Ensemble Weights" + "Parameter Importance" sections —
#  one card per feature, two stacked bars, ranked by importance.)
# ════════════════════════════════════════════════════════════════════════════

# Human-readable labels for the engine-output features
_FEATURE_LABELS = {
    "mood":          "Mood Score (raw)",
    "mood_smooth":   "Smoothed Mood",
    "mood_diverge":  "Mood Divergence",
    "mood_squared":  "Mood² (amplified)",
    "mood_sqrt":     "√Mood (damped)",
    "msf_spread":    "MSF Spread",
    "msf_momentum":  "MSF · Momentum",
    "msf_structure": "MSF · Structure",
    "msf_regime":    "MSF · Regime",
    "msf_flow":      "MSF · Flow",
}


def _feature_card(
    rank: int,
    name: str,
    weight: float,
    importance: float,
    max_abs_w: float,
    max_imp: float,
) -> str:
    """One consolidated feature card. Tier strip + name + weight + two bars.

    Bars:
      • Weight bar  — directional (emerald/rose) showing magnitude of
        the linear coefficient relative to the largest |weight|
      • Importance bar — neutral violet→amber gradient showing fANOVA %
        relative to the most-explanatory feature
    """
    if weight >= 0.15:
        tier, badge_cls, badge_label = "tier-strong-buy", "badge-strong-buy", "Bullish+"
    elif weight >= 0.05:
        tier, badge_cls, badge_label = "tier-buy",        "badge-buy",        "Bullish"
    elif weight <= -0.15:
        tier, badge_cls, badge_label = "tier-caution",    "badge-caution",    "Bearish+"
    elif weight <= -0.05:
        tier, badge_cls, badge_label = "tier-caution",    "badge-caution",    "Bearish"
    else:
        tier, badge_cls, badge_label = "tier-hold",       "badge-hold",       "Neutral"

    val_cls = "pos" if weight > 0 else "neg" if weight < 0 else "neutral"
    w_bar_pct = (abs(weight) / max_abs_w * 100.0) if max_abs_w > 0 else 0.0
    i_bar_pct = (importance / max_imp * 100.0) if max_imp > 0 else 0.0
    label = _FEATURE_LABELS.get(name, name)

    return f"""\
<div class="position-card feature-card {tier}">
  <div class="feature-card-head">
    <span class="feature-rank">{rank:02d}</span>
    <div class="feature-card-id">
      <div class="feature-card-name">{html_mod.escape(label)}</div>
      <div class="feature-card-key">{html_mod.escape(name)}</div>
    </div>
    <span class="position-card-badge {badge_cls}">{badge_label}</span>
  </div>
  <div class="feature-card-value {val_cls}">{weight:+.3f}</div>
  <div class="feature-card-bar-row">
    <span class="feature-card-bar-label">Weight</span>
    <div class="feature-card-bar"><div class="feature-card-bar-fill {val_cls}" style="width:{w_bar_pct:.0f}%;"></div></div>
    <span class="feature-card-bar-pct">{w_bar_pct:.0f}%</span>
  </div>
  <div class="feature-card-bar-row">
    <span class="feature-card-bar-label">Importance</span>
    <div class="feature-card-bar"><div class="feature-card-bar-fill importance" style="width:{i_bar_pct:.0f}%;"></div></div>
    <span class="feature-card-bar-pct">{importance:.1f}%</span>
  </div>
</div>
"""


def _render_feature_grid(weights: dict, importance: dict, defaults: dict) -> None:
    """Render all features as a single CSS-grid of consolidated cards.

    Cards are ranked by importance (descending). All cards in the same
    grid row have equal height — that's what makes the layout feel
    cohesive instead of looking like 'things thrown on the page'.
    """
    feature_order = list(defaults.keys())
    if importance:
        feature_order = sorted(
            feature_order,
            key=lambda k: -importance.get(k, 0.0),
        )

    # Use a single CSS Grid container — guarantees equal-height rows and
    # stable layout across any combination of feature counts.
    values = [float(weights.get(k, 0.0)) for k in feature_order]
    max_abs_w = max((abs(v) for v in values), default=1.0) or 1.0
    imp_values = [float(importance.get(k, 0.0)) for k in feature_order]
    max_imp = max(imp_values, default=1.0) or 1.0

    cards_html = "".join(
        _feature_card(
            rank=i + 1,
            name=k,
            weight=float(weights.get(k, 0.0)),
            importance=float(importance.get(k, 0.0)),
            max_abs_w=max_abs_w,
            max_imp=max_imp,
        )
        for i, k in enumerate(feature_order)
    )
    st.markdown(
        f'<div class="feature-grid">{cards_html}</div>',
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# Profile metadata — stat-card grid
# ════════════════════════════════════════════════════════════════════════════

def _stat_card(label: str, value: str, sub: str = "") -> str:
    sub_html = (
        f'<div class="profile-stat-sub">{html_mod.escape(sub)}</div>'
        if sub else ""
    )
    return f"""\
<div class="profile-stat">
  <div class="profile-stat-label">{html_mod.escape(label)}</div>
  <div class="profile-stat-value">{html_mod.escape(value)}</div>
  {sub_html}
</div>
"""


def _render_profile_grid(profile: intel.CalibrationProfile) -> None:
    """Profile metadata as a 3-column grid of stat-cards."""
    age = intel.profile_age_days(profile)
    if age < 1.0:
        age_str = "today"
    elif age < 2.0:
        age_str = "yesterday"
    elif age < 30:
        age_str = f"{age:.0f}d ago"
    else:
        age_str = f"{age / 30:.1f}mo ago"

    horizons_str = " · ".join(f"+{h}D" for h in profile.horizons)
    pieces = [
        ("Last Calibration",  profile.timestamp.replace("T", " ").rstrip("Z")[:16],
         f"Fit {age_str}"),
        ("Engine Version",    profile.arthagati_version, "Arthagati build"),
        ("Profile Schema",    f"v{profile.schema_version}", "JSON envelope"),
        ("Predictors",        f"{profile.n_predictors}", "active in calibration"),
        ("Data Window",       profile.data_end, f"from {profile.data_start}"),
        ("Trials Run",        f"{profile.n_trials}",    "Optuna TPE"),
        ("CV Folds",          f"{profile.n_folds}",     f"embargo {profile.embargo_days}d"),
        ("Train Rows",        f"{profile.n_dates_train:,}", "expanding windows"),
        ("Val Rows",          f"{profile.n_dates_val:,}",   "purged validation"),
        ("Horizons",          horizons_str, "forward NIFTY return"),
    ]
    rows_html = "".join(_stat_card(lbl, val, sub) for lbl, val, sub in pieces)
    st.markdown(
        f'<div class="profile-stat-grid">{rows_html}</div>',
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# Calibration Impact — split into composable helpers
# ════════════════════════════════════════════════════════════════════════════

def _render_impact_strip(
    profile: intel.CalibrationProfile,
    mood_df: pd.DataFrame,
    msf_df: pd.DataFrame,
) -> np.ndarray:
    """Render the 4-card impact strip (Raw / Calibrated / Shift / Direction).

    Returns the full calibrated_series so callers can reuse it for the
    Predictive Power Lift table below.
    """
    render_section_header(
        title="Calibration Impact",
        description="What the post-engine ensemble changes vs the raw engine output",
        icon="zap",
        accent="emerald",
    )
    section_gap()

    raw_last = float(mood_df["Mood_Score"].iloc[-1])
    calibrated_series = intel.apply_calibration(mood_df, msf_df, profile.weights)
    cal_last = float(calibrated_series[-1])
    shift    = cal_last - raw_last

    flipped = (
        abs(raw_last) > 5 and abs(cal_last) > 5 and (raw_last > 0) != (cal_last > 0)
    )
    if flipped:
        dir_label, dir_sub, dir_cls = "Flipped", "Sign reversed by calibration", "warning"
    elif abs(shift) > 30:
        dir_label, dir_sub, dir_cls = "Amplified", "Large magnitude shift", "info"
    elif abs(shift) > 5:
        dir_label, dir_sub, dir_cls = "Adjusted",  "Moderate shift",          "info"
    else:
        dir_label, dir_sub, dir_cls = "Preserved", "Calibration broadly agrees", "success"

    raw_zone = "Bullish" if raw_last > 20 else "Bearish" if raw_last < -20 else "Neutral"
    cal_zone = "Bullish" if cal_last > 20 else "Bearish" if cal_last < -20 else "Neutral"

    c1, c2, c3, c4 = st.columns(4, gap="small")
    with c1:
        render_metric_card(
            "Raw Mood (Engine)", f"{raw_last:+.2f}",
            f"{raw_zone} · factory pipeline",
            "success" if raw_last > 20 else "danger" if raw_last < -20 else "neutral",
            icon="activity",
        )
    with c2:
        render_metric_card(
            "Calibrated Conviction", f"{cal_last:+.2f}",
            f"{cal_zone} · post-engine ensemble",
            "success" if cal_last > 20 else "danger" if cal_last < -20 else "warning",
            icon="target",
        )
    with c3:
        render_metric_card(
            "Net Shift", f"{shift:+.2f}",
            "Δ vs raw mood",
            "warning" if abs(shift) > 30 else "info",
            icon="trending-up" if shift > 0 else "trending-down",
        )
    with c4:
        render_metric_card(
            "Signal Direction", dir_label, dir_sub, dir_cls, icon="compass",
        )

    return calibrated_series


def _render_predictive_power_table(
    profile: intel.CalibrationProfile,
    mood_df: pd.DataFrame,
    calibrated_series: np.ndarray,
) -> None:
    """Per-horizon Spearman IR: raw Mood vs Calibrated · with lift column."""
    raw_train_ir, raw_val_ir, raw_per_h = intel.score_series_ir(
        mood_df["Mood_Score"].to_numpy(dtype=np.float64),
        mood_df,
        horizons=profile.horizons,
        n_folds=profile.n_folds,
        embargo_days=profile.embargo_days,
    )
    cal_train_ir, cal_val_ir, cal_per_h = intel.score_series_ir(
        calibrated_series,
        mood_df,
        horizons=profile.horizons,
        n_folds=profile.n_folds,
        embargo_days=profile.embargo_days,
    )

    rows_html = []
    for h in profile.horizons:
        h = int(h)
        raw_v = raw_per_h.get(h, 0.0)
        cal_v = cal_per_h.get(h, 0.0)
        lift  = cal_v - raw_v
        lift_cls = "pos" if lift > 0 else "neg" if lift < 0 else "neutral"
        lift_pct = (lift / abs(raw_v) * 100.0) if abs(raw_v) > 1e-6 else 0.0
        lift_str = f"{lift:+.3f}  ({lift_pct:+.0f}%)" if abs(raw_v) > 1e-6 else f"{lift:+.3f}"
        rows_html.append(
            f"<tr>"
            f"<td class='key'>+{h}D</td>"
            f"<td class='value'>{raw_v:+.3f}</td>"
            f"<td class='value'>{cal_v:+.3f}</td>"
            f"<td class='value delta {lift_cls}'>{lift_str}</td>"
            f"</tr>"
        )
    ov_lift = cal_val_ir - raw_val_ir
    ov_cls  = "pos" if ov_lift > 0 else "neg" if ov_lift < 0 else "neutral"
    ov_pct  = (ov_lift / abs(raw_val_ir) * 100.0) if abs(raw_val_ir) > 1e-6 else 0.0
    ov_str  = f"{ov_lift:+.3f}  ({ov_pct:+.0f}%)" if abs(raw_val_ir) > 1e-6 else f"{ov_lift:+.3f}"
    rows_html.append(
        f"<tr class='total'>"
        f"<td class='key'>Overall IR</td>"
        f"<td class='value'>{raw_val_ir:+.3f}</td>"
        f"<td class='value'>{cal_val_ir:+.3f}</td>"
        f"<td class='value delta {ov_cls}'>{ov_str}</td>"
        f"</tr>"
    )

    st.markdown(
        f"""\
<div class="weights-table-wrap">
  <table class="weights-table impact-table">
    <thead><tr><th>Horizon</th><th>Raw Mood</th><th>Calibrated</th><th>Lift</th></tr></thead>
    <tbody>{"".join(rows_html)}</tbody>
  </table>
</div>
""",
        unsafe_allow_html=True,
    )


def _render_profile_table(profile: intel.CalibrationProfile) -> None:
    """Profile provenance rendered as a 2-column key/value table (matches
    the visual rhythm of the Predictive Power Lift table next to it)."""
    age = intel.profile_age_days(profile)
    if age < 1.0:
        age_str = "today"
    elif age < 2.0:
        age_str = "yesterday"
    elif age < 30:
        age_str = f"{age:.0f}d ago"
    else:
        age_str = f"{age / 30:.1f}mo ago"
    horizons_str = " · ".join(f"+{h}D" for h in profile.horizons)
    pieces = [
        ("Last Calibration",  profile.timestamp.replace("T", " ").rstrip("Z")[:16],
         age_str),
        ("Engine Version",    profile.arthagati_version, "Arthagati build"),
        ("Profile Schema",    f"v{profile.schema_version}", "JSON envelope"),
        ("Predictors",        f"{profile.n_predictors}", "active in calibration"),
        ("Data Window End",   profile.data_end, f"from {profile.data_start}"),
        ("Trials Run",        f"{profile.n_trials}", "Optuna TPE"),
        ("CV Folds",          f"{profile.n_folds}", f"embargo {profile.embargo_days}d"),
        ("Train Rows",        f"{profile.n_dates_train:,}", "expanding"),
        ("Val Rows",          f"{profile.n_dates_val:,}",   "purged"),
        ("Horizons",          horizons_str, "forward NIFTY return"),
    ]
    rows_html = "".join(
        f"<tr>"
        f"<td class='key'>{html_mod.escape(lbl)}</td>"
        f"<td class='value'>{html_mod.escape(val)}</td>"
        f"<td class='value sub'>{html_mod.escape(sub)}</td>"
        f"</tr>"
        for lbl, val, sub in pieces
    )
    st.markdown(
        f"""\
<div class="weights-table-wrap">
  <table class="weights-table profile-table">
    <thead><tr><th>Field</th><th>Value</th><th>Note</th></tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>
""",
        unsafe_allow_html=True,
    )


# (Top Drivers panel removed — Feature Analysis already ranks features by
#  importance and surfaces the same information without the duplicate
#  side-by-side panel.)


# ════════════════════════════════════════════════════════════════════════════
# Main view
# ════════════════════════════════════════════════════════════════════════════

def render(
    raw_df: pd.DataFrame,
    active_predictors,
    *,
    defaults: dict,
) -> None:
    """Intelligence Center: read-only dashboard.

    Calibration is fired automatically by ``arthagati.main`` when the user
    clicks **Run Analysis**. Import/Export/Reset live in the sidebar
    Model Passport — this view is purely diagnostic.
    """
    render_section_header(
        title="Intelligence Center",
        description="Self-tuning mood-engine calibration · walk-forward Bayesian search",
        icon="cpu",
        accent="violet",
    )

    intel_on   = bool(st.session_state.get("intelligence_mode"))
    profile    = intel.load_active_profile()
    last_run   = st.session_state.get("intel_last_profile")
    if profile is None and last_run is not None:
        profile = last_run

    # Need the engine output for the impact section
    mood_df = st.session_state.get("_engine_mood_df")
    msf_df  = st.session_state.get("_engine_msf_df")

    n_rows  = int(len(raw_df))
    n_dates = int(raw_df["DATE"].nunique())
    n_pred  = len(active_predictors)
    start_d = raw_df["DATE"].min().strftime("%d %b %Y")
    end_d   = raw_df["DATE"].max().strftime("%d %b %Y")

    if profile is None:
        profile_label = "Not Calibrated" if intel_on else "Default"
        profile_color = "warning"        if intel_on else "neutral"
        profile_sub   = "Active engine config"
    else:
        profile_label = "Calibrated" if intel_on else "Available · Inactive"
        profile_color = "success"    if intel_on else "warning"
        age_d = intel.profile_age_days(profile)
        if age_d < 1.0:
            age_str = "today"
        elif age_d < 2.0:
            age_str = "yesterday"
        else:
            age_str = f"{age_d:.0f}d ago"
        profile_sub = f"Fit {age_str} · {profile.quality_check}"

    # ── Dataset strip ───────────────────────────────────────────────────
    section_gap()
    c1, c2, c3, c4, c5 = st.columns(5, gap="small")
    with c1:
        render_metric_card("Observations", f"{n_rows:,}", "Daily rows", "info", icon="database")
    with c2:
        render_metric_card("Date Span",    f"{n_dates}",  f"{start_d} → {end_d}", "info", icon="globe")
    with c3:
        render_metric_card("Predictors",   f"{n_pred}",   "Active in pipeline", "info",  icon="layers")
    with c4:
        render_metric_card("Horizons",     "30 · 60 · 90", "Forward NIFTY return (days)", "warning", icon="target")
    with c5:
        render_metric_card("Profile State", profile_label, profile_sub, profile_color, icon="shield")

    section_divider()

    # ── No-profile early-out ────────────────────────────────────────────
    if profile is None:
        if intel_on:
            render_interpretation_card(
                title="No Calibrated Profile",
                body=(
                    "Intelligence Mode is <strong>ON</strong> but no profile has been saved — "
                    "either the dataset was too small for walk-forward CV, the quality gate "
                    "rejected the search result, or the run encountered an error. "
                    "Check the terminal log for details and try "
                    "<strong>Refresh Data</strong> to re-run."
                ),
                color="warning",
            )
        else:
            render_interpretation_card(
                title="Intelligence Mode Off",
                body=(
                    "Calibration runs automatically when <strong>Intelligence Mode</strong> is "
                    "enabled in the sidebar Model Passport. With the toggle off, the engine uses "
                    "factory-default hyperparameters and this dashboard stays empty."
                ),
                color="info",
            )
        return

    # ── Calibration Diagnostics strip ───────────────────────────────────
    render_section_header(
        title="Calibration Diagnostics",
        description=(
            f"Run {html_mod.escape(profile.timestamp)} · "
            f"{profile.n_trials} trials · {profile.n_folds} folds · "
            f"embargo {profile.embargo_days}d"
        ),
        icon="activity",
        accent="emerald",
    )
    section_gap()
    m1, m2, m3, m4 = st.columns(4, gap="small")
    with m1:
        render_metric_card(
            "Train IR", f"{profile.train_ir:+.4f}",
            "In-sample Spearman IR",
            "success" if profile.train_ir > 0 else "danger",
            icon="trending-up",
        )
    with m2:
        render_metric_card(
            "Validation IR", f"{profile.val_ir:+.4f}",
            "OOS Spearman IR · all folds × horizons",
            _val_ir_severity(profile.val_ir),
            icon="target",
        )
    with m3:
        render_metric_card(
            "Stability", f"{profile.stability * 100:.0f}%",
            "Val / Train ratio",
            _stability_severity(profile.stability, profile.train_ir),
            icon="zap",
        )
    with m4:
        render_metric_card(
            "Quality Check", profile.quality_check,
            _quality_subtext(profile),
            _quality_severity(profile.quality_check),
            icon="shield",
        )

    section_divider()

    # ── Calibration Impact strip (4 metric cards) ───────────────────────
    if mood_df is None or msf_df is None or not profile.weights:
        st.caption("Calibration Impact requires cached engine output (run analysis first).")
        return

    calibrated_series = _render_impact_strip(profile, mood_df, msf_df)
    section_gap()

    # ── Feature Analysis (left)  │  Profile + Predictive Power Lift (right)
    # Feature Analysis owns the left column (tall, 10 cards stacked).
    # The right column stacks Profile Provenance on top of Predictive
    # Power Lift — both are KV tables sharing the same .weights-table
    # chassis, so they read as a single coherent dossier.
    left, mid, right = st.columns([12, 1, 10], gap="small")
    with mid:
        vertical_divider()

    with left:
        render_section_header(
            title="Feature Analysis",
            description="Per-feature weight + fANOVA importance · ranked by explanatory power",
            icon="layers",
            accent="amber",
        )
        section_gap()
        _render_feature_grid(
            profile.weights or defaults,
            profile.importance or {},
            defaults,
        )

    with right:
        # Top of the right column: Predictive Power Lift table
        render_section_header(
            title="Predictive Power Lift",
            description="Spearman IR — raw Mood vs Calibrated · per horizon",
            icon="bar-chart",
            accent="emerald",
        )
        _render_predictive_power_table(profile, mood_df, calibrated_series)

        section_divider()

        # Below it: Profile Provenance table
        render_section_header(
            title="Profile Provenance",
            description="Run details · CV configuration · dataset window",
            icon="file-text",
            accent="cyan",
        )
        _render_profile_table(profile)


def _quality_subtext(profile: intel.CalibrationProfile) -> str:
    if profile.quality_check == "No Edge":
        return "Val IR ≤ 0 — profile not auto-activated"
    if profile.quality_check == "Overfit":
        return "Train ≫ Val — fewer trials or broader window"
    return "No overfit / no-edge issues detected"
