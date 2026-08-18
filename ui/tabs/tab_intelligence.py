"""
Arthagati — Intelligence Center (read-only dashboard).

Three sections, all using the Obsidian Quant card system:
  1. Calibration Diagnostics (Holdout IR · Significance · Stability · Quality)
  2. Calibration Impact (4-card strip: Raw Mood · Calibrated · Shift · Direction)
  3. Feature Analysis grid (left)  │  Predictive Power Lift + Profile (right)

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
        intel.QUALITY_OK:           "success",
        intel.QUALITY_OVERFIT:      "warning",
        intel.QUALITY_NO_EDGE:      "danger",
        intel.QUALITY_INSUFFICIENT: "neutral",
    }.get(label, "neutral")


def _stability_severity(stability: float, train_ir: float) -> str:
    if train_ir <= 0:
        return "neutral"
    if stability >= intel.GATE_OVERFIT_STABILITY:
        return "info"
    return "warning"


def _holdout_severity(holdout_ir: float, p_value: float) -> str:
    if holdout_ir > 0 and p_value <= intel.GATE_MAX_P_VALUE:
        return "success"
    if holdout_ir > 0:
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
    "mood_diverge":  "Mood Divergence",
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
      • Contribution bar — share of total |weight| carried by this
        feature, relative to the largest contributor
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
    <span class="feature-card-bar-label">Share</span>
    <div class="feature-card-bar"><div class="feature-card-bar-fill importance" style="width:{i_bar_pct:.0f}%;"></div></div>
    <span class="feature-card-bar-pct">{importance:.1f}%</span>
  </div>
</div>
"""


def _render_feature_grid(weights: dict, importance: dict, defaults: dict) -> None:
    """Render all features as a single CSS-grid of consolidated cards.

    Cards are ranked by contribution share (descending). All cards in the same
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

    # The two numbers answer different questions and routinely disagree —
    # they are 46% sign-discordant on typical data. Say which one governs
    # rather than leaving the reader to guess.
    st.caption(
        "**Mood Score is the primary reading** — it describes the market's current "
        "sentiment state. **Calibrated Conviction is a forward-return overlay**: it is "
        "fitted to predict NIFTY returns at 5–90 days and is deliberately allowed to "
        "disagree with the state reading. When they conflict, the Mood Score describes "
        "where the market *is*; the conviction score is a bet about where it goes next, "
        "and carries the wider error bars."
    )

    return calibrated_series


def _render_predictive_power_table(
    profile: intel.CalibrationProfile,
    mood_df: pd.DataFrame,
    calibrated_series: np.ndarray,
) -> None:
    """Per-horizon Spearman IR on HELD-OUT data: raw Mood vs Calibrated.

    Both series are scored over the same window — the holdout the optimiser
    never saw — so the comparison can come out either way.

    This table previously scored both signals on the very CV folds the
    ensemble weights had been fitted to. The calibrated signal was chosen to
    maximise that quantity and raw Mood was not, so a positive "lift" was
    guaranteed by construction: on data whose forward returns were an
    independent random walk it still reported +0.92 overall.
    """
    n = len(mood_df)
    holdout_start = _holdout_index(profile, mood_df)
    if holdout_start is None or holdout_start >= n - intel.MIN_SPEARMAN_OBS:
        st.caption(
            "Not enough held-out data to compare predictive power. "
            "Re-run the calibration once the sheet has more history."
        )
        return

    window = slice(holdout_start, n)
    raw_ir, raw_per_h = intel.score_series_ir(
        mood_df["Mood_Score"].to_numpy(dtype=np.float64), mood_df,
        horizons=profile.horizons, window=window,
    )
    cal_ir, cal_per_h = intel.score_series_ir(
        calibrated_series, mood_df, horizons=profile.horizons, window=window,
    )

    rows_html = []
    for h in profile.horizons:
        h = int(h)
        raw_v = raw_per_h.get(h, 0.0)
        cal_v = cal_per_h.get(h, 0.0)
        lift = cal_v - raw_v
        lift_cls = "pos" if lift > 0 else "neg" if lift < 0 else "neutral"
        lift_str = (
            f"{lift:+.3f}  ({lift / abs(raw_v) * 100.0:+.0f}%)"
            if abs(raw_v) > 1e-6 else f"{lift:+.3f}"
        )
        rows_html.append(
            f"<tr><td class='key'>+{h}D</td>"
            f"<td class='value'>{raw_v:+.3f}</td>"
            f"<td class='value'>{cal_v:+.3f}</td>"
            f"<td class='value delta {lift_cls}'>{lift_str}</td></tr>"
        )

    ov_lift = cal_ir - raw_ir
    ov_cls = "pos" if ov_lift > 0 else "neg" if ov_lift < 0 else "neutral"
    ov_str = (
        f"{ov_lift:+.3f}  ({ov_lift / abs(raw_ir) * 100.0:+.0f}%)"
        if abs(raw_ir) > 1e-6 else f"{ov_lift:+.3f}"
    )
    rows_html.append(
        f"<tr class='total'><td class='key'>Overall IR</td>"
        f"<td class='value'>{raw_ir:+.3f}</td>"
        f"<td class='value'>{cal_ir:+.3f}</td>"
        f"<td class='value delta {ov_cls}'>{ov_str}</td></tr>"
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
    st.caption(
        f"Scored on {profile.n_rows_holdout:,} held-out rows from "
        f"{profile.holdout_start} — data the optimiser never saw. "
        "A negative lift is a real possible outcome here."
    )


def _holdout_index(profile: intel.CalibrationProfile, mood_df: pd.DataFrame) -> int | None:
    """Row index where the profile's holdout begins, matched by date."""
    if not profile.holdout_start:
        return None
    try:
        cutoff = pd.Timestamp(profile.holdout_start)
    except (ValueError, TypeError):
        return None
    pos = int((mood_df["DATE"] < cutoff).sum())
    return pos if 0 < pos < len(mood_df) else None


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
        ("CV Folds",          f"{profile.n_folds}", f"embargo {profile.embargo_days}d = max horizon"),
        # Unique rows, not the sum over overlapping expanding folds — that
        # used to report more "train rows" than the dataset contained.
        ("Search Rows",       f"{profile.n_rows_train:,}", "unique, expanding folds"),
        ("Holdout Rows",      f"{profile.n_rows_holdout:,}", f"from {profile.holdout_start}"),
        ("Holdout IR",        f"{profile.holdout_ir:+.3f}", f"p = {profile.holdout_p_value:.3f}"),
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

def render(*, defaults: dict) -> None:
    """Intelligence Center: read-only dashboard.

    Calibration is fired automatically by ``arthagati.main`` when the user
    clicks **Run Analysis**. Import/Export/Reset live in the sidebar
    Model Passport — this view is purely diagnostic.
    """
    intel_on = bool(st.session_state.get("intelligence_mode"))
    disabled = bool(st.session_state.get("_intel_profile_disabled"))

    # Show the profile from THIS run when there is one. Falling back to
    # whatever sits on disk used to mean that a run rejected by the quality
    # gate left the previous profile on screen — the top metric strip hid the
    # Calibrated Conviction card while this view kept rendering the old
    # profile's conviction as though it were current.
    last_run = st.session_state.get("intel_last_profile")
    profile = last_run if last_run is not None else intel.load_active_profile()
    is_live = (
        profile is not None and profile.is_activatable and intel_on and not disabled
    )

    # Need the engine output for the impact section
    mood_df = st.session_state.get("_engine_mood_df")
    msf_df  = st.session_state.get("_engine_msf_df")

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

    # A profile that did not clear the gate is shown for diagnosis only.
    if not is_live:
        if disabled:
            body = (
                "Calibration is <strong>disabled for this session</strong> "
                "(Reset to Defaults). The figures below are the last run's, kept "
                "for reference. Click <strong>Run Analysis</strong> to recalibrate."
            )
        elif not intel_on:
            body = (
                "Intelligence Mode is <strong>off</strong>. The figures below are "
                "from the last saved calibration and are not being applied."
            )
        else:
            body = (
                f"This calibration was graded <strong>{html_mod.escape(profile.quality_check)}</strong> "
                "and is <strong>not applied</strong>. The Calibrated Conviction card is hidden "
                "everywhere else in the app; the diagnostics below are shown so you can see "
                "why it was rejected. The raw Mood Score is the operative signal."
            )
        render_interpretation_card("Calibration Not Active", body, color="warning")
        section_gap()

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
            "Holdout IR", f"{profile.holdout_ir:+.4f}",
            f"{profile.n_rows_holdout:,} rows never seen by the search",
            _holdout_severity(profile.holdout_ir, profile.holdout_p_value),
            icon="target",
        )
    with m2:
        render_metric_card(
            "Significance", f"p = {profile.holdout_p_value:.3f}",
            f"vs {intel.N_PERMUTATIONS} circular-shift permutations",
            "success" if profile.holdout_p_value <= intel.GATE_MAX_P_VALUE else "danger",
            icon="shield",
        )
    with m3:
        render_metric_card(
            "Stability", f"{profile.stability * 100:.0f}%",
            "Holdout / Train ratio",
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
    st.caption(
        f"Search diagnostics (not a performance claim): train IR {profile.train_ir:+.3f} · "
        f"optimised validation IR {profile.val_ir:+.3f}. The optimiser maximised the "
        "validation figure, so only the holdout above is evidence of generalisation."
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
            description="Per-feature weight + share of total |weight| · ranked by contribution",
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
    if profile.quality_check == intel.QUALITY_INSUFFICIENT:
        return "Holdout too short to validate — not activated"
    if profile.quality_check == intel.QUALITY_NO_EDGE:
        return "Failed the holdout test — not activated"
    if profile.quality_check == intel.QUALITY_OVERFIT:
        return "Generalises weakly — not activated"
    return "Cleared the holdout and the permutation null"
