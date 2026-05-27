"""
Arthagati — Intelligence Center view (dashboard only).

The actual calibration runs automatically when the user clicks
*Run Analysis* (see ``arthagati.main`` for the pipeline integration).
This view is a read-only dashboard surfacing:

  • Dataset shape & profile state
  • Calibration diagnostics (Train IR / Val IR / Stability / Quality)
  • Default vs Calibrated weights table
  • Parameter importance (fANOVA)
  • Profile provenance + export / import / reset controls
"""

from __future__ import annotations

import html as html_mod
import json
import time

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
# Weights table + importance rows
# ════════════════════════════════════════════════════════════════════════════

def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.2f}" if abs(v) < 10 else f"{v:.0f}"
    return f"{v}"


def _render_weights_table(weights: dict, defaults: dict) -> None:
    body: list[str] = []
    for k, df in defaults.items():
        cur = weights.get(k, df)
        delta = float(cur) - float(df)
        delta_pct = (delta / df * 100.0) if df not in (0, None) else 0.0
        changed = abs(delta) > 1e-9
        cls = "pos" if delta > 0 else "neg" if delta < 0 else "neutral"
        delta_str = f"{delta_pct:+.0f}%" if changed else "—"
        cur_html = (
            f'<span style="color:var(--amber-bright); font-weight:700;">{_fmt(cur)}</span>'
            if changed else
            f'<span style="color:var(--ink-secondary);">{_fmt(cur)}</span>'
        )
        body.append(
            f"<tr>"
            f"<td class=\"key\">{html_mod.escape(k)}</td>"
            f"<td class=\"value\">{_fmt(df)}</td>"
            f"<td class=\"value\">{cur_html}</td>"
            f"<td class=\"value delta {cls}\">{delta_str}</td>"
            f"</tr>"
        )

    html = f"""\
<div class="weights-table-wrap">
  <table class="weights-table">
    <thead><tr><th>Hyperparameter</th><th>Default</th><th>Calibrated</th><th>Δ</th></tr></thead>
    <tbody>{"".join(body)}</tbody>
  </table>
</div>
"""
    st.markdown(html, unsafe_allow_html=True)


def _render_importance(importance: dict) -> None:
    if not importance:
        st.caption("Importance data not yet available — calibration produces this on completion.")
        return
    rows = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
    max_v = max((v for _, v in rows), default=1.0) or 1.0
    parts: list[str] = []
    for k, v in rows:
        pct = (v / max_v) * 100.0
        parts.append(f"""\
<div class="importance-row">
  <span class="importance-key">{html_mod.escape(k)}</span>
  <div class="importance-track"><div class="importance-fill" style="width:{pct:.0f}%;"></div></div>
  <span class="importance-val">{v:.1f}%</span>
</div>
""")
    st.markdown("".join(parts), unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# Public render
# ════════════════════════════════════════════════════════════════════════════

def render(
    raw_df: pd.DataFrame,
    active_predictors,
    *,
    defaults: dict,
) -> None:
    """Intelligence Center: read-only dashboard.

    Calibration is fired automatically by ``arthagati.main`` when the user
    clicks **Run Analysis**, gated by the sidebar Intelligence Mode toggle
    and a one-shot session-state flag.
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

    n_rows  = int(len(raw_df))
    n_dates = int(raw_df["DATE"].nunique())
    n_pred  = len(active_predictors)
    start_d = raw_df["DATE"].min().strftime("%d %b %Y")
    end_d   = raw_df["DATE"].max().strftime("%d %b %Y")

    if profile is None:
        profile_label = "Calibrating…" if intel_on else "Default"
        profile_color = "warning"      if intel_on else "neutral"
    else:
        profile_label = "Calibrated" if intel_on else "Available · Inactive"
        profile_color = "success"    if intel_on else "warning"

    # ── Dataset strip ───────────────────────────────────────────────────
    section_gap()
    c1, c2, c3, c4, c5 = st.columns(5, gap="small")
    with c1:
        render_metric_card("Observations", f"{n_rows:,}", "Daily rows",      "info",     icon="database")
    with c2:
        render_metric_card("Date Span",    f"{n_dates}",  f"{start_d} → {end_d}", "info", icon="globe")
    with c3:
        render_metric_card("Predictors",   f"{n_pred}",   "Active in pipeline", "info",  icon="layers")
    with c4:
        render_metric_card("Horizons",     "30 · 60 · 90", "Forward NIFTY return (days)", "warning", icon="target")
    with c5:
        render_metric_card("Profile State", profile_label, "Active engine config", profile_color, icon="shield")

    section_divider()

    # ── No-profile state ────────────────────────────────────────────────
    if profile is None:
        if intel_on:
            render_interpretation_card(
                title="Calibration In Progress",
                body=(
                    "Intelligence Mode is <strong>ON</strong> but no calibrated profile has been "
                    "saved yet. The next time you click <strong>Run Analysis</strong> the engine "
                    "will perform a full walk-forward Bayesian search and persist the result here."
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

    # ── 4-metric diagnostics strip ──────────────────────────────────────
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

    section_gap()

    # ── Active weights + importance side by side ────────────────────────
    weights_col, mid_col, imp_col = st.columns([10, 1, 9], gap="small")
    with mid_col:
        vertical_divider()
    with weights_col:
        render_section_header(
            title="Active Weights",
            description="Default vs calibrated hyperparameters · Δ vs factory",
            icon="grid",
            accent="amber",
        )
        _render_weights_table(profile.weights or defaults, defaults)
    with imp_col:
        render_section_header(
            title="Parameter Importance",
            description="fANOVA importance (Optuna) · weight-share fallback",
            icon="bar-chart",
            accent="violet",
        )
        _render_importance(profile.importance)

    section_divider()

    # ── Provenance + lifecycle controls ────────────────────────────────
    render_section_header(
        title="Profile",
        description="Provenance · integrity · lifecycle",
        icon="file-text",
        accent="cyan",
    )

    meta_col, action_col = st.columns([3, 2], gap="small")
    with meta_col:
        meta_rows = [
            ("Run timestamp",     profile.timestamp),
            ("Arthagati version", profile.arthagati_version),
            ("Predictors",        f"{profile.n_predictors}"),
            ("Data window",       f"{profile.data_start} → {profile.data_end}"),
            ("CV folds",          f"{profile.n_folds} (embargo {profile.embargo_days}d)"),
            ("Train rows",        f"{profile.n_dates_train:,}"),
            ("Val rows",          f"{profile.n_dates_val:,}"),
            ("Horizons",          " · ".join(f"+{h}D" for h in profile.horizons)),
            ("Schema version",    f"v{profile.schema_version}"),
        ]
        rows_html = "".join(
            f"<div class='lookback-row'><span class='label'>{html_mod.escape(k)}</span>"
            f"<span class='value'>{html_mod.escape(str(v))}</span></div>"
            for k, v in meta_rows
        )
        st.markdown(
            f"<div class='intel-meta-card'>{rows_html}</div>",
            unsafe_allow_html=True,
        )

    with action_col:
        st.download_button(
            label="⤓  Export Profile (JSON)",
            data=profile.to_json(),
            file_name=f"arthagati_profile_{profile.timestamp.replace(':', '').replace('-', '')[:15]}.json",
            mime="application/json",
            use_container_width=True,
        )
        uploaded = st.file_uploader(
            "Import a profile JSON",
            type=["json"],
            label_visibility="visible",
        )
        if uploaded is not None:
            _handle_profile_upload(uploaded)

        if intel.load_active_profile() is not None:
            if st.button(
                "Reset to Default Hyperparameters",
                use_container_width=True,
                type="secondary",
            ):
                intel.delete_active_profile()
                st.session_state.pop("intel_last_profile", None)
                st.session_state.pop("_intel_calibration_done", None)
                st.success("Active profile cleared. Engine reverted to factory defaults.")
                time.sleep(0.4)
                st.rerun()


def _quality_subtext(profile: intel.CalibrationProfile) -> str:
    if profile.quality_check == "No Edge":
        return "Val IR ≤ 0 — profile not auto-activated"
    if profile.quality_check == "Overfit":
        return "Train ≫ Val — fewer trials or broader window"
    return "No overfit / no-edge issues detected"


def _handle_profile_upload(uploaded) -> None:
    from core.logger_config import console
    try:
        raw = json.loads(uploaded.read().decode("utf-8"))
        profile = intel.CalibrationProfile.from_dict(raw)
        intel.save_active_profile(profile)
        st.session_state["intel_last_profile"] = profile
        st.session_state.pop("_intel_calibration_done", None)
        console.success(f"Profile imported from {uploaded.name}")
        st.success(f"Profile from {uploaded.name} is now active.")
        time.sleep(0.4)
        st.rerun()
    except Exception as exc:
        console.error(f"Profile import failed: {exc}")
        st.error(f"Import failed: {exc}")
