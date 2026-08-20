"""
Arthagati — Configuration: what the engine is running on.

The predictor set is the engine's only structural input; every other
hyperparameter is fixed in config.py and is not tunable from the interface.

Eight presets, each carrying a recorded holdout rho, a margin over the −PE
baseline and a p-value, is a comparison table — so it is one. It lived in the
sidebar, where a 260px column had to carry a four-row statistics block, a
blurb, a missing-columns warning and a multiselect, which made the most
consequential control in the product the hardest thing on screen to read. The
rail keeps the quick-switch; this page carries the evidence.

Reading order:

  1 SCAN     what is running?          Active Model
  2 SIGNAL   what else could run?      Profile Comparison
  3 STATE    what exactly is on?       Active Columns
  4 DETAIL   why is a column missing?  Eligibility Rules
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from config import PREDICTOR_PROFILES, PROFILE_MEASUREMENT_CONTEXT
from ui import format as fmt
from ui.components import (
    render_empty_state,
    render_kpi_strip,
    render_note,
    render_section_header,
    render_table_panel,
)


def _preset_frame(available: list[str], resolve, current: str) -> pd.DataFrame:
    base = PROFILE_MEASUREMENT_CONTEXT["baseline_rho"]
    rows = []
    for key, spec in PREDICTOR_PROFILES.items():
        present, missing = resolve(key, available)
        m = spec["measured"]
        rows.append({
            "Profile": spec["label"],
            "Columns": len(present),
            "Missing": len(missing),
            "Holdout rho": float(m["holdout_rho"]),
            "vs −PE": float(m["holdout_rho"]) - float(base),
            "p": float(m["p_value"]),
            "Selected": "Yes" if key == current else "",
        })
    return pd.DataFrame(rows)


def _custom_picker(available: list[str], on_change) -> None:
    """Hand-picked set. Staged, then applied.

    The multiselect fires on every checkbox and each change would otherwise
    trigger a full engine recompute, so the commit is explicit. The preset
    dropdown applies immediately because it is one discrete choice.
    """
    staged = st.multiselect(
        "Predictor columns", options=available,
        default=list(st.session_state["active_predictors"]),
        help="NIFTY-derived columns are withheld — using one would make the "
             "valuation score a function of the price it is scored against.")
    if not staged:
        render_note("Select at least one predictor. The active set is unchanged "
                    "until a valid selection is applied.")
        return

    staged_set = set(staged)
    active_set = set(st.session_state["active_predictors"])
    changed = staged_set != active_set
    if changed:
        added, removed = staged_set - active_set, active_set - staged_set
        parts = ([f"+{len(added)}"] if added else []) + ([f"−{len(removed)}"] if removed else [])
        render_note(f"Pending: {' · '.join(parts)} — {len(staged)} columns after apply.")

    if st.button("Apply configuration" if changed else "No changes",
                 width="stretch", disabled=not changed,
                 type="primary" if changed else "secondary") and changed:
        on_change(tuple(staged))


def render(*, available_predictors, resolve_profile, detect_profile,
           on_profile_change, on_predictors_change) -> None:
    active = tuple(st.session_state.get("active_predictors", ()))
    detected = detect_profile(active, available_predictors)
    ctx = PROFILE_MEASUREMENT_CONTEXT

    # ── 1 · SCAN ──────────────────────────────────────────────────────────
    render_section_header(
        "Active Model",
        "The predictor set is the engine's only structural input. Everything else — "
        "half-lives, band edges, Kalman parameters, the MSF weighting rule — is fixed "
        "in config.py and deliberately not tunable from the interface.",
        icon="cpu",
    )
    render_kpi_strip([
        {"label": "Active set", "value": str(len(active)),
         "subtext": "columns feeding the engine", "color_class": "accent", "icon": "cpu"},
        {"label": "Eligible", "value": str(len(available_predictors)),
         "subtext": "pass coverage and uniqueness", "color_class": "neutral",
         "icon": "database"},
        {"label": "Profile",
         "value": PREDICTOR_PROFILES.get(detected, {}).get("label", "Custom"),
         "subtext": "matched exactly against this sheet", "color_class": "neutral",
         "icon": "layers"},
        {"label": "Baseline rho", "value": fmt.rho(ctx["baseline_rho"]),
         "subtext": "−PE alone, no engine", "color_class": "warning", "icon": "zap"},
    ], max_cols=4)

    # ── 2 · SIGNAL ────────────────────────────────────────────────────────
    render_section_header(
        "Profile Comparison",
        "Each preset carries the measurement recorded for it. These are a RECORD "
        "from the reference sheet, not a live claim about the loaded data — "
        "Validation re-measures whatever is actually active.",
        icon="target",
        accent="cyan",
    )
    render_table_panel(
        _preset_frame(available_predictors, resolve_profile, detected),
        key="cfg-presets", label_col="Profile",
        context=f"Recorded {ctx['measured_date']} · {ctx['rows']:,} rows "
                f"({ctx['span']}) · holdout {ctx['holdout']}",
        chip=("HISTORICAL RECORD", "warning"),
        sign_color_cols={"Holdout rho", "vs −PE"},
        col_precision={"Holdout rho": 3, "vs −PE": 3, "p": 3,
                       "Columns": 0, "Missing": 0},
        max_height=340,
    )

    left, right = st.columns([5, 7], gap="small")
    with left:
        render_section_header("Profile Selection", "Applies immediately and recomputes.",
                              icon="settings")
        keys = list(PREDICTOR_PROFILES) + ["custom"]
        labels = {k: PREDICTOR_PROFILES[k]["label"] for k in PREDICTOR_PROFILES}
        labels["custom"] = "Custom"
        chosen = st.selectbox(
            "Predictor profile", options=keys,
            index=keys.index(detected) if detected in keys else len(keys) - 1,
            format_func=lambda k: (
                labels[k] if k == "custom"
                else f"{labels[k]} · {len(resolve_profile(k, available_predictors)[0])} cols"),
            label_visibility="collapsed")
        if chosen != "custom":
            preds, missing = resolve_profile(chosen, available_predictors)
            render_note(PREDICTOR_PROFILES[chosen]["blurb"])
            if missing:
                render_note(
                    f"<strong>{len(missing)} column(s) not in this sheet</strong> — "
                    f"{', '.join(missing[:6])}{'…' if len(missing) > 6 else ''}. The "
                    "preset applies without them, so its recorded measurement no "
                    "longer describes the set you would be running.")
            if preds and set(preds) != set(active):
                on_profile_change(chosen, tuple(preds))
        else:
            st.session_state["predictor_profile"] = "custom"
            _custom_picker(available_predictors, on_predictors_change)

    # ── 3 · STATE ─────────────────────────────────────────────────────────
    with right:
        render_section_header(
            "Active Columns",
            f"{len(active)} columns currently feeding the correlation layer.",
            icon="grid", accent="emerald")
        if not active:
            render_empty_state(
                "No predictors selected",
                "The engine cannot build a composite without at least one column.",
                eyebrow="Empty set")
        else:
            render_table_panel(
                pd.DataFrame([
                    {"Column": c,
                     "Status": "Eligible" if c in available_predictors else "Missing"}
                    for c in sorted(active)]),
                key="cfg-active", label_col="Column",
                context="Sorted alphabetically", max_height=340)

    # ── 4 · DETAIL ────────────────────────────────────────────────────────
    render_section_header(
        "Eligibility Rules",
        "Why a column present in the sheet may not appear in the universe. Applied "
        "in load order, before any profile resolves.",
        icon="file-text",
        accent="violet",
    )
    render_table_panel(
        pd.DataFrame([
            {"Rule": "Anchor or index key",
             "Effect": "Excluded — the score is measured against it"},
            {"Rule": "Derived from NIFTY",
             "Effect": "Excluded — would make the valuation score a function of the "
                       "price it is scored against, so any measured edge would be "
                       "partly price predicting itself"},
            {"Rule": "Duplicate of a derived column",
             "Effect": "Excluded — the loader derives it already"},
            {"Rule": "Coverage below threshold",
             "Effect": "Excluded — too few real values to estimate a correlation from"},
            {"Rule": "Too few unique values",
             "Effect": "Excluded — a near-constant column carries no rank information"},
        ]),
        key="cfg-rules", label_col="Rule", max_height=260,
    )
