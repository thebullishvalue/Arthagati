"""
Arthagati — Drivers: which inputs carry the signal, and which are noise.

The engine exposes a ranked universe of candidate predictors, each with several
measures — correlation against two anchors, entropy, coverage, a composite
quality score, and whether it is currently active. That is a screener, and it
gets a screener's interface: a filter bar, one dense table, tier badges.

Quality is computed over EVERY eligible column, not just the active set. A
panel whose stated job is to guide predictor selection could otherwise never
recommend anything the user had not already selected.

Reading order:

  1 SCAN     how large is the universe?    Tier counts
  2 SIGNAL   which columns rank?           The screen and the ranked table
  3 DETAIL   what does each anchor say?    Per-anchor correlations
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from ui import format as fmt
from ui.components import (
    render_chip,
    render_control_hint,
    render_empty_state,
    render_kpi_strip,
    render_note,
    render_section_header,
    render_table_panel,
)

_TIER_LABEL = {"keep": "Keep", "useful": "Useful", "weak": "Weak", "nodata": "No data"}


def _tier(row, max_q: float) -> str:
    if row["Coverage"] < 10:
        return "nodata"
    if row["Quality"] >= max_q * 0.5 and row["Coverage"] > 50:
        return "keep"
    if row["Quality"] >= max_q * 0.2 and row["Coverage"] > 30:
        return "useful"
    return "weak"


def _quality_frame(raw_df, all_vars, pe_lookup, ey_lookup, entropy_fn) -> pd.DataFrame:
    rows = []
    for var in all_vars:
        pe_raw = float(pe_lookup.get(var, np.nan))
        ey_raw = float(ey_lookup.get(var, np.nan))
        avg = (abs(pe_raw if np.isfinite(pe_raw) else 0.0)
               + abs(ey_raw if np.isfinite(ey_raw) else 0.0)) / 2

        # First differences, not pct_change. Several predictors cross zero by
        # construction (both term spreads, the deviation pairs), and a
        # near-zero denominator produced changes of several hundred x, which
        # dominated the histogram bin width and corrupted the entropy.
        diffs = raw_df[var].diff().dropna().to_numpy()
        entropy = entropy_fn(diffs) if len(diffs) > 10 else 0.5
        col = raw_df[var]
        rows.append({
            "Variable": var,
            "Rho PE": pe_raw,
            "Rho EY": ey_raw,
            "Entropy": float(entropy),
            # `col != 0` counts NaN as True in pandas, so an all-NaN column
            # used to report 100% coverage.
            "Coverage": float((col.notna() & (col != 0)).mean() * 100),
            "Quality": float(avg * max(1.0 - entropy, 0.1)),
        })
    return pd.DataFrame(rows).sort_values("Quality", ascending=False).reset_index(drop=True)


def render(raw_df, *, active_preds, non_predictor_cols,
           calculate_anchor_correlations, shannon_entropy) -> None:
    active = set(active_preds)

    # ── Anchor health ─────────────────────────────────────────────────────
    bad = [label for col, label in (("NIFTY50_PE", "PE Ratio"),
                                    ("NIFTY50_EY", "Earnings Yield"))
           if not (col in raw_df.columns and raw_df[col].nunique() > 3
                   and raw_df[col].std() > 1e-6)]

    all_vars = [c for c in raw_df.columns
                if c not in non_predictor_cols and pd.api.types.is_numeric_dtype(raw_df[c])]
    if not all_vars:
        render_empty_state(
            "No eligible numeric columns",
            "Every column in the sheet is an anchor, an index key, or non-numeric.",
            eyebrow="Empty universe")
        return

    pe_all = calculate_anchor_correlations(raw_df, "NIFTY50_PE", all_vars)
    ey_all = calculate_anchor_correlations(raw_df, "NIFTY50_EY", all_vars)
    pe_lookup = dict(zip(pe_all["variable"], pe_all["correlation"])) if not pe_all.empty else {}
    ey_lookup = dict(zip(ey_all["variable"], ey_all["correlation"])) if not ey_all.empty else {}

    q = _quality_frame(raw_df, all_vars, pe_lookup, ey_lookup, shannon_entropy)
    if q.empty:
        render_empty_state("No measurable predictors",
                           "Every eligible column resolved to a zero-quality score.",
                           eyebrow="Empty universe")
        return

    max_q = float(q["Quality"].max()) or 1.0
    q["Tier"] = q.apply(lambda r: _TIER_LABEL[_tier(r, max_q)], axis=1)
    q["State"] = np.where(q["Variable"].isin(active), "Active", "Off")
    counts = q["Tier"].value_counts().to_dict()

    # ── 1 · SCAN ──────────────────────────────────────────────────────────
    render_section_header(
        "Predictor Universe",
        "Every eligible column ranked by the same shape the engine weights with: "
        "|rho| against the PE and Earnings Yield anchors, penalised by the entropy "
        "of its own increments.",
        icon="database",
    )
    if bad:
        render_note(
            f"<strong>{', '.join(bad)}</strong> carries insufficient variance in the "
            "source data, so every correlation measured against it is unreliable. If "
            "Earnings Yield is empty in the sheet it is auto-derived from PE "
            "(1/PE × 100).")
    render_kpi_strip([
        {"label": "Universe", "value": str(len(q)),
         "subtext": "eligible columns", "color_class": "neutral", "icon": "database"},
        {"label": "Active", "value": str(len(active)),
         "subtext": "feeding the engine", "color_class": "accent", "icon": "cpu"},
        {"label": "Keep", "value": str(counts.get("Keep", 0)),
         "subtext": "high rho, low entropy", "color_class": "success", "icon": "check-circle"},
        {"label": "Useful", "value": str(counts.get("Useful", 0)),
         "subtext": "moderate signal", "color_class": "info", "icon": "circle"},
        {"label": "Weak", "value": str(counts.get("Weak", 0)),
         "subtext": "low signal or noisy", "color_class": "neutral", "icon": "minus-circle"},
        {"label": "No data", "value": str(counts.get("No data", 0)),
         "subtext": "coverage under 10%", "color_class": "warning", "icon": "alert-triangle"},
    ], max_cols=6)

    # ── 2 · SIGNAL ────────────────────────────────────────────────────────
    render_section_header(
        "Screen",
        "Filters apply to the ranked table below and to nothing else — the engine "
        "keeps running on the active set whatever is screened here.",
        icon="search",
        accent="cyan",
    )
    f1, f2, f3, f4 = st.columns([3, 3, 3, 3], gap="small")
    with f1:
        tiers = st.multiselect("Tier", options=list(_TIER_LABEL.values()),
                               default=["Keep", "Useful"])
    with f2:
        state = st.selectbox("State", ["All", "Active only", "Inactive only"])
    with f3:
        min_cov = st.slider("Min coverage %", 0, 100, 30, step=5)
    with f4:
        query = st.text_input("Name contains", placeholder="e.g. YIELD")

    view = q.copy()
    if tiers:
        view = view[view["Tier"].isin(tiers)]
    if state == "Active only":
        view = view[view["State"] == "Active"]
    elif state == "Inactive only":
        view = view[view["State"] == "Off"]
    view = view[view["Coverage"] >= min_cov]
    if query.strip():
        view = view[view["Variable"].str.contains(query.strip(), case=False, regex=False)]
    view = view.reset_index(drop=True)

    if view.empty:
        render_empty_state(
            "No variables match this screen",
            f"Widen the tier selection or lower the coverage floor. "
            f"{len(q)} variables exist in the universe.",
            eyebrow="Empty screen",
            action_label="Reset the filters above",
        )
    else:
        render_table_panel(
            view, key="dr-universe",
            context=f"{len(view)} of {len(q)} · ranked by quality = |rho| × (1 − H)",
            chip=("FULL SAMPLE", "warning"),
            label_col="Variable",
            sign_color_cols={"Rho PE", "Rho EY"},
            col_precision={"Rho PE": 2, "Rho EY": 2, "Entropy": 2,
                           "Coverage": 0, "Quality": 3},
            max_height=520,
            footer="Full-sample figures. The engine weights on WALK-FORWARD "
                   "correlations blended across quarterly checkpoints, so its live "
                   "coefficients differ in magnitude — read this as a ranking, not "
                   "as the engine's weights.",
        )

    # ── 3 · DETAIL ────────────────────────────────────────────────────────
    render_section_header(
        "Anchor Detail",
        "Active predictors only, ranked by |rho| against each valuation anchor "
        "separately. A column that loads on one anchor but not the other is doing "
        "less work than its average suggests.",
        icon="bar-chart",
        accent="violet",
    )
    a1, a2 = st.columns(2, gap="small")
    for col, anchor, title in ((a1, "NIFTY50_PE", "PE Ratio"),
                               (a2, "NIFTY50_EY", "Earnings Yield")):
        with col:
            corrs = calculate_anchor_correlations(raw_df, anchor, list(active_preds))
            if corrs is None or corrs.empty:
                render_empty_state(
                    f"No correlations against {title}",
                    "The anchor column is missing from the sheet or carries no variance.",
                    eyebrow=title)
                continue
            frame = (corrs.sort_values("correlation", key=abs, ascending=False)
                     .rename(columns={"variable": "Variable", "correlation": "Rho"})
                     .reset_index(drop=True))
            render_table_panel(
                frame, key=f"dr-anchor-{anchor.lower()}",
                title=title,
                context=f"Decay-weighted Spearman · {len(frame)} active",
                label_col="Variable", sign_color_cols={"Rho"},
                precision=2, max_height=340,
            )
