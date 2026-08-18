"""
Arthagati — cold start: a description of the product, built from the product's
own parts.

Every block here uses the same components the analysis pages use — a section
header for each division, ``render_kpi_strip`` for the coverage numbers, and
``panel()`` for each system — so the landing page sits on the same
section-rhythm contract as everything else and cannot drift into reading like
a different product's marketing page bolted to the front of this one.

The claim leads, because a reader who has not run anything needs to know what
the thing IS before they are shown what it covers.
"""

from __future__ import annotations

import html as html_mod

import streamlit as st

from config import (
    CORR_HALF_LIFE,
    CORR_REBALANCE_PERIOD,
    KALMAN_HALF_LIFE,
    MSF_OB_LEVEL_1,
    MSF_OB_LEVEL_2,
    PCT_HALF_LIFE,
    PREDICTOR_PROFILES,
    PROFILE_MEASUREMENT_CONTEXT,
    SIMILAR_MIN_SEPARATION,
    SIMILAR_W_MAHA,
    SIMILAR_W_RECV,
    SIMILAR_W_TRAJ,
)
from ui.components import (
    panel,
    render_header,
    render_kpi_strip,
    render_notice_rail,
    render_section_header,
)

#: The three engines, as the cold-start screen describes them. Data, not
#: markup — the landing page renders them through one template, so the three
#: panels cannot drift apart in structure the way three hand-written HTML
#: blocks would.
#:
#: The order is the order of the argument: MOOD makes the claim, MSF says
#: whether to believe it, PRECEDENT checks both against history without
#: depending on either being right.
_SYSTEM_PANELS = (
    ("mood", "MOOD", "The claim · valuation",
     "Scores where the market sits against its own recent history, anchored to the PE "
     "ratio and Earnings Yield, through five causal layers: decay-weighted rank "
     "correlation, an entropy penalty on noisy inputs, a decay-weighted percentile, an "
     "Ornstein-Uhlenbeck fit, and a Kalman filter. It moves AGAINST recent price by "
     "construction — cheap scores high — so a falling score during a rally is the "
     "instrument working, not failing.",
     (("Weights", f"Decay-Spearman · {CORR_HALF_LIFE}d half-life"),
      ("Normalisation", "Ornstein-Uhlenbeck"),
      ("Filter", f"Kalman · {KALMAN_HALF_LIFE}d fading memory"))),

    ("msf", "MSF", "The confirmation · oscillator",
     "Four components — momentum, structure, regime and flow — blended by inverse "
     "variance and auto-calibrated, built to be independent of the score above. That "
     "independence is the point: when the two disagree, the disagreement is "
     "information rather than an error, and it is the constraint that most often caps "
     "conviction.",
     (("Components", "Momentum · Structure · Regime · Flow"),
      ("Weights", "Inverse-variance, auto-calibrated"),
      ("Bands", f"±{MSF_OB_LEVEL_2:.0f} / ±{MSF_OB_LEVEL_1:.0f}, fixed"))),

    ("precedent", "PRECEDENT", "The check · historical analogs",
     "Finds the days whose market state most resembles the current one by Mahalanobis "
     "distance and trajectory shape, under a minimum-separation window so each match is "
     "a distinct episode rather than an adjacent day of the same one. It is the only "
     "read in the app that does not depend on the engine being right.",
     ((f"Distance", f"Mahalanobis · {SIMILAR_W_MAHA:.0%}"),
      ("Shape", f"Trajectory cosine · {SIMILAR_W_TRAJ:.0%}"),
      ("Separation", f"{SIMILAR_MIN_SEPARATION} trading days minimum"))),
)

#: What a completed run puts on the screen. Four static cards, so ONE markdown
#: block rather than four Streamlit containers — static text gains nothing from
#: a container and loses something real to it, since the anonymous row
#: Streamlit wraps markdown in does not grow to fit its content and each panel
#: clips its own last line by a different amount. A grid of plain divs has no
#: wrapper to collapse.
_OUTCOMES = (
    ("A directional claim",
     "One verdict, with the six gates that condition it and the single binding "
     "constraint named — the specific reason conviction is not higher."),
    ("A measured edge",
     "Holdout Spearman rho against a permutation null, and against the negated PE "
     "ratio: the same claim with no engine at all. The margin between them is a gate, "
     "not a footnote."),
    ("An independent check",
     "A forward-return base rate from the most similar historical states, reported "
     "with its own spread so a thin sample cannot read as agreement."),
    ("The evidence",
     "Every candidate predictor's correlation, entropy and coverage, ranked by the "
     "same quality shape the engine weights with — including the ones it rejected."),
)


def render_landing_page(version: str, n_predictors: int, sheet_configured: bool) -> None:
    ctx = PROFILE_MEASUREMENT_CONTEXT

    render_header(
        "ARTHAGATI",
        "Valuation-Anchored Sentiment · Ornstein-Uhlenbeck · Kalman · Walk-Forward Spearman",
    )

    # A missing data source is a caveat on everything below it, so it renders
    # in the app's own notice grammar, under the masthead rather than over it.
    if not sheet_configured:
        render_notice_rail([{
            "kind": "warning",
            "title": "Data source not configured",
            "body": "<code>ARTHAGATI_SHEET_ID</code> and <code>ARTHAGATI_SHEET_GID</code> "
                    "are unset, so a run will fail at ingestion. Point them at the "
                    "spreadsheet ID and worksheet GID of a sheet readable through the "
                    "Google Visualization API, then restart.",
        }])

    # ── The proposition ───────────────────────────────────────────────────
    st.markdown(
        """<div class="lede">
  <div class="lede-claim">One score prices market sentiment against its own
    history, a second instrument says whether to believe it, and a held-out
    test says whether either has ever been worth anything.</div>
  <div class="lede-cta">Press <strong>Run Analysis</strong> in the rail.</div>
</div>""",
        unsafe_allow_html=True,
    )

    # ── Coverage — the app's own KPI grammar, not a bespoke number row ─────
    render_section_header("Coverage", icon="layers")
    render_kpi_strip(
        [
            {"label": "Predictor Profiles", "value": str(len(PREDICTOR_PROFILES)),
             "subtext": "Each carrying the out-of-sample correlation it actually "
                        "achieved on the reference sheet, alongside the no-engine "
                        "baseline it has to beat"},
            {"label": "Default Predictors", "value": str(n_predictors),
             "subtext": "Macro, breadth and valuation series. NIFTY-derived columns are "
                        "withheld — using one would make the score a function of the "
                        "price it is scored against"},
            {"label": "Daily History Per Run", "value": "~20y",
             "subtext": f"Walk-forward throughout: statistics for a segment are "
                        f"estimated on data through the previous checkpoint, "
                        f"rebalanced every {CORR_REBALANCE_PERIOD} days"},
        ],
        max_cols=3,
        key="landing-coverage",
    )

    # ── The three engines, as panels ──────────────────────────────────────
    render_section_header(
        "Systems",
        "Three readings of the same market, in the order the argument runs.",
        icon="cpu",
    )
    cols = st.columns(3, gap="small")
    for col, (cls, name, kicker, body, specs) in zip(cols, _SYSTEM_PANELS):
        with col:
            with panel(f"landing-{cls}", name, context=kicker):
                st.markdown(
                    f'<div class="panel-copy">{html_mod.escape(body)}</div>'
                    '<div class="panel-specs">'
                    + "".join(
                        f'<div class="lookback-row"><span class="lbl">{html_mod.escape(k)}</span>'
                        f'<span class="val">{html_mod.escape(v)}</span></div>'
                        for k, v in specs
                    )
                    + "</div>",
                    unsafe_allow_html=True,
                )

    # ── What a run returns ────────────────────────────────────────────────
    render_section_header(
        "What a run returns",
        f"Measured on the reference sheet {ctx['measured_date']}: "
        f"{ctx['rows']:,} rows spanning {ctx['span']}, holdout {ctx['holdout']}, "
        f"validated on {ctx['validated_on']} against {ctx['permutations']} permutations.",
        icon="target",
    )
    st.markdown(
        '<div class="outcome-grid">'
        + "".join(
            f'<div class="outcome"><div class="o-t">{html_mod.escape(t)}</div>'
            f'<div class="o-d">{html_mod.escape(d)}</div></div>'
            for t, d in _OUTCOMES
        )
        + "</div>",
        unsafe_allow_html=True,
    )
