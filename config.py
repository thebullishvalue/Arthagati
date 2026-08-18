# -*- coding: utf-8 -*-
"""
ARTHAGATI — Shared configuration.

Single source of truth for column schemas, model hyperparameters, and
display constants. Both ``arthagati.py`` (the Streamlit entrypoint) and
the ``ui/`` view modules import from here.

Why this module exists
----------------------
``ui/tabs/*.py`` used to do ``from arthagati import MSF_OB_LEVEL_1``.
Streamlit executes the entrypoint under the module name ``__main__``, so
that import does not hit ``sys.modules`` — it re-executes the entire
entrypoint as a *second* module object, duplicating every module-level
side effect (page config, CSS injection) and creating a divergent copy of
the engine's globals. Constants live here so no module ever has to import
the running script.
"""

from __future__ import annotations

# ══════════════════════════════════════════════════════════════════════════════
# IDENTITY — single source of truth for every version string in the repo
# ══════════════════════════════════════════════════════════════════════════════

VERSION      = "v2.11.0"
PRODUCT_NAME = "Arthagati"
COMPANY      = "@thebullishvalue"

# ══════════════════════════════════════════════════════════════════════════════
# DATA SCHEMA
# ══════════════════════════════════════════════════════════════════════════════

EXPECTED_COLUMNS: list[str] = [
    'DATE', 'NIFTY',
    'AD_RATIO', 'REL_AD_RATIO', 'REL_BREADTH', 'BREADTH', 'COUNT',
    'NIFTY50_PE', 'NIFTY50_EY', 'NIFTY50_DY', 'NIFTY50_PB',
    'IN10Y', 'IN02Y', 'IN30Y', 'INIRYY',
    'REPO', 'CRR',
    'US02Y', 'US10Y', 'US30Y', 'US_FED',
    'PE_DEV', 'EY_DEV',
]

# Default predictor set — chosen by measurement, not by argument.
#
# Selection protocol (research/): 65 sheet columns -> 37 eligible after
# removing NIFTY-derived columns (a valuation score must not be a function of
# price when it is then scored against price returns) and duplicates of
# columns the app derives itself. Redundant predictors were collapsed by
# correlation cluster (|rho| >= 0.90) to 24 representatives, then greedy
# forward selection maximised out-of-sample Spearman rho against forward
# NIFTY returns on the DEVELOPMENT window only (2006-2021). The holdout
# (2021-2026) was scored once, at the end.
#
#   development rho:  current 12 +0.189  ->  selected 4 +0.334
#   holdout rho:      current 12 +0.526  ->  selected 4 +0.544  (p = 0.005)
#
# All four are rate and liquidity variables. The breadth family — AD_RATIO,
# REL_AD_RATIO, REL_BREADTH, BREADTH, COUNT — ranked at the bottom of the
# univariate screen (rho +0.10 to +0.14 against +0.29 for SPREAD_02Y) and a
# breadth-only engine scores +0.434 on the holdout against +0.544 here.
#
# Honest caveat: the holdout gap between predictor sets is small (+0.53 to
# +0.55 across every set tested, breadth-only excepted). Most of the dev-set
# improvement did not transfer. See the ablation note in README — the edge
# belongs mainly to the PE anchor, not to the predictor mix.
DEPENDENT_VARS: list[str] = [
    'SPREAD_02Y',        # India 2Y minus US 2Y — cross-market policy spread
    'US_TERM_SPREAD',    # US 10Y - 2Y, derived in load_data()
    'CRR',               # Cash reserve ratio — domestic liquidity
    'US02Y',             # US 2-year yield — global rate anchor
]

# Everything eligible, offered in the sidebar multiselect. Columns derived
# from NIFTY are deliberately absent: selecting them would make the score a
# function of the price it is meant to be evaluated against.
CIRCULAR_COLUMNS: frozenset[str] = frozenset({
    'GAIN', 'LOSS', 'AVG GAIN', 'AVG LOSS', 'RS', 'RSI',
    'NIFTY MA20', 'STD MA20', 'BOL TEST MA20', '% CHNG',
    'OSC.', 'OSC MA50', 'NIFTY MA 90', 'NIFTY MA200 (8d lag)',
    'SPREAD90', 'SPREAD200', 'RVOL_20D',
    'COR. PE', 'PE_DEV', 'COR. EY', 'EY_DEV',
    'REGIME', 'SIGNAL_STR', 'BTD', 'STT', 'SPREAD',
})

# Sheet columns that duplicate something load_data() derives itself. Offering
# both would let the same yield-curve information be counted twice.
DUPLICATE_COLUMNS: frozenset[str] = frozenset({
    'IN_YC (10-2)',   # == IN_TERM_SPREAD
    'US_YC (10-2)',   # == US_TERM_SPREAD
})

# Minimum quality for a column to be offered as a predictor at all.
PREDICTOR_MIN_COVERAGE = 60.0   # percent of rows non-null and non-zero
PREDICTOR_MIN_UNIQUE   = 10     # distinct values

# Columns that are anchors or index keys, never predictors.
NON_PREDICTOR_COLS: frozenset[str] = frozenset({'DATE', 'NIFTY', 'NIFTY50_PE', 'NIFTY50_EY'})

# Columns the engine cannot synthesise a safe fallback for. Absence is a
# hard data-quality fault, surfaced in the UI rather than papered over.
REQUIRED_COLUMNS: tuple[str, ...] = ('DATE', 'NIFTY', 'NIFTY50_PE')

# Components the MSF oscillator needs real data for. A missing source used
# to be silently replaced with a constant, which then captured ~100% of the
# inverse-variance weight and flatlined the oscillator.
MSF_SOURCE_COLUMNS: dict[str, str] = {
    'momentum': 'NIFTY',
    'flow':     'AD_RATIO',
}

# ══════════════════════════════════════════════════════════════════════════════
# TIMEFRAMES
# ══════════════════════════════════════════════════════════════════════════════
# Values are CALENDAR days and are applied as a date filter against the last
# available observation — never as a row count. Taking `.tail(365)` on a
# trading-day series yields ~510 calendar days, so every window used to be
# ~1.4x longer than its label claimed.

TIMEFRAMES: dict[str, int | None] = {
    '1W':  7,
    '1M':  30,
    '3M':  90,
    '6M':  180,
    'YTD': None,   # resolved at runtime from 1 Jan of the last data year
    '1Y':  365,
    '2Y':  730,
    '5Y':  1825,
    'MAX': None,   # all available rows
}

# ══════════════════════════════════════════════════════════════════════════════
# MOOD ENGINE
# ══════════════════════════════════════════════════════════════════════════════

CORR_HALF_LIFE   = 504    # ~2 trading years; exponential recency weight for Spearman
PCT_HALF_LIFE    = 252    # ~1 trading year;  recency weight for adaptive ECDF
MOOD_SCALE       = 30.0   # maps the normalised signal → mood score
KALMAN_CI_Z      = 1.96   # Kalman confidence band (≈95%)
KALMAN_HALF_LIFE = 126    # Kalman fading memory half-life (trading days)

# Mood classification bands — fixed, so "Bullish" means the same thing every
# day (VISION.md §6). Sized from the score's realised distribution: over
# 2006-2026 the 1st-99th percentile range is about -49 to +56, so the former
# +/-60 outer band produced zero "Very Bearish" readings in twenty years,
# through both the GFC and the COVID crash.
MOOD_BAND_INNER = 20      # Neutral <-> Bullish / Bearish
MOOD_BAND_OUTER = 45      # Bullish / Bearish <-> Very Bullish / Very Bearish
DATA_TTL         = 3600   # Streamlit cache TTL for the Google Sheets fetch (seconds)

# Walk-forward correlation rebalancing.
# Statistics for segment k are estimated on data through checkpoint k−1, so
# every score at or after CORR_MIN_WARMUP is strictly causal. Rows before
# CORR_MIN_WARMUP are flagged Is_Warmup and excluded from all evaluation.
CORR_MIN_WARMUP       = 252   # warm-up length; scores before this are flagged
CORR_REBALANCE_PERIOD = 63    # expanding-window rebalance interval (≈quarterly)

# Regime diagnostics. Hurst on a smoothed sentiment index sits far above the
# 0.5 random-walk boundary (empirically ~84% of observations), so a fixed
# theoretical threshold collapses the four-quadrant scheme into one label.
# Both axes are therefore split at their own EXPANDING median — causal, and
# self-calibrating to the series. Labels are relative, not absolute.
REGIME_WINDOW       = 90    # Hurst / entropy lookback (trading days)
REGIME_MIN_HISTORY  = 252   # observations before adaptive thresholds engage
REGIME_SMOOTH       = 10    # smoothing applied to both axes before classification

# ══════════════════════════════════════════════════════════════════════════════
# MSF SPREAD
# ══════════════════════════════════════════════════════════════════════════════

MSF_WINDOW      = 20     # rolling window for all MSF components
MSF_ROC_LEN     = 14     # NIFTY rate-of-change period
MSF_ZSCORE_CLIP = 3.0    # Z-score clipping threshold
MSF_SCALE       = 10.0   # output scaling factor

# Inverse-variance weighting guard rails. Without these a zero-variance
# component takes 1/1e-6 inverse-variance, wins ~100% of the weight, and
# drags the composite to a flat zero.
MSF_MIN_WEIGHT      = 0.10   # floor per component after normalisation
MSF_MAX_WEIGHT      = 0.50   # cap per component after normalisation
MSF_MIN_WARMUP      = 60     # observations before inverse-variance engages
MSF_DEGENERATE_STD  = 1e-3   # component std below this ⇒ treated as dead

# Reference bands. One ladder, used by the chart, the divergence markers and
# the metric card alike (previously three different threshold sets).
MSF_OB_LEVEL_1  = 5      # Overbought primary
MSF_OB_LEVEL_2  = 3      # Overbought secondary
MSF_OS_LEVEL_1  = -5     # Oversold primary
MSF_OS_LEVEL_2  = -3     # Oversold secondary
MSF_SIGNAL_Y    = 4      # Divergence-triangle y-coordinate magnitude

# ══════════════════════════════════════════════════════════════════════════════
# WAVETREND
# ══════════════════════════════════════════════════════════════════════════════

WT_CHANNEL_LEN  = 10        # Channel length (PineScript: n1)
WT_AVERAGE_LEN  = 21        # Average length (PineScript: n2)
WT_SIGNAL_LEN   = 20        # Signal-line length (PineScript: wt2_len)
WT_SIGNAL_TYPE  = "ALMA"    # Signal-line smoother (PineScript: wt2_type)
WT_ALMA_OFFSET  = 0.85      # ALMA offset (ta.alma default)
WT_ALMA_SIGMA   = 6         # ALMA sigma  (ta.alma default)

# OB/OS levels are CALIBRATED FROM THE DATA, not inherited from LazyBear.
# The original ±80 / ±60 assume `ci` built from hlc3; driven by Mood_Score
# instead, |wt1| empirically peaks around 70 and never reaches 80, leaving
# the primary band permanently unreachable. These quantiles of |wt1| over
# the full history are used instead, with the constants below as fallbacks
# for short series.
WT_OB_QUANTILE_1 = 0.95     # primary band  = this quantile of |wt1|
WT_OB_QUANTILE_2 = 0.80     # secondary band
WT_OB_LEVEL_1   = 60        # fallback overbought primary
WT_OB_LEVEL_2   = 40        # fallback overbought secondary
WT_OS_LEVEL_1   = -60       # fallback oversold primary
WT_OS_LEVEL_2   = -40       # fallback oversold secondary

# ══════════════════════════════════════════════════════════════════════════════
# SIMILAR PERIODS
# ══════════════════════════════════════════════════════════════════════════════

SIMILAR_W_MAHA  = 0.55   # Mahalanobis distance weight
SIMILAR_W_TRAJ  = 0.35   # trajectory cosine-similarity weight
SIMILAR_W_RECV  = 0.10   # recency decay weight
TRAJ_WINDOW     = 20     # trajectory comparison window (trading days)

# Adjacent trading days are near-identical states. Without a separation
# constraint the "top 10 analogs" routinely collapse onto two or three
# episodes, and the median forward return is then quoted as if it came from
# ten independent observations.
SIMILAR_MIN_SEPARATION = 20   # trading days between accepted analogs
SIMILAR_EXCLUDE_TAIL   = 90   # trailing rows excluded (max forward horizon)

BACKTEST_HORIZON = 20    # forward-return horizon for the backtest scatter

# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

OU_PROJ_DAYS    = 90     # OU mean-reversion projection horizon (calendar days)
STALE_DATA_DAYS = 4      # calendar-day age before the staleness banner fires

# ══════════════════════════════════════════════════════════════════════════════
# PREDICTOR PROFILES
# ══════════════════════════════════════════════════════════════════════════════
#
# Presets for the sidebar dropdown, each carrying the measurement that
# justifies it. The numbers below were recorded on the reference sheet
# (see PROFILE_MEASUREMENT_CONTEXT) — they are a RECORD, not a live claim.
# The Signal Validation view re-measures whatever set is actually active.
#
# `predictors = None` means "every eligible column in the loaded sheet",
# resolved at runtime. Names absent from a sheet are dropped with a note
# rather than causing an error.

PROFILE_MEASUREMENT_CONTEXT: dict = {
    "sheet":         "NIFTY · reference sheet",
    "rows":          4985,
    "span":          "2006-06-08 → 2026-08-18",
    "holdout":       "2021-08-05 onward (1246 rows)",
    "validated_on":  "+20D, +60D",
    "descriptive":   "+125D, +250D — holdout too short to validate at that length",
    "baseline_rho":  0.5319,   # -PE alone, no engine, same window
    "measured_date": "2026-08-18",
    "permutations":  200,
}

PREDICTOR_PROFILES: dict[str, dict] = {
    "measured": {
        "label":       "Measured",
        "blurb":       "Chosen by measurement: greedy forward selection on 2006–2021, holdout scored once. Rate and liquidity variables only.",
        "predictors":  [
            "SPREAD_02Y",
            "US_TERM_SPREAD",
            "CRR",
            "US02Y",
        ],
        "measured": {
            "n":            4,
            "dev_rho":      0.3344,
            "holdout_rho":  0.5382,
            "p_value":      0.005,
            "verdict":      "Edge Confirmed",
            "per_horizon":  {20: 0.419, 60: 0.657, 125: 0.474, 250: 0.64},
        },
    },
    "rates": {
        "label":       "Rates & Liquidity",
        "blurb":       "Thematic: the policy-rate and yield-curve family, including the ones selection did not keep.",
        "predictors":  [
            "SPREAD_02Y",
            "US02Y",
            "CRR",
            "RATE_SPREAD",
            "IN_REAL_Y",
            "IN10Y",
            "YIELD_SPREAD",
            "REPO",
        ],
        "measured": {
            "n":            8,
            "dev_rho":      0.2879,
            "holdout_rho":  0.4915,
            "p_value":      0.005,
            "verdict":      "Edge Confirmed",
            "per_horizon":  {20: 0.384, 60: 0.599, 125: 0.482, 250: 0.676},
        },
    },
    "broad": {
        "label":       "Broad",
        "blurb":       "Every eligible column in the sheet. Maximum coverage, most dilution — the engine normalises weights, so weak predictors take weight from strong ones.",
        "predictors":  None,  # resolved at runtime: all eligible columns
        "measured": {
            "n":            37,
            "dev_rho":      0.2383,
            "holdout_rho":  0.4895,
            "p_value":      0.005,
            "verdict":      "Edge Confirmed",
            "per_horizon":  {20: 0.357, 60: 0.622, 125: 0.539, 250: 0.73},
        },
    },
    "valuation": {
        "label":       "Valuation & Volatility",
        "blurb":       "Thematic: price-to-book, dividend yield, VIX, USDINR, real yield.",
        "predictors":  [
            "NIFTY50_PB",
            "NIFTY50_DY",
            "INDIAVIX",
            "USDINR",
            "IN_REAL_Y",
        ],
        "measured": {
            "n":            5,
            "dev_rho":      0.1823,
            "holdout_rho":  0.4733,
            "p_value":      0.005,
            "verdict":      "Edge Confirmed",
            "per_horizon":  {20: 0.36, 60: 0.587, 125: 0.49, 250: 0.665},
        },
    },
    "legacy_29": {
        "label":       "Legacy · v2.9",
        "blurb":       "The v2.9.0 default. Term spreads promoted on the strength of VISION §2-I — which measurement did not support.",
        "predictors":  [
            "AD_RATIO",
            "REL_AD_RATIO",
            "REL_BREADTH",
            "COUNT",
            "IN_TERM_SPREAD",
            "US_TERM_SPREAD",
            "IN10Y",
            "US10Y",
            "INIRYY",
            "REPO",
            "NIFTY50_DY",
            "NIFTY50_PB",
        ],
        "measured": {
            "n":            12,
            "dev_rho":      0.1886,
            "holdout_rho":  0.444,
            "p_value":      0.005,
            "verdict":      "Edge Confirmed",
            "per_horizon":  {20: 0.31, 60: 0.578, 125: 0.536, 250: 0.71},
        },
    },
    "classic_28": {
        "label":       "Classic · v2.8",
        "blurb":       "The original shipped default: breadth, the full yield ladder, and two valuation ratios.",
        "predictors":  [
            "AD_RATIO",
            "REL_AD_RATIO",
            "REL_BREADTH",
            "COUNT",
            "IN10Y",
            "IN02Y",
            "IN30Y",
            "INIRYY",
            "REPO",
            "US02Y",
            "US10Y",
            "US30Y",
            "NIFTY50_DY",
            "NIFTY50_PB",
        ],
        "measured": {
            "n":            14,
            "dev_rho":      0.2131,
            "holdout_rho":  0.4315,
            "p_value":      0.005,
            "verdict":      "Edge Confirmed",
            "per_horizon":  {20: 0.297, 60: 0.566, 125: 0.536, 250: 0.663},
        },
    },
    "minimal": {
        "label":       "Minimal",
        "blurb":       "The single strongest predictor from the univariate screen. Closest to the pure PE anchor.",
        "predictors":  [
            "SPREAD_02Y",
        ],
        "measured": {
            "n":            1,
            "dev_rho":      0.291,
            "holdout_rho":  0.4177,
            "p_value":      0.005,
            "verdict":      "Edge Confirmed",
            "per_horizon":  {20: 0.359, 60: 0.477, 125: 0.324, 250: 0.312},
        },
    },
    "breadth": {
        "label":       "Breadth Only",
        "blurb":       "Market participation alone. Included as a contrast — it measurably underperforms, and the engine's anchors do the work.",
        "predictors":  [
            "AD_RATIO",
            "REL_AD_RATIO",
            "REL_BREADTH",
            "BREADTH",
            "COUNT",
            "A/(A+D)",
        ],
        "measured": {
            "n":            6,
            "dev_rho":      0.1409,
            "holdout_rho":  0.3239,
            "p_value":      0.005,
            "verdict":      "Edge Confirmed",
            "per_horizon":  {20: 0.218, 60: 0.43, 125: 0.457, 250: 0.672},
        },
    },
}

DEFAULT_PROFILE = "measured"
