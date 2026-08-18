# -*- coding: utf-8 -*-
"""
ARTHAGATI (अर्थगति) — Market Sentiment Analysis | An @thebullishvalue Product
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Physics-informed quantitative market-mood engine.

Pipeline (per Run Analysis):
    1. Data ingestion         — Google Sheets gviz API
    2. Correlation engine     — Decay-weighted Spearman vs PE & EY anchors
    3. Sentiment engine       — OU normalisation + Kalman smoothing
    4. MSF Spread             — Momentum · Structure · Regime · Flow oscillator
    5. WaveTrend              — Mood-Score-driven secondary oscillator

Views:
    • Historical Mood       — 3-pane TradingView-style chart (Mood + MSF + WT)
    • Similar Periods       — Mahalanobis + trajectory analog matching with
                              forward-return tiles at 5D / 20D / 60D / 90D
    • Correlation Analysis  — PE / EY decay-Spearman + entropy-weighted
                              predictor quality assessment
    • Signal Validation     — Holdout Spearman rho + permutation null
"""

import logging
import os
import sys
import time
from datetime import datetime
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
import requests
import streamlit as st

# ── Make ui/ + core/ importable when running `streamlit run arthagati.py`
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Structured terminal console (banner / phase / step / checkpoint / summary).
from core.logger_config import console, generate_run_id

# Quiet noisy library loggers — the structured console handles user-facing output
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
for _noisy in ("urllib3", "requests", "streamlit", "matplotlib"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="ARTHAGATI | Market Sentiment",
    page_icon="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iMTAiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzRDN0RGMCIgc3Ryb2tlLXdpZHRoPSIyIi8+PHBhdGggZD0iTTggMTRsMy01IDIgMyAzLTQiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzRDN0RGMCIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz48L3N2Zz4=",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# UI MODULE — Obsidian Quant design system
# ══════════════════════════════════════════════════════════════════════════════

from ui.theme import (
    inject_css,
    progress_bar,
    chart_color,
    VERSION,
    PRODUCT_NAME,
    COMPANY,
    SANSKRIT,
)
from ui.components import (
    render_ticker,
    render_top_bar,
    render_nav_brand,
    render_notice_rail,
    render_rail_readout,
    render_section_header,
    render_metric_card,
    render_kpi_strip,
    render_chip,
    render_empty_state,
    render_note,
    render_hero_card,
    build_hero_verdict,
    warmup_note,
    panel,
)
from ui import format as fmt
from ui import signals as sig
from ui.tabs.tab_landing import render_landing_page
from ui.tabs.tab_overview import render as render_overview
from ui.tabs.tab_mood import render as render_mood
from ui.tabs.tab_analogs import render as render_analogs
from ui.tabs.tab_drivers import render as render_drivers
from ui.tabs.tab_validation import render as render_validation
from ui.tabs.tab_config import render as render_config


# ══════════════════════════════════════════════════════════════════════════════
# DATA SOURCE
# ══════════════════════════════════════════════════════════════════════════════

# Google Sheets coordinates are read from environment variables.
# Set these in your deployment environment or .env file:
#   ARTHAGATI_SHEET_ID  = "<spreadsheet-id>"
#   ARTHAGATI_SHEET_GID = "<worksheet-gid>"
#
# The sheet must be accessible via the Google Visualization API (public with link).
# No service account authentication is needed — the gviz endpoint works without auth.

SHEET_ID  = os.environ.get("ARTHAGATI_SHEET_ID", "")
SHEET_GID = os.environ.get("ARTHAGATI_SHEET_GID", "")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — schemas, hyperparameters, display constants
# ══════════════════════════════════════════════════════════════════════════════
# All of these live in config.py so the ui/ package can read them without
# importing this module. Streamlit runs this file as `__main__`, so a
# `from arthagati import ...` elsewhere re-executes it as a second module
# object rather than hitting sys.modules.

from config import (  # noqa: E402
    EXPECTED_COLUMNS,
    DEPENDENT_VARS,
    NON_PREDICTOR_COLS,
    CIRCULAR_COLUMNS,
    DUPLICATE_COLUMNS,
    PREDICTOR_MIN_COVERAGE,
    PREDICTOR_MIN_UNIQUE,
    PREDICTOR_PROFILES,
    PROFILE_MEASUREMENT_CONTEXT,
    DEFAULT_PROFILE,
    REQUIRED_COLUMNS,
    MSF_SOURCE_COLUMNS,
    TIMEFRAMES,
    CORR_HALF_LIFE,
    PCT_HALF_LIFE,
    MOOD_SCALE,
    KALMAN_CI_Z,
    KALMAN_HALF_LIFE,
    DATA_TTL,
    CORR_MIN_WARMUP,
    CORR_REBALANCE_PERIOD,
    REGIME_WINDOW,
    REGIME_MIN_HISTORY,
    REGIME_SMOOTH,
    MSF_WINDOW,
    MSF_ROC_LEN,
    MSF_ZSCORE_CLIP,
    MSF_SCALE,
    MSF_MIN_WEIGHT,
    MSF_MAX_WEIGHT,
    MSF_MIN_WARMUP,
    MSF_DEGENERATE_STD,
    MSF_OB_LEVEL_1,
    MSF_OB_LEVEL_2,
    MSF_OS_LEVEL_1,
    MSF_OS_LEVEL_2,
    MSF_SIGNAL_Y,
    WT_CHANNEL_LEN,
    WT_AVERAGE_LEN,
    WT_SIGNAL_LEN,
    WT_SIGNAL_TYPE,
    WT_ALMA_OFFSET,
    WT_ALMA_SIGMA,
    WT_OB_QUANTILE_1,
    WT_OB_QUANTILE_2,
    WT_OB_LEVEL_1,
    WT_OB_LEVEL_2,
    WT_OS_LEVEL_1,
    WT_OS_LEVEL_2,
    SIMILAR_W_MAHA,
    SIMILAR_W_TRAJ,
    SIMILAR_W_RECV,
    SIMILAR_MIN_SEPARATION,
    SIMILAR_EXCLUDE_TAIL,
    TRAJ_WINDOW,
    BACKTEST_HORIZON,
    OU_PROJ_DAYS,
    STALE_DATA_DAYS,
    MOOD_BAND_INNER,
    MOOD_BAND_OUTER,
)

# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN LOOK-UP TABLES
# ══════════════════════════════════════════════════════════════════════════════

# Regime label → chip tone. One table, read by every surface that shows a
# regime, so the colour and the word cannot disagree between views. Colour
# itself is resolved per render through ui.theme.chart_color so it follows
# the appearance; only the SEMANTIC name is fixed here.
REGIME_TONE: dict[str, tuple[str, str]] = {
    "Trending":       ("emerald", "success"),
    "Volatile Trend": ("amber",   "warning"),
    "Mean-Reverting": ("cyan",    "info"),
    "Choppy":         ("rose",    "danger"),
    "Unknown":        ("slate",   "neutral"),
}
REGIME_STYLES = REGIME_TONE


# ══════════════════════════════════════════════════════════════════════════════
# ENGINE LOG VERBOSITY
# ══════════════════════════════════════════════════════════════════════════════

from contextlib import contextmanager as _contextmanager

# Suppresses the engine's trailing per-call detail line. Used by the test
# suite and by any batch caller that runs the engine in a loop.
_CALIBRATION_QUIET: bool = False


@_contextmanager
def quiet_engine_logs():
    """Suppress engine detail logs for the duration of a batch run."""
    global _CALIBRATION_QUIET
    prev = _CALIBRATION_QUIET
    _CALIBRATION_QUIET = True
    try:
        yield
    finally:
        _CALIBRATION_QUIET = prev


# ══════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM — ui/theme.css + ui/theme.py
# ══════════════════════════════════════════════════════════════════════════════
# inject_css() is called from main(), AFTER the appearance is resolved: the
# stylesheet and the charts must agree on one theme for the whole run, and the
# appearance control lives in the rail, i.e. further down the script.

# ── Boot banner (printed once per Streamlit process) ─────────────────────────
if not st.session_state.get("_arthagati_banner_printed"):
    console.header(f"ARTHAGATI {VERSION}", version="")
    console.item("Product", f"{PRODUCT_NAME} · अर्थगति")
    console.item("Maintainer", COMPANY)
    console.item("Engine", "OU · Kalman · Decay-Spearman · MSF Spread")
    console.line("─", 70)
    st.session_state["_arthagati_banner_printed"] = True


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

# The engine paths below call _progress_bar(); route it to the themed
# progress card, which derives its phase number from the percentage
# (see RUN_PHASES in ui/theme.py).
_progress_bar = progress_bar


def sigmoid(x, scale=1.0):
    """Sigmoid normalization to [-1, 1] range — overflow-safe."""
    z = np.clip(np.asarray(x, dtype=np.float64) / max(scale, 1e-12), -500, 500)
    return 2.0 / (1.0 + np.exp(-z)) - 1.0

def rolling_mean_fast(series, window):
    """O(N) rolling mean using numpy cumsums — NaN-aware (NaN values excluded from both sum and count)."""
    a = series.values if hasattr(series, 'values') else np.asarray(series, dtype=np.float64)
    n = len(a)
    if n == 0:
        return series

    valid = np.isfinite(a)
    a_clean = np.where(valid, a, 0.0)

    cs = np.cumsum(a_clean)
    cs_valid = np.cumsum(valid.astype(np.float64))

    cs_shifted = np.zeros(n, dtype=np.float64)
    cs_shifted[window:] = cs[:-window]
    cv_shifted = np.zeros(n, dtype=np.float64)
    cv_shifted[window:] = cs_valid[:-window]

    sums = cs - cs_shifted
    counts = cs_valid - cv_shifted

    # np.maximum prevents 0/0 division evaluation before np.where masks it
    means = np.where(counts > 0, sums / np.maximum(counts, 1.0), np.nan)
    return pd.Series(means, index=series.index) if hasattr(series, 'index') else means

def alma(series, window, offset=0.85, sigma=6.0):
    """Arnaud Legoux Moving Average — exact match for TradingView ``ta.alma(src, len, offset, sigma)``.

    A Gaussian-weighted MA whose weight peak is shifted toward the most recent
    bars by ``offset`` (0.85 ⇒ responsive) with spread controlled by ``sigma``.
    Weights are indexed oldest→newest: ``w[j] = exp(-(j - m)² / (2s²))`` with
    ``m = floor(offset·(window-1))`` and ``s = window / sigma``, then normalised.

    Returns NaN for the first ``window-1`` bars (partial windows), matching
    Pine's requirement of a full window before emitting a value.
    """
    a = series.values if hasattr(series, 'values') else np.asarray(series, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    n = len(a)
    idx = series.index if hasattr(series, 'index') else None

    if window <= 1 or n < window:
        # Degenerate: window 1 is the identity; too-short series is all-NaN.
        out = a.copy() if window <= 1 else np.full(n, np.nan, dtype=np.float64)
        return pd.Series(out, index=idx) if idx is not None else out

    m = np.floor(offset * (window - 1))
    s = window / sigma
    j = np.arange(window, dtype=np.float64)            # 0 = oldest … window-1 = newest
    w = np.exp(-((j - m) ** 2) / (2.0 * s * s))
    w /= w.sum()

    # Sliding dot product: each output is the weighted sum of the trailing `window` values.
    out = np.full(n, np.nan, dtype=np.float64)
    windows = np.lib.stride_tricks.sliding_window_view(a, window)   # (n-window+1, window), oldest→newest
    out[window - 1:] = windows @ w
    return pd.Series(out, index=idx) if idx is not None else out


def zscore_clipped(series, window, clip=3.0):
    """Z-score with rolling window and clipping — NaN-aware O(N) numpy cumsums."""
    a = series.values if hasattr(series, 'values') else np.asarray(series, dtype=np.float64)
    n = len(a)
    if n == 0:
        return series

    valid = np.isfinite(a)
    a_clean = np.where(valid, a, 0.0)

    cs = np.cumsum(a_clean)
    cs2 = np.cumsum(a_clean ** 2)
    cs_valid = np.cumsum(valid.astype(np.float64))

    cs_shifted = np.zeros(n, dtype=np.float64)
    cs_shifted[window:] = cs[:-window]
    cs2_shifted = np.zeros(n, dtype=np.float64)
    cs2_shifted[window:] = cs2[:-window]
    cv_shifted = np.zeros(n, dtype=np.float64)
    cv_shifted[window:] = cs_valid[:-window]

    sums = cs - cs_shifted
    sums2 = cs2 - cs2_shifted
    counts = cs_valid - cv_shifted

    means = np.where(counts > 0, sums / np.maximum(counts, 1.0), 0.0)
    var = np.where(counts > 1, (sums2 - (sums ** 2) / np.maximum(counts, 1)) / np.maximum(counts - 1, 1), 0.0)
    stds = np.sqrt(np.maximum(var, 0))

    with np.errstate(divide='ignore', invalid='ignore'):
        z = np.where(stds > 1e-12, (a_clean - means) / stds, 0.0)

    z = np.where(~valid, 0.0, z)
    z = np.clip(z, -clip, clip)
    return pd.Series(z, index=series.index) if hasattr(series, 'index') else z

# ══════════════════════════════════════════════════════════════════════════════
# v2.0 MATHEMATICAL PRIMITIVES
# ══════════════════════════════════════════════════════════════════════════════
#
# Design principle: every function has exactly ONE callsite and ONE job.
#
#   Function                        → Used in                   → Purpose
#   ─────────────────────────────────────────────────────────────────────────
#   exponential_decay_weights       → correlations              → recency weighting
#   weighted_spearman               → correlations              → robust rank correlation
#   shannon_entropy                 → variable weighting        → penalize noisy variables
#   adaptive_percentile             → mood scoring              → decay-weighted CDF
#   kalman_filter_1d                → mood smoothing            → adaptive noise filtering
#   rolling_hurst                   → diagnostics (output only) → trending vs reverting
#   rolling_entropy                 → diagnostics (output only) → market disorder
#   mahalanobis_distance_batch      → similar periods           → covariance-aware matching
#   cosine_similarity               → similar periods           → trajectory shape matching
#   detect_regime_transitions       → regime diagnostics        → quadrant classification
# ══════════════════════════════════════════════════════════════════════════════

def exponential_decay_weights(n, half_life):
    """
    Generate exponential decay weights for n observations.
    w_i = exp(-λ * i), λ = ln(2) / half_life.
    Most recent observation has weight 1.0, oldest decays toward 0.
    
    Used in: calculate_anchor_correlations (Layer 1)
    """
    if n <= 0:
        return np.array([])
    lam = np.log(2) / max(half_life, 1)
    indices = np.arange(n - 1, -1, -1, dtype=np.float64)
    weights = np.exp(-lam * indices)
    return weights / weights.sum()

def weighted_spearman(x, y, weights):
    """
    Exponential-decay-weighted Spearman rank correlation.
    Computes weighted Pearson on ranks — preserves rank-robustness
    while adding recency weighting.
    
    Why Spearman over Pearson:
      - Robust to outliers (rank-based, one extreme day doesn't dominate)
      - Captures monotonic nonlinear relationships (PE compression, yield inversions)
      - Invariant to marginal distribution shape
    
    Used in: calculate_anchor_correlations (Layer 1)
    """
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return 0.0
    x, y, w = x[valid], y[valid], weights[valid]
    
    def _rank(arr):
        sorter = np.argsort(arr)
        inv = np.empty(sorter.size, dtype=np.intp)
        inv[sorter] = np.arange(sorter.size, dtype=np.intp)
        
        arr_sorted = arr[sorter]
        obs = np.r_[True, arr_sorted[1:] != arr_sorted[:-1]]
        
        tie_indices = np.nonzero(obs)[0]
        if len(tie_indices) == len(arr):
            # Fast path: No ties, return standard ordinal rank
            return inv.astype(np.float64) + 1.0
            
        # Exact average-tie rank computation (fully C-vectorised)
        dense = np.cumsum(obs) - 1
        tie_counts = np.diff(np.r_[tie_indices, len(arr)])
        avg_ranks = tie_indices + (tie_counts + 1) / 2.0
        
        ranks_sorted = avg_ranks[dense]
        return ranks_sorted[inv]
    
    rx, ry = _rank(x), _rank(y)
    w_sum = w.sum()
    if w_sum == 0:
        return 0.0
    w_norm = w / w_sum
    
    mean_rx = np.sum(w_norm * rx)
    mean_ry = np.sum(w_norm * ry)
    
    cov_xy = np.sum(w_norm * (rx - mean_rx) * (ry - mean_ry))
    var_x = np.sum(w_norm * (rx - mean_rx) ** 2)
    var_y = np.sum(w_norm * (ry - mean_ry) ** 2)
    
    denom = np.sqrt(var_x * var_y)
    if denom < 1e-12:
        return 0.0
    return np.clip(cov_xy / denom, -1.0, 1.0)

def shannon_entropy(values, n_bins=20):
    """
    Shannon entropy H = -Σ p_i * log₂(p_i), normalized to [0, 1],
    with Miller-Madow bias correction: H_corrected = H_naive + (k-1)/(2·n·ln2)
    where k = number of non-empty bins, n = sample size.

    Beirlant et al. (1997) show the naive histogram plug-in estimator is biased
    by O(k/n); the Miller (1955) correction removes the first-order term.

    Bin count selection: Freedman-Diaconis rule (bin_width = 2·IQR·n^{-1/3}),
    floored at 5 and capped at n_bins to avoid overfitting on small samples.

    Used in: calculate_historical_mood → _build_weights (Layer 2)
    """
    clean = values[np.isfinite(values)]
    n_obs = len(clean)
    if n_obs < 5:
        return 0.5
    # Freedman-Diaconis bin-width: 2 × IQR × n^{-1/3}
    q75, q25 = np.percentile(clean, [75, 25])
    iqr = q75 - q25
    data_range = clean.max() - clean.min()
    if iqr > 1e-12 and data_range > 1e-12:
        fd_width = 2.0 * iqr * (n_obs ** (-1.0 / 3.0))
        fd_bins = int(np.ceil(data_range / fd_width))
    else:
        fd_bins = int(np.sqrt(n_obs))
    adaptive_bins = max(5, min(n_bins, fd_bins))
    counts, _ = np.histogram(clean, bins=adaptive_bins)
    probs = counts / counts.sum()
    non_empty = probs[probs > 0]
    if len(non_empty) <= 1:
        return 0.0
    h_naive = -np.sum(non_empty * np.log2(non_empty))
    # Miller-Madow first-order bias correction
    k = len(non_empty)
    h_corrected = h_naive + (k - 1) / (2.0 * n_obs * np.log(2))
    h_max = np.log2(adaptive_bins)
    return np.clip(h_corrected / h_max, 0.0, 1.0) if h_max > 0 else 0.0

class _FenwickTree:
    """Binary-indexed tree over value ranks, holding decayed observation mass.

    Supports O(log N) point-update and prefix-sum, plus an O(N) rescale used
    to keep the exponentially growing mass representable in float64.
    """

    __slots__ = ("_n", "_t")

    def __init__(self, n: int) -> None:
        self._n = n
        self._t = np.zeros(n + 1, dtype=np.float64)

    def add(self, i: int, value: float) -> None:
        """Add ``value`` at 1-based index ``i``."""
        t, n = self._t, self._n
        while i <= n:
            t[i] += value
            i += i & (-i)

    def prefix(self, i: int) -> float:
        """Sum over 1-based indices [1, i]."""
        t, s = self._t, 0.0
        while i > 0:
            s += t[i]
            i -= i & (-i)
        return s

    def rescale(self, factor: float) -> None:
        """Divide every stored mass by ``factor`` (the tree is linear)."""
        self._t /= factor


def adaptive_percentile(series, half_life=252):
    """
    Exponential-decay-weighted empirical CDF — O(N log N).

    For each time t the percentile of x_t is

        P(t) = Σ_{i≤t} w_i · 1(x_i ≤ x_t) / Σ_{i≤t} w_i,   w_i = exp(-λ(t-i))

    Implementation. Writing w_i = exp(-λt)·exp(λi), the exp(-λt) factor is
    common to numerator and denominator and cancels. What remains is a
    prefix-sum over value rank of the mass exp(λi), which a Fenwick tree
    answers in O(log N) per step — O(N log N) overall.

    The previous implementation re-derived the full decay vector at every
    step (O(N) per step, O(N²) overall) and was the single largest cost in
    the engine: ~436 ms at N=4000 versus ~9 ms here.

    exp(λi) grows without bound, so the tree is rescaled whenever the running
    total exceeds ``_RESCALE_AT``; because the tree is linear in its stored
    values, an elementwise divide is exact.

    Ties are handled by ranking against the sorted unique values with
    ``side='right'``, which counts every observation ≤ x_t.

    Used in: calculate_historical_mood (Layer 3)
    """
    values = np.asarray(series, dtype=np.float64)
    n = len(values)
    if n == 0:
        return np.array([])

    valid = np.isfinite(values)
    if not np.any(valid):
        return np.full(n, 0.5)

    lam = np.log(2) / max(half_life, 1)

    # Rank space: 1-based index of each value among the sorted unique values.
    uniques = np.unique(values[valid])
    ranks = np.searchsorted(uniques, values, side="right")  # 0 for NaN-ish lows
    n_ranks = len(uniques)

    tree = _FenwickTree(n_ranks)
    result = np.full(n, np.nan)

    total = 0.0
    _RESCALE_AT = 1e150
    log_scale = 0.0   # accumulated log of the rescale factors already applied

    for t in range(n):
        if not valid[t]:
            continue

        # Mass of this observation in the shifted frame.
        mass = np.exp(lam * t - log_scale)
        if not np.isfinite(mass):
            # Frame has drifted too far; re-anchor on the current step.
            shift = lam * t - log_scale
            tree.rescale(np.exp(shift))
            total /= np.exp(shift)
            log_scale += shift
            mass = 1.0

        tree.add(int(ranks[t]), mass)
        total += mass

        if total <= 0.0:
            continue
        # Clip to [0, 1]: the Fenwick prefix and the running total accumulate
        # in different orders, so their ratio can overshoot the closed unit
        # interval by an ulp or two. A percentile outside [0, 1] would flow
        # straight into the (1 - 2*pct) mapping in Layer 3.
        result[t] = min(max(tree.prefix(int(ranks[t])) / total, 0.0), 1.0)

        if total > _RESCALE_AT:
            tree.rescale(_RESCALE_AT)
            total /= _RESCALE_AT
            log_scale += np.log(_RESCALE_AT)

    return pd.Series(result).ffill().fillna(0.5).values

def kalman_filter_1d(
    observations: np.ndarray | pd.Series,
    process_var: float | None = None,
    measurement_var: float | None = None,
    half_life: int = KALMAN_HALF_LIFE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    1D Fading Memory Kalman Filter (Sorenson & Sacks).

    Uses an exponential fading factor to discount past data,
    preventing filter divergence in non-stationary regimes.

    Returns:
        filtered_state: Smoothed state estimates for each observation.
        estimate_variances: Posterior variance estimates (used for confidence bands).
    """
    obs = np.asarray(observations, dtype=np.float64)
    n = len(obs)
    if n == 0:
        return np.array([]), np.array([])
    
    # Causal noise estimation flags
    auto_process = process_var is None
    auto_measure = measurement_var is None
    
    s_obs = pd.Series(obs)
    
    # O(N) Causal variance estimations with burn-in bootstrap.
    # Harvey (1990): early expanding variance estimates are unreliable;
    # bootstrap the first BURN_IN observations from the first stable window.
    _BURN_IN = min(50, n // 4) if n > 20 else 1
    if auto_measure:
        m_vars = s_obs.expanding().var().fillna(1.0).values * 0.5
        m_vars = np.maximum(m_vars, 1e-8)
        if _BURN_IN > 1 and n > _BURN_IN:
            m_vars[:_BURN_IN] = m_vars[_BURN_IN]
    else:
        m_vars = np.full(n, measurement_var)

    if auto_process:
        p_vars = s_obs.diff().expanding().var().fillna(1e-3).values * 0.1
        p_vars = np.maximum(p_vars, 1e-8)
        if _BURN_IN > 1 and n > _BURN_IN:
            p_vars[:_BURN_IN] = p_vars[_BURN_IN]
    else:
        p_vars = np.full(n, process_var)
        
    state = obs[0] if np.isfinite(obs[0]) else 0.0
    estimate_var = m_vars[0]
    
    filtered = np.zeros(n)
    variances = np.zeros(n)
    filtered[0] = state
    variances[0] = estimate_var

    # Sorenson & Sacks Fading Memory parameter
    lam = np.log(2) / max(half_life, 1)
    alpha_sq = np.exp(2 * lam)  # Fading factor > 1

    for i in range(1, n):
        # Fading memory predict step
        pred_var = alpha_sq * estimate_var + p_vars[i]

        if np.isfinite(obs[i]):
            # Update step
            K = pred_var / (pred_var + m_vars[i])
            state = state + K * (obs[i] - state)
            estimate_var = (1 - K) * pred_var
        else:
            estimate_var = pred_var

        filtered[i] = state
        variances[i] = estimate_var

    return filtered, variances

def _hurst_dfa(series, max_lag=None):
    """
    Hurst exponent via Detrended Fluctuation Analysis (DFA-1).
    H > 0.5 → persistent (trending), H < 0.5 → anti-persistent (mean-reverting).

    DFA is more robust than R/S for short series and correctly distinguishes
    long-range dependence from short-range ARMA effects.
    Reference: Peng et al. (1994), "Mosaic organization of DNA nucleotides."
               Weron (2002) shows DFA outperforms R/S for n < 256.

    Internal helper for rolling_hurst.
    """
    ts = np.asarray(series, dtype=np.float64)
    ts = ts[np.isfinite(ts)]
    n = len(ts)
    if n < 20:
        return 0.5

    # Integrated profile: cumulative deviation from mean
    profile = np.cumsum(ts - ts.mean())

    min_scale = 10
    if max_lag is None:
        max_lag = min(n // 4, 200)
    if max_lag <= min_scale:
        return 0.5

    scales = np.unique(np.logspace(
        np.log10(min_scale), np.log10(max_lag), num=20,
    ).astype(int))
    scales = scales[(scales >= min_scale) & (scales <= max_lag)]

    if len(scales) < 3:
        return 0.5

    flucts = []
    for s in scales:
        n_seg = n // s
        if n_seg < 4:
            continue
        # Non-overlapping segments
        segments = profile[:n_seg * s].reshape(n_seg, s)

        # Vectorised linear detrend across all segments
        x = np.arange(s, dtype=np.float64)
        x_mean = x.mean()
        x_var = np.sum((x - x_mean) ** 2)
        if x_var < 1e-12:
            continue

        seg_means = segments.mean(axis=1, keepdims=True)
        slopes = np.sum((segments - seg_means) * (x - x_mean), axis=1) / x_var
        intercepts = seg_means.ravel() - slopes * x_mean

        trends = intercepts[:, None] + slopes[:, None] * x[None, :]
        residuals = segments - trends

        fluct = np.sqrt(np.mean(residuals ** 2))
        if fluct > 1e-12:
            flucts.append((s, fluct))

    if len(flucts) < 3:
        return 0.5

    log_s = np.log(np.array([f[0] for f in flucts], dtype=np.float64))
    log_f = np.log(np.array([f[1] for f in flucts], dtype=np.float64))

    valid = np.isfinite(log_s) & np.isfinite(log_f)
    if valid.sum() < 3:
        return 0.5
    log_s, log_f = log_s[valid], log_f[valid]
    mean_x, mean_y = log_s.mean(), log_f.mean()
    var_x = np.sum((log_s - mean_x) ** 2)
    H = np.sum((log_s - mean_x) * (log_f - mean_y)) / var_x if var_x > 1e-12 else 0.5
    return np.clip(H, 0.01, 0.99)

def rolling_hurst(series, window=90, step=5):
    """
    Rolling Hurst exponent via DFA. Computed every `step` points, forward-filled.
    Uses a sentinel to distinguish "not yet computed" from a legitimate H=0.5 estimate.
    Used in: calculate_historical_mood → diagnostics output
    """
    values = np.asarray(series, dtype=np.float64)
    n = len(values)
    _SENTINEL = -1.0  # impossible Hurst value — marks "not yet computed"
    result = np.full(n, _SENTINEL)
    for i in range(window, n, step):
        result[i] = _hurst_dfa(values[i - window:i])
    # Forward-fill only sentinel gaps (preserves legitimate H=0.5 estimates)
    for i in range(1, n):
        if result[i] == _SENTINEL and result[i - 1] != _SENTINEL:
            result[i] = result[i - 1]
    # Replace any remaining sentinels (before first computation) with 0.5
    result[result == _SENTINEL] = 0.5
    return result

def rolling_entropy(series, window=60, n_bins=15):
    """
    Rolling Shannon entropy of a series. Normalized to [0, 1].
    Used in: calculate_historical_mood → diagnostics output
    """
    from numpy.lib.stride_tricks import sliding_window_view
    
    values = series.values if hasattr(series, 'values') else np.asarray(series, dtype=np.float64)
    n = len(values)
    result = np.full(n, 0.5)
    if n < 5:
        return result
        
    if n >= window:
        # sliding_window_view on full array: windows[i] = values[i:i+window]
        # result[i+window-1] = entropy of values[i:i+window] (aligned to window end)
        windows = sliding_window_view(values, window)
        result[window - 1:window - 1 + len(windows)] = [shannon_entropy(w, n_bins) for w in windows]

    for i in range(5, min(window - 1, n)):
        result[i] = shannon_entropy(values[:i + 1], n_bins)
        
    return result

def _ledoit_wolf_shrinkage(S, n):
    """
    Ledoit & Wolf (2004) analytical shrinkage estimator.
    Σ* = δ·F + (1−δ)·S  where F = (tr(S)/p)·I  (scaled identity target).
    Optimal δ minimises E[‖Σ*−Σ‖²_F] under standard asymptotics.
    Returns the shrunk covariance matrix — always well-conditioned.
    """
    p = S.shape[0]
    if p == 0 or n < 2:
        return S
    trace_S = np.trace(S)
    mu = trace_S / p                       # target = μ·I
    delta_mat = S - mu * np.eye(p)
    sum_sq = np.sum(delta_mat ** 2)        # ‖S − μI‖²_F
    # Optimal shrinkage intensity (OAS closed-form, Chen et al. 2010)
    rho_num = ((1.0 - 2.0 / p) * sum_sq + trace_S ** 2)
    rho_den = ((n + 1.0 - 2.0 / p) * (sum_sq + trace_S ** 2 / p))
    rho = np.clip(rho_num / max(rho_den, 1e-12), 0.0, 1.0)
    return (1.0 - rho) * S + rho * mu * np.eye(p)

def mahalanobis_distance_batch(features, center, cov_matrix):
    """
    Mahalanobis distance: d_M = √((x−μ)ᵀ Σ⁻¹ (x−μ))
    Uses Ledoit-Wolf analytical shrinkage (2004) for a well-conditioned
    covariance inverse, replacing ad-hoc diagonal regularization.
    Used in: find_similar_periods
    """
    diff = features - center
    n_samples = features.shape[0]
    shrunk_cov = _ledoit_wolf_shrinkage(cov_matrix, n_samples)
    try:
        cov_inv = np.linalg.inv(shrunk_cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(shrunk_cov)
    left = diff @ cov_inv
    d_sq = np.maximum(np.sum(left * diff, axis=1), 0)
    return np.sqrt(d_sq)

def cosine_similarity(a, b):
    """
    Cosine similarity — measures trajectory shape match irrespective of magnitude.
    Used in: find_similar_periods → trajectory matching
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)

def detect_regime_transitions(
    hurst_values,
    entropy_values,
    window: int = REGIME_SMOOTH,
    min_history: int = REGIME_MIN_HISTORY,
):
    """
    Classify each observation into a Hurst x entropy quadrant.

        High H, Low S  -> Trending        (momentum works)
        High H, High S -> Volatile Trend  (directional, large swings)
        Low  H, Low S  -> Mean-Reverting  (range-bound)
        Low  H, High S -> Choppy          (hardest to trade)

    Thresholds are ADAPTIVE and CAUSAL.

    Both axes are split at their own *expanding* median rather than at a
    fixed constant. Two reasons:

      1. The theoretical H = 0.5 random-walk boundary does not apply here.
         Hurst is measured on the mood score, which is a smoothed composite
         of percentiles — empirically ~84% of observations sit above 0.5 and
         the upper quartile pins to the 0.99 clip, so a 0.5 split assigns
         nearly everything to "trending" and the four quadrants collapse to
         one. Entropy is similarly compressed (interquartile range ~0.89 to
         ~0.96).

      2. The previous implementation used the median of the *whole* series,
         which meant the regime label at time t depended on data from after
         t. The expanding median uses only observations up to and including
         t, so labels are reproducible and never revised by future data.

    Labels are therefore RELATIVE: "Trending" means persistent relative to
    this series' own history, not H > 0.5 in the absolute sense.

    Until ``min_history`` observations are available the classification is
    withheld ('Unknown') rather than guessed from a handful of points.

    Returns: (array of regime labels, list of transition records)
    """
    h = np.asarray(hurst_values, dtype=np.float64)
    s = np.asarray(entropy_values, dtype=np.float64)
    n = len(h)

    if n < max(window * 2, 4):
        return np.full(n, 'Unknown', dtype=object), []

    # Smooth both axes so single-point jitter doesn't trigger a transition.
    h_smooth = pd.Series(h).rolling(window=window, min_periods=1).mean()
    s_smooth = pd.Series(s).rolling(window=window, min_periods=1).mean()

    # Causal, self-calibrating thresholds.
    h_thresh = h_smooth.expanding(min_periods=1).median().to_numpy()
    s_thresh = s_smooth.expanding(min_periods=1).median().to_numpy()
    h_arr = h_smooth.to_numpy()
    s_arr = s_smooth.to_numpy()

    trending = h_arr > h_thresh
    ordered  = s_arr < s_thresh

    regimes = np.where(
        trending,
        np.where(ordered, 'Trending', 'Volatile Trend'),
        np.where(ordered, 'Mean-Reverting', 'Choppy'),
    ).astype(object)

    # Withhold a verdict until the adaptive thresholds have enough support.
    warm = min(min_history, n)
    regimes[:warm] = 'Unknown'

    # Transition records (a change of quadrant between consecutive points).
    major_pairs = {
        ('Trending', 'Choppy'), ('Choppy', 'Trending'),
        ('Trending', 'Mean-Reverting'), ('Mean-Reverting', 'Trending'),
    }
    transitions = []
    for i in range(1, n):
        prev, curr = regimes[i - 1], regimes[i]
        if prev == curr or 'Unknown' in (prev, curr):
            continue
        transitions.append({
            'index': i,
            'from': prev,
            'to': curr,
            'major': (prev, curr) in major_pairs,
            'hurst': float(h_arr[i]),
            'entropy': float(s_arr[i]),
        })

    return regimes, transitions

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_sheet_csv(max_retries: int = 3) -> str:
    """
    Fetch the Google Sheet as CSV via the Google Visualization API.

    Uses the /gviz/tq?tqx=out:csv endpoint — no OAuth/service account needed.
    The sheet must be set to "Anyone with the link can view" in sharing settings.

    Retries with exponential backoff on transient network failures.
    """
    if not SHEET_ID or not SHEET_GID:
        raise RuntimeError(
            "ARTHAGATI_SHEET_ID and ARTHAGATI_SHEET_GID environment variables are not set.\n"
            '  export ARTHAGATI_SHEET_ID="<spreadsheet-id>"   # from the sheet URL\n'
            '  export ARTHAGATI_SHEET_GID="<worksheet-gid>"   # the ?gid= parameter'
        )

    url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&gid={SHEET_GID}"

    last_exception = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            return resp.text
        except requests.exceptions.Timeout as e:
            last_exception = e
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt * 2
                console.warning(
                    f"Sheets request timed out (attempt {attempt + 1}/{max_retries}) — retrying in {wait_time}s"
                )
                time.sleep(wait_time)
            else:
                console.error(f"Sheets request failed after {max_retries} attempts: {e}")
        except requests.exceptions.RequestException as e:
            last_exception = e
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt * 2
                console.warning(
                    f"Sheets request failed (attempt {attempt + 1}/{max_retries}) — retrying in {wait_time}s"
                )
                time.sleep(wait_time)
            else:
                console.error(f"Sheets request failed after {max_retries} attempts: {e}")

    raise RuntimeError(f"Failed to load sheet data after {max_retries} attempts: {last_exception}")


@st.cache_data(ttl=DATA_TTL, show_spinner=False)
def load_data() -> pd.DataFrame | None:
    """
    Fetch and parse market data from the private Google Sheet.

    Returns a clean DataFrame with:
      - All columns present in the sheet (none fabricated from EXPECTED_COLUMNS)
      - DATE parsed, all other columns coerced to float
      - Derived columns: IN_TERM_SPREAD, US_TERM_SPREAD, NIFTY50_EY (if absent)
      - Rows with NIFTY ≤ 0 or unparseable DATE dropped
    """
    start_time = time.time()
    try:
        csv_text = _fetch_sheet_csv()
        df = pd.read_csv(StringIO(csv_text), dtype=str)

        # Normalise column names: strip whitespace, drop unnamed padding columns
        df.columns = [c.strip() for c in df.columns]
        df = df[[c for c in df.columns if not c.startswith('Unnamed')]]

        # Hard requirements — nothing works without these two
        if 'DATE' not in df.columns or 'NIFTY' not in df.columns:
            raise ValueError("Required columns DATE and NIFTY not found in the sheet.")

        # Warn about any known-schema columns absent in the sheet, but do NOT fabricate them.
        # The predictor dropdown will only show columns that genuinely exist in the data.
        missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
        if missing:
            console.issue(
                "SCHEMA",
                "load_data",
                f"{len(missing)} expected column(s) absent: {', '.join(missing)}",
            )

        df['DATE'] = pd.to_datetime(df['DATE'], format='%d/%m/%Y', errors='coerce')

        non_date_cols = [c for c in df.columns if c != 'DATE']
        df[non_date_cols] = df[non_date_cols].apply(pd.to_numeric, errors='coerce')
        # Forward-fill only: persistent data (rates, yields) carries forward.
        # No back-fill — it would leak future values into early observations.
        # NaN-only columns and series starts remain NaN; all downstream math
        # primitives have np.isfinite() guards that handle missing data correctly.
        df[non_date_cols] = df[non_date_cols].ffill()

        df = df[df['NIFTY'] > 0].dropna(subset=['DATE']).copy()
        if df.empty:
            raise ValueError("No valid rows after filtering on NIFTY > 0 and a parseable DATE.")

        # Preserve column order: DATE and NIFTY first, then everything else
        core = ['DATE', 'NIFTY']
        df = df[core + [c for c in df.columns if c not in core]].sort_values('DATE').reset_index(drop=True)

        # Derive NIFTY50_EY from PE if the sheet omits it or populates it as a constant.
        # EY = 1/PE × 100.
        if 'NIFTY50_PE' in df.columns and df['NIFTY50_PE'].gt(0).any():
            if 'NIFTY50_EY' not in df.columns or df['NIFTY50_EY'].nunique() <= 1:
                df['NIFTY50_EY'] = (1.0 / df['NIFTY50_PE'].replace(0, np.nan) * 100).fillna(0)
                console.detail("NIFTY50_EY absent or constant — derived from PE (EY = 1/PE × 100)")

        # Derive yield-curve term spreads (10Y − 2Y).
        # Positive = normal curve (expansion). Negative = inverted (recession signal).
        if 'IN10Y' in df.columns and 'IN02Y' in df.columns:
            df['IN_TERM_SPREAD'] = df['IN10Y'] - df['IN02Y']
        else:
            df['IN_TERM_SPREAD'] = 0.0
        if 'US10Y' in df.columns and 'US02Y' in df.columns:
            df['US_TERM_SPREAD'] = df['US10Y'] - df['US02Y']
        else:
            df['US_TERM_SPREAD'] = 0.0

        elapsed = time.time() - start_time
        date_range = f"{df['DATE'].iloc[0].strftime('%Y-%m-%d')} → {df['DATE'].iloc[-1].strftime('%Y-%m-%d')}"
        console.detail(
            f"Parsed {len(df):,} rows × {len(df.columns)} cols  ·  "
            f"{date_range}  ·  {elapsed:.2f}s"
        )
        return df

    except Exception as exc:
        console.failure("Data load", str(exc))
        st.error(f"Failed to load sheet data: {exc}")
        return None

# ══════════════════════════════════════════════════════════════════════════════
# MOOD SCORE CALCULATION
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(max_entries=5, show_spinner=False)
def calculate_anchor_correlations(df, anchor, dependent_vars=None):
    """
    Layer 1: Exponential-decay-weighted Spearman rank correlations.
    
    Half-life ~504 days (~2 trading years). This means:
    - Data from 2 years ago has half the weight of today
    - Data from 4 years ago has 1/4 the weight
    - Very old regimes fade naturally without being discarded
    """
    if dependent_vars is None:
        dependent_vars = DEPENDENT_VARS
    cols_to_check = [col for col in dependent_vars if col in df.columns]
    
    if anchor not in df.columns or not cols_to_check:
        return pd.DataFrame(columns=['variable', 'correlation', 'strength', 'type'])
    
    analysis_df = df[[anchor] + cols_to_check].select_dtypes(include=[np.number])
    if anchor not in analysis_df.columns:
        return pd.DataFrame(columns=['variable', 'correlation', 'strength', 'type'])
    
    anchor_vals = analysis_df[anchor].values
    n = len(anchor_vals)
    half_life = min(CORR_HALF_LIFE, n // 2) if n > 20 else max(n // 2, 5)
    weights = exponential_decay_weights(n, half_life)
    
    correlations = []
    for var in cols_to_check:
        if var == anchor or var not in analysis_df.columns:
            continue
        corr = weighted_spearman(anchor_vals, analysis_df[var].values, weights)
        if not np.isfinite(corr):
            corr = 0.0
        abs_corr = abs(corr)
        strength = ('Strong' if abs_corr >= 0.7 else
                   'Moderate' if abs_corr >= 0.5 else
                   'Weak' if abs_corr >= 0.3 else 'Very weak')
        correlations.append({
            'variable': var, 'correlation': corr,
            'strength': strength, 'type': 'positive' if corr > 0 else 'negative'
        })
    
    return pd.DataFrame(correlations)

def _calculate_historical_mood_impl(df, dependent_vars=None):
    """
    v2.3 Mood Score Engine — 5-layer architecture with walk-forward weights.

    Fixes vs v2.2:
      - Layers 1+2 now use EXPANDING-WINDOW correlations and entropy at periodic
        checkpoints (CORR_REBALANCE_PERIOD), eliminating look-ahead bias.
      - Layer 3 percentile semantics corrected: adjustments are symmetric [-1,+1]
        around zero (was [0,+1], creating asymmetric bearish/bullish capacity).
      - Layer 4 OU bias correction applied (Kendall-Marriott-Pope on AR(1) coef).
      - Layer 5 Kalman uses its own half-life (KALMAN_HALF_LIFE), decoupled from PCT.

    Diagnostics (output-only, do NOT modify the score):
      Hurst exponent (DFA), market entropy, OU half-life
    """
    if dependent_vars is None:
        dependent_vars = DEPENDENT_VARS
    start_time = time.time()

    if 'DATE' not in df.columns or 'NIFTY50_PE' not in df.columns or 'NIFTY50_EY' not in df.columns:
        console.failure(
            "Mood engine",
            "required anchor columns missing — sheet must contain DATE, NIFTY50_PE, NIFTY50_EY",
        )
        return pd.DataFrame(columns=['DATE', 'Mood_Score', 'Mood', 'Smoothed_Mood_Score', 'Mood_Volatility'])

    n = len(df)
    vars_to_check = [col for col in dependent_vars
                     if col in df.columns and col not in NON_PREDICTOR_COLS]

    # ── Layer 3 (computed first): Adaptive Percentiles ────────────────
    # These are already expanding-window (no look-ahead).
    pct_hl = min(PCT_HALF_LIFE, n // 2) if n > 20 else max(n // 2, 5)

    pe_percentiles = adaptive_percentile(df['NIFTY50_PE'].values, half_life=pct_hl)
    ey_percentiles = adaptive_percentile(df['NIFTY50_EY'].values, half_life=pct_hl)

    var_percentiles = {}
    for var in vars_to_check:
        var_percentiles[var] = adaptive_percentile(df[var].values, half_life=pct_hl)

    # ── Layers 1+2: Walk-Forward Correlations & Entropy ───────────────
    # At each checkpoint, compute expanding Spearman correlations and expanding
    # entropy using ONLY data available up to that point — no look-ahead.
    anchor_pe = df['NIFTY50_PE'].values
    anchor_ey = df['NIFTY50_EY'].values

    min_warmup = min(CORR_MIN_WARMUP, n // 2) if n > 50 else max(n // 3, 10)
    rebal = max(min(CORR_REBALANCE_PERIOD, max((n - min_warmup) // 3, 1)), 1)

    checkpoints = list(range(min_warmup, n, rebal))
    if not checkpoints or checkpoints[-1] != n - 1:
        checkpoints.append(n - 1)

    # Pre-compute first differences for the expanding entropy estimate.
    #
    # These used to be relative changes (diff / |prev|). Several predictors
    # cross zero — both term spreads by construction, plus PE_DEV / EY_DEV —
    # and a denominator near zero produced returns of several hundred x. A
    # single such spike dominates the Freedman-Diaconis bin width and
    # corrupts that variable's entropy weight for the rest of the run.
    #
    # Entropy is estimated from a histogram whose bin width already adapts to
    # the data's scale, so it is invariant to the units of the differenced
    # series. Plain first differences are therefore both safe and sufficient.
    var_returns_all = {}
    for var in vars_to_check:
        vals = df[var].values
        rets = np.empty(len(vals))
        rets[0] = np.nan
        rets[1:] = np.diff(vals)
        rets = np.where(np.isfinite(rets), rets, np.nan)
        var_returns_all[var] = rets

    # Accumulate adjustments and strengths segment-by-segment
    pe_base = 1.0 - 2.0 * pe_percentiles     # High PE → low score (bearish)
    ey_base = 2.0 * ey_percentiles - 1.0      # High EY → high score (bullish)

    pe_adjustments = np.zeros(n)
    ey_adjustments = np.zeros(n)
    pe_strength_arr = np.zeros(n)
    ey_strength_arr = np.zeros(n)

    # Exponential weight blending across checkpoints to smooth discontinuities.
    # At each checkpoint, new weights are blended with previous:
    #   w_eff = α·w_new + (1−α)·w_prev,  α = 1 − exp(−ln(2)/blend_hl)
    # First checkpoint uses α=1 (no prior to blend with).
    _BLEND_HL = 2.0  # in checkpoint units (≈2 rebalance periods to fully converge)
    _blend_alpha = 1.0 - np.exp(-np.log(2) / max(_BLEND_HL, 0.5))
    prev_pe_w: dict[str, float] = {}
    prev_ey_w: dict[str, float] = {}
    prev_pe_corrs: dict[str, float] = {}
    prev_ey_corrs: dict[str, float] = {}

    for cp_idx, cp in enumerate(checkpoints):
        seg_start = checkpoints[cp_idx - 1] + 1 if cp_idx > 0 else 0
        seg_end = cp + 1

        # CAUSALITY: the statistics applied to a segment must be estimated
        # on data that ends BEFORE the segment begins.
        #
        # This block previously used `cp_n = cp + 1` — data through the END
        # of the very segment being scored — so a score at time t depended
        # on up to CORR_REBALANCE_PERIOD days of its own future. Measured on
        # synthetic data, perturbing only rows after index 300 moved scores
        # inside the untouched prefix by as much as 12.75 points.
        #
        # Segment k now reads its weights from checkpoint k-1. The first
        # segment has no prior checkpoint, so it borrows the first one and
        # is flagged Is_Warmup; those rows are excluded from every
        # evaluation path (validation blocks, backtest, rho scoring).
        cp_n = (checkpoints[cp_idx - 1] + 1) if cp_idx > 0 else (checkpoints[0] + 1)
        cp_half_life = min(CORR_HALF_LIFE, cp_n // 2) if cp_n > 20 else max(cp_n // 2, 5)
        cp_weights = exponential_decay_weights(cp_n, cp_half_life)

        # Expanding correlations and entropy at this checkpoint
        cp_pe_corrs = {}
        cp_ey_corrs = {}
        cp_entropies = {}

        for var in vars_to_check:
            var_vals = df[var].values[:cp_n]

            # Expanding entropy on returns available up to checkpoint
            rets_cp = var_returns_all[var][1:cp_n]
            clean_rets = rets_cp[np.isfinite(rets_cp)]
            cp_entropies[var] = shannon_entropy(clean_rets) if len(clean_rets) > 10 else 0.5

            # Expanding Spearman with PE and EY
            pe_c = weighted_spearman(anchor_pe[:cp_n], var_vals, cp_weights)
            ey_c = weighted_spearman(anchor_ey[:cp_n], var_vals, cp_weights)
            cp_pe_corrs[var] = pe_c if np.isfinite(pe_c) else 0.0
            cp_ey_corrs[var] = ey_c if np.isfinite(ey_c) else 0.0

        # Build raw weights: |corr| × (1 − entropy)
        pe_raw_w, ey_raw_w = {}, {}
        for var in vars_to_check:
            entropy_pen = 1.0 - cp_entropies.get(var, 0.5)
            pe_raw_w[var] = abs(cp_pe_corrs[var]) * max(entropy_pen, 0.1)
            ey_raw_w[var] = abs(cp_ey_corrs[var]) * max(entropy_pen, 0.1)

        pe_total = max(sum(pe_raw_w.values()), 1e-10)
        ey_total = max(sum(ey_raw_w.values()), 1e-10)
        pe_w_new = {k: v / pe_total for k, v in pe_raw_w.items()}
        ey_w_new = {k: v / ey_total for k, v in ey_raw_w.items()}

        # Blend with previous checkpoint weights (first checkpoint: α=1, use raw)
        if prev_pe_w:
            pe_w = {v: _blend_alpha * pe_w_new.get(v, 0.0) + (1.0 - _blend_alpha) * prev_pe_w.get(v, 0.0) for v in vars_to_check}
            ey_w = {v: _blend_alpha * ey_w_new.get(v, 0.0) + (1.0 - _blend_alpha) * prev_ey_w.get(v, 0.0) for v in vars_to_check}
        else:
            pe_w = pe_w_new
            ey_w = ey_w_new
        prev_pe_w = dict(pe_w)
        prev_ey_w = dict(ey_w)

        # Also blend correlations for sign stability
        if prev_pe_corrs:
            blended_pe_corrs = {v: _blend_alpha * cp_pe_corrs.get(v, 0.0) + (1.0 - _blend_alpha) * prev_pe_corrs.get(v, 0.0) for v in vars_to_check}
            blended_ey_corrs = {v: _blend_alpha * cp_ey_corrs.get(v, 0.0) + (1.0 - _blend_alpha) * prev_ey_corrs.get(v, 0.0) for v in vars_to_check}
        else:
            blended_pe_corrs = dict(cp_pe_corrs)
            blended_ey_corrs = dict(cp_ey_corrs)
        prev_pe_corrs = dict(blended_pe_corrs)
        prev_ey_corrs = dict(blended_ey_corrs)

        pe_str = sum(abs(blended_pe_corrs[v]) for v in vars_to_check)
        ey_str = sum(abs(blended_ey_corrs[v]) for v in vars_to_check)

        # Compute adjustments for this segment using blended correlations/weights
        seg_pe = np.zeros(seg_end - seg_start)
        seg_ey = np.zeros(seg_end - seg_start)

        for var in vars_to_check:
            vpct = var_percentiles[var][seg_start:seg_end]

            # FIXED percentile semantics (L1):
            # PE: positive corr + high var_pct → high PE → bearish → push score DOWN
            #     Adjustment = sign × weight × (1 − 2·pct) maps [0,1] → [+1,−1]
            pe_sign = 1.0 if blended_pe_corrs[var] >= 0 else -1.0
            seg_pe += pe_sign * pe_w[var] * (1.0 - 2.0 * vpct)

            # EY: positive corr + high var_pct → high EY → bullish → push score UP
            #     Adjustment = sign × weight × (2·pct − 1) maps [0,1] → [−1,+1]
            ey_sign = 1.0 if blended_ey_corrs[var] >= 0 else -1.0
            seg_ey += ey_sign * ey_w[var] * (2.0 * vpct - 1.0)

        pe_adjustments[seg_start:seg_end] = seg_pe
        ey_adjustments[seg_start:seg_end] = seg_ey
        pe_strength_arr[seg_start:seg_end] = pe_str
        ey_strength_arr[seg_start:seg_end] = ey_str

    pe_scores = np.clip(0.5 * pe_base + 0.5 * pe_adjustments, -1, 1)
    ey_scores = np.clip(0.5 * ey_base + 0.5 * ey_adjustments, -1, 1)

    total_strength = pe_strength_arr + ey_strength_arr
    total_strength = np.where(total_strength > 0, total_strength, 1.0)
    raw_mood = (pe_strength_arr / total_strength) * pe_scores + (ey_strength_arr / total_strength) * ey_scores

    # ── Layer 4: OU Normalization ───────────────────────────────────────
    # Expanding z-score to get rough scale
    counts = np.arange(1, n + 1)
    cum_sum = np.cumsum(raw_mood)
    expanding_mean = cum_sum / counts

    cum_sq_sum = np.cumsum(raw_mood ** 2)
    var_expanding = (cum_sq_sum - (cum_sum ** 2) / counts) / np.maximum(counts - 1, 1)
    expanding_std = np.maximum(np.sqrt(np.maximum(var_expanding, 0)), 1e-6)
    expanding_std[0] = 1.0

    rough_scaled = (raw_mood - expanding_mean) / expanding_std

    # Vectorised Expanding OU Estimation with bias correction.
    #
    # H2 Fix: The previous algebraic expanding RSS (cumsum(y²) + n·a² + ...)
    # is only correct when (a, b) are constant; with per-step expanding estimates
    # that change at every index, the cross-terms are inconsistent.
    #
    # Correct approach: compute the per-observation residual e²_i = (y_i − a_i − b_i·x_i)²
    # using the current expanding (a, b) at each step, then EMA-smooth these squared
    # residuals for a stable variance estimate.
    ou_thetas = np.full(n, 0.05)
    ou_mus = np.zeros(n)
    ou_sigmas = np.ones(n)

    x_ou = rough_scaled[:-1]
    y_ou = rough_scaled[1:]
    n_points = np.arange(1, n)

    sum_x = np.cumsum(x_ou)
    sum_y = np.cumsum(y_ou)
    sum_x2 = np.cumsum(x_ou ** 2)
    sum_xy = np.cumsum(x_ou * y_ou)

    mean_x_ou = sum_x / n_points
    mean_y_ou = sum_y / n_points

    var_x_ou = sum_x2 - (sum_x ** 2) / n_points
    cov_xy_ou = sum_xy - (sum_x * sum_y) / n_points

    var_x_safe = np.where(var_x_ou < 1e-12, 1e-12, var_x_ou)
    b_hat = cov_xy_ou / var_x_safe

    # Kendall-Marriott-Pope first-order bias correction (vectorised)
    b = b_hat + (1.0 + 3.0 * b_hat) / np.maximum(n_points, 1)
    b = np.clip(b, 1e-6, 1.0 - 1e-6)

    a_ou = mean_y_ou - b * mean_x_ou

    theta_vals = np.clip(-np.log(b), 1e-4, 10.0)
    mu_vals = a_ou / (1.0 - b)

    # Per-observation residuals using the current expanding (a, b) at each step.
    # e²_i = (y_i − a_i − b_i·x_i)² — each residual uses the correct parameters.
    per_residual_sq = (y_ou - a_ou - b * x_ou) ** 2
    # Expanding mean of squared residuals (correct RSS regardless of how a,b vary)
    var_eps = np.maximum(np.cumsum(per_residual_sq) / n_points, 0)

    denom_ou = np.maximum(1.0 - b ** 2, 1e-12)
    sigma_sq = np.where((1.0 - b ** 2) > 1e-12, 2.0 * theta_vals * var_eps / denom_ou, var_eps)
    sigma_vals = np.sqrt(np.maximum(sigma_sq, 1e-12))

    valid_idx = n_points >= 50
    ou_thetas[1:][valid_idx] = theta_vals[valid_idx]
    ou_mus[1:][valid_idx] = mu_vals[valid_idx]
    ou_sigmas[1:][valid_idx] = sigma_vals[valid_idx]

    t_std = np.maximum(ou_sigmas / np.sqrt(2.0 * np.maximum(ou_thetas, 1e-4)), 1e-6)
    mood_scores = np.clip((rough_scaled - ou_mus) / t_std * MOOD_SCALE, -100, 100)

    theta, mu, sigma_ou = ou_thetas[-1], ou_mus[-1], ou_sigmas[-1]
    ou_half_life = np.log(2) / max(theta, 1e-4)

    # ── Layer 5: Kalman Smoothing ───────────────────────────────────────
    smoothed_mood_scores, kalman_variances = kalman_filter_1d(mood_scores)

    # Confidence band: ±KALMAN_CI_Z × √variance (~95% interval)
    kalman_std = np.sqrt(np.maximum(kalman_variances, 0))
    confidence_upper = smoothed_mood_scores + KALMAN_CI_Z * kalman_std
    confidence_lower = smoothed_mood_scores - KALMAN_CI_Z * kalman_std

    # Traditional volatility (backward compatible)
    mood_volatility = pd.Series(mood_scores).rolling(window=30, min_periods=1).std().fillna(0)

    # ── Classification (fixed thresholds — see VISION.md §6 for why) ───
    # Classification bands.
    #
    # Fixed, not adaptive — per VISION.md §6, "Bullish" must mean the same
    # thing today as it did last year. But a fixed band still has to be
    # reachable: at the previous +/-60 outer threshold, twenty years of real
    # NIFTY data (2006-2026, spanning the GFC and the COVID crash) produced
    # ZERO "Very Bearish" readings and only 29 "Very Bullish" days, because
    # the score's actual 1st-99th percentile range is about -49 to +56, not
    # -100 to +100. An extreme label that never fires is not a stable
    # classification, it is a dead one.
    #
    # MOOD_BAND_OUTER sits near the 2nd/98th percentile of the long-run
    # distribution, so all five classes are attainable while the extremes
    # stay genuinely rare.
    moods = np.where(mood_scores > MOOD_BAND_OUTER, 'Very Bullish',
            np.where(mood_scores > MOOD_BAND_INNER, 'Bullish',
            np.where(mood_scores > -MOOD_BAND_INNER, 'Neutral',
            np.where(mood_scores > -MOOD_BAND_OUTER, 'Bearish', 'Very Bearish'))))

    # ── Diagnostics (output-only — do NOT modify scores) ───────────────
    # Mood-domain regime panel: Hurst, entropy, and OU half-life all describe
    # the same series (mood score) over a unified 90d window, so the three
    # readings stay internally consistent. Computing Hurst on price *levels*
    # produces H≈1.0 trivially (integrated random walk) — that's why this
    # operates on mood_scores, which are OU-normalized and stationary.
    # Hurst is measured on mood INCREMENTS, not on the level.
    #
    # DFA applied to a strongly persistent series returns H > 1, which the
    # estimator clips to 0.99. The mood score is exactly that kind of series
    # — a smoothed composite of slow-moving percentiles — so on real data the
    # level-Hurst pinned to the clip on 87% of observations (p5/p50/p95 =
    # 0.77/0.99/0.99) and carried no information at all. Every regime
    # comparison then reduced to `0.99 > 0.99` = False, collapsing 90% of
    # history into the two low-Hurst quadrants.
    #
    # For an integrated series, H(level) = H(increments) + 1, so the
    # increments carry the same information in a range the estimator can
    # actually resolve: on the same data, p5/p50/p95 = 0.24/0.48/0.78 with
    # 0.3% at the clip. The 0.5 boundary recovers its textbook meaning —
    # increments that persist (trending mood) versus increments that reverse.
    mood_increments = np.diff(mood_scores, prepend=mood_scores[0])
    hurst_vals = rolling_hurst(mood_increments, window=REGIME_WINDOW, step=5)
    entropy_vals = rolling_entropy(mood_scores, window=REGIME_WINDOW, n_bins=15)

    # ── Regime Detection ────────────────────────────────────────────────
    regime_labels, regime_transitions = detect_regime_transitions(hurst_vals, entropy_vals)

    result_df = pd.DataFrame({
        'DATE': df['DATE'].values,
        'Mood_Score': mood_scores,
        'Mood': moods,
        'Smoothed_Mood_Score': smoothed_mood_scores,
        'Mood_Volatility': mood_volatility.values,
        'NIFTY': df['NIFTY'].values,
        'AD_RATIO': df['AD_RATIO'].values if 'AD_RATIO' in df.columns else np.ones(n),
        # v2.0 diagnostics
        'Hurst': hurst_vals,
        'Market_Entropy': entropy_vals,
        'OU_Half_Life': ou_half_life,
        'OU_Theta': theta,
        'OU_Mu': mu,
        # v2.1 additions
        'OU_Sigma': sigma_ou,
        # Soft-clip: tanh preserves band *width* near the extremes so users
        # still see how uncertain the reading is, unlike a hard clip at ±100
        # which would make the band appear artificially narrow.
        'Confidence_Upper': np.tanh(confidence_upper / 100.0) * 100.0,
        'Confidence_Lower': np.tanh(confidence_lower / 100.0) * 100.0,
        'Regime': regime_labels,
        # True for rows scored with borrowed (non-causal) warm-up statistics.
        # Every evaluation path filters these out; the chart still draws them
        # so the Kalman and MSF chains stay continuous.
        'Is_Warmup': np.arange(n) < min(min_warmup, n),
    })

    if not _CALIBRATION_QUIET:
        console.detail(
            f"Mood engine complete — {n:,} rows in {time.time() - start_time:.2f}s  ·  "
            f"OU: θ={theta:.3f} μ={mu:.2f} t½={ou_half_life:.0f}d  ·  "
            f"Hurst={hurst_vals[-1]:.2f} Entropy={entropy_vals[-1]:.2f} Regime={regime_labels[-1]}  ·  "
            f"Walk-forward checkpoints: {len(checkpoints)}"
        )
    return result_df


@st.cache_data(max_entries=5, show_spinner=False)
def calculate_historical_mood(df, dependent_vars=None):
    """Cached public entry — delegates to ``_calculate_historical_mood_impl``.

    The engine always runs on factory hyperparameters. Intelligence Mode
    calibrates a small ensemble on TOP of this output rather than tuning the
    engine's internals, so there is no swapped-globals path to invalidate.
    """
    return _calculate_historical_mood_impl(df, dependent_vars)


# ══════════════════════════════════════════════════════════════════════════════
# MSF-ENHANCED SPREAD INDICATOR
# ══════════════════════════════════════════════════════════════════════════════

def _calculate_msf_spread_impl(df, mood_col='Mood_Score', nifty_col='NIFTY', breadth_col='AD_RATIO'):
    """
    v2.0 MSF-Enhanced Spread Indicator.
    
    4 components (same purpose as v1.x — momentum/structure alignment detection):
      1. Momentum  — NIFTY ROC z-score (price velocity)
      2. Structure — Mood trend divergence + acceleration (mood curvature)
      3. Regime    — Adaptive-threshold directional count (market character)
      4. Flow      — Breadth divergence from mean (participation width)
    
    v2.0 changes:
      - Regime threshold adapts to local volatility (was: fixed 0.0033)
      - Inverse-variance weighting (was: fixed 30/25/25/20)
        Stable components get more weight — minimum-variance portfolio of signals.
    """
    start_time = time.time()
    result = pd.DataFrame(index=df.index)
    n = len(df)
    
    # Missing sources used to be replaced with constants (or, for NIFTY,
    # with the mood score itself). Both choices produced a plausible-looking
    # but meaningless oscillator. Absence is now recorded so the component is
    # excluded from the composite and reported to the user.
    missing_sources: list[str] = []
    mood = df[mood_col].values if mood_col in df.columns else np.zeros(n)
    if nifty_col in df.columns:
        nifty = df[nifty_col].values
    else:
        missing_sources.append(nifty_col)
        nifty = np.full(n, np.nan)
    if breadth_col in df.columns:
        breadth = df[breadth_col].values
    else:
        missing_sources.append(breadth_col)
        breadth = np.full(n, np.nan)
    
    mood_series = pd.Series(mood, index=df.index)
    nifty_series = pd.Series(nifty, index=df.index)
    breadth_series = pd.Series(breadth, index=df.index)
    
    if n == 0:
        console.failure("MSF Spread", "received an empty DataFrame — no rows to process")
        return result

    if missing_sources and not _CALIBRATION_QUIET:
        console.issue(
            "SCHEMA", "MSF Spread",
            f"source column(s) absent: {', '.join(missing_sources)} — "
            f"the dependent component(s) will be excluded from the composite",
        )
    
    # ── Component 1: Momentum (NIFTY ROC z-score) ──────────────────────
    roc_raw = nifty_series.pct_change(MSF_ROC_LEN)
    roc_z = zscore_clipped(roc_raw, MSF_WINDOW, MSF_ZSCORE_CLIP)
    momentum_norm = sigmoid(roc_z, 1.5)

    # ── Component 2: Structure (Mood trend divergence + acceleration) ──
    trend_fast = rolling_mean_fast(mood_series, 5)
    trend_slow = rolling_mean_fast(mood_series, MSF_WINDOW)
    trend_diff_z = zscore_clipped(trend_fast - trend_slow, MSF_WINDOW, MSF_ZSCORE_CLIP)
    mood_accel_raw = mood_series.diff(5).diff(5)
    mood_accel_z = zscore_clipped(mood_accel_raw, MSF_WINDOW, MSF_ZSCORE_CLIP)
    structure_z = (trend_diff_z + mood_accel_z) / np.sqrt(2.0)
    structure_norm = sigmoid(structure_z, 1.5)

    # ── Component 3: Regime (Adaptive threshold) ────────────────────────
    # v1.x: fixed 0.0033 threshold. v2.0: scales with local volatility.
    # A move is "directional" only if it exceeds half a local std.
    pct_vals = nifty_series.pct_change().fillna(0).values
    
    cs_pct = np.cumsum(pct_vals)
    cs2_pct = np.cumsum(pct_vals**2)
    cs_pct_shift = np.zeros(n, dtype=np.float64)
    cs_pct_shift[MSF_WINDOW:] = cs_pct[:-MSF_WINDOW]
    cs2_pct_shift = np.zeros(n, dtype=np.float64)
    cs2_pct_shift[MSF_WINDOW:] = cs2_pct[:-MSF_WINDOW]
    
    sums_pct = cs_pct - cs_pct_shift
    sums2_pct = cs2_pct - cs2_pct_shift
    counts_pct = np.minimum(np.arange(1, n + 1), MSF_WINDOW)
    
    var_pct = (sums2_pct - (sums_pct**2) / counts_pct) / np.maximum(counts_pct - 1, 1)
    rolling_vol = np.sqrt(np.maximum(var_pct, 0))
    
    rolling_vol[:4] = 0.003  # min_periods=5 fallback
    rolling_vol = np.where(rolling_vol < 1e-12, 0.003, rolling_vol)
    adaptive_threshold = np.clip(rolling_vol * 0.5, 0.001, None)

    regime_signals = np.where(pct_vals > adaptive_threshold, 1,
                     np.where(pct_vals < -adaptive_threshold, -1, 0))
    # Windowed sum (not cumsum) — prevents unbounded growth that creates
    # trend artifacts when cumsum drifts far from its rolling mean.
    regime_count = pd.Series(regime_signals, index=df.index).rolling(MSF_WINDOW, min_periods=1).sum()
    regime_raw = regime_count - rolling_mean_fast(regime_count, MSF_WINDOW)
    regime_z = zscore_clipped(regime_raw, MSF_WINDOW, MSF_ZSCORE_CLIP)
    regime_norm = sigmoid(regime_z, 1.5)

    # ── Component 4: Breadth Flow ───────────────────────────────────────
    breadth_ma = rolling_mean_fast(breadth_series, MSF_WINDOW)
    # Guard against near-zero denominators (not just exact zero)
    breadth_ma_safe = breadth_ma.where(breadth_ma.abs() > 1e-6, 1.0)
    breadth_ratio = breadth_series / breadth_ma_safe
    breadth_z = zscore_clipped(breadth_ratio - 1, MSF_WINDOW, MSF_ZSCORE_CLIP)
    flow_norm = sigmoid(breadth_z, 1.5)
    
    # ── Inverse-Variance Weighting (causal + guarded) ───────────────────
    # Markowitz for signals: stable (low variance) components get more weight.
    #
    # Two defects are fixed here.
    #
    # 1. LOOK-AHEAD. Weights were derived from the variance of the LAST 60
    #    rows and applied across all of history, so historical MSF values
    #    shifted every time new data arrived (measured: up to 0.52 on a +/-5
    #    band). The variance is now EXPANDING — the weight at time t uses
    #    only observations up to t.
    #
    # 2. DEGENERATE CAPTURE. A component with zero variance took
    #    1/1e-6 = 1e6 inverse-variance and won ~100% of the weight; because a
    #    constant component is identically zero after the z-score/sigmoid
    #    chain, the whole composite collapsed to a flat line (measured std
    #    0.0001 versus a healthy 1.95). A missing AD_RATIO column alone was
    #    enough to trigger it. Weights are now clamped to
    #    [MSF_MIN_WEIGHT, MSF_MAX_WEIGHT] and renormalised, and any component
    #    whose full-sample std falls below MSF_DEGENERATE_STD is reported.
    components = {
        'momentum': momentum_norm,
        'structure': structure_norm,
        'regime': regime_norm,
        'flow': flow_norm,
    }
    names = list(components)
    comp_mat = np.column_stack([
        np.asarray(components[k].values if hasattr(components[k], 'values') else components[k],
                   dtype=np.float64)
        for k in names
    ])
    comp_mat = np.where(np.isfinite(comp_mat), comp_mat, 0.0)

    # Components that carry no information at all over the full sample.
    full_std = comp_mat.std(axis=0)
    degenerate = [names[i] for i in range(len(names)) if full_std[i] < MSF_DEGENERATE_STD]

    # Expanding mean/variance per component, O(N) via cumulative sums.
    counts = np.arange(1, n + 1, dtype=np.float64)[:, None]
    cs = np.cumsum(comp_mat, axis=0)
    cs2 = np.cumsum(comp_mat ** 2, axis=0)
    exp_var = (cs2 - (cs ** 2) / counts) / np.maximum(counts - 1.0, 1.0)
    exp_var = np.maximum(exp_var, 1e-6)

    inv_var = 1.0 / exp_var
    w = inv_var / inv_var.sum(axis=1, keepdims=True)

    # Clamp, then renormalise. Iterate twice so a clamp on one component
    # cannot push another back outside the band.
    for _ in range(2):
        w = np.clip(w, MSF_MIN_WEIGHT, MSF_MAX_WEIGHT)
        w = w / w.sum(axis=1, keepdims=True)

    # Equal weights until there is enough history to estimate variance.
    warm = min(MSF_MIN_WARMUP, n)
    w[:warm, :] = 1.0 / len(names)

    # A dead component contributes nothing but must not absorb weight from
    # the live ones either — zero it out and redistribute.
    if degenerate:
        for i, name in enumerate(names):
            if name in degenerate:
                w[:, i] = 0.0
        row_sum = w.sum(axis=1, keepdims=True)
        w = np.where(row_sum > 1e-12, w / np.maximum(row_sum, 1e-12), 1.0 / len(names))

    msf_raw = np.sum(w * comp_mat, axis=1)
    msf_spread = pd.Series(msf_raw * MSF_SCALE, index=df.index)

    result['msf_spread'] = msf_spread
    result['momentum']   = momentum_norm  * MSF_SCALE
    result['structure']  = structure_norm * MSF_SCALE
    result['regime']     = regime_norm    * MSF_SCALE
    result['flow']       = flow_norm      * MSF_SCALE

    # Latest weights, for the console line and the component breakdown card.
    weights = {name: float(w[-1, i]) for i, name in enumerate(names)}
    result.attrs['weights'] = weights
    result.attrs['degenerate_components'] = degenerate

    weight_str = '  '.join(f"{k}={v:.0%}" for k, v in weights.items())
    if not _CALIBRATION_QUIET:
        console.detail(
            f"MSF Spread complete — {time.time() - start_time:.2f}s  ·  "
            f"Latest inverse-variance weights: {weight_str}"
        )
        if degenerate:
            console.issue(
                "DATA",
                "MSF Spread",
                f"{len(degenerate)} component(s) carry no signal and were excluded: "
                f"{', '.join(degenerate)}. Check the source columns "
                f"({', '.join(sorted(set(MSF_SOURCE_COLUMNS.values())))}) in the sheet.",
            )
    return result


@st.cache_data(max_entries=5, show_spinner=False)
def calculate_msf_spread(df, mood_col='Mood_Score', nifty_col='NIFTY', breadth_col='AD_RATIO'):
    """Cached public entry — delegates to ``_calculate_msf_spread_impl``."""
    return _calculate_msf_spread_impl(df, mood_col, nifty_col, breadth_col)


# ══════════════════════════════════════════════════════════════════════════════
# WAVETREND OSCILLATOR  (Adapted to Mood Score instead of HLC3)
# ══════════════════════════════════════════════════════════════════════════════
# WRCI v3.5.0 core PineScript:
#   ap  = hlc3
#   esa = ema(ap, n1)                            n1 = 10
#   d   = ema(abs(ap - esa), n1)
#   ci  = (ap - esa) / math.max(0.015 * d, 1e-6) ← denom floored at 1e-6
#   tci = ema(ci, n2)                            n2 = 21
#   wt1 = tci
#   wt2 = f_smooth(wt1, wt2_len, wt2_type)       ← default ALMA, len 20
#
# Arthagati adaptation: ``ap = Mood_Score`` — the OU-normalized, Kalman-
# smoothed sentiment signal already lives in the same [-100, +100] range
# that WaveTrend expects, so the oscillator levels carry their original
# interpretation. EMA is computed via pandas' ewm(span=N) which matches
# TradingView's ema() exactly, and ALMA via the ``alma()`` helper which
# matches ta.alma(src, len, offset, sigma) exactly.

def _calculate_wavetrend_impl(
    df,
    source_col: str = 'Mood_Score',
    n1: int = WT_CHANNEL_LEN,
    n2: int = WT_AVERAGE_LEN,
    signal_len: int = WT_SIGNAL_LEN,
):
    """Compute the WRCI v3.5.0 WaveTrend core on the given source column.
    Returns a DataFrame with columns ``DATE`` (passthrough), ``wt1`` (the
    Composite Index wave) and ``wt2`` (the ALMA signal line).

    Engineering notes vs. the v3.5.0 PineScript:
      • The divisor ``0.015 * d`` (d = EMA of |ap − esa|) under-flows on
        flat / warm-up segments. v3.5.0 floors it at 1e-6 — matching the
        WRCI run_full_analysis reference — rather than the legacy d.clip(0.5).
      • The WT2 signal line is ALMA(wt1, ``signal_len``, 0.85, 6) — the
        f_smooth default in v3.5.0 — replacing the legacy SMA(wt1, 4).
      • The first ``n1 + n2`` rows are masked (NaN) — Plotly skips them
        in the chart automatically, and a properly warmed-up EMA/ALMA
        emerges cleanly afterwards (ALMA's own ``signal_len``-bar warm-up
        is fully covered since signal_len ≤ n1 + n2).
    """
    if source_col not in df.columns or len(df) == 0:
        return pd.DataFrame(columns=['DATE', 'wt1', 'wt2'])

    ap  = pd.Series(df[source_col].values, dtype=np.float64)
    esa = ap.ewm(span=n1, adjust=False).mean()
    d   = (ap - esa).abs().ewm(span=n1, adjust=False).mean()
    # v3.5.0: floor the denominator (0.015 * d) at 1e-6, not d itself at 0.5.
    denom = (0.015 * d).clip(lower=1e-6)
    ci  = (ap - esa) / denom
    tci = ci.ewm(span=n2, adjust=False).mean()

    wt1 = tci
    # v3.5.0: WT2 = ALMA signal line (f_smooth default), replacing SMA(4).
    wt2 = alma(wt1, signal_len, WT_ALMA_OFFSET, WT_ALMA_SIGMA)

    # Mask warmup period — the EMAs need at least n1+n2 bars before the
    # oscillator stabilises. NaN values are skipped by Plotly.
    warmup = n1 + n2
    if len(wt1) > warmup:
        wt1.iloc[:warmup] = np.nan
        wt2.iloc[:warmup] = np.nan

    out = pd.DataFrame({
        'DATE': df['DATE'].values,
        'wt1':  wt1.values,
        'wt2':  wt2.values,
    })
    return out


def wavetrend_bands(wt1: np.ndarray) -> tuple[float, float]:
    """Overbought/oversold levels for the WaveTrend pane, from the data.

    LazyBear's +/-80 and +/-60 assume ``ci`` built from hlc3. Driven by
    Mood_Score the oscillator has a different scale: measured over synthetic
    and production-shaped data, |wt1| peaks near 70 and never reaches 80, so
    the primary band was unreachable and roughly 40% of the pane was
    permanently empty.

    The levels are quantiles of |wt1| over the FULL history — not the visible
    window — so they stay put as the user switches timeframes, and they move
    only slowly as new data arrives. Falls back to the config constants when
    there is too little history to estimate them.
    """
    finite = wt1[np.isfinite(wt1)]
    if len(finite) < 100:
        return float(WT_OB_LEVEL_1), float(WT_OB_LEVEL_2)
    mag = np.abs(finite)
    primary = float(np.quantile(mag, WT_OB_QUANTILE_1))
    secondary = float(np.quantile(mag, WT_OB_QUANTILE_2))
    # Round to a readable gridline and keep the bands apart.
    primary = max(round(primary / 5.0) * 5.0, 10.0)
    secondary = max(round(secondary / 5.0) * 5.0, 5.0)
    if secondary >= primary:
        secondary = max(primary - 5.0, 5.0)
    return primary, secondary


@st.cache_data(max_entries=5, show_spinner=False)
def calculate_wavetrend(df, source_col: str = 'Mood_Score'):
    """Cached public entry — delegates to ``_calculate_wavetrend_impl``
    with module-level constants for channel/average/signal lengths."""
    return _calculate_wavetrend_impl(
        df, source_col,
        n1=WT_CHANNEL_LEN, n2=WT_AVERAGE_LEN, signal_len=WT_SIGNAL_LEN,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SIMILAR PERIODS FINDER
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(max_entries=5, show_spinner=False)
def find_similar_periods(
    df,
    top_n: int = 10,
    recency_weight: float = SIMILAR_W_RECV,
    min_separation: int = SIMILAR_MIN_SEPARATION,
):
    """
    Historical analog matching.

    3-part scoring:
      1. Mahalanobis distance   — covariance-aware state matching
         Features: mood, volatility, NIFTY momentum, Hurst, entropy
      2. Trajectory cosine      — detrended mood path shape
      3. Exponential recency    — prefer recent analogs

    Two corrections versus the previous implementation.

    SEPARATION. ``nlargest`` returned whichever rows scored highest, and
    adjacent trading days describe near-identical states — so the "top 10"
    routinely collapsed onto two or three episodes (measured: five of ten
    inside a 32-row window). Downstream the UI quotes a median forward
    return and a hit rate over those ten rows as though they were ten
    independent observations. Selection is now greedy with a
    ``min_separation`` gap, and the count of distinct episodes is returned
    so the UI can report the effective sample size honestly.

    RECENCY WEIGHT. ``recency_weight`` was scaled into the recency term and
    then normalised straight back out, making the parameter a no-op —
    passing 0.1 or 99.0 returned identical analogs. It is now the actual
    blend weight.
    """
    if df.empty or 'Mood_Score' not in df.columns:
        return []

    latest = df.iloc[-1]
    n = len(df)

    # Exclude the trailing window: those rows cannot have a full set of
    # forward returns, so including them biases the outcome tiles.
    tail = min(SIMILAR_EXCLUDE_TAIL, max(n - 5, 1))
    historical = df.iloc[:-tail].copy() if n > tail else df.iloc[:-1].copy()
    # Warm-up rows are scored with borrowed statistics — not comparable states.
    if 'Is_Warmup' in historical.columns:
        historical = historical[~historical['Is_Warmup'].astype(bool)]
    if historical.empty or len(historical) < 5:
        return []

    # ── Feature vectors ─────────────────────────────────────────────────
    nifty_roc = df['NIFTY'].pct_change(MSF_ROC_LEN).fillna(0).values
    hist_pos = np.array([df.index.get_loc(i) for i in historical.index], dtype=int)

    current_features = [latest['Mood_Score'], latest['Mood_Volatility']]
    hist_arrays = [historical['Mood_Score'].values, historical['Mood_Volatility'].values]

    current_features.append(nifty_roc[-1] if len(nifty_roc) > 0 else 0.0)
    hist_arrays.append(nifty_roc[hist_pos])

    for col in ('Hurst', 'Market_Entropy'):
        if col in df.columns:
            current_features.append(latest[col])
            hist_arrays.append(historical[col].values)

    current_vec = np.array(current_features, dtype=np.float64)
    hist_matrix = np.column_stack(hist_arrays)

    for col in range(hist_matrix.shape[1]):
        col_data = hist_matrix[:, col]
        valid = np.isfinite(col_data)
        median_val = np.median(col_data[valid]) if valid.any() else 0.0
        hist_matrix[~valid, col] = median_val
    current_vec = np.where(np.isfinite(current_vec), current_vec, 0.0)

    # ── Part 1: Mahalanobis distance ────────────────────────────────────
    cov_matrix = np.cov(hist_matrix, rowvar=False)
    if cov_matrix.ndim < 2:
        cov_matrix = np.array([[max(float(cov_matrix), 1e-6)]])

    maha_dist = mahalanobis_distance_batch(hist_matrix, current_vec, cov_matrix)
    max_dist = maha_dist.max() if maha_dist.max() > 0 else 1.0
    maha_sim = 1.0 - (maha_dist / max_dist)

    # ── Part 2: Trajectory cosine similarity ────────────────────────────
    traj_sim = np.zeros(len(historical))
    if n > TRAJ_WINDOW:
        _traj_x = np.arange(TRAJ_WINDOW, dtype=np.float64)
        _traj_xm = _traj_x - _traj_x.mean()
        _traj_xvar = np.sum(_traj_xm ** 2)

        def _ls_detrend(traj):
            if _traj_xvar < 1e-12:
                return traj - traj.mean()
            slope = np.sum(_traj_xm * (traj - traj.mean())) / _traj_xvar
            return traj - (traj.mean() + slope * _traj_xm)

        mood_vals = df['Mood_Score'].values
        ct_detrended = _ls_detrend(mood_vals[-TRAJ_WINDOW:])
        for j, pos in enumerate(hist_pos):
            if pos >= TRAJ_WINDOW:
                ht = _ls_detrend(mood_vals[pos - TRAJ_WINDOW:pos])
                traj_sim[j] = (cosine_similarity(ct_detrended, ht) + 1) / 2

    # ── Part 3: Exponential recency decay ───────────────────────────────
    days_since = (latest['DATE'] - historical['DATE']).dt.days.values.astype(float)
    recency_norm = np.exp(-np.log(2) * days_since / 365.0)

    # ── Combined ────────────────────────────────────────────────────────
    w_state = SIMILAR_W_MAHA
    w_traj  = SIMILAR_W_TRAJ
    w_recv  = float(recency_weight)
    w_total = max(w_state + w_traj + w_recv, 1e-9)
    combined = (w_state * maha_sim + w_traj * traj_sim + w_recv * recency_norm) / w_total

    # ── Greedy selection with a minimum separation ──────────────────────
    order = np.argsort(-combined)
    chosen: list[int] = []
    chosen_pos: list[int] = []
    for j in order:
        pos = int(hist_pos[j])
        if all(abs(pos - p) >= min_separation for p in chosen_pos):
            chosen.append(int(j))
            chosen_pos.append(pos)
        if len(chosen) >= top_n:
            break

    nifty_vals = df['NIFTY'].values
    hist_rows = historical.iloc[chosen]
    results = []
    for rank, (j, pos) in enumerate(zip(chosen, chosen_pos)):
        row = hist_rows.iloc[rank]
        nifty_at = row['NIFTY'] if 'NIFTY' in row and row['NIFTY'] > 0 else None

        fwd_returns = {}
        for horizon in (5, 20, 60, 90):
            fwd_idx = pos + horizon
            if fwd_idx < len(nifty_vals) and nifty_at and nifty_at > 0:
                fwd_returns[horizon] = (nifty_vals[fwd_idx] / nifty_at - 1) * 100
            else:
                fwd_returns[horizon] = None

        results.append({
            'date': row['DATE'].strftime('%Y-%m-%d'),
            'similarity': float(combined[j]),
            'mood_score': row['Mood_Score'],
            'mood': row['Mood'],
            'mood_volatility': row['Mood_Volatility'],
            'nifty': nifty_at or 0,
            'fwd_5d':  fwd_returns.get(5),
            'fwd_20d': fwd_returns.get(20),
            'fwd_60d': fwd_returns.get(60),
            'fwd_90d': fwd_returns.get(90),
            'separation_days': min_separation,
        })

    return results


# ══════════════════════════════════════════════════════════════════════════════
# APPLICATION ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════

def resolve_profile(key: str, available: list[str]) -> tuple[list[str], list[str]]:
    """Return (predictors present in this sheet, names that were missing).

    A profile names columns; a sheet may not carry all of them. Missing names
    are reported rather than silently dropped, so a preset that only half
    applies is visible as such.
    """
    spec = PREDICTOR_PROFILES.get(key, {}).get("predictors")
    if spec is None:                       # "broad" — everything eligible
        return list(available), []
    present = [p for p in spec if p in available]
    missing = [p for p in spec if p not in available]
    return present, missing


def detect_profile(active: tuple, available: list[str]) -> str:
    """Name the preset matching the active set exactly, else 'custom'.

    A profile only matches when it resolved COMPLETELY against this sheet.
    A partially-resolved preset — say two of the five Valuation columns
    present — would otherwise match a two-column active set and display the
    five-column measurement beside it, advertising evidence for a set the
    user is not running.
    """
    cur = set(active)
    for key in PREDICTOR_PROFILES:
        present, missing = resolve_profile(key, available)
        if missing:
            continue
        if present and set(present) == cur:
            return key
    return "custom"


def _dataset_fingerprint(raw_df: pd.DataFrame, predictors) -> tuple:
    """Hashable fingerprint for the session engine cache."""
    return (
        int(len(raw_df)),
        str(raw_df["DATE"].iloc[0].date()) if len(raw_df) else "",
        str(raw_df["DATE"].iloc[-1].date()) if len(raw_df) else "",
        tuple(sorted(predictors)),
    )


def _compute_engine_output(
    raw_df: pd.DataFrame,
    selected_preds,
    prog_slot,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (mood_df, msf_df), session-cached by input fingerprint.

    The engine ALWAYS runs on factory hyperparameters. Intelligence Mode
    no longer tunes structural hyperparameters (that approach was 1000×
    too expensive for Streamlit Cloud's wake-and-forget model). Instead,
    IM tunes a small post-engine ensemble on TOP of this output — see
    ``_auto_calibrate_if_needed`` and ``apply_calibration``.

    The cache lives in ``st.session_state`` so view/timeframe switches
    return in ~150ms instead of re-running the 30-second engine.
    """
    fp = _dataset_fingerprint(raw_df, selected_preds)
    total_phases = int(st.session_state.get("_phase_total", 4))
    cached_fp = st.session_state.get("_engine_fp")
    if cached_fp == fp:
        cached_mood = st.session_state.get("_engine_mood_df")
        cached_msf  = st.session_state.get("_engine_msf_df")
        if cached_mood is not None and cached_msf is not None:
            console.section("Engine Cache HIT — fast-path", phase="CACHE")
            console.item("Reused rows",    f"{len(cached_mood):,}")
            console.item("Recompute",      "skipped — view/timeframe switches are O(1) from here")
            _progress_bar(
                prog_slot, 90,
                "Engine Output Cached",
                "Re-using mood + MSF frames from this session",
            )
            return cached_mood, cached_msf

    # ── Compute mood ────────────────────────────────────────────────────
    console.start_phase("Sentiment Engine", num=3, total=total_phases)
    console.step(3, "OU normalisation · Kalman smoothing · 5-layer pipeline")
    console.item("Mode", "Factory defaults (post-engine ensemble tunes the output layer)")
    _progress_bar(
        prog_slot, 50,
        "Running Sentiment Engine",
        "OU Normalisation · Kalman Smoothing · 5-Layer Pipeline",
    )
    mood_df = calculate_historical_mood(raw_df, dependent_vars=selected_preds)
    if mood_df.empty:
        prog_slot.empty()
        console.error("calculate_historical_mood returned empty DataFrame")
        console.end_phase("Sentiment Engine")
        st.error("Failed to calculate mood scores.")
        st.stop()
    latest_mood = float(mood_df["Mood_Score"].iloc[-1])
    console.success(f"Mood score computed: {latest_mood:+.2f}")
    console.checkpoint("Mood frame integrity", "OK")
    console.end_phase("Sentiment Engine")

    # ── Compute MSF Spread ──────────────────────────────────────────────
    console.start_phase("MSF Spread", num=4, total=total_phases)
    console.step(4, "Momentum · Structure · Regime · Flow (inverse-variance weights)")
    _progress_bar(prog_slot, 80, "Computing MSF Spread", "Momentum · Structure · Regime · Flow")
    msf_df = calculate_msf_spread(mood_df)
    mood_df["MSF_Spread"] = msf_df["msf_spread"].values if not msf_df.empty else 0
    # A component with no variance is excluded from the composite rather than
    # capturing all of the inverse-variance weight. Record it so the view can
    # tell the user their oscillator is running on fewer inputs.
    _degenerate = list(msf_df.attrs.get("degenerate_components", []))
    st.session_state["_msf_degenerate"] = _degenerate
    if _degenerate:
        console.warning(
            f"MSF running on {4 - len(_degenerate)}/4 components — "
            f"no signal in: {', '.join(_degenerate)}"
        )
    latest_msf = float(mood_df["MSF_Spread"].iloc[-1]) if not mood_df.empty else 0.0
    console.success(f"MSF Spread computed: {latest_msf:+.2f}")
    console.end_phase("MSF Spread")

    # ── Compute WaveTrend (LazyBear · Mood-driven) ──────────────────────
    _progress_bar(
        prog_slot, 87,
        "Computing WaveTrend",
        f"LazyBear · Mood-driven · n1={WT_CHANNEL_LEN}, n2={WT_AVERAGE_LEN}",
    )
    wt_df = calculate_wavetrend(mood_df)
    mood_df["WT1"] = wt_df["wt1"].values if not wt_df.empty else 0.0
    mood_df["WT2"] = wt_df["wt2"].values if not wt_df.empty else 0.0
    st.session_state["_wt_bands"] = wavetrend_bands(mood_df["WT1"].to_numpy(dtype=float))
    if not wt_df.empty:
        latest_wt1 = float(mood_df["WT1"].iloc[-1])
        latest_wt2 = float(mood_df["WT2"].iloc[-1])
        console.detail(
            f"WaveTrend: WT1={latest_wt1:+.2f} · WT2={latest_wt2:+.2f}  ·  "
            f"levels ±{WT_OB_LEVEL_1}/{WT_OB_LEVEL_2}"
        )
    console.detail("Engine output cached for session — subsequent UI interactions are instant")

    # ── Persist into session cache ──────────────────────────────────────
    st.session_state["_engine_fp"]      = fp
    st.session_state["_engine_mood_df"] = mood_df
    st.session_state["_engine_msf_df"]  = msf_df

    return mood_df, msf_df


def _invalidate_engine_cache() -> None:
    """Drop session-cached engine frames. Call when inputs change
    (data refreshed, predictor set changed)."""
    for k in ("_engine_fp", "_engine_mood_df", "_engine_msf_df",
              "_validation", "_msf_degenerate", "_wt_bands"):
        st.session_state.pop(k, None)


def _clear_engine_caches() -> None:
    """Clear this app's @st.cache_data entries — and only this app's.

    ``st.cache_data.clear()`` is process-global. On a shared deployment one
    user pressing Refresh Data used to flush every other user's cached
    frames, forcing a full recompute in sessions that had changed nothing.
    Clearing the specific wrapped functions keeps the blast radius local to
    the data this app owns.
    """
    for fn in (load_data, calculate_anchor_correlations, calculate_historical_mood,
               calculate_msf_spread, calculate_wavetrend, find_similar_periods):
        try:
            fn.clear()
        except Exception:  # pragma: no cover — older Streamlit without .clear()
            st.cache_data.clear()
            break
    _invalidate_engine_cache()


# ══════════════════════════════════════════════════════════════════════════════
# APPLICATION SHELL
# ══════════════════════════════════════════════════════════════════════════════

#: The two appearances, in order. PAPER LEADS, and the order IS the default:
#: `theme_choice()` falls back to APPEARANCES[0] for any unset or unrecognised
#: value, so first-in-tuple is first-run. Kept as one fact rather than a
#: separate DEFAULT_ constant, so the toggle's left-to-right order and the
#: default can never disagree.
APPEARANCES = ("Paper", "Slate")

#: The appearance is stored under a DURABLE key rather than read back off the
#: widget. Streamlit discards a widget's state on any run that does not reach
#: it — and the rail is not reached on the cold-start branch — so reading the
#: widget key directly loses the choice the first time a run short-circuits.
_THEME_CHOICE = "appearance_choice"

#: The forward window the conviction chain is stated over. The analog engine
#: reports +5/20/60/90D; 90 is the horizon at which validation finds the
#: signal strongest, so it is the one the verdict is framed on.
HERO_HORIZON = 90


def theme_choice() -> str:
    """The appearance the user last chose, always one of ``APPEARANCES``.

    A value not in the list is treated as unset. That matters across a rename:
    a session opened before this list changed still holds the old string, and
    handing an unknown option to the segmented control as its default is an
    error rather than a fallback.
    """
    choice = st.session_state.get(_THEME_CHOICE)
    return choice if choice in APPEARANCES else APPEARANCES[0]


def _render_appearance_control() -> None:
    """The theme switch — LAST control in the rail, deliberately.

    It is the least consequential switch in the application, so it does not
    get the most valuable position in the rail. Slate is the working theme;
    Paper is for reading a result and for print.
    """
    with st.container(key="appearance"):
        st.markdown('<div class="sidebar-title">Appearance</div>', unsafe_allow_html=True)
        mode = st.segmented_control(
            "Appearance", list(APPEARANCES), key="theme_mode",
            default=theme_choice(), label_visibility="collapsed",
            help="Slate — dark, for working. Paper — light, for reading and print.",
        )
        # Mirror into the DURABLE key and rerun, so the stylesheet at the top
        # of main() is re-injected with the new value. Without the rerun the
        # change lands half-way down the page and the run renders as a mix of
        # both themes.
        if mode is not None and mode != theme_choice():
            st.session_state[_THEME_CHOICE] = mode
            st.rerun()


def _render_footer() -> None:
    ist_now = datetime.now(pytz.UTC).astimezone(pytz.timezone("Asia/Kolkata"))
    st.markdown(
        f'<div class="app-footer"><div class="content">'
        f'\u00a9 {ist_now.year} <strong>{PRODUCT_NAME}</strong> &nbsp;\u00b7&nbsp; {COMPANY}'
        f' &nbsp;\u00b7&nbsp; {VERSION} &nbsp;\u00b7&nbsp; '
        f'{ist_now.strftime("%Y-%m-%d %H:%M:%S IST")}'
        f'</div></div>',
        unsafe_allow_html=True,
    )


def _apply_profile(key: str, preds: tuple) -> None:
    """Commit a preset. Predictor set changed ⇒ engine output no longer applies."""
    st.session_state["active_predictors"] = preds
    st.session_state["predictor_profile"] = key
    _clear_engine_caches()
    st.rerun()


def _apply_predictors(preds: tuple) -> None:
    st.session_state["active_predictors"] = preds
    st.session_state["predictor_profile"] = "custom"
    _clear_engine_caches()
    st.rerun()


def main():
    # ─── Resolve the appearance BEFORE anything is styled ──────────────────
    # This must run first. `theme` is written by the appearance control, which
    # renders deep in the rail — i.e. AFTER this line. On the rerun following a
    # click, inject_css() would otherwise still see the PREVIOUS appearance
    # while every chart, which resolves its palette at render time further down
    # the script, already saw the new one: a page whose chrome and whose plots
    # disagree about which theme is active. Deriving it here, from the durable
    # choice, makes the whole run agree on one value.
    st.session_state["theme"] = "light" if theme_choice() == "Paper" else "dark"
    inject_css(theme=st.session_state["theme"])

    st.session_state.setdefault("analysis_started", False)
    st.session_state.setdefault("active_predictors", None)
    st.session_state.setdefault("active_instrument", "NIFTY 50")

    # One main-area progress slot, created up front so the same themed bar
    # drives the fetch, the correlations, the engine and the oscillators
    # rather than a spinner handing off to a bar with a gap between them.
    progress_slot = st.empty()

    # ─── Cold start ────────────────────────────────────────────────────────
    if not st.session_state["analysis_started"]:
        with st.sidebar:
            render_nav_brand()
            st.markdown('<div class="sidebar-title">Session</div>', unsafe_allow_html=True)
            if st.button("Run analysis", width="stretch", type="primary"):
                st.session_state["analysis_started"] = True
                st.rerun()
            render_rail_readout([
                ("Status", "Idle", "caution"),
                ("Source", "Sheets" if (SHEET_ID and SHEET_GID) else "Not set",
                 "" if (SHEET_ID and SHEET_GID) else "short"),
                ("Version", VERSION, ""),
            ])
            _render_appearance_control()
        render_landing_page(
            version=VERSION,
            n_predictors=len(DEPENDENT_VARS),
            sheet_configured=bool(SHEET_ID and SHEET_GID),
        )
        _render_footer()
        return

    # ─── Ingestion ─────────────────────────────────────────────────────────
    run_id = generate_run_id()
    console.main_header(
        f"Analysis Run · {run_id}",
        details={
            "Started": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Mode":    "Streamlit interactive",
            "Sheet":   f"…{SHEET_ID[-8:]}" if SHEET_ID else "(env not set)",
        },
    )
    _total = 4
    st.session_state["_phase_total"] = _total
    console.start_phase("Data Ingestion", num=1, total=_total)
    console.step(1, "Fetching market data from Google Sheets (GViz API)")
    _progress_bar(progress_slot, 5, "Fetching market data", "Google Sheets · gviz API · CSV decode")
    raw_df = load_data()

    if raw_df is None:
        progress_slot.empty()
        console.error("Data fetch returned None — aborting run.")
        console.end_phase("Data Ingestion")
        render_empty_state(
            "Data source unreachable",
            "The Google Sheets gviz endpoint returned no usable CSV after the configured "
            "retries, so there is nothing to score. Check ARTHAGATI_SHEET_ID and "
            "ARTHAGATI_SHEET_GID, and that the sheet is shared as readable by link.",
            eyebrow="Ingestion failed",
            action_label="Press Refresh data in the rail to retry",
        )
        st.stop()
    console.success(f"Loaded {len(raw_df):,} rows × {len(raw_df.columns)} columns")
    console.item("Date range", f"{raw_df['DATE'].min().date()} → {raw_df['DATE'].max().date()}")
    console.end_phase("Data Ingestion")

    # Columns derived from NIFTY are withheld from selection: using one makes
    # the valuation score partly a function of the price it is then evaluated
    # against, and any measured edge would be partly price predicting itself.
    available_predictors = [
        col for col in raw_df.columns
        if col not in NON_PREDICTOR_COLS
        and col not in CIRCULAR_COLUMNS
        and col not in DUPLICATE_COLUMNS
        and pd.api.types.is_numeric_dtype(raw_df[col])
        and float((raw_df[col].notna() & (raw_df[col] != 0)).mean() * 100) >= PREDICTOR_MIN_COVERAGE
        and raw_df[col].nunique() >= PREDICTOR_MIN_UNIQUE
    ]
    current_preds = st.session_state.get("active_predictors")
    if not current_preds:
        default_preds, _ = resolve_profile(DEFAULT_PROFILE, available_predictors)
        if not default_preds:
            default_preds = [p for p in DEPENDENT_VARS if p in available_predictors]
        st.session_state["active_predictors"] = (
            tuple(default_preds) if default_preds else tuple(available_predictors)
        )
        st.session_state.setdefault("predictor_profile", DEFAULT_PROFILE)
    else:
        valid = tuple(p for p in current_preds if p in available_predictors)
        st.session_state["active_predictors"] = valid if valid else tuple(available_predictors)

    selected_preds = st.session_state["active_predictors"]

    # ─── Engine ────────────────────────────────────────────────────────────
    console.start_phase("Correlation Engine", num=2, total=_total)
    console.step(2, "Computing decay-weighted Spearman vs PE & EY anchors")
    console.item("Active predictors", f"{len(selected_preds)}/{len(available_predictors)}")
    _progress_bar(progress_slot, 30, "Computing correlations",
                  "Decay-weighted Spearman · PE & EY anchors")
    console.success("Correlations computed")
    console.end_phase("Correlation Engine")

    mood_df, msf_df = _compute_engine_output(raw_df, selected_preds, progress_slot)
    _progress_bar(progress_slot, 100, "Ready", "All stages complete")
    time.sleep(0.12)
    progress_slot.empty()

    latest = mood_df.iloc[-1]
    latest_date = raw_df["DATE"].max()
    today_ist = datetime.now(pytz.timezone("Asia/Kolkata")).date()
    data_age = (pd.Timestamp(today_ist) - latest_date).days
    selected_tf = st.session_state.get("tf_selected", "1Y")

    console.summary(f"Run {run_id} · Pipeline Summary", {
        "Predictors":     f"{len(selected_preds)} active",
        "Rows":           f"{len(mood_df):,}",
        "Mood Score":     f"{latest['Mood_Score']:+.2f} ({latest.get('Mood', '—')})",
        "MSF Spread":     f"{latest['MSF_Spread']:+.2f}",
        "Regime":         str(latest.get("Regime", "Unknown")),
        "OU Half-Life":   f"{latest.get('OU_Half_Life', 0):.0f}d",
        "Hurst":          f"{latest.get('Hurst', 0.5):.2f}",
        "Market Entropy": f"{latest.get('Market_Entropy', 0.5):.2f}",
    })

    # ─── The control rail ──────────────────────────────────────────────────
    # Everything GLOBAL lives here — which model, what to do with the session,
    # how the app looks. Everything LOCAL to a page (the chart window) lives on
    # that page, in the panel header of the chart it reframes. A control's
    # position is the only reliable statement of its scope, so the two are
    # never mixed.
    #
    # Rail order is by frequency of use: Model (every few visits) → Session
    # (occasionally) → Readout (read-only) → Appearance (almost never).
    #
    # st.navigation pins its page-nav to the TOP of the sidebar by design, so
    # this content renders below it and `.nav-brand` is lifted above the nav
    # by CSS.
    with st.sidebar:
        render_nav_brand()

        st.markdown('<div class="sidebar-title">Model</div>', unsafe_allow_html=True)
        _active = tuple(st.session_state["active_predictors"])
        _detected = detect_profile(_active, available_predictors)
        _keys = list(PREDICTOR_PROFILES) + ["custom"]
        _labels = {k: PREDICTOR_PROFILES[k]["label"] for k in PREDICTOR_PROFILES}
        _labels["custom"] = "Custom"
        _chosen = st.selectbox(
            "Predictor profile", options=_keys,
            index=_keys.index(_detected) if _detected in _keys else len(_keys) - 1,
            format_func=lambda k: (
                _labels[k] if k == "custom"
                else f"{_labels[k]} · {len(resolve_profile(k, available_predictors)[0])}"
            ),
            label_visibility="collapsed",
            help="Preset predictor mixes. Model Configuration carries each one's "
                 "recorded measurement and the custom column picker.",
        )
        if _chosen != "custom":
            _preds, _ = resolve_profile(_chosen, available_predictors)
            if _preds and set(_preds) != set(_active):
                _apply_profile(_chosen, tuple(_preds))

        st.markdown('<div class="sidebar-title">Session</div>', unsafe_allow_html=True)
        if st.button("Refresh data", width="stretch"):
            _clear_engine_caches()
            _invalidate_engine_cache()
            st.rerun()

        _status_label, _status_tone = fmt.age_label(data_age)
        render_rail_readout([
            ("Predictors", f"{len(selected_preds)}/{len(available_predictors)}", "accent"),
            ("Rows", f"{len(mood_df):,}", ""),
            ("As of", fmt.when(latest_date, "%d %b %y"), ""),
            ("Freshness", _status_label,
             {"pos": "long", "warn": "caution", "neg": "short"}.get(_status_tone, "")),
            ("Version", VERSION, ""),
        ])
        _render_appearance_control()

    # ─── Notices: computed once, rendered under the command bar ────────────
    # These used to render as full-width boxes at the very top of the page,
    # which put the reading itself below the fold on exactly the days the data
    # most needed scrutiny. One row each, severity on the left rule, and now
    # BELOW the thing they qualify rather than above it.
    notices: list[dict] = []
    if data_age > STALE_DATA_DAYS:
        console.warning(f"Stale data — last point is {latest_date.date()} ({data_age}d old)")
        notices.append({
            "kind": "warning", "title": "Stale data",
            "body": f"The last observation is {fmt.when(latest_date)} — {data_age} days old. "
                    "Every reading on this screen describes that date, not the current "
                    "market. Update the source sheet, then press Refresh data.",
        })
    _degenerate = st.session_state.get("_msf_degenerate") or []
    if _degenerate:
        notices.append({
            "kind": "warning", "title": "Degraded oscillator",
            "body": f"MSF Spread is running on {4 - len(_degenerate)} of 4 components — no "
                    f"signal in {', '.join(_degenerate)}. Check that NIFTY and AD_RATIO are "
                    "populated in the source sheet.",
        })
    _warm = warmup_note(mood_df)
    if _warm:
        notices.append({"kind": "info", "title": "Warm-up rows present", "body": _warm})

    # ─── The conviction chain, built once and shared ───────────────────────
    # Built here rather than inside a page so the Overview's verdict and any
    # other surface that shows a gate read the same object. Pure data in, pure
    # data out — see ui.components.build_hero_verdict.
    try:
        _periods = find_similar_periods(mood_df)
    except Exception:                    # engine unavailable — a gate, not a crash
        _periods = []
    _fwd = [p.get("fwd_90d") for p in _periods if p.get("fwd_90d") is not None]
    _precedent = ({"n": len(_fwd),
                   "positive_pct": sum(1 for v in _fwd if v > 0) / len(_fwd) * 100}
                  if _fwd else None)
    _validation = st.session_state.get("_validation_summary")

    verdict = build_hero_verdict(
        mood=float(latest["Mood_Score"]),
        msf=float(latest["MSF_Spread"]),
        regime=str(latest.get("Regime", "Unknown")),
        entropy=float(latest.get("Market_Entropy", 0.5)),
        hurst=float(latest.get("Hurst", 0.5)),
        ou_half_life=float(latest.get("OU_Half_Life", 0.0)),
        precedent=_precedent,
        validation=_validation,
        data_age_days=int(data_age),
        is_warmup=bool(latest.get("Is_Warmup", False)),
        horizon_days=HERO_HORIZON,
        bands=(float(MOOD_BAND_INNER), float(MOOD_BAND_OUTER), float(MSF_OB_LEVEL_1)),
    )

    # ─── Page shell ────────────────────────────────────────────────────────
    def _shell() -> None:
        """The page chrome, identical on every page.

        Order is fixed and means something: the TAPE (the world) sits above the
        COMMAND BAR (this reading), which sits above the NOTICE RAIL (the
        caveats on it). Page content follows.
        """
        render_ticker(raw_df)
        _mood = float(latest["Mood_Score"])
        _band, _ = sig.mood_state(_mood)
        render_top_bar(
            target=f"NIFTY 50 · {_band}",
            price=float(latest["NIFTY"]),
            change_pct=_nifty_change(mood_df),
            status_label=_status_label,
            status_tone={"pos": "success", "warn": "warning",
                         "neg": "danger"}.get(_status_tone, "neutral"),
            meta_items=[
                ("Mood", f"{_mood:+.1f}"),
                ("MSF", f"{float(latest['MSF_Spread']):+.2f}"),
                ("Regime", str(latest.get("Regime", "Unknown"))),
                ("Window", selected_tf),
                ("As of", fmt.when(latest_date, "%d %b %Y")),
            ],
        )
        render_notice_rail(notices)

    def _safe(name: str, fn) -> None:
        """Render a page's content with graceful error handling.

        A page that raises must not take the shell with it — the command bar
        and the tape carry state the reader needs in order to understand that
        something failed at all.
        """
        try:
            fn()
        except Exception as exc:                       # noqa: BLE001 — boundary
            console.error(f"{name} raised: {exc}")
            render_empty_state(
                f"{name} could not be rendered",
                str(exc),
                eyebrow="Page error",
                action_label="Try Refresh data, or a different predictor profile",
            )

    # ─── Pages — thin wrappers. None recomputes the pipeline above. ────────
    def _page_overview() -> None:
        _shell()
        _safe("Overview", lambda: render_overview(
            mood_df, msf_df, verdict=verdict, timeframes=TIMEFRAMES,
            tf=st.session_state.get("tf_selected", "1Y"),
            periods=_periods, data_age=data_age))
        _render_footer()

    def _page_mood() -> None:
        _shell()
        _safe("Mood Engine", lambda: render_mood(
            mood_df, msf_df, timeframes=TIMEFRAMES, mood_scale=MOOD_SCALE,
            ou_proj_days=OU_PROJ_DAYS))
        _render_footer()

    def _page_analogs() -> None:
        _shell()
        _safe("Analogs", lambda: render_analogs(
            mood_df, periods=_periods, backtest_horizon=BACKTEST_HORIZON))
        _render_footer()

    def _page_drivers() -> None:
        _shell()
        _safe("Drivers", lambda: render_drivers(
            raw_df, active_preds=selected_preds, non_predictor_cols=NON_PREDICTOR_COLS,
            calculate_anchor_correlations=calculate_anchor_correlations,
            shannon_entropy=shannon_entropy))
        _render_footer()

    def _page_validation() -> None:
        _shell()
        _safe("Validation", lambda: render_validation(mood_df, raw_df))
        _render_footer()

    def _page_config() -> None:
        _shell()
        _safe("Configuration", lambda: render_config(
            available_predictors=available_predictors,
            resolve_profile=resolve_profile, detect_profile=detect_profile,
            on_profile_change=_apply_profile, on_predictors_change=_apply_predictors))
        _render_footer()

    # Overview leads: it is the read that combines everything else, so it is
    # the page a returning user opens first. The engine surfaces follow as its
    # inputs, Validation as the independent check on all of them.
    pages = {
        "": [st.Page(_page_overview, title="Overview",
                     icon=":material/dashboard:", default=True)],
        "Engine": [
            st.Page(_page_mood, title="Mood Engine", icon=":material/monitoring:"),
            st.Page(_page_analogs, title="Analogs", icon=":material/history:"),
            st.Page(_page_drivers, title="Drivers", icon=":material/hub:"),
        ],
        "System": [
            st.Page(_page_validation, title="Validation", icon=":material/verified:"),
            st.Page(_page_config, title="Configuration", icon=":material/tune:"),
        ],
    }
    st.navigation(pages, position="sidebar").run()


def _nifty_change(mood_df) -> float | None:
    """Session change in the index, in PERCENT POINTS.

    Percent points, not a fraction: the command bar prints it with "%.2f%%",
    and handing it a fraction is how a sub-1% session — i.e. most of them —
    ends up printing as "0.00%".
    """
    if mood_df is None or len(mood_df) < 2 or "NIFTY" not in mood_df.columns:
        return None
    tail = mood_df["NIFTY"].to_numpy(dtype=float)[-2:]
    if not np.isfinite(tail).all() or tail[0] == 0:
        return None
    return float((tail[1] / tail[0] - 1.0) * 100.0)


if __name__ == "__main__":
    main()
