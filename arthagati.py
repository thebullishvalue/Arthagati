# -*- coding: utf-8 -*-
"""
ARTHAGATI (अर्थगति) - Market Sentiment Analysis | A Hemrek Capital Product
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Quantitative market mood analysis with MSF-enhanced indicators.
TradingView-style charting with institutional-grade analytics.
"""

import logging
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytz
import streamlit as st
from plotly.subplots import make_subplots

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="ARTHAGATI | Market Sentiment Analysis",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════════════
# IDENTITY
# ══════════════════════════════════════════════════════════════════════════════

VERSION      = "v2.1.0"
PRODUCT_NAME = "Arthagati"
COMPANY      = "Hemrek Capital"

# ══════════════════════════════════════════════════════════════════════════════
# DATA SOURCE
# ══════════════════════════════════════════════════════════════════════════════

SHEET_ID  = "1po7z42n3dYIQGAvn0D1-a4pmyxpnGPQ13TrNi3DB5_c"
SHEET_GID = "1938234952"

EXPECTED_COLUMNS = [
    'DATE', 'NIFTY',
    'AD_RATIO', 'REL_AD_RATIO', 'REL_BREADTH', 'BREADTH', 'COUNT',
    'NIFTY50_PE', 'NIFTY50_EY', 'NIFTY50_DY', 'NIFTY50_PB',
    'IN10Y', 'IN02Y', 'IN30Y', 'INIRYY',
    'REPO', 'CRR',
    'US02Y', 'US10Y', 'US30Y', 'US_FED',
    'PE_DEV', 'EY_DEV',
]

DEPENDENT_VARS = [
    'AD_RATIO', 'REL_AD_RATIO', 'REL_BREADTH', 'BREADTH', 'COUNT',
    'IN10Y', 'IN02Y', 'IN30Y', 'INIRYY',
    'REPO', 'CRR',
    'US02Y', 'US10Y', 'US30Y', 'US_FED',
    'NIFTY50_DY', 'NIFTY50_PB',
    'PE_DEV', 'EY_DEV',
    'IN_TERM_SPREAD', 'US_TERM_SPREAD',  # derived yield-curve slopes
]

# Columns that are anchors or index keys, never predictors
NON_PREDICTOR_COLS: frozenset[str] = frozenset({'DATE', 'NIFTY', 'NIFTY50_PE', 'NIFTY50_EY'})

# Timeframe labels → calendar-day window (None = use all data / special handling)
TIMEFRAMES: dict[str, int | None] = {
    '1W':  7,
    '1M':  30,
    '3M':  90,
    '6M':  180,
    'YTD': None,   # computed at runtime from Jan 1
    '1Y':  365,
    '2Y':  730,
    '5Y':  1825,
    'MAX': None,   # all available rows
}

# ══════════════════════════════════════════════════════════════════════════════
# COLOUR PALETTE  (mirrors CSS :root variables — keep both in sync)
# ══════════════════════════════════════════════════════════════════════════════

C_PRIMARY = '#FFC300'
C_GREEN   = '#10b981'
C_RED     = '#ef4444'
C_AMBER   = '#f59e0b'
C_CYAN    = '#06b6d4'
C_MUTED   = '#888888'
C_BG_CARD = '#1A1A1A'
C_BG_GRID = '#2A2A2A'
C_TEXT    = '#EAEAEA'

# ══════════════════════════════════════════════════════════════════════════════
# MODEL HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

# Correlation engine
CORR_HALF_LIFE  = 504    # ~2 trading years; exponential recency weight for Spearman
PCT_HALF_LIFE   = 252    # ~1 trading year;  recency weight for adaptive ECDF
MOOD_SCALE      = 30.0   # maps OU-normalised signal → mood score
KALMAN_CI_Z     = 1.96   # Kalman confidence band (≈95%)
DATA_TTL        = 3600   # Streamlit cache TTL for the Google Sheets fetch (seconds)

# MSF Spread indicator
MSF_WINDOW      = 20     # rolling window for all MSF components
MSF_ROC_LEN     = 14     # NIFTY rate-of-change period
MSF_ZSCORE_CLIP = 3.0    # Z-score clipping threshold
MSF_SCALE       = 10.0   # output scaling factor

# Similar-period finder
SIMILAR_W_MAHA  = 0.55   # Mahalanobis distance weight
SIMILAR_W_TRAJ  = 0.35   # trajectory cosine-similarity weight
SIMILAR_W_RECV  = 0.10   # recency decay weight
TRAJ_WINDOW     = 20     # trajectory comparison window (trading days)
BACKTEST_HORIZON = 30    # default forward-return horizon (trading days)

# Chart display
OU_PROJ_DAYS    = 90     # OU mean-reversion projection horizon (calendar days)

# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN LOOK-UP TABLES
# ══════════════════════════════════════════════════════════════════════════════

# Maps regime label → (hex colour, CSS card class)
REGIME_STYLES: dict[str, tuple[str, str]] = {
    'Trending':       (C_GREEN, 'success'),
    'Volatile Trend': (C_AMBER, 'warning'),
    'Mean-Reverting': (C_CYAN,  'info'),
    'Choppy':         (C_RED,   'danger'),
    'Unknown':        (C_MUTED, 'neutral'),
}

# Shared Plotly dark-theme base for all figures
PLOTLY_BASE: dict = dict(
    template='plotly_dark',
    plot_bgcolor=C_BG_CARD,
    paper_bgcolor=C_BG_CARD,
    font=dict(color=C_TEXT, family='Inter'),
)

# ══════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

_DESIGN_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    :root {
        --primary-color: #FFC300;
        --primary-rgb: 255, 195, 0;
        --background-color: #0F0F0F;
        --secondary-background-color: #1A1A1A;
        --bg-card: #1A1A1A;
        --bg-elevated: #2A2A2A;
        --text-primary: #EAEAEA;
        --text-secondary: #EAEAEA;
        --text-muted: #888888;
        --border-color: #2A2A2A;
        --border-light: #3A3A3A;
        --success-green: #10b981;
        --danger-red: #ef4444;
        --warning-amber: #f59e0b;
        --info-cyan: #06b6d4;
        --neutral: #888888;
    }
    
    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    .main, [data-testid="stSidebar"] { background-color: var(--background-color); color: var(--text-primary); }
    .stApp > header { background-color: transparent; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    .block-container { padding-top: 3.5rem; max-width: 90%; padding-left: 2rem; padding-right: 2rem; }
    
    /* Sidebar toggle button - always visible */
    [data-testid="collapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        background-color: var(--secondary-background-color) !important;
        border: 2px solid var(--primary-color) !important;
        border-radius: 8px !important;
        padding: 10px !important;
        margin: 12px !important;
        box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.4) !important;
        z-index: 999999 !important;
        position: fixed !important;
        top: 14px !important;
        left: 14px !important;
        width: 40px !important;
        height: 40px !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    [data-testid="collapsedControl"]:hover {
        background-color: rgba(var(--primary-rgb), 0.2) !important;
        box-shadow: 0 0 20px rgba(var(--primary-rgb), 0.6) !important;
        transform: scale(1.05);
    }
    
    [data-testid="collapsedControl"] svg {
        stroke: var(--primary-color) !important;
        width: 20px !important;
        height: 20px !important;
    }
    
    [data-testid="stSidebar"] button[kind="header"] {
        background-color: transparent !important;
        border: none !important;
    }
    
    [data-testid="stSidebar"] button[kind="header"] svg {
        stroke: var(--primary-color) !important;
    }
    
    button[kind="header"] {
        z-index: 999999 !important;
    }
    
    .premium-header {
        background: var(--secondary-background-color);
        padding: 1.25rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 0 20px rgba(var(--primary-rgb), 0.1);
        border: 1px solid var(--border-color);
        position: relative;
        overflow: hidden;
        margin-top: 1rem;
    }
    
    .premium-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(circle at 20% 50%, rgba(var(--primary-rgb),0.08) 0%, transparent 50%);
        pointer-events: none;
    }
    
    .premium-header h1 { margin: 0; font-size: 2rem; font-weight: 700; color: var(--text-primary); letter-spacing: -0.50px; position: relative; }
    .premium-header .tagline { color: var(--text-muted); font-size: 0.9rem; margin-top: 0.25rem; font-weight: 400; position: relative; }
    .premium-header .product-badge { display: inline-block; background: rgba(var(--primary-rgb), 0.15); color: var(--primary-color); padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem; }
    
    .metric-card {
        background-color: var(--bg-card);
        padding: 1.25rem;
        border-radius: 12px;
        border: 1px solid var(--border-color);
        box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.08);
        margin-bottom: 0.5rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(0,0,0,0.3); border-color: var(--border-light); }
    .metric-card h4 { color: var(--text-muted); font-size: 0.75rem; margin-bottom: 0.5rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
    .metric-card h2 { color: var(--text-primary); font-size: 1.75rem; font-weight: 700; margin: 0; line-height: 1; }
    .metric-card .sub-metric { font-size: 0.75rem; color: var(--text-muted); margin-top: 0.5rem; font-weight: 500; }
    .metric-card.success h2 { color: var(--success-green); }
    .metric-card.danger h2 { color: var(--danger-red); }
    .metric-card.warning h2 { color: var(--warning-amber); }
    .metric-card.info h2 { color: var(--info-cyan); }
    .metric-card.neutral h2 { color: var(--neutral); }
    .metric-card.primary h2 { color: var(--primary-color); }
    
    .signal-card {
        background-color: var(--bg-card);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid var(--border-color);
        box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.08);
        margin-bottom: 1rem;
        position: relative;
        overflow: hidden;
    }
    
    .signal-card::before { content: ''; position: absolute; top: 0; left: 0; width: 4px; height: 100%; }
    .signal-card.bullish::before { background: var(--success-green); }
    .signal-card.bearish::before { background: var(--danger-red); }
    .signal-card.neutral::before { background: var(--neutral); }
    
    .status-badge { display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.4rem 0.8rem; border-radius: 20px; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; }
    .status-badge.bullish { background: rgba(16, 185, 129, 0.15); color: var(--success-green); border: 1px solid rgba(16, 185, 129, 0.3); }
    .status-badge.bearish { background: rgba(239, 68, 68, 0.15); color: var(--danger-red); border: 1px solid rgba(239, 68, 68, 0.3); }
    .status-badge.oversold { background: rgba(6, 182, 212, 0.15); color: var(--info-cyan); border: 1px solid rgba(6, 182, 212, 0.3); }
    .status-badge.overbought { background: rgba(245, 158, 11, 0.15); color: var(--warning-amber); border: 1px solid rgba(245, 158, 11, 0.3); }
    .status-badge.neutral { background: rgba(136, 136, 136, 0.15); color: var(--neutral); border: 1px solid rgba(136, 136, 136, 0.3); }
    
    .stButton>button { border: 2px solid var(--primary-color); background: transparent; color: var(--primary-color); font-weight: 700; border-radius: 12px; padding: 0.75rem 2rem; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); text-transform: uppercase; letter-spacing: 0.5px; }
    .stButton>button:hover { box-shadow: 0 0 25px rgba(var(--primary-rgb), 0.6); background: var(--primary-color); color: #1A1A1A; transform: translateY(-2px); }
    .stButton>button:active { transform: translateY(0); }
    
    .stTabs [data-baseweb="tab-list"] { gap: 24px; background: transparent; }
    .stTabs [data-baseweb="tab"] { color: var(--text-muted); border-bottom: 2px solid transparent; transition: color 0.3s, border-bottom 0.3s; background: transparent; font-weight: 600; }
    .stTabs [aria-selected="true"] { color: var(--primary-color); border-bottom: 2px solid var(--primary-color); background: transparent !important; }
    
    .stPlotlyChart { border-radius: 12px; background-color: var(--secondary-background-color); padding: 10px; border: 1px solid var(--border-color); box-shadow: 0 0 25px rgba(var(--primary-rgb), 0.1); }
    .stDataFrame { border-radius: 12px; background-color: var(--secondary-background-color); border: 1px solid var(--border-color); }
    .section-divider { height: 1px; background: linear-gradient(90deg, transparent 0%, var(--border-color) 50%, transparent 100%); margin: 1.5rem 0; }
    
    .sidebar-title { font-size: 0.75rem; font-weight: 700; color: var(--primary-color); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.75rem; }
    
    [data-testid="stSidebar"] { background: var(--secondary-background-color); border-right: 1px solid var(--border-color); }
    
    .stTextInput > div > div > input { background: var(--bg-elevated) !important; border: 1px solid var(--border-color) !important; border-radius: 8px !important; color: var(--text-primary) !important; }
    .stTextInput > div > div > input:focus { border-color: var(--primary-color) !important; box-shadow: 0 0 0 2px rgba(var(--primary-rgb), 0.2) !important; }
    
    .info-box { background: var(--secondary-background-color); border: 1px solid var(--border-color); border-left: 0px solid var(--primary-color); padding: 1.25rem; border-radius: 12px; margin: 0.5rem 0; box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.08); }
    .info-box h4 { color: var(--primary-color); margin: 0 0 0.5rem 0; font-size: 1rem; font-weight: 700; }
    .info-box p { color: var(--text-muted); margin: 0; font-size: 0.9rem; line-height: 1.6; }
    
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--background-color); }
    ::-webkit-scrollbar-thumb { background: var(--border-color); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--border-light); }
</style>
"""

st.markdown(_DESIGN_CSS, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def sigmoid(x, scale=1.0):
    """Sigmoid normalization to [-1, 1] range"""
    return 2.0 / (1.0 + np.exp(-x / scale)) - 1.0

def zscore_clipped(series, window, clip=3.0):
    """Z-score with rolling window and clipping"""
    roll_mean = series.rolling(window=window, min_periods=1).mean()
    roll_std = series.rolling(window=window, min_periods=1).std()
    z = (series - roll_mean) / roll_std.replace(0, np.nan)
    return z.clip(-clip, clip).fillna(0)

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
#   ornstein_uhlenbeck_estimate     → mood normalization        → physics-based scaling
#   kalman_filter_1d                → mood smoothing            → adaptive noise filtering
#   rolling_hurst                   → diagnostics (output only) → trending vs reverting
#   rolling_entropy                 → diagnostics (output only) → market disorder
#   mahalanobis_distance_batch      → similar periods           → covariance-aware matching
#   cosine_similarity               → similar periods           → trajectory shape matching
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
        order = arr.argsort()
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(arr) + 1, dtype=float)
        for val in np.unique(arr):
            mask = arr == val
            if mask.sum() > 1:
                ranks[mask] = ranks[mask].mean()
        return ranks
    
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
    Shannon entropy H = -Σ p_i * log₂(p_i), normalized to [0, 1].
    H≈1 → maximum disorder (uniform dist), H≈0 → perfect order (delta).
    
    Applied to variable returns to measure how "noisy" a predictor is.
    
    Used in: calculate_historical_mood → _build_weights (Layer 2)
    """
    clean = values[np.isfinite(values)]
    if len(clean) < 5:
        return 0.5
    counts, _ = np.histogram(clean, bins=n_bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    if len(probs) <= 1:
        return 0.0
    h = -np.sum(probs * np.log2(probs))
    h_max = np.log2(n_bins)
    return h / h_max if h_max > 0 else 0.0

def adaptive_percentile(series, half_life=252):
    """
    Exponential-decay-weighted empirical CDF.
    
    For each time t, the percentile of x_t is:
        P(t) = Σ_{i≤t} w_i · 𝟙(x_i ≤ x_t) / Σ_{i≤t} w_i
    where w_i = exp(-λ·(t-i)), λ = ln(2)/half_life.
    
    This makes recent data count more in determining "where are we historically."
    A PE of 22 is judged against recent-ish history, not against 2005.
    
    Used in: calculate_historical_mood (Layer 3)
    """
    values = np.asarray(series, dtype=np.float64)
    n = len(values)
    result = np.full(n, 0.5)
    lam = np.log(2) / max(half_life, 1)
    
    for t in range(1, n):
        if not np.isfinite(values[t]):
            result[t] = result[t - 1] if t > 0 else 0.5
            continue
        
        ages = np.arange(t, -1, -1, dtype=np.float64)
        weights = np.exp(-lam * ages)
        indicators = (values[:t + 1] <= values[t]).astype(np.float64)
        valid = np.isfinite(values[:t + 1])
        w_valid = weights[valid]
        ind_valid = indicators[valid]
        w_sum = w_valid.sum()
        if w_sum > 1e-12:
            result[t] = np.sum(w_valid * ind_valid) / w_sum
    
    return result

def ornstein_uhlenbeck_estimate(series, dt=1.0):
    """
    Estimate Ornstein-Uhlenbeck process parameters from discrete observations.
    
    The OU process: dx = θ(μ − x)dt + σdW
    Discrete AR(1): x_{t+1} = a + b·x_t + ε
        b = exp(−θ·dt), a = μ·(1−b), Var(ε) = σ²·(1−b²)/(2θ)
    
    Returns (theta, mu, sigma):
        theta : mean-reversion speed (higher = faster snap-back)
        mu    : long-run equilibrium level
        sigma : process volatility
    
    Used in: calculate_historical_mood (Layer 4)
    """
    ts = np.asarray(series, dtype=np.float64)
    ts = ts[np.isfinite(ts)]
    if len(ts) < 10:
        return (0.01, 0.0, 1.0)
    
    x, y = ts[:-1], ts[1:]
    mean_x, mean_y = x.mean(), y.mean()
    
    cov_xy = np.sum((x - mean_x) * (y - mean_y))
    var_x = np.sum((x - mean_x) ** 2)
    if var_x < 1e-12:
        return (0.01, mean_x, 1.0)
    
    b = np.clip(cov_xy / var_x, 1e-6, 1.0 - 1e-6)
    a = mean_y - b * mean_x
    
    theta = np.clip(-np.log(b) / dt, 1e-4, 10.0)
    mu = a / (1.0 - b)
    
    residuals = y - (a + b * x)
    var_eps = np.var(residuals)
    sigma_sq = 2.0 * theta * var_eps / (1.0 - b ** 2) if (1.0 - b ** 2) > 1e-12 else var_eps
    sigma = np.sqrt(max(sigma_sq, 1e-12))
    
    return (theta, mu, sigma)

def kalman_filter_1d(observations, process_var=None, measurement_var=None):
    """
    1D Kalman filter for adaptive smoothing.
    
    The state model: "there exists a true underlying mood being observed with noise."
    - When SNR is low (noisy): Kalman gain ↓, smooths aggressively
    - When SNR is high (clean): Kalman gain ↑, tracks signal closely
    
    Auto-estimates noise parameters from data if not provided.
    
    Used in: calculate_historical_mood (Layer 5)
    Returns: (filtered_state, kalman_gains, estimate_variances)
    """
    obs = np.asarray(observations, dtype=np.float64)
    n = len(obs)
    if n == 0:
        return np.array([]), np.array([]), np.array([])
    
    if process_var is None:
        diffs = np.diff(obs[np.isfinite(obs)])
        process_var = max(np.var(diffs) * 0.1, 1e-8) if len(diffs) > 1 else 1e-3
    if measurement_var is None:
        clean = obs[np.isfinite(obs)]
        measurement_var = max(np.var(clean) * 0.5, 1e-8) if len(clean) > 1 else 1.0
    
    state = obs[0] if np.isfinite(obs[0]) else 0.0
    estimate_var = measurement_var
    filtered = np.zeros(n)
    gains = np.zeros(n)
    variances = np.zeros(n)
    filtered[0] = state
    variances[0] = estimate_var
    
    for i in range(1, n):
        pred_var = estimate_var + process_var
        if np.isfinite(obs[i]):
            K = pred_var / (pred_var + measurement_var)
            state = state + K * (obs[i] - state)
            estimate_var = (1 - K) * pred_var
            gains[i] = K
        else:
            estimate_var = pred_var
        filtered[i] = state
        variances[i] = estimate_var
    
    return filtered, gains, variances

def _hurst_rs(series, max_lag=None):
    """
    Hurst exponent via Rescaled Range (R/S) analysis.
    H > 0.5 → persistent (trending), H < 0.5 → anti-persistent (mean-reverting).
    Internal helper for rolling_hurst.
    """
    ts = np.asarray(series, dtype=np.float64)
    ts = ts[np.isfinite(ts)]
    n = len(ts)
    if n < 20:
        return 0.5
    if max_lag is None:
        max_lag = min(n // 2, 200)
    
    lags, rs_values = [], []
    for lag in range(10, max_lag + 1, max(1, (max_lag - 10) // 25)):
        n_blocks = n // lag
        if n_blocks < 1:
            continue
        rs_block = []
        for b in range(n_blocks):
            block = ts[b * lag:(b + 1) * lag]
            deviations = block - block.mean()
            cumulative = np.cumsum(deviations)
            R = cumulative.max() - cumulative.min()
            S = block.std(ddof=1)
            if S > 1e-12:
                rs_block.append(R / S)
        if rs_block:
            lags.append(lag)
            rs_values.append(np.mean(rs_block))
    
    if len(lags) < 3:
        return 0.5
    log_lags = np.log(np.array(lags, dtype=np.float64))
    log_rs = np.log(np.array(rs_values, dtype=np.float64))
    valid = np.isfinite(log_lags) & np.isfinite(log_rs)
    if valid.sum() < 3:
        return 0.5
    log_lags, log_rs = log_lags[valid], log_rs[valid]
    mean_x, mean_y = log_lags.mean(), log_rs.mean()
    var_x = np.sum((log_lags - mean_x) ** 2)
    H = np.sum((log_lags - mean_x) * (log_rs - mean_y)) / var_x if var_x > 1e-12 else 0.5
    return np.clip(H, 0.01, 0.99)

def rolling_hurst(series, window=90, step=5):
    """
    Rolling Hurst exponent. Computed every `step` points, forward-filled.
    Used in: calculate_historical_mood → diagnostics output
    """
    values = np.asarray(series, dtype=np.float64)
    n = len(values)
    result = np.full(n, 0.5)
    for i in range(window, n, step):
        result[i] = _hurst_rs(values[i - window:i])
    # Forward-fill stepped values
    for i in range(1, n):
        if result[i] == 0.5 and i > window and result[i - 1] != 0.5:
            result[i] = result[i - 1]
    return result

def rolling_entropy(series, window=60, n_bins=15):
    """
    Rolling Shannon entropy of a series. Normalized to [0, 1].
    Used in: calculate_historical_mood → diagnostics output
    """
    values = series.values if hasattr(series, 'values') else np.asarray(series)
    n = len(values)
    result = np.full(n, 0.5)
    for i in range(window, n):
        result[i] = shannon_entropy(values[i - window:i], n_bins)
    for i in range(5, min(window, n)):
        result[i] = shannon_entropy(values[:i], n_bins)
    return result

def mahalanobis_distance_batch(features, center, cov_matrix):
    """
    Mahalanobis distance: d_M = √((x−μ)ᵀ Σ⁻¹ (x−μ))
    Accounts for correlations — two periods close in correlated dimensions
    are less similar than Euclidean distance would suggest.
    Used in: find_similar_periods
    """
    diff = features - center
    reg = 1e-6 * np.eye(cov_matrix.shape[0])
    try:
        cov_inv = np.linalg.inv(cov_matrix + reg)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov_matrix + reg)
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

def detect_regime_transitions(hurst_values, entropy_values, window=10):
    """
    Detect regime transitions using Hurst exponent + entropy jointly.
    
    The idea: market operates in one of 4 quadrants:
      High H, Low S  → Trending/Ordered   (momentum works, strong directional move)
      High H, High S → Trending/Disordered (volatile trend, large swings in one direction)
      Low H, Low S   → Mean-reverting/Ordered (range-bound, predictable oscillation)
      Low H, High S  → Mean-reverting/Disordered (choppy chaos, hardest to trade)
    
    A regime TRANSITION is when the market crosses quadrant boundaries.
    Specifically, the most important transitions are:
      Trending→Choppy : H drops below 0.5 while S rises → trend exhaustion
      Choppy→Trending : H rises above 0.5 while S drops → new trend emerging
    
    We smooth both signals and detect crossover events.
    
    Returns: array of regime labels + transition indices
    """
    h = np.asarray(hurst_values, dtype=np.float64)
    s = np.asarray(entropy_values, dtype=np.float64)
    n = len(h)
    
    if n < window * 2:
        return np.full(n, 'Unknown', dtype=object), []
    
    # Smooth both signals to avoid noise-triggered transitions
    h_smooth = pd.Series(h).rolling(window=window, min_periods=1).mean().values
    s_smooth = pd.Series(s).rolling(window=window, min_periods=1).mean().values
    
    # Median thresholds (adaptive to the data, not hardcoded)
    h_threshold = 0.5   # Theoretical random walk boundary
    s_median = np.median(s_smooth[s_smooth > 0]) if np.any(s_smooth > 0) else 0.5
    
    # Classify each point into regime quadrant
    regimes = np.full(n, 'Unknown', dtype=object)
    for i in range(n):
        trending = h_smooth[i] > h_threshold
        ordered = s_smooth[i] < s_median
        
        if trending and ordered:
            regimes[i] = 'Trending'         # Best for momentum
        elif trending and not ordered:
            regimes[i] = 'Volatile Trend'   # Momentum with risk
        elif not trending and ordered:
            regimes[i] = 'Mean-Reverting'   # Best for contrarian
        else:
            regimes[i] = 'Choppy'           # Hardest to trade
    
    # Detect transition points (regime[i] != regime[i-1])
    transitions = []
    for i in range(1, n):
        if regimes[i] != regimes[i - 1]:
            prev = regimes[i - 1]
            curr = regimes[i]
            
            # Classify transition significance
            # Major: Trending↔Choppy (complete character flip)
            # Minor: adjacent quadrant shifts
            major_pairs = {
                ('Trending', 'Choppy'), ('Choppy', 'Trending'),
                ('Trending', 'Mean-Reverting'), ('Mean-Reverting', 'Trending'),
            }
            is_major = (prev, curr) in major_pairs
            
            transitions.append({
                'index': i,
                'from': prev,
                'to': curr,
                'major': is_major,
                'hurst': h_smooth[i],
                'entropy': s_smooth[i],
            })
    
    return regimes, transitions

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=DATA_TTL, show_spinner=False)
def load_data():
    """Load market data from Google Sheets."""
    start_time = time.time()
    try:
        url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={SHEET_GID}"
        df = pd.read_csv(url, dtype=str)

        # Clean column names: strip whitespace, drop unnamed columns
        df.columns = [c.strip() for c in df.columns]
        df = df[[c for c in df.columns if not c.startswith('Unnamed')]]

        if 'DATE' not in df.columns or 'NIFTY' not in df.columns:
            raise ValueError("Required columns DATE and NIFTY not found in the Sheet.")

        if not any(col in df.columns for col in EXPECTED_COLUMNS):
            raise ValueError("None of the expected columns found in the Sheet.")

        # Fill any missing EXPECTED_COLUMNS with 0
        missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]
        if missing_columns:
            logging.warning(f"Missing columns: {missing_columns}. Setting to 0.0.")
            for col in missing_columns:
                df[col] = "0.0"

        df['DATE'] = pd.to_datetime(df['DATE'], format='%m/%d/%Y', errors='coerce')

        # Convert all non-DATE columns to numeric
        non_date_cols = [col for col in df.columns if col != 'DATE']
        df[non_date_cols] = df[non_date_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

        df = df[df['NIFTY'] > 0].dropna(subset=['DATE']).copy()
        if df.empty:
            raise ValueError("No valid rows with positive NIFTY or valid DATE.")

        # Keep DATE and NIFTY first, then all other columns (preserving extra sheet columns)
        core_cols = ['DATE', 'NIFTY']
        extra_cols = [c for c in df.columns if c not in core_cols]
        df = df[core_cols + extra_cols].sort_values('DATE').reset_index(drop=True)
        
        # v2.0: Ensure NIFTY50_EY has real data.
        # EY (Earnings Yield) = 1/PE × 100. If the sheet has PE but EY is
        # missing or constant (all zeros), derive it. This is the most common
        # cause of "empty EY correlations" — the sheet simply doesn't populate it.
        if 'NIFTY50_PE' in df.columns and df['NIFTY50_PE'].gt(0).any():
            if 'NIFTY50_EY' not in df.columns or df['NIFTY50_EY'].nunique() <= 1:
                df['NIFTY50_EY'] = (1.0 / df['NIFTY50_PE'].replace(0, np.nan) * 100).fillna(0)
                logging.info("Derived NIFTY50_EY from NIFTY50_PE (1/PE × 100).")
        
        # v2.0: Derive yield curve term spreads.
        # Formula: Term Spread = 10-Year Bond Yield − 2-Year Bond Yield
        # IN_TERM_SPREAD = IN10Y − IN02Y  (India yield curve slope)
        # US_TERM_SPREAD = US10Y − US02Y  (US yield curve slope)
        # Positive = normal curve (expansion), Negative = inverted (recession signal).
        # The 10Y−2Y spread is the single most validated macro predictor:
        # every US recession since 1960 was preceded by inversion.
        if 'IN10Y' in df.columns and 'IN02Y' in df.columns:
            df['IN_TERM_SPREAD'] = df['IN10Y'] - df['IN02Y']
        else:
            df['IN_TERM_SPREAD'] = 0.0
        
        if 'US10Y' in df.columns and 'US02Y' in df.columns:
            df['US_TERM_SPREAD'] = df['US10Y'] - df['US02Y']
        else:
            df['US_TERM_SPREAD'] = 0.0
        
        logging.info(f"Data loaded in {time.time() - start_time:.2f} seconds.")
        return df
    except Exception as e:
        logging.error(f"Failed to load data: {str(e)}")
        st.error(f"Failed to load data. Ensure the Google Sheet is 'Public' and the ID is correct. Error: {str(e)}")
        return None

# ══════════════════════════════════════════════════════════════════════════════
# MOOD SCORE CALCULATION
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
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

@st.cache_data
def calculate_historical_mood(df, dependent_vars=None):
    """
    v2.0 Mood Score Engine — 5-layer architecture.
    
    Layer 1: Adaptive Correlations    (decay-weighted Spearman)
    Layer 2: Information Weighting    (|corr| × entropy penalty)
    Layer 3: Adaptive Percentiles     (decay-weighted ECDF)
    Layer 4: OU Normalization         (physics-based scaling)
    Layer 5: Kalman Smoothing         (adaptive noise filtering)
    
    Diagnostics (output-only, do NOT modify the score):
      Hurst exponent, market entropy, OU half-life
    """
    if dependent_vars is None:
        dependent_vars = DEPENDENT_VARS
    start_time = time.time()
    
    if 'DATE' not in df.columns or 'NIFTY50_PE' not in df.columns or 'NIFTY50_EY' not in df.columns:
        logging.error("Required columns missing.")
        return pd.DataFrame(columns=['DATE', 'Mood_Score', 'Mood', 'Smoothed_Mood_Score', 'Mood_Volatility'])
    
    n = len(df)
    
    # ── Layer 1: Adaptive Correlations ──────────────────────────────────
    pe_corrs = calculate_anchor_correlations(df, 'NIFTY50_PE', dependent_vars)
    ey_corrs = calculate_anchor_correlations(df, 'NIFTY50_EY', dependent_vars)
    
    # ── Layer 2: Information-Theoretic Weighting ────────────────────────
    # weight = |correlation| × (1 − normalized_entropy_of_variable)
    # Entropy is computed ONCE per variable on its returns distribution.
    # Low-entropy (structured) variables get amplified.
    # High-entropy (noisy/random) variables get suppressed.
    # This is the ONE place entropy belongs — in variable selection.
    var_entropies = {}
    for var in [col for col in dependent_vars if col in df.columns]:
        var_returns = df[var].pct_change().dropna().values
        var_entropies[var] = shannon_entropy(var_returns) if len(var_returns) > 10 else 0.5
    
    def _build_weights(corr_df):
        raw = {}
        for _, row in corr_df.iterrows():
            var = row['variable']
            corr_mag = abs(row['correlation'])
            entropy_penalty = 1.0 - var_entropies.get(var, 0.5)
            raw[var] = corr_mag * max(entropy_penalty, 0.1)
        total = max(sum(raw.values()), 1e-10)
        return {k: v / total for k, v in raw.items()}
    
    pe_weights = _build_weights(pe_corrs)
    ey_weights = _build_weights(ey_corrs)
    
    # ── Layer 3: Adaptive Percentiles ───────────────────────────────────
    # Half-life = PCT_HALF_LIFE trading days (~1 year). Recent market structure
    # matters more than ancient history for positioning.
    pct_hl = min(PCT_HALF_LIFE, n // 2) if n > 20 else max(n // 2, 5)
    
    pe_percentiles = adaptive_percentile(df['NIFTY50_PE'].values, half_life=pct_hl)
    ey_percentiles = adaptive_percentile(df['NIFTY50_EY'].values, half_life=pct_hl)
    
    pe_base = -1.0 + 2.0 * (1.0 - pe_percentiles)  # High PE = bearish
    ey_base = -1.0 + 2.0 * ey_percentiles             # High EY = bullish
    
    pe_adjustments = np.zeros(n)
    ey_adjustments = np.zeros(n)
    
    for var in [col for col in dependent_vars if col in df.columns]:
        var_pct = adaptive_percentile(df[var].values, half_life=pct_hl)
        
        if var in pe_weights:
            pe_match = pe_corrs.loc[pe_corrs['variable'] == var]
            pe_type = pe_match.iloc[0]['type'] if len(pe_match) > 0 else 'positive'
            sign = 1.0 if pe_type == 'positive' else -1.0
            pe_adjustments += sign * pe_weights[var] * (1.0 - var_pct)
        
        if var in ey_weights:
            ey_match = ey_corrs.loc[ey_corrs['variable'] == var]
            ey_type = ey_match.iloc[0]['type'] if len(ey_match) > 0 else 'positive'
            sign = 1.0 if ey_type == 'positive' else -1.0
            ey_adjustments += sign * ey_weights[var] * var_pct
    
    pe_scores = np.clip(0.5 * pe_base + 0.5 * pe_adjustments, -1, 1)
    ey_scores = np.clip(0.5 * ey_base + 0.5 * ey_adjustments, -1, 1)
    
    pe_strength = sum(abs(r['correlation']) for _, r in pe_corrs.iterrows())
    ey_strength = sum(abs(r['correlation']) for _, r in ey_corrs.iterrows())
    total_strength = pe_strength + ey_strength or 1
    raw_mood = (pe_strength / total_strength) * pe_scores + (ey_strength / total_strength) * ey_scores
    
    # ── Layer 4: OU Normalization ───────────────────────────────────────
    # Instead of global z-score (adding 1 point shifts ALL history),
    # model the mood as Ornstein-Uhlenbeck: dx = θ(μ-x)dt + σdW
    # Normalize by the OU stationary std: σ_∞ = σ/√(2θ)
    # This is LOCAL and has physical meaning: "distance from equilibrium in natural units"
    
    # First: expanding z-score to get rough scale
    expanding_mean = pd.Series(raw_mood).expanding().mean().values
    expanding_std = np.maximum(pd.Series(raw_mood).expanding().std().fillna(1).values, 1e-6)
    rough_scaled = (raw_mood - expanding_mean) / expanding_std
    
    # OU estimation on the rough-scaled series
    theta, mu, sigma_ou = ornstein_uhlenbeck_estimate(rough_scaled) if n > 50 else (0.05, 0.0, 1.0)
    sigma_ou = max(sigma_ou, 1e-6)
    ou_stationary_std = max(sigma_ou / np.sqrt(2.0 * max(theta, 1e-4)), 1e-6)
    
    mood_scores = np.clip((rough_scaled - mu) / ou_stationary_std * MOOD_SCALE, -100, 100)
    
    # OU half-life: the expected time for the current deviation to halve
    ou_half_life = np.log(2) / max(theta, 1e-4)
    
    # ── Layer 5: Kalman Smoothing ───────────────────────────────────────
    smoothed_mood_scores, kalman_gains, kalman_variances = kalman_filter_1d(mood_scores)
    
    # Confidence band: ±KALMAN_CI_Z * sqrt(variance) for ~95% interval
    kalman_std = np.sqrt(np.maximum(kalman_variances, 0))
    confidence_upper = smoothed_mood_scores + KALMAN_CI_Z * kalman_std
    confidence_lower = smoothed_mood_scores - KALMAN_CI_Z * kalman_std
    
    # Traditional volatility (backward compatible)
    mood_volatility = pd.Series(mood_scores).rolling(window=30, min_periods=1).std().fillna(0)
    
    # ── Classification (fixed thresholds — see VISION.md §6 for why) ───
    moods = np.where(mood_scores > 60, 'Very Bullish',
            np.where(mood_scores > 20, 'Bullish',
            np.where(mood_scores > -20, 'Neutral',
            np.where(mood_scores > -60, 'Bearish', 'Very Bearish'))))
    
    # ── Diagnostics (output-only — do NOT modify scores) ───────────────
    nifty_returns = df['NIFTY'].pct_change().fillna(0).values
    hurst_vals = rolling_hurst(mood_scores, window=90, step=5)
    entropy_vals = rolling_entropy(nifty_returns, window=60, n_bins=15)
    
    # ── Regime Detection ────────────────────────────────────────────────
    regime_labels, regime_transitions = detect_regime_transitions(hurst_vals, entropy_vals)
    
    result_df = pd.DataFrame({
        'DATE': df['DATE'].values,
        'Mood_Score': mood_scores,
        'Mood': moods,
        'Smoothed_Mood_Score': smoothed_mood_scores,
        'Mood_Volatility': mood_volatility.values,
        'NIFTY': df['NIFTY'].values,
        'AD_RATIO': df['AD_RATIO'].values if 'AD_RATIO' in df.columns else 1.0,
        # v2.0 diagnostics
        'Hurst': hurst_vals,
        'Market_Entropy': entropy_vals,
        'OU_Half_Life': ou_half_life,  # Scalar, broadcast
        'OU_Theta': theta,
        'OU_Mu': mu,
        # v2.1 additions
        'OU_Sigma': sigma_ou,
        'Confidence_Upper': np.clip(confidence_upper, -100, 100),
        'Confidence_Lower': np.clip(confidence_lower, -100, 100),
        'Regime': regime_labels,
    })
    
    logging.info(f"v2.0 mood [{n} rows] in {time.time() - start_time:.2f}s | "
                 f"θ={theta:.3f} μ={mu:.2f} t½={ou_half_life:.0f}d "
                 f"H={hurst_vals[-1]:.2f} S={entropy_vals[-1]:.2f}")
    return result_df

# ══════════════════════════════════════════════════════════════════════════════
# MSF-ENHANCED SPREAD INDICATOR
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def calculate_msf_spread(df, mood_col='Mood_Score', nifty_col='NIFTY', breadth_col='AD_RATIO'):
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
    
    mood = df[mood_col].values if mood_col in df.columns else np.zeros(n)
    nifty = df[nifty_col].values if nifty_col in df.columns else mood
    breadth = df[breadth_col].values if breadth_col in df.columns else np.ones(n)
    
    mood_series = pd.Series(mood, index=df.index)
    nifty_series = pd.Series(nifty, index=df.index)
    breadth_series = pd.Series(breadth, index=df.index)
    
    if n == 0:
        logging.error("Empty data for MSF Spread calculation.")
        return result
    
    # ── Component 1: Momentum (NIFTY ROC z-score) ──────────────────────
    roc_raw = nifty_series.pct_change(MSF_ROC_LEN, fill_method=None)
    roc_z = zscore_clipped(roc_raw, MSF_WINDOW, MSF_ZSCORE_CLIP)
    momentum_norm = sigmoid(roc_z, 1.5)

    # ── Component 2: Structure (Mood trend divergence + acceleration) ──
    trend_fast = mood_series.rolling(5, min_periods=1).mean()
    trend_slow = mood_series.rolling(MSF_WINDOW, min_periods=1).mean()
    trend_diff_z = zscore_clipped(trend_fast - trend_slow, MSF_WINDOW, MSF_ZSCORE_CLIP)
    mood_accel_raw = mood_series.diff(5).diff(5)
    mood_accel_z = zscore_clipped(mood_accel_raw, MSF_WINDOW, MSF_ZSCORE_CLIP)
    structure_z = (trend_diff_z + mood_accel_z) / np.sqrt(2.0)
    structure_norm = sigmoid(structure_z, 1.5)

    # ── Component 3: Regime (Adaptive threshold) ────────────────────────
    # v1.x: fixed 0.0033 threshold. v2.0: scales with local volatility.
    # A move is "directional" only if it exceeds half a local std.
    pct_change = nifty_series.pct_change(fill_method=None).fillna(0)
    rolling_vol = pct_change.rolling(window=MSF_WINDOW, min_periods=5).std().fillna(0.003)
    adaptive_threshold = (rolling_vol * 0.5).clip(lower=0.001)

    regime_signals = np.where(pct_change > adaptive_threshold, 1,
                     np.where(pct_change < -adaptive_threshold, -1, 0))
    regime_count = pd.Series(regime_signals, index=df.index).cumsum()
    regime_raw = regime_count - regime_count.rolling(MSF_WINDOW, min_periods=1).mean()
    regime_z = zscore_clipped(regime_raw, MSF_WINDOW, MSF_ZSCORE_CLIP)
    regime_norm = sigmoid(regime_z, 1.5)

    # ── Component 4: Breadth Flow ───────────────────────────────────────
    breadth_ma = breadth_series.rolling(MSF_WINDOW, min_periods=1).mean()
    breadth_ratio = breadth_series / breadth_ma.replace(0, 1)
    breadth_z = zscore_clipped(breadth_ratio - 1, MSF_WINDOW, MSF_ZSCORE_CLIP)
    flow_norm = sigmoid(breadth_z, 1.5)
    
    # ── Inverse-Variance Weighting ──────────────────────────────────────
    # Markowitz for signals: stable (low variance) components get more weight.
    components = {
        'momentum': momentum_norm,
        'structure': structure_norm,
        'regime': regime_norm,
        'flow': flow_norm,
    }
    
    tail_window = min(60, n)
    inv_vars = {}
    for name, comp in components.items():
        comp_vals = comp.values if hasattr(comp, 'values') else np.asarray(comp)
        tail = comp_vals[-tail_window:]
        tail_clean = tail[np.isfinite(tail)]
        var = np.var(tail_clean) if len(tail_clean) > 5 else 1.0
        inv_vars[name] = 1.0 / max(var, 1e-6)
    
    total_inv_var = sum(inv_vars.values())
    weights = {k: v / total_inv_var for k, v in inv_vars.items()}
    
    msf_raw = sum(weights[name] * comp for name, comp in components.items())
    msf_spread = msf_raw * MSF_SCALE

    result['msf_spread'] = msf_spread
    result['momentum']   = momentum_norm  * MSF_SCALE
    result['structure']  = structure_norm * MSF_SCALE
    result['regime']     = regime_norm    * MSF_SCALE
    result['flow']       = flow_norm      * MSF_SCALE
    
    weight_str = ', '.join(f"{k}={v:.0%}" for k, v in weights.items())
    logging.info(f"v2.0 MSF in {time.time() - start_time:.2f}s [{weight_str}]")
    return result

# ══════════════════════════════════════════════════════════════════════════════
# SIMILAR PERIODS FINDER
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def find_similar_periods(df, top_n=10, recency_weight=0.1):
    """
    v2.0 Similar Period Finder.
    
    3-part scoring:
      1. Mahalanobis distance (55%) — covariance-aware state matching
         Features: mood, volatility, NIFTY momentum, Hurst, entropy
      2. Trajectory cosine similarity (35%) — detrended mood path shape
      3. Exponential recency decay (10%) — prefer recent analogs
    """
    if df.empty or 'Mood_Score' not in df.columns:
        return []
    
    latest = df.iloc[-1]
    n = len(df)
    
    historical = df.iloc[:-30].copy() if n > 30 else df.iloc[:-1].copy()
    if historical.empty or len(historical) < 5:
        return []
    
    # ── Build Feature Vectors ───────────────────────────────────────────
    nifty_roc = df['NIFTY'].pct_change(MSF_ROC_LEN).fillna(0).values
    
    feature_cols = ['Mood_Score', 'Mood_Volatility']
    current_features = [latest['Mood_Score'], latest['Mood_Volatility']]
    hist_arrays = [historical['Mood_Score'].values, historical['Mood_Volatility'].values]
    
    # NIFTY momentum
    feature_cols.append('NIFTY_ROC')
    current_features.append(nifty_roc[-1] if len(nifty_roc) > 0 else 0.0)
    h_roc = nifty_roc[:len(historical)]
    if len(h_roc) < len(historical):
        h_roc = np.pad(h_roc, (len(historical) - len(h_roc), 0), constant_values=0)
    hist_arrays.append(h_roc[:len(historical)])
    
    # Hurst (if available from v2.0 engine)
    if 'Hurst' in df.columns:
        feature_cols.append('Hurst')
        current_features.append(latest['Hurst'])
        hist_arrays.append(historical['Hurst'].values)
    
    # Market Entropy (if available)
    if 'Market_Entropy' in df.columns:
        feature_cols.append('Market_Entropy')
        current_features.append(latest['Market_Entropy'])
        hist_arrays.append(historical['Market_Entropy'].values)
    
    current_vec = np.array(current_features, dtype=np.float64)
    hist_matrix = np.column_stack(hist_arrays)
    
    # Clean NaN/Inf → column medians
    for col in range(hist_matrix.shape[1]):
        col_data = hist_matrix[:, col]
        valid = np.isfinite(col_data)
        median_val = np.median(col_data[valid]) if valid.any() else 0.0
        hist_matrix[~valid, col] = median_val
    current_vec = np.where(np.isfinite(current_vec), current_vec, 0.0)
    
    # ── Part 1: Mahalanobis Distance (55%) ──────────────────────────────
    cov_matrix = np.cov(hist_matrix, rowvar=False)
    if cov_matrix.ndim < 2:
        cov_matrix = np.array([[max(float(cov_matrix), 1e-6)]])
    
    maha_dist = mahalanobis_distance_batch(hist_matrix, current_vec, cov_matrix)
    max_dist = maha_dist.max() if maha_dist.max() > 0 else 1.0
    maha_sim = 1.0 - (maha_dist / max_dist)
    
    # ── Part 2: Trajectory Cosine Similarity (SIMILAR_W_TRAJ) ──────────
    traj_sim = np.zeros(len(historical))

    if n > TRAJ_WINDOW:
        current_traj = df['Mood_Score'].values[-TRAJ_WINDOW:]
        ct_detrended = current_traj - np.linspace(current_traj[0], current_traj[-1], TRAJ_WINDOW)

        for j, idx in enumerate(historical.index):
            pos = df.index.get_loc(idx)
            if pos >= TRAJ_WINDOW:
                hist_traj = df['Mood_Score'].values[pos - TRAJ_WINDOW:pos]
                ht_detrended = hist_traj - np.linspace(hist_traj[0], hist_traj[-1], TRAJ_WINDOW)
                traj_sim[j] = (cosine_similarity(ct_detrended, ht_detrended) + 1) / 2

    # ── Part 3: Exponential Recency Decay (SIMILAR_W_RECV) ──────────────
    days_since = (latest['DATE'] - historical['DATE']).dt.days.values.astype(float)
    recency = np.exp(-np.log(2) * days_since / 365.0) * recency_weight
    recency_norm = recency / max(recency.max(), 1e-6)

    # ── Combined ────────────────────────────────────────────────────────
    combined = SIMILAR_W_MAHA * maha_sim + SIMILAR_W_TRAJ * traj_sim + SIMILAR_W_RECV * recency_norm
    
    historical = historical.copy()
    historical['similarity'] = combined
    top_similar = historical.nlargest(top_n, 'similarity')
    
    results = []
    nifty_vals = df['NIFTY'].values
    for _, row in top_similar.iterrows():
        idx_pos = df.index.get_loc(row.name)
        nifty_at = row['NIFTY'] if 'NIFTY' in row and row['NIFTY'] > 0 else None
        
        # Forward returns: what happened to NIFTY 30/60/90 days after this analog?
        fwd_returns = {}
        for horizon in [30, 60, 90]:
            fwd_idx = idx_pos + horizon
            if fwd_idx < len(nifty_vals) and nifty_at and nifty_at > 0:
                fwd_returns[horizon] = (nifty_vals[fwd_idx] / nifty_at - 1) * 100
            else:
                fwd_returns[horizon] = None
        
        results.append({
            'date': row['DATE'].strftime('%Y-%m-%d'),
            'similarity': row['similarity'],
            'mood_score': row['Mood_Score'],
            'mood': row['Mood'],
            'mood_volatility': row['Mood_Volatility'],
            'nifty': nifty_at or 0,
            'fwd_30d': fwd_returns.get(30),
            'fwd_60d': fwd_returns.get(60),
            'fwd_90d': fwd_returns.get(90),
        })
    
    return results

# ══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ═══════════════════════════════════════════════════════════════════════════
    # LOAD DATA FIRST — needed to populate dynamic predictor options
    # ═══════════════════════════════════════════════════════════════════════════
    with st.spinner("Loading market data..."):
        raw_df = load_data()

    if raw_df is None:
        st.stop()

    available_predictors = [
        col for col in raw_df.columns
        if col not in NON_PREDICTOR_COLS and pd.api.types.is_numeric_dtype(raw_df[col])
    ]

    # Initialize or validate session-state predictors against actual columns
    if 'active_predictors' not in st.session_state:
        st.session_state['active_predictors'] = tuple(available_predictors)
    else:
        valid = tuple(p for p in st.session_state['active_predictors'] if p in available_predictors)
        st.session_state['active_predictors'] = valid if valid else tuple(available_predictors)

    # ═══════════════════════════════════════════════════════════════════════════
    # SIDEBAR
    # ═══════════════════════════════════════════════════════════════════════════
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; margin-bottom: 1rem;">
            <div style="font-size: 1.75rem; font-weight: 800; color: #FFC300;">ARTHAGATI</div>
            <div style="color: #888888; font-size: 0.75rem; margin-top: 0.25rem;">अर्थगति | Market Sentiment</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        view_mode = st.radio(
            "View Mode",
            ["📈 Historical Mood", "🔍 Similar Periods", "📋 Correlation Analysis"],
            label_visibility="collapsed"
        )
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-title">⚙️ Controls</div>', unsafe_allow_html=True)
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # ── Model Configuration ──
        st.markdown('<div class="sidebar-title">🧠 Model Configuration</div>', unsafe_allow_html=True)

        with st.expander("Predictor Columns", expanded=False):
            st.caption("Select predictors, then click Apply to recompute.")

            # Staging multiselect — options are all numeric columns from the sheet
            staging_predictors = st.multiselect(
                "Predictor Columns",
                options=available_predictors,
                default=list(st.session_state['active_predictors']),
                label_visibility="collapsed",
                help="These columns are used as dependent variables for PE & EY correlation-weighted mood scoring."
            )

            if not staging_predictors:
                st.warning("⚠️ Select at least one predictor.")
                staging_predictors = list(st.session_state['active_predictors'])

            # Show diff between staging and active
            staging_set = set(staging_predictors)
            active_set = set(st.session_state['active_predictors'])
            has_changes = staging_set != active_set

            if has_changes:
                added = staging_set - active_set
                removed = active_set - staging_set
                changes = []
                if added:
                    changes.append(f"+{len(added)} added")
                if removed:
                    changes.append(f"−{len(removed)} removed")
                st.caption(f"Pending: {', '.join(changes)}")

            # Apply button — only this triggers recomputation
            apply_clicked = st.button(
                "✅ Apply Configuration" if has_changes else "No changes",
                use_container_width=True,
                disabled=not has_changes,
                type="primary" if has_changes else "secondary"
            )

            if apply_clicked and has_changes:
                st.session_state['active_predictors'] = tuple(staging_predictors)
                st.cache_data.clear()
                st.rerun()

            active_count = len(st.session_state['active_predictors'])
            total_count = len(available_predictors)
            if active_count != total_count:
                st.info(f"Active: {active_count}/{total_count} predictors")
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class='info-box'>
            <p style='font-size: 0.8rem; margin: 0; color: var(--text-muted); line-height: 1.5;'>
                <strong>Version:</strong> {VERSION}<br>
                <strong>Engine:</strong> OU · Kalman · Spearman<br>
                <strong>Data:</strong> {COMPANY}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # HEADER
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown(f"""
        <div class="premium-header">
            <h1>ARTHAGATI : Market Sentiment Analysis</h1>
            <div class="tagline">Ornstein-Uhlenbeck · Kalman · Decay-Spearman · Adaptive Percentiles | Quantitative Market Physics</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Separator between header and cards
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # ── Data Staleness Check ────────────────────────────────────────────
    latest_date = raw_df['DATE'].max()
    ist_tz = pytz.timezone('Asia/Kolkata')
    today_ist = datetime.now(ist_tz).date()
    data_age_days = (pd.Timestamp(today_ist) - latest_date).days
    
    # >3 days gap (accounts for weekends: Fri data on Mon = 3 days, fine)
    if data_age_days > 3:
        st.markdown(f"""
        <div style="background: rgba(239,68,68,0.1); border: 1px solid #ef4444; border-radius: 10px; 
                    padding: 0.75rem 1.25rem; margin-bottom: 1rem; display: flex; align-items: center; gap: 12px;">
            <span style="font-size: 1.4rem;">⚠️</span>
            <div>
                <span style="color: #ef4444; font-weight: 700;">Stale Data</span>
                <span style="color: #888; font-size: 0.85rem;"> — Last data point is <b>{latest_date.strftime('%d %b %Y')}</b> ({data_age_days} days ago). 
                Scores reflect the last available data, not current market state. Update your Google Sheet.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with st.spinner("Calculating mood scores..."):
        selected_preds = st.session_state.get('active_predictors', tuple(available_predictors))
        mood_df = calculate_historical_mood(raw_df, dependent_vars=selected_preds)
    
    if mood_df.empty:
        st.error("Failed to calculate mood scores.")
        st.stop()
    
    # Calculate MSF Spread
    msf_df = calculate_msf_spread(mood_df)
    mood_df['MSF_Spread'] = msf_df['msf_spread'].values if not msf_df.empty else 0
    
    # ═══════════════════════════════════════════════════════════════════════════
    # METRIC CARDS
    # ═══════════════════════════════════════════════════════════════════════════
    latest = mood_df.iloc[-1]
    mood_score = latest['Mood_Score']
    msf_spread = latest['MSF_Spread']
    
    # Mood card styling
    if mood_score > 60:
        mood_class = "success"
    elif mood_score > 20:
        mood_class = "warning"
    elif mood_score < -60:
        mood_class = "danger"
    elif mood_score < -20:
        mood_class = "info"
    else:
        mood_class = "neutral"
    
    # MSF card styling (thresholds at ±4)
    if msf_spread > 4:
        msf_class = "danger"
        msf_label = "Overbought"
    elif msf_spread > 2:
        msf_class = "warning"
        msf_label = "Bullish"
    elif msf_spread < -4:
        msf_class = "success"
        msf_label = "Oversold"
    elif msf_spread < -2:
        msf_class = "info"
        msf_label = "Bearish"
    else:
        msf_class = "neutral"
        msf_label = "Neutral"
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card primary">
            <h4>Mood Score</h4>
            <h2>{mood_score:.2f}</h2>
            <div class="sub-metric">{latest['Mood']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card {msf_class}">
            <h4>MSF Spread</h4>
            <h2 style="color: {C_CYAN};">{msf_spread:+.2f}</h2>
            <div class="sub-metric">{msf_label}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        nifty_val = latest['NIFTY']
        st.markdown(f"""
        <div class="metric-card primary">
            <h4>NIFTY 50</h4>
            <h2>{nifty_val:,.0f}</h2>
            <div class="sub-metric">Index Level</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card neutral">
            <h4>Analysis Date</h4>
            <h2>{latest['DATE'].strftime('%d %b')}</h2>
            <div class="sub-metric">{latest['DATE'].strftime('%Y')}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # ── Diagnostics Row ─────────────────────────────────────────────────
    d1, d2, d3, d4 = st.columns(4)
    
    current_regime = latest.get('Regime', 'Unknown')
    reg_color, reg_class = REGIME_STYLES.get(current_regime, (C_MUTED, 'neutral'))
    
    with d1:
        st.markdown(f"""
        <div class="metric-card {reg_class}">
            <h4>Market Regime</h4>
            <h2 style="font-size: 1.25rem;">{current_regime}</h2>
            <div class="sub-metric">Hurst + Entropy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with d2:
        ou_hl = latest.get('OU_Half_Life', 0)
        st.markdown(f"""
        <div class="metric-card primary">
            <h4>OU Half-Life</h4>
            <h2>{ou_hl:.0f}d</h2>
            <div class="sub-metric">Expected reversion time</div>
        </div>
        """, unsafe_allow_html=True)
    
    with d3:
        h_val = latest.get('Hurst', 0.5)
        h_label = 'Trending' if h_val > 0.55 else 'Random' if h_val > 0.45 else 'Reverting'
        h_class = 'success' if h_val > 0.55 else 'neutral' if h_val > 0.45 else 'info'
        st.markdown(f"""
        <div class="metric-card {h_class}">
            <h4>Hurst Exponent</h4>
            <h2>{h_val:.2f}</h2>
            <div class="sub-metric">{h_label}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with d4:
        s_val = latest.get('Market_Entropy', 0.5)
        s_label = 'Disordered' if s_val > 0.6 else 'Ordered' if s_val < 0.4 else 'Mixed'
        s_class = 'danger' if s_val > 0.6 else 'success' if s_val < 0.4 else 'neutral'
        st.markdown(f"""
        <div class="metric-card {s_class}">
            <h4>Market Entropy</h4>
            <h2>{s_val:.2f}</h2>
            <div class="sub-metric">{s_label}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Separator between cards and chart section
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # VIEW MODES
    # ═══════════════════════════════════════════════════════════════════════════
    
    if view_mode == "📈 Historical Mood":
        render_historical_mood(mood_df, msf_df)
    elif view_mode == "🔍 Similar Periods":
        render_similar_periods(mood_df)
    else:
        render_correlation_analysis(raw_df)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FOOTER
    # ═══════════════════════════════════════════════════════════════════════════
    utc_now = datetime.now(pytz.UTC)
    ist_now = utc_now.astimezone(pytz.timezone('Asia/Kolkata'))
    current_time_ist = ist_now.strftime("%Y-%m-%d %H:%M:%S IST")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.caption(f"© 2026 {PRODUCT_NAME} | {COMPANY} | {VERSION} | {current_time_ist}")

# ══════════════════════════════════════════════════════════════════════════════
# HISTORICAL MOOD VIEW (TradingView Style)
# ══════════════════════════════════════════════════════════════════════════════

def render_historical_mood(mood_df, msf_df):
    """Render TradingView-style historical mood chart with timeframe selector."""
    
    st.markdown("""
        <div style="margin-bottom: 1rem;">
            <h3 style="color: #FFC300; margin: 0;">📈 Market Mood Terminal</h3>
            <p style="color: #888888; font-size: 0.85rem; margin: 0;">TradingView-Style Analysis • Mood Score + MSF Spread Indicator</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state for timeframe
    if 'tf_selected' not in st.session_state:
        st.session_state.tf_selected = '1Y'
    
    # Timeframe selector row (Google Finance style)
    tf_cols = st.columns(len(TIMEFRAMES))
    for i, tf in enumerate(TIMEFRAMES.keys()):
        with tf_cols[i]:
            btn_type = "primary" if st.session_state.tf_selected == tf else "secondary"
            if st.button(tf, key=f"tf_{tf}", use_container_width=True, type=btn_type):
                st.session_state.tf_selected = tf
                st.rerun()
    
    # Calculate days for selected timeframe
    selected_tf = st.session_state.tf_selected
    if selected_tf == 'YTD':
        today = datetime.now()
        days_back = (today - datetime(today.year, 1, 1)).days + 1
    else:
        days_back = TIMEFRAMES[selected_tf]
    
    # Filter data based on timeframe
    if days_back and days_back < len(mood_df):
        df = mood_df.tail(days_back).copy()
        msf_filtered = msf_df.tail(days_back).copy()
    else:
        df = mood_df.copy()
        msf_filtered = msf_df.copy()
    
    if df.empty:
        st.warning("No data available for selected timeframe.")
        return
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TRADINGVIEW-STYLE CHART (2 panes: Mood Score + MSF Spread)
    # ═══════════════════════════════════════════════════════════════════════════
    
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,  # Increased spacing for separator
        row_heights=[0.65, 0.35]
    )
    
    # ─────────────────────────────────────────────────────────────────────────
    # ROW 1: MOOD SCORE (Main Chart) - YELLOW
    # ─────────────────────────────────────────────────────────────────────────
    
    # Kalman Confidence Band (±KALMAN_CI_Z σ, ~95% interval)
    if 'Confidence_Upper' in df.columns and 'Confidence_Lower' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['DATE'], y=df['Confidence_Upper'],
                mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip',
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df['DATE'], y=df['Confidence_Lower'],
                mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip',
                fill='tonexty', fillcolor='rgba(255,195,0,0.08)',
                name='95% Confidence',
            ),
            row=1, col=1
        )

    # Mood Score line
    fig.add_trace(
        go.Scattergl(
            x=df['DATE'], y=df['Mood_Score'],
            mode='lines', name='Mood Score',
            line=dict(color=C_PRIMARY, width=2),
            hovertemplate='<b>%{x|%d %b %Y}</b><br>Mood: %{y:.2f}<extra></extra>',
        ),
        row=1, col=1
    )

    # Zero reference
    fig.add_hline(y=0, line_color='#757575', line_width=1, line_dash='dash', row=1, col=1)

    # Current-value annotation
    last_point = df.iloc[-1]
    fig.add_annotation(
        x=last_point['DATE'], y=last_point['Mood_Score'],
        text=f"<b>{last_point['Mood_Score']:.1f}</b>",
        showarrow=True, arrowhead=2, arrowcolor=C_PRIMARY,
        ax=40, ay=0,
        bgcolor=C_BG_CARD, bordercolor=C_PRIMARY, borderwidth=1,
        font=dict(color=C_PRIMARY, size=11),
        row=1, col=1
    )
    
    # ── OU Forward Projection (dotted line) ─────────────────────────────
    # E[x(t+n)] = μ + (x_current − μ) · exp(−θ·n)
    ou_theta = float(last_point.get('OU_Theta', 0.05))
    ou_mu    = float(last_point.get('OU_Mu',    0.0))
    ou_sigma = float(last_point.get('OU_Sigma', 1.0))
    ou_stationary_std_proj = ou_sigma / np.sqrt(2.0 * max(ou_theta, 1e-4))

    last_date = last_point['DATE']
    proj_dates = pd.date_range(start=last_date, periods=OU_PROJ_DAYS + 1, freq='D')[1:]
    proj_n     = np.arange(1, OU_PROJ_DAYS + 1, dtype=np.float64)

    # Convert current mood → OU-space, project, convert back.
    # mood_score = (rough_scaled − μ) / ou_stationary_std × MOOD_SCALE
    # → OU-space: x = mood / MOOD_SCALE × ou_stationary_std + μ
    x_current_ou = last_point['Mood_Score'] / MOOD_SCALE * max(ou_stationary_std_proj, 1e-6) + ou_mu
    proj_ou   = ou_mu + (x_current_ou - ou_mu) * np.exp(-ou_theta * proj_n)
    proj_mood = np.clip((proj_ou - ou_mu) / max(ou_stationary_std_proj, 1e-6) * MOOD_SCALE, -100, 100)

    fig.add_trace(
        go.Scatter(
            x=proj_dates, y=proj_mood,
            mode='lines', name='OU Projection',
            line=dict(color=C_PRIMARY, width=1.5, dash='dot'),
            opacity=0.5,
            hovertemplate='<b>%{x|%d %b %Y}</b><br>Projected: %{y:.1f}<extra></extra>',
        ),
        row=1, col=1
    )
    
    # OU equilibrium line
    ou_eq_mood = 0.0  # μ maps to 0 in normalized space
    fig.add_annotation(
        x=proj_dates[-1], y=ou_eq_mood,
        text=f"EQ ({last_point.get('OU_Half_Life', 0):.0f}d t½)",
        showarrow=False,
        font=dict(color='#888', size=9),
        xanchor='left', xshift=5,
        row=1, col=1
    )
    
    # ── Regime Transition Markers ───────────────────────────────────────
    if 'Regime' in df.columns:
        regimes = df['Regime'].values
        dates   = df['DATE'].values
        for i in range(1, len(regimes)):
            if regimes[i] != regimes[i - 1] and regimes[i] != 'Unknown':
                reg_color = REGIME_STYLES.get(regimes[i], (C_MUTED, 'neutral'))[0]
                fig.add_vline(
                    x=dates[i], line_color=reg_color,
                    line_width=1, line_dash='dot', opacity=0.5,
                    row=1, col=1
                )
                fig.add_annotation(
                    x=dates[i], y=df['Mood_Score'].values[i],
                    text=regimes[i][:4], showarrow=False,
                    font=dict(size=7, color=reg_color),
                    yshift=12, row=1, col=1
                )
    
    # ─────────────────────────────────────────────────────────────────────────
    # ROW 2: MSF SPREAD INDICATOR (Oscillator Pane) - CYAN
    # ─────────────────────────────────────────────────────────────────────────
    
    # MSF Spread line
    msf_values = msf_filtered['msf_spread'].values

    fig.add_trace(
        go.Scattergl(
            x=df['DATE'], y=msf_values,
            mode='lines', name='MSF Spread',
            line=dict(color=C_CYAN, width=2),
            hovertemplate='<b>%{x|%d %b %Y}</b><br>MSF: %{y:.2f}<extra></extra>',
        ),
        row=2, col=1
    )

    fig.add_hline(y=0, line_color='#757575', line_width=1, row=2, col=1)
    
    # ─────────────────────────────────────────────────────────────────────────
    # DIVERGENCE SIGNALS (Triangles)
    # ─────────────────────────────────────────────────────────────────────────
    
    # Detect divergences between Mood Score and MSF Spread
    # Condition A (was bullish): Mood making lower lows, MSF making higher lows -> RED (top)
    # Condition B (was bearish): Mood making higher highs, MSF making lower highs -> GREEN (bottom)
    
    lookback = 10  # Lookback period for local extrema
    mood_vals = df['Mood_Score'].values
    
    red_signals = []    # Mood lower low + MSF higher low -> bearish signal (red at top)
    green_signals = []  # Mood higher high + MSF lower high -> bullish signal (green at bottom)
    
    for i in range(lookback * 2, len(df) - 1):
        # Get local windows
        mood_window = mood_vals[i - lookback:i + 1]
        msf_window = msf_values[i - lookback:i + 1]
        
        # Check for local minimum
        if mood_vals[i] == mood_window.min() and i > lookback:
            prev_mood_window = mood_vals[i - lookback * 2:i - lookback + 1]
            prev_msf_window = msf_values[i - lookback * 2:i - lookback + 1]
            
            if len(prev_mood_window) > 0 and len(prev_msf_window) > 0:
                prev_mood_min = prev_mood_window.min()
                prev_msf_min = prev_msf_window.min()
                curr_msf_min = msf_window.min()
                
                # Mood lower low, MSF higher low -> RED signal (inverted)
                if mood_vals[i] < prev_mood_min and curr_msf_min > prev_msf_min:
                    red_signals.append(i)
        
        # Check for local maximum
        if mood_vals[i] == mood_window.max() and i > lookback:
            prev_mood_window = mood_vals[i - lookback * 2:i - lookback + 1]
            prev_msf_window = msf_values[i - lookback * 2:i - lookback + 1]
            
            if len(prev_mood_window) > 0 and len(prev_msf_window) > 0:
                prev_mood_max = prev_mood_window.max()
                prev_msf_max = prev_msf_window.max()
                curr_msf_max = msf_window.max()
                
                # Mood higher high, MSF lower high -> GREEN signal (inverted)
                if mood_vals[i] > prev_mood_max and curr_msf_max < prev_msf_max:
                    green_signals.append(i)
    
    # Red triangles at y=5 (bearish divergence, top)
    if red_signals:
        fig.add_trace(
            go.Scatter(
                x=[df['DATE'].iloc[i] for i in red_signals], y=[5] * len(red_signals),
                mode='markers', name='Bearish Signal',
                marker=dict(symbol='triangle-down', size=8, color=C_RED, line=dict(color=C_RED, width=1)),
                hoverinfo='skip', showlegend=False,
            ),
            row=2, col=1
        )

    # Green triangles at y=-5 (bullish divergence, bottom)
    if green_signals:
        fig.add_trace(
            go.Scatter(
                x=[df['DATE'].iloc[i] for i in green_signals], y=[-5] * len(green_signals),
                mode='markers', name='Bullish Signal',
                marker=dict(symbol='triangle-up', size=8, color=C_GREEN, line=dict(color=C_GREEN, width=1)),
                hoverinfo='skip', showlegend=False,
            ),
            row=2, col=1
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # LAYOUT
    # ─────────────────────────────────────────────────────────────────────────
    
    fig.update_layout(
        **PLOTLY_BASE,
        height=750,
        hovermode='x unified',
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='right', x=1, font=dict(size=11)),
        margin=dict(l=60, r=20, t=80, b=40),
        xaxis2=dict(showgrid=True, gridcolor=C_BG_GRID, type='date'),
        yaxis=dict(
            title=dict(text='Mood Score', font=dict(size=11, color=C_MUTED)),
            showgrid=True, gridcolor=C_BG_GRID, zeroline=False, autorange='reversed',
        ),
        yaxis2=dict(
            title=dict(text='MSF Spread', font=dict(size=11, color=C_MUTED)),
            showgrid=True, gridcolor=C_BG_GRID, zeroline=False,
        ),
    )
    
    # Add separator line between charts (horizontal line at the boundary)
    fig.add_shape(
        type="line",
        xref="paper",
        yref="paper",
        x0=0,
        y0=0.38,  # Position between the two charts
        x1=1,
        y1=0.38,
        line=dict(color="#555555", width=2)
    )
    
    # Remove x-axis grid on row 1 for cleaner look
    fig.update_xaxes(showgrid=False, row=1, col=1)
    fig.update_xaxes(showgrid=True, gridcolor=C_BG_GRID, row=2, col=1)
    
    st.plotly_chart(fig, config={
        'displayModeBar': True,
        'scrollZoom': True,
        'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape']
    })
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PERIOD SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    period_high = df['Mood_Score'].max()
    period_low = df['Mood_Score'].min()
    period_avg = df['Mood_Score'].mean()
    msf_avg = msf_filtered['msf_spread'].mean()
    
    with col1:
        st.markdown(f"""
        <div class="metric-card success">
            <h4>Period High</h4>
            <h2>{period_high:.1f}</h2>
            <div class="sub-metric">Most Bullish</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card danger">
            <h4>Period Low</h4>
            <h2>{period_low:.1f}</h2>
            <div class="sub-metric">Most Bearish</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_color = 'success' if period_avg > 0 else 'danger' if period_avg < 0 else 'neutral'
        st.markdown(f"""
        <div class="metric-card {avg_color}">
            <h4>Average Mood</h4>
            <h2>{period_avg:.1f}</h2>
            <div class="sub-metric">{selected_tf} Period</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        msf_color = 'success' if msf_avg < 0 else 'danger' if msf_avg > 0 else 'neutral'
        st.markdown(f"""
        <div class="metric-card {msf_color}">
            <h4>Avg MSF Spread</h4>
            <h2>{msf_avg:+.2f}</h2>
            <div class="sub-metric">{selected_tf} Period</div>
        </div>
        """, unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MSF COMPONENT DECOMPOSITION
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
        <div style="margin-bottom: 0.75rem;">
            <h4 style="color: #06b6d4; margin: 0;">MSF Component Breakdown</h4>
            <p style="color: #888; font-size: 0.8rem; margin: 0;">Current contribution of each component to the MSF Spread reading · Weights are inverse-variance (auto-calibrated)</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Get latest MSF component values
    msf_latest_idx = min(len(msf_filtered) - 1, len(df) - 1)
    if msf_latest_idx >= 0 and not msf_filtered.empty:
        comp_names = ['momentum', 'structure', 'regime', 'flow']
        comp_labels = ['Momentum', 'Structure', 'Regime', 'Flow']
        comp_colors = ['#f59e0b', '#a78bfa', '#10b981', '#06b6d4']
        comp_icons = ['🚀', '🏗️', '📊', '🌊']
        
        c_cols = st.columns(4)
        for j, (name, label, color, icon) in enumerate(zip(comp_names, comp_labels, comp_colors, comp_icons)):
            val = msf_filtered[name].iloc[msf_latest_idx] if name in msf_filtered.columns else 0
            # Compute period average for context
            period_val = msf_filtered[name].mean() if name in msf_filtered.columns else 0
            
            bar_pct = (val + 10) / 20 * 100  # Map [-10, +10] → [0%, 100%]
            bar_pct = max(0, min(100, bar_pct))
            
            with c_cols[j]:
                st.markdown(f"""
                <div style="background: #1A1A1A; border-radius: 10px; padding: 0.75rem; border: 1px solid #2A2A2A;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                        <span style="font-size: 0.75rem; color: #888; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">{icon} {label}</span>
                        <span style="font-size: 1.1rem; font-weight: 700; color: {color};">{val:+.1f}</span>
                    </div>
                    <div style="height: 6px; background: #2A2A2A; border-radius: 3px; position: relative;">
                        <div style="position: absolute; left: 50%; top: 0; width: 1px; height: 6px; background: #555;"></div>
                        <div style="width: {bar_pct:.0f}%; height: 100%; background: {color}; border-radius: 3px; opacity: 0.8;"></div>
                    </div>
                    <div style="font-size: 0.65rem; color: #555; margin-top: 0.3rem;">Period avg: {period_val:+.1f}</div>
                </div>
                """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIMILAR PERIODS VIEW
# ══════════════════════════════════════════════════════════════════════════════

def render_similar_periods(mood_df):
    """Render similar historical periods with forward returns and backtest validation."""
    
    st.markdown("""
        <div style="margin-bottom: 1rem;">
            <h3 style="color: #FFC300; margin: 0;">🔍 Similar Historical Periods</h3>
            <p style="color: #888888; font-size: 0.85rem; margin: 0;">Mahalanobis + trajectory matching · Forward NIFTY returns from each analog</p>
        </div>
    """, unsafe_allow_html=True)
    
    similar_periods = find_similar_periods(mood_df)
    
    if not similar_periods:
        st.warning("Not enough historical data to find similar periods.")
        return
    
    # ── Forward Return Summary ──────────────────────────────────────────
    # Aggregate: "In N similar periods, what was the median NIFTY return?"
    fwd_30 = [p['fwd_30d'] for p in similar_periods if p['fwd_30d'] is not None]
    fwd_60 = [p['fwd_60d'] for p in similar_periods if p['fwd_60d'] is not None]
    fwd_90 = [p['fwd_90d'] for p in similar_periods if p['fwd_90d'] is not None]
    
    if fwd_30 or fwd_60 or fwd_90:
        sc1, sc2, sc3 = st.columns(3)
        
        def _fwd_card(col, horizon, values):
            if not values:
                return
            median_ret = np.median(values)
            positive_pct = sum(1 for v in values if v > 0) / len(values) * 100
            ret_color = '#10b981' if median_ret > 0 else '#ef4444'
            card_class = 'success' if median_ret > 0 else 'danger'
            with col:
                st.markdown(f"""
                <div class="metric-card {card_class}">
                    <h4>+{horizon}D Median Return</h4>
                    <h2>{median_ret:+.1f}%</h2>
                    <div class="sub-metric">{positive_pct:.0f}% positive ({len(values)} analogs)</div>
                </div>
                """, unsafe_allow_html=True)
        
        _fwd_card(sc1, 30, fwd_30)
        _fwd_card(sc2, 60, fwd_60)
        _fwd_card(sc3, 90, fwd_90)
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # ── Period Cards with Forward Returns ───────────────────────────────
    cols = st.columns(2)
    for i, period in enumerate(similar_periods[:10]):
        col = cols[i % 2]
        with col:
            similarity_pct = period['similarity'] * 100
            mood_val = period['mood_score']
            mood_class = 'bullish' if mood_val > 20 else 'bearish' if mood_val < -20 else 'neutral'
            
            # Forward return badges
            fwd_badges = ''
            for horizon, key in [(30, 'fwd_30d'), (60, 'fwd_60d'), (90, 'fwd_90d')]:
                val = period.get(key)
                if val is not None:
                    fwd_color = '#10b981' if val > 0 else '#ef4444'
                    fwd_badges += f'<span style="font-size:0.7rem; color:{fwd_color}; margin-left:8px;">+{horizon}d: <b>{val:+.1f}%</b></span>'
                else:
                    fwd_badges += f'<span style="font-size:0.7rem; color:#555; margin-left:8px;">+{horizon}d: —</span>'
            
            st.markdown(f"""
            <div class="signal-card {mood_class}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <span style="font-weight: 700; color: #EAEAEA;">{period['date']}</span>
                    <span class="status-badge {mood_class}">{period['mood']}</span>
                </div>
                <div style="display: flex; justify-content: space-between; color: #888888; font-size: 0.85rem;">
                    <span>Similarity: <b style="color: #FFC300;">{similarity_pct:.1f}%</b></span>
                    <span>Mood: <b>{mood_val:.1f}</b></span>
                    <span>NIFTY: <b>{period['nifty']:,.0f}</b></span>
                </div>
                <div style="margin-top: 0.4rem; padding-top: 0.4rem; border-top: 1px solid #2A2A2A;">
                    <span style="font-size: 0.7rem; color: #666;">NIFTY After:</span>{fwd_badges}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # BACKTEST SANITY CHECK
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
        <div style="margin-bottom: 1rem;">
            <h3 style="color: #FFC300; margin: 0;">📊 Backtest: Mood Score vs Forward NIFTY Return</h3>
            <p style="color: #888888; font-size: 0.85rem; margin: 0;">
                Does today's mood score predict tomorrow's market? Each dot = one historical day.
                If there's a relationship, the scatter should show a pattern.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Compute mood_score at T vs NIFTY return at T+30 for all historical points
    n = len(mood_df)
    horizon = 30
    if n > horizon + 10:
        bt_mood = mood_df['Mood_Score'].values[:n - horizon]
        bt_nifty = mood_df['NIFTY'].values
        bt_fwd_return = (bt_nifty[horizon:] / bt_nifty[:n - horizon] - 1) * 100
        bt_dates = mood_df['DATE'].values[:n - horizon]
        
        # Remove NaN/Inf
        valid = np.isfinite(bt_mood) & np.isfinite(bt_fwd_return)
        bt_mood_clean = bt_mood[valid]
        bt_fwd_clean = bt_fwd_return[valid]
        
        if len(bt_mood_clean) > 20:
            # Compute correlation
            bt_corr = np.corrcoef(bt_mood_clean, bt_fwd_clean)[0, 1] if len(bt_mood_clean) > 2 else 0
            
            colors = np.where(bt_mood_clean > 0, C_GREEN, C_RED)
            
            fig_bt = go.Figure()
            
            fig_bt.add_trace(go.Scattergl(
                x=bt_mood_clean, y=bt_fwd_clean,
                mode='markers',
                marker=dict(size=4, color=colors, opacity=0.5),
                hovertemplate='Mood: %{x:.1f}<br>+30d Return: %{y:.1f}%<extra></extra>',
                showlegend=False,
            ))
            
            # Add regression line
            if len(bt_mood_clean) > 10:
                z = np.polyfit(bt_mood_clean, bt_fwd_clean, 1)
                x_range = np.linspace(bt_mood_clean.min(), bt_mood_clean.max(), 50)
                fig_bt.add_trace(go.Scatter(
                    x=x_range, y=z[0] * x_range + z[1],
                    mode='lines', line=dict(color=C_PRIMARY, width=2, dash='dash'),
                    name=f'ρ = {bt_corr:.2f}', showlegend=True,
                ))
            
            # Zero lines
            fig_bt.add_hline(y=0, line_color='#555', line_width=1, line_dash='dot')
            fig_bt.add_vline(x=0, line_color='#555', line_width=1, line_dash='dot')
            
            fig_bt.update_layout(
                **PLOTLY_BASE,
                height=400,
                xaxis=dict(title='Mood Score at T', showgrid=True, gridcolor=C_BG_GRID),
                yaxis=dict(title='NIFTY Return T+30d (%)', showgrid=True, gridcolor=C_BG_GRID),
                margin=dict(l=60, r=20, t=30, b=50),
                legend=dict(
                    x=0.02, y=0.98,
                    bgcolor='rgba(26,26,26,0.8)',
                    bordercolor=C_BG_GRID, borderwidth=1,
                    font=dict(size=11),
                ),
            )
            
            st.plotly_chart(fig_bt, config={'displayModeBar': False})
            
            # Interpretation
            if abs(bt_corr) > 0.3:
                strength = 'strong' if abs(bt_corr) > 0.5 else 'moderate'
                direction = 'positive' if bt_corr > 0 else 'negative'
                st.markdown(f"""
                <div class="info-box">
                    <b>Correlation: ρ = {bt_corr:.2f}</b> — {strength} {direction} relationship.
                    {'Higher mood scores have historically been followed by positive NIFTY returns.' if bt_corr > 0 else 'Higher mood scores have historically been followed by negative NIFTY returns (contrarian signal).'}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="info-box">
                    <b>Correlation: ρ = {bt_corr:.2f}</b> — weak linear relationship at 30-day horizon.
                    The mood score's predictive power may be non-linear or work better at different horizons.
                </div>
                """, unsafe_allow_html=True)
        else:
            st.caption("Insufficient data points for backtest.")

# ══════════════════════════════════════════════════════════════════════════════
# CORRELATION ANALYSIS VIEW
# ══════════════════════════════════════════════════════════════════════════════

def render_correlation_analysis(raw_df):
    """Render correlation analysis with data diagnostics and predictor recommendations."""
    
    all_avail = tuple(
        c for c in raw_df.columns
        if c not in NON_PREDICTOR_COLS and pd.api.types.is_numeric_dtype(raw_df[c])
    )
    active_preds = st.session_state.get('active_predictors', all_avail)
    
    st.markdown("""
        <div style="margin-bottom: 1rem;">
            <h3 style="color: #FFC300; margin: 0;">📋 Correlation & Predictor Analysis</h3>
            <p style="color: #888888; font-size: 0.85rem; margin: 0;">Decay-weighted Spearman correlations with PE and EY anchors · Predictor quality assessment</p>
        </div>
    """, unsafe_allow_html=True)
    
    # ── Data Quality Check ──────────────────────────────────────────────
    # Detect anchors with insufficient variance (root cause of "empty" correlations)
    anchors = {'NIFTY50_PE': 'PE Ratio', 'NIFTY50_EY': 'Earnings Yield'}
    anchor_health = {}
    for col, label in anchors.items():
        if col in raw_df.columns:
            nunique = raw_df[col].nunique()
            has_variance = nunique > 3 and raw_df[col].std() > 1e-6
            anchor_health[col] = {'label': label, 'ok': has_variance, 'nunique': nunique}
        else:
            anchor_health[col] = {'label': label, 'ok': False, 'nunique': 0}
    
    # Show diagnostic if any anchor is unhealthy
    bad_anchors = [v['label'] for v in anchor_health.values() if not v['ok']]
    if bad_anchors:
        st.markdown(f"""
        <div class="info-box" style="border-left: 4px solid #ef4444;">
            <h4 style="color: #ef4444;">⚠️ Data Quality Issue</h4>
            <p><b>{', '.join(bad_anchors)}</b> has insufficient variance in the source data.
            If Earnings Yield is empty in the sheet, it is auto-derived from PE (1/PE × 100).
            Check that your Google Sheet has valid data for these columns.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # ── Correlation Bars ────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    
    def _render_corr_bars(parent_col, anchor_col, title):
        with parent_col:
            st.markdown(f"#### {title}")
            if not anchor_health.get(anchor_col, {}).get('ok', False):
                st.caption(f"⚠️ {anchor_col} has insufficient data variance — correlations may be unreliable.")
            
            corrs = calculate_anchor_correlations(raw_df, anchor_col, active_preds)
            if corrs.empty:
                st.caption("No correlations computed. Check data source.")
                return corrs
            
            corrs_display = corrs.sort_values('correlation', key=abs, ascending=False)
            for _, row in corrs_display.iterrows():
                corr_val = row['correlation']
                color = '#10b981' if corr_val > 0 else '#ef4444'
                bar_width = abs(corr_val) * 100
                strength_dot = '🟢' if abs(corr_val) >= 0.5 else '🟡' if abs(corr_val) >= 0.3 else '⚪'
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem; padding: 0.5rem; background: #1A1A1A; border-radius: 8px;">
                    <span style="width: 14px; font-size: 0.6rem;">{strength_dot}</span>
                    <span style="width: 130px; font-size: 0.8rem; color: #EAEAEA;">{row['variable']}</span>
                    <div style="flex: 1; height: 8px; background: #2A2A2A; border-radius: 4px; margin: 0 10px;">
                        <div style="width: {bar_width}%; height: 100%; background: {color}; border-radius: 4px;"></div>
                    </div>
                    <span style="width: 60px; text-align: right; font-size: 0.8rem; color: {color}; font-weight: 600;">{corr_val:+.2f}</span>
                </div>
                """, unsafe_allow_html=True)
            return corrs
    
    pe_corrs = _render_corr_bars(col1, 'NIFTY50_PE', 'PE Ratio Correlations')
    ey_corrs = _render_corr_bars(col2, 'NIFTY50_EY', 'Earnings Yield Correlations')
    
    # ── Predictor Recommendations ───────────────────────────────────────
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
        <div style="margin-bottom: 1rem;">
            <h3 style="color: #FFC300; margin: 0;">🎯 Predictor Quality Assessment</h3>
            <p style="color: #888888; font-size: 0.85rem; margin: 0;">
                Each predictor scored by: correlation strength × information quality (1 − entropy).
                High-entropy (noisy) variables are penalized. This is how the mood engine weights them internally.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Build quality scores for all predictors
    all_vars = [col for col in raw_df.columns if col not in NON_PREDICTOR_COLS and pd.api.types.is_numeric_dtype(raw_df[col])]
    
    quality_rows = []
    for var in all_vars:
        # PE correlation
        pe_corr = 0.0
        if not pe_corrs.empty:
            pe_match = pe_corrs.loc[pe_corrs['variable'] == var]
            if len(pe_match) > 0:
                pe_corr = abs(pe_match.iloc[0]['correlation'])
        
        # EY correlation
        ey_corr = 0.0
        if not ey_corrs.empty:
            ey_match = ey_corrs.loc[ey_corrs['variable'] == var]
            if len(ey_match) > 0:
                ey_corr = abs(ey_match.iloc[0]['correlation'])
        
        avg_corr = (pe_corr + ey_corr) / 2
        
        # Entropy of variable's returns
        var_returns = raw_df[var].pct_change().dropna().values
        entropy = shannon_entropy(var_returns) if len(var_returns) > 10 else 0.5
        
        # Quality score: same formula the engine uses for weighting
        info_quality = 1.0 - entropy
        quality_score = avg_corr * max(info_quality, 0.1)
        
        # Data coverage
        non_zero_pct = (raw_df[var] != 0).mean() * 100
        
        is_active = var in active_preds
        
        quality_rows.append({
            'variable': var,
            'pe_corr': pe_corr,
            'ey_corr': ey_corr,
            'avg_corr': avg_corr,
            'entropy': entropy,
            'quality': quality_score,
            'coverage': non_zero_pct,
            'active': is_active
        })
    
    # Sort by quality score descending
    quality_rows.sort(key=lambda x: x['quality'], reverse=True)
    
    if quality_rows:
        max_quality = max(r['quality'] for r in quality_rows) or 1.0
        
        # Render as ranked cards
        for rank, row in enumerate(quality_rows, 1):
            bar_pct = (row['quality'] / max_quality) * 100
            
            # Recommendation logic
            if row['quality'] >= max_quality * 0.5 and row['coverage'] > 50:
                rec = '✅ KEEP'
                rec_color = '#10b981'
            elif row['quality'] >= max_quality * 0.2 and row['coverage'] > 30:
                rec = '🟡 USEFUL'
                rec_color = '#f59e0b'
            elif row['coverage'] < 10:
                rec = '❌ NO DATA'
                rec_color = '#ef4444'
            else:
                rec = '⚪ WEAK'
                rec_color = '#888888'
            
            active_badge = '● Active' if row['active'] else '○ Inactive'
            active_color = '#FFC300' if row['active'] else '#555555'
            
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin-bottom: 0.4rem; padding: 0.6rem 0.75rem; background: #1A1A1A; border-radius: 8px; border: 1px solid {'#2A2A2A' if row['active'] else '#1A1A1A'};">
                <span style="width: 24px; font-size: 0.75rem; color: #555; font-weight: 700;">{rank}</span>
                <span style="width: 140px; font-size: 0.8rem; color: #EAEAEA; font-weight: 600;">{row['variable']}</span>
                <div style="flex: 1; height: 6px; background: #2A2A2A; border-radius: 3px; margin: 0 12px;">
                    <div style="width: {bar_pct:.0f}%; height: 100%; background: linear-gradient(90deg, #FFC300, #f59e0b); border-radius: 3px;"></div>
                </div>
                <span style="width: 50px; font-size: 0.7rem; color: #888; text-align: center;">|ρ| {row['avg_corr']:.2f}</span>
                <span style="width: 50px; font-size: 0.7rem; color: #888; text-align: center;">H {row['entropy']:.2f}</span>
                <span style="width: 55px; font-size: 0.7rem; color: {rec_color}; font-weight: 700; text-align: center;">{rec}</span>
                <span style="width: 65px; font-size: 0.65rem; color: {active_color}; text-align: right;">{active_badge}</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Summary recommendation
        keep_count = sum(1 for r in quality_rows if r['quality'] >= max_quality * 0.5 and r['coverage'] > 50)
        useful_count = sum(1 for r in quality_rows if max_quality * 0.2 <= r['quality'] < max_quality * 0.5 and r['coverage'] > 30)
        weak_count = len(quality_rows) - keep_count - useful_count
        
        st.markdown(f"""
        <div class="info-box" style="margin-top: 1rem;">
            <h4>Recommendation Summary</h4>
            <p>
                <b style="color: #10b981;">✅ {keep_count} strong</b> predictors (high correlation × low entropy) ·
                <b style="color: #f59e0b;">🟡 {useful_count} useful</b> (moderate signal) ·
                <b style="color: #888;">⚪ {weak_count} weak</b> (low signal or noisy)<br>
                <span style="font-size: 0.8rem; color: #666;">
                    |ρ| = average |correlation| with PE & EY anchors · H = Shannon entropy of returns (lower = more structured) ·
                    Quality = |ρ| × (1−H) — same formula the mood engine uses internally for weighting.
                </span>
            </p>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# RUN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
