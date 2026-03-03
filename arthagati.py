# -*- coding: utf-8 -*-
"""
ARTHAGATI (अर्थगति) - Market Sentiment Analysis | A Hemrek Capital Product
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Quantitative market mood analysis with MSF-enhanced indicators.
TradingView-style charting with institutional-grade analytics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pandas_ta as ta
from io import BytesIO
import logging
import time
import pytz

# Configure logging
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

VERSION = "v2.0.0"
PRODUCT_NAME = "Arthagati"
COMPANY = "Hemrek Capital"

# ══════════════════════════════════════════════════════════════════════════════
# GOOGLE SHEETS CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

SHEET_ID = "1po7z42n3dYIQGAvn0D1-a4pmyxpnGPQ13TrNi3DB5_c"
SHEET_GID = "0"

EXPECTED_COLUMNS = [
    'DATE', 'NIFTY', 'AD_RATIO', 'REL_AD_RATIO', 'REL_BREADTH', 'BREADTH', 'COUNT', 
    'NIFTY50_PE', 'NIFTY50_EY', 'NIFTY50_DY', 'NIFTY50_PB', 'IN10Y', 'IN02Y', 'IN30Y', 
    'INIRYY', 'REPO', 'CRR', 'US02Y', 'US10Y', 'US30Y', 'US_FED', 'PE_DEV', 'EY_DEV'
]

DEPENDENT_VARS = [
    'AD_RATIO', 'REL_AD_RATIO', 'REL_BREADTH', 'BREADTH', 'COUNT', 'IN10Y', 'IN02Y', 
    'IN30Y', 'INIRYY', 'REPO', 'CRR', 'US02Y', 'US30Y', 'US10Y', 'US_FED', 'NIFTY50_DY',
    'NIFTY50_PB', 'PE_DEV', 'EY_DEV'
]

# Timeframe Configuration
TIMEFRAMES = {
    '1W': 7,
    '1M': 30,
    '3M': 90,
    '6M': 180,
    'YTD': None,
    '1Y': 365,
    '2Y': 730,
    '5Y': 1825,
    'MAX': None  # Show all data
}

# ══════════════════════════════════════════════════════════════════════════════
# HEMREK CAPITAL DESIGN SYSTEM (Nirnay-Grade)
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
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
""", unsafe_allow_html=True)

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
# MATHEMATICAL PRIMITIVES (v2.0 Engine)
# ══════════════════════════════════════════════════════════════════════════════
#
# AUDIT LOG — problems in v1.x and what v2.0 fixes:
#
# 1. STATIC PEARSON CORRELATION — Pearson assumes linearity & stationarity.
#    Financial correlations are non-stationary and often nonlinear-monotonic.
#    → Fix: Exponential-decay-weighted Spearman rank correlation.
#
# 2. EXPANDING PERCENTILES — expanding().rank() weights 2005 data equal to 2024.
#    Market structure evolves; old regimes pollute current percentile estimates.
#    → Fix: Half-life-decay-weighted empirical CDF (adaptive percentiles).
#
# 3. GLOBAL Z-SCORE NORMALIZATION — (x - mean) / std over the FULL sample
#    means adding 1 new data point shifts ALL historical mood scores.
#    Path-dependent and non-local.
#    → Fix: Ornstein-Uhlenbeck process parameters for physics-based normalization.
#
# 4. FIXED SMOOTHING WINDOW — rolling(7).mean() doesn't adapt to volatility.
#    In high-vol regimes you need more smoothing; in low-vol, less.
#    → Fix: 1D Kalman filter with adaptive process noise.
#
# 5. ARBITRARY MSF WEIGHTS — 30/25/25/20 with no empirical basis.
#    → Fix: Inverse-variance weighting (minimum-variance portfolio of signals).
#
# 6. FIXED REGIME THRESHOLD — 0.0033 is arbitrary.
#    → Fix: Adaptive threshold = rolling_std * multiplier.
#
# 7. CRUDE SIMILARITY SEARCH — only 2 features, Manhattan distance,
#    no trajectory matching, no covariance awareness.
#    → Fix: Mahalanobis distance on enriched feature vector with trajectory.
#
# 8. NO REGIME PHYSICS — no entropy, no Hurst, no mean-reversion model.
#    → Fix: Shannon entropy, Hurst exponent, OU process estimation,
#           Fisher information — all as new first-class signals.
# ══════════════════════════════════════════════════════════════════════════════

def exponential_decay_weights(n, half_life):
    """
    Generate exponential decay weights for n observations.
    w_i = exp(-λ * i) where λ = ln(2) / half_life.
    Most recent observation has weight 1.0, oldest decays toward 0.
    """
    if n <= 0:
        return np.array([])
    lam = np.log(2) / max(half_life, 1)
    indices = np.arange(n - 1, -1, -1, dtype=np.float64)  # [n-1, n-2, ..., 0]
    weights = np.exp(-lam * indices)
    return weights / weights.sum()

def spearman_rank_correlation(x, y):
    """
    Spearman rank correlation — robust to outliers, captures monotonic
    nonlinear relationships (e.g., yield curve inversions, PE compression).
    Pure numpy implementation (no scipy dependency).
    """
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return 0.0
    x, y = x[valid], y[valid]
    
    def _rank(arr):
        order = arr.argsort()
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(arr) + 1, dtype=float)
        # Handle ties: average rank
        for val in np.unique(arr):
            mask = arr == val
            if mask.sum() > 1:
                ranks[mask] = ranks[mask].mean()
        return ranks
    
    rx, ry = _rank(x), _rank(y)
    n = len(rx)
    d = rx - ry
    rho = 1.0 - (6.0 * np.sum(d ** 2)) / (n * (n ** 2 - 1))
    return np.clip(rho, -1.0, 1.0)

def weighted_spearman(x, y, weights):
    """
    Weighted Spearman rank correlation with exponential decay.
    Approximation: compute weighted Pearson on ranks.
    This preserves the rank-robustness while adding recency weighting.
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
    Shannon entropy H = -Σ p_i * log(p_i), normalized to [0, 1].
    H=1 → maximum disorder (uniform), H=0 → perfect order (delta function).
    Measures how "uncertain" or "disordered" the distribution of values is.
    Applied to returns: high entropy = confused market, low entropy = trending.
    """
    clean = values[np.isfinite(values)]
    if len(clean) < 5:
        return 0.5  # Agnostic prior
    
    # Adaptive binning: use data range
    counts, _ = np.histogram(clean, bins=n_bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # Exclude zero-probability bins
    
    if len(probs) <= 1:
        return 0.0
    
    h = -np.sum(probs * np.log2(probs))
    h_max = np.log2(n_bins)
    return h / h_max if h_max > 0 else 0.0

def rolling_entropy(series, window=60, n_bins=15):
    """
    Rolling Shannon entropy of a series. Returns normalized [0, 1] entropy
    at each point using a lookback window.
    """
    values = series.values if hasattr(series, 'values') else np.asarray(series)
    n = len(values)
    result = np.full(n, 0.5)  # Default: maximum uncertainty
    
    for i in range(window, n):
        window_vals = values[i - window:i]
        result[i] = shannon_entropy(window_vals, n_bins)
    
    # Fill the initial period with expanding entropy
    for i in range(5, min(window, n)):
        result[i] = shannon_entropy(values[:i], n_bins)
    
    return result

def hurst_exponent(series, max_lag=None):
    """
    Hurst exponent via Rescaled Range (R/S) analysis.
    H > 0.5 → persistent (trending), momentum works
    H < 0.5 → anti-persistent (mean-reverting), contrarian works
    H ≈ 0.5 → random walk, no edge from direction
    
    Uses the classic Mandelbrot R/S method with regression on log-log plot.
    """
    ts = np.asarray(series, dtype=np.float64)
    ts = ts[np.isfinite(ts)]
    n = len(ts)
    
    if n < 20:
        return 0.5  # Insufficient data → agnostic
    
    if max_lag is None:
        max_lag = min(n // 2, 200)
    
    lags = []
    rs_values = []
    
    for lag in range(10, max_lag + 1, max(1, (max_lag - 10) // 30)):
        n_blocks = n // lag
        if n_blocks < 1:
            continue
        
        rs_block = []
        for b in range(n_blocks):
            block = ts[b * lag:(b + 1) * lag]
            mean_block = block.mean()
            deviations = block - mean_block
            cumulative = np.cumsum(deviations)
            R = cumulative.max() - cumulative.min()
            S = block.std(ddof=1)
            if S > 1e-12:
                rs_block.append(R / S)
        
        if len(rs_block) > 0:
            lags.append(lag)
            rs_values.append(np.mean(rs_block))
    
    if len(lags) < 3:
        return 0.5
    
    # Linear regression on log-log: log(R/S) = H * log(lag) + c
    log_lags = np.log(np.array(lags, dtype=np.float64))
    log_rs = np.log(np.array(rs_values, dtype=np.float64))
    
    valid = np.isfinite(log_lags) & np.isfinite(log_rs)
    if valid.sum() < 3:
        return 0.5
    
    log_lags, log_rs = log_lags[valid], log_rs[valid]
    
    # OLS: H = cov(x,y) / var(x)
    mean_x, mean_y = log_lags.mean(), log_rs.mean()
    cov_xy = np.sum((log_lags - mean_x) * (log_rs - mean_y))
    var_x = np.sum((log_lags - mean_x) ** 2)
    
    H = cov_xy / var_x if var_x > 1e-12 else 0.5
    return np.clip(H, 0.01, 0.99)

def rolling_hurst(series, window=120, step=1):
    """Rolling Hurst exponent with given window."""
    values = np.asarray(series, dtype=np.float64)
    n = len(values)
    result = np.full(n, 0.5)
    
    for i in range(window, n, step):
        result[i] = hurst_exponent(values[i - window:i])
    
    # Forward-fill stepped values
    if step > 1:
        for i in range(window, n):
            if result[i] == 0.5 and i > window:
                result[i] = result[i - 1]
    
    return result

def ornstein_uhlenbeck_estimate(series, dt=1.0):
    """
    Estimate Ornstein-Uhlenbeck process parameters from discrete data.
    
    The OU process: dx = θ(μ - x)dt + σdW
    Discrete: x_{t+1} = a + b*x_t + ε  where:
        b = exp(-θ*dt)
        a = μ*(1 - b)
        var(ε) = σ² * (1 - b²) / (2θ)
    
    Returns: (theta, mu, sigma)
        theta: mean-reversion speed (higher = faster reversion)
        mu: long-run equilibrium level
        sigma: volatility of the process
    """
    ts = np.asarray(series, dtype=np.float64)
    valid = np.isfinite(ts)
    ts = ts[valid]
    
    if len(ts) < 10:
        return (0.01, 0.0, 1.0)  # Default: slow reversion, zero mean, unit vol
    
    x = ts[:-1]
    y = ts[1:]
    
    n = len(x)
    mean_x = x.mean()
    mean_y = y.mean()
    
    # OLS: y = a + b*x
    cov_xy = np.sum((x - mean_x) * (y - mean_y))
    var_x = np.sum((x - mean_x) ** 2)
    
    if var_x < 1e-12:
        return (0.01, mean_x, 1.0)
    
    b = cov_xy / var_x
    a = mean_y - b * mean_x
    
    # Extract OU parameters
    b = np.clip(b, 1e-6, 1.0 - 1e-6)  # Ensure stationarity
    theta = -np.log(b) / dt
    theta = np.clip(theta, 1e-4, 10.0)  # Physical bounds
    
    mu = a / (1.0 - b)
    
    # Residual variance → sigma
    residuals = y - (a + b * x)
    var_eps = np.var(residuals)
    sigma_sq = 2.0 * theta * var_eps / (1.0 - b ** 2) if (1.0 - b ** 2) > 1e-12 else var_eps
    sigma = np.sqrt(max(sigma_sq, 1e-12))
    
    return (theta, mu, sigma)

def kalman_filter_1d(observations, process_var=None, measurement_var=None):
    """
    1D Kalman filter for adaptive smoothing.
    
    Automatically estimates noise parameters if not provided.
    Adapts the smoothing bandwidth based on the signal-to-noise ratio:
    high noise → more smoothing, low noise → tracks signal closely.
    
    Returns: (filtered_state, kalman_gain_series)
    """
    obs = np.asarray(observations, dtype=np.float64)
    n = len(obs)
    
    if n == 0:
        return np.array([]), np.array([])
    
    # Auto-estimate noise parameters from data if not provided
    if process_var is None:
        diffs = np.diff(obs[np.isfinite(obs)])
        process_var = np.var(diffs) * 0.1 if len(diffs) > 1 else 1e-3
        process_var = max(process_var, 1e-8)
    
    if measurement_var is None:
        clean = obs[np.isfinite(obs)]
        measurement_var = np.var(clean) * 0.5 if len(clean) > 1 else 1.0
        measurement_var = max(measurement_var, 1e-8)
    
    # Initialize
    state = obs[0] if np.isfinite(obs[0]) else 0.0
    estimate_var = measurement_var
    
    filtered = np.zeros(n)
    gains = np.zeros(n)
    filtered[0] = state
    
    for i in range(1, n):
        # Predict
        pred_state = state
        pred_var = estimate_var + process_var
        
        if np.isfinite(obs[i]):
            # Update
            K = pred_var / (pred_var + measurement_var)
            state = pred_state + K * (obs[i] - pred_state)
            estimate_var = (1 - K) * pred_var
            gains[i] = K
        else:
            state = pred_state
            estimate_var = pred_var
            gains[i] = 0.0
        
        filtered[i] = state
    
    return filtered, gains

def fisher_information_rolling(series, window=30):
    """
    Rolling Fisher information proxy: I_F = 1 / variance.
    High Fisher info = tightly concentrated data = confident signal.
    Low Fisher info = dispersed data = uncertain signal.
    
    Normalized to [0, 1] via sigmoid transform.
    """
    s = pd.Series(series) if not isinstance(series, pd.Series) else series
    rolling_var = s.rolling(window=window, min_periods=5).var()
    
    # Fisher info = 1 / var, then normalize
    fi = 1.0 / rolling_var.replace(0, np.nan)
    fi = fi.fillna(0)
    
    # Normalize to [0, 1] using the percentile within its own history
    fi_expanding_rank = fi.expanding().rank(pct=True).fillna(0.5)
    return fi_expanding_rank.values

def adaptive_percentile(series, half_life=252):
    """
    Exponential-decay-weighted empirical CDF.
    
    For each time t, the percentile of x_t is:
        P(t) = Σ_{i≤t} w_i * I(x_i ≤ x_t) / Σ_{i≤t} w_i
    where w_i = exp(-λ*(t-i)), λ = ln(2)/half_life.
    
    This makes recent data count more in determining "where are we historically."
    A PE of 22 is judged against recent-ish history, not all-time.
    """
    values = np.asarray(series, dtype=np.float64)
    n = len(values)
    result = np.full(n, 0.5)
    lam = np.log(2) / max(half_life, 1)
    
    for t in range(1, n):
        if not np.isfinite(values[t]):
            result[t] = result[t - 1] if t > 0 else 0.5
            continue
        
        # Weights: exponential decay from current point backward
        ages = np.arange(t, -1, -1, dtype=np.float64)  # [t, t-1, ..., 0]
        weights = np.exp(-lam * ages)
        
        # Weighted empirical CDF at x_t
        indicators = (values[:t + 1] <= values[t]).astype(np.float64)
        valid = np.isfinite(values[:t + 1])
        
        w_valid = weights[valid]
        ind_valid = indicators[valid]
        
        w_sum = w_valid.sum()
        if w_sum > 1e-12:
            result[t] = np.sum(w_valid * ind_valid) / w_sum
        else:
            result[t] = 0.5
    
    return result

def mahalanobis_distance_batch(features, center, cov_matrix):
    """
    Mahalanobis distance from each row of features to center.
    d_M = sqrt((x - μ)ᵀ Σ⁻¹ (x - μ))
    
    Accounts for correlations between features — two periods that are
    "close" in correlated dimensions are less similar than they appear
    in Euclidean space.
    """
    diff = features - center
    
    # Regularize covariance for numerical stability
    reg = 1e-6 * np.eye(cov_matrix.shape[0])
    cov_reg = cov_matrix + reg
    
    try:
        cov_inv = np.linalg.inv(cov_reg)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse
        cov_inv = np.linalg.pinv(cov_reg)
    
    # d² = (x - μ)ᵀ Σ⁻¹ (x - μ) for each row
    left = diff @ cov_inv
    d_sq = np.sum(left * diff, axis=1)
    d_sq = np.maximum(d_sq, 0)  # Numerical safety
    return np.sqrt(d_sq)

def cosine_similarity(a, b):
    """Cosine similarity between two vectors — measures trajectory shape match."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    """Load market data from Google Sheets."""
    start_time = time.time()
    try:
        url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&gid={SHEET_GID}"
        df = pd.read_csv(url, usecols=lambda x: x in EXPECTED_COLUMNS, dtype=str)
        
        if not any(col in df.columns for col in EXPECTED_COLUMNS):
            raise ValueError("None of the expected columns found in the Sheet.")
        
        missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]
        if missing_columns:
            logging.warning(f"Missing columns: {missing_columns}. Setting to 0.0.")
            for col in missing_columns:
                df[col] = "0.0"
        
        df['DATE'] = pd.to_datetime(df['DATE'], format='%m/%d/%Y', errors='coerce')
        
        numeric_cols = [col for col in EXPECTED_COLUMNS if col != 'DATE']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        
        df = df[df['NIFTY'] > 0].dropna(subset=['DATE']).copy()
        if df.empty:
            raise ValueError("No valid rows with positive NIFTY or valid DATE.")
        
        df = df[EXPECTED_COLUMNS].sort_values('DATE').reset_index(drop=True)
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
    v2.0: Exponential-decay-weighted Spearman rank correlations.
    
    Why Spearman over Pearson:
      - Robust to outliers (rank-based)
      - Captures monotonic nonlinear relationships
      - Invariant to marginal distribution shape
    
    Why exponential decay:
      - Financial correlation structure is non-stationary
      - Recent regime correlations matter more than decade-old ones
      - Half-life of ~504 days (~2 trading years) balances stability vs adaptiveness
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
    decay_half_life = min(504, n // 2) if n > 20 else max(n // 2, 5)
    weights = exponential_decay_weights(n, decay_half_life)
    
    correlations = []
    for var in cols_to_check:
        if var == anchor:
            continue
        
        var_vals = analysis_df[var].values if var in analysis_df.columns else None
        if var_vals is None:
            continue
        
        # Weighted Spearman rank correlation
        corr = weighted_spearman(anchor_vals, var_vals, weights)
        
        if not np.isfinite(corr):
            corr = 0.0
        
        abs_corr = abs(corr)
        strength = ('Strong' if abs_corr >= 0.7 else
                   'Moderate' if abs_corr >= 0.5 else
                   'Weak' if abs_corr >= 0.3 else 'Very weak')
        
        correlations.append({
            'variable': var,
            'correlation': corr,
            'strength': strength,
            'type': 'positive' if corr > 0 else 'negative'
        })
    
    return pd.DataFrame(correlations)

@st.cache_data
def calculate_historical_mood(df, dependent_vars=None):
    """
    v2.0: Physics-Informed Mood Score Engine.
    
    Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │ Layer 1: Adaptive Correlations (decay-weighted Spearman)           │
    │ Layer 2: Adaptive Percentiles (half-life weighted empirical CDF)   │
    │ Layer 3: Entropy-Augmented Scoring (confident regimes amplified)   │
    │ Layer 4: Ornstein-Uhlenbeck Normalization (physics-based scaling)  │
    │ Layer 5: Kalman Smoothing (adaptive bandwidth)                     │
    │ Layer 6: Hurst-Informed Classification (trending vs reverting)     │
    └─────────────────────────────────────────────────────────────────────┘
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
    
    # ── Information-Theoretic Weight Allocation ─────────────────────────
    # Weight = |correlation| * (1 - entropy_of_variable)
    # Variables with low entropy (structured, non-random) and high
    # correlation get the most weight. Pure noise variables get suppressed.
    var_entropies = {}
    for var in [col for col in dependent_vars if col in df.columns]:
        var_returns = df[var].pct_change().dropna().values
        var_entropies[var] = shannon_entropy(var_returns) if len(var_returns) > 10 else 0.5
    
    def _build_weights(corr_df):
        raw = {}
        for _, row in corr_df.iterrows():
            var = row['variable']
            corr_mag = abs(row['correlation'])
            entropy_penalty = 1.0 - var_entropies.get(var, 0.5)  # Low entropy → high weight
            raw[var] = corr_mag * max(entropy_penalty, 0.1)
        total = max(sum(raw.values()), 1e-10)
        return {k: v / total for k, v in raw.items()}
    
    pe_weights = _build_weights(pe_corrs)
    ey_weights = _build_weights(ey_corrs)
    
    # ── Layer 2: Adaptive Percentiles ───────────────────────────────────
    # Half-life = 504 trading days (~2 years). Recent market structure
    # matters more than ancient history for "where are we historically."
    pct_half_life = min(504, n // 2) if n > 20 else max(n // 2, 5)
    
    pe_percentiles = adaptive_percentile(df['NIFTY50_PE'].values, half_life=pct_half_life)
    ey_percentiles = adaptive_percentile(df['NIFTY50_EY'].values, half_life=pct_half_life)
    
    # Base scores: PE is inverse (high PE = bearish), EY is direct (high EY = bullish)
    pe_base = -1.0 + 2.0 * (1.0 - pe_percentiles)
    ey_base = -1.0 + 2.0 * ey_percentiles
    
    # ── Weighted Variable Adjustments ───────────────────────────────────
    pe_adjustments = np.zeros(n)
    ey_adjustments = np.zeros(n)
    
    vars_to_process = [col for col in dependent_vars if col in df.columns]
    
    for var in vars_to_process:
        var_pct = adaptive_percentile(df[var].values, half_life=pct_half_life)
        
        if var in pe_weights:
            pe_type = 'positive'
            pe_match = pe_corrs.loc[pe_corrs['variable'] == var]
            if len(pe_match) > 0:
                pe_type = pe_match.iloc[0]['type']
            weight = pe_weights.get(var, 0)
            sign = 1.0 if pe_type == 'positive' else -1.0
            pe_adjustments += sign * weight * (1.0 - var_pct)
        
        if var in ey_weights:
            ey_type = 'positive'
            ey_match = ey_corrs.loc[ey_corrs['variable'] == var]
            if len(ey_match) > 0:
                ey_type = ey_match.iloc[0]['type']
            weight = ey_weights.get(var, 0)
            sign = 1.0 if ey_type == 'positive' else -1.0
            ey_adjustments += sign * weight * var_pct
    
    pe_scores = np.clip(0.5 * pe_base + 0.5 * pe_adjustments, -1, 1)
    ey_scores = np.clip(0.5 * ey_base + 0.5 * ey_adjustments, -1, 1)
    
    # Anchor blending (correlation-strength-weighted)
    pe_strength = sum(abs(row['correlation']) for _, row in pe_corrs.iterrows())
    ey_strength = sum(abs(row['correlation']) for _, row in ey_corrs.iterrows())
    total_strength = pe_strength + ey_strength or 1
    raw_mood = (pe_strength / total_strength) * pe_scores + (ey_strength / total_strength) * ey_scores
    
    # ── Layer 3: Entropy-Augmented Scoring ──────────────────────────────
    # When market entropy is high (disordered), compress scores toward 0.
    # When entropy is low (ordered/trending), let the signal through fully.
    # This prevents strong mood signals during confused, choppy markets.
    nifty_returns = df['NIFTY'].pct_change().fillna(0).values
    market_entropy = rolling_entropy(nifty_returns, window=60, n_bins=15)
    
    # Entropy gate: maps entropy [0,1] to confidence [0.4, 1.0]
    # Even at max entropy, we don't fully suppress (floor at 0.4)
    entropy_confidence = 1.0 - 0.6 * market_entropy
    raw_mood_gated = raw_mood * entropy_confidence
    
    # ── Layer 4: Ornstein-Uhlenbeck Normalization ───────────────────────
    # Instead of global z-score (which shifts all history when you add data),
    # use OU process physics. Estimate theta, mu, sigma from the series itself,
    # then express the score as "distance from equilibrium in sigma units."
    # This is local, path-independent, and has physical meaning.
    
    # First pass: rough scaling to get into a reasonable range
    expanding_mean = pd.Series(raw_mood_gated).expanding().mean().values
    expanding_std = pd.Series(raw_mood_gated).expanding().std().fillna(1).values
    expanding_std = np.maximum(expanding_std, 1e-6)
    rough_scaled = (raw_mood_gated - expanding_mean) / expanding_std
    
    # OU estimation on the rough-scaled series
    if n > 50:
        theta, mu, sigma_ou = ornstein_uhlenbeck_estimate(rough_scaled)
    else:
        theta, mu, sigma_ou = (0.05, 0.0, 1.0)
    
    sigma_ou = max(sigma_ou, 1e-6)
    
    # OU-normalized: (x - mu) / (sigma / sqrt(2*theta))
    # This is the stationary standard deviation of the OU process
    ou_stationary_std = sigma_ou / np.sqrt(2.0 * max(theta, 1e-4))
    ou_stationary_std = max(ou_stationary_std, 1e-6)
    
    mood_scores = (rough_scaled - mu) / ou_stationary_std * 30.0
    mood_scores = np.clip(mood_scores, -100, 100)
    
    # ── Layer 5: Kalman Smoothing ───────────────────────────────────────
    # Replaces fixed rolling(7).mean() with adaptive Kalman filter.
    # Smoothing bandwidth auto-adjusts based on signal-to-noise ratio.
    smoothed_mood_scores, kalman_gains = kalman_filter_1d(mood_scores)
    
    # Traditional volatility measure (kept for backward compatibility)
    mood_series = pd.Series(mood_scores)
    mood_volatility = mood_series.rolling(window=30, min_periods=1).std().fillna(0)
    
    # ── Layer 6: Hurst-Informed Classification ──────────────────────────
    # When H > 0.5 (trending): trust the score direction.
    # When H < 0.5 (mean-reverting): the score is likely to reverse,
    #   so we tighten the thresholds for extreme classifications.
    hurst_values = rolling_hurst(mood_scores, window=120, step=5)
    
    # Adaptive thresholds: base ±20/±60, tighten when mean-reverting
    def _classify_mood(score, h):
        # When mean-reverting (H<0.5), widen the "neutral" band
        # because extreme readings are likely to snap back.
        # When trending (H>0.5), narrow it — momentum is real.
        h_factor = np.clip(1.0 + (h - 0.5) * 1.0, 0.6, 1.4)
        thresh_mild = 20.0 / h_factor   # Shrinks when trending, expands when reverting
        thresh_extreme = 60.0 / h_factor
        
        if score > thresh_extreme:
            return 'Very Bullish'
        elif score > thresh_mild:
            return 'Bullish'
        elif score > -thresh_mild:
            return 'Neutral'
        elif score > -thresh_extreme:
            return 'Bearish'
        else:
            return 'Very Bearish'
    
    moods = np.array([_classify_mood(mood_scores[i], hurst_values[i]) for i in range(n)])
    
    # ── Fisher Information (signal confidence) ──────────────────────────
    fisher_info = fisher_information_rolling(mood_scores, window=30)
    
    # ── Assemble Output ─────────────────────────────────────────────────
    result_df = pd.DataFrame({
        'DATE': df['DATE'].values,
        'Mood_Score': mood_scores,
        'Mood': moods,
        'Smoothed_Mood_Score': smoothed_mood_scores,
        'Mood_Volatility': mood_volatility.values,
        'NIFTY': df['NIFTY'].values,
        'AD_RATIO': df['AD_RATIO'].values if 'AD_RATIO' in df.columns else 1.0,
        # v2.0 enriched columns
        'Hurst': hurst_values,
        'Market_Entropy': market_entropy,
        'Fisher_Info': fisher_info,
        'Entropy_Confidence': entropy_confidence,
        'OU_Theta': theta,
        'OU_Mu': mu,
        'OU_Sigma': sigma_ou,
        'Kalman_Gain': kalman_gains,
    })
    
    logging.info(f"v2.0 mood engine completed in {time.time() - start_time:.2f}s "
                 f"[θ={theta:.3f}, μ={mu:.2f}, H={hurst_values[-1]:.2f}, "
                 f"S={market_entropy[-1]:.2f}]")
    return result_df

# ══════════════════════════════════════════════════════════════════════════════
# MSF-ENHANCED SPREAD INDICATOR
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def calculate_msf_spread(df, mood_col='Mood_Score', nifty_col='NIFTY', breadth_col='AD_RATIO'):
    """
    v2.0: MSF-Enhanced Spread Indicator with Physics Extensions.
    
    Components:
      1. Momentum  — NIFTY ROC z-score (price velocity)
      2. Structure — Mood trend divergence + acceleration (shape of curve)
      3. Regime    — Adaptive-threshold directional count (market character)
      4. Flow      — Breadth divergence from mean (participation)
      5. Entropy   — Shannon entropy of returns (disorder measure)    [NEW]
      6. Persistence — Hurst exponent deviation from 0.5 (memory)    [NEW]
    
    Weighting: Inverse-variance (minimum-variance portfolio of signals).
    Components that are more stable get higher weight because they carry
    more information per unit of noise. This is the signal-processing
    analog of Markowitz optimization.
    """
    start_time = time.time()
    
    length = 20
    roc_len = 14
    clip = 3.0
    
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
    roc_raw = nifty_series.pct_change(roc_len, fill_method=None)
    roc_z = zscore_clipped(roc_raw, length, clip)
    momentum_norm = sigmoid(roc_z, 1.5)
    
    # ── Component 2: Structure (Mood trend divergence) ──────────────────
    trend_fast = mood_series.rolling(5, min_periods=1).mean()
    trend_slow = mood_series.rolling(length, min_periods=1).mean()
    trend_diff_z = zscore_clipped(trend_fast - trend_slow, length, clip)
    mood_accel_raw = mood_series.diff(5).diff(5)
    mood_accel_z = zscore_clipped(mood_accel_raw, length, clip)
    structure_z = (trend_diff_z + mood_accel_z) / np.sqrt(2.0)
    structure_norm = sigmoid(structure_z, 1.5)
    
    # ── Component 3: Regime (Adaptive threshold) ────────────────────────
    # v1.x used fixed 0.0033 threshold. v2.0 adapts to local volatility.
    # Threshold = rolling_std * 0.5 — a move is "directional" if it exceeds
    # half a local standard deviation.
    pct_change = nifty_series.pct_change(fill_method=None).fillna(0)
    rolling_vol = pct_change.rolling(window=length, min_periods=5).std().fillna(0.003)
    adaptive_threshold = (rolling_vol * 0.5).clip(lower=0.001)
    
    regime_signals = np.where(pct_change > adaptive_threshold, 1,
                     np.where(pct_change < -adaptive_threshold, -1, 0))
    regime_count = pd.Series(regime_signals, index=df.index).cumsum()
    regime_raw = regime_count - regime_count.rolling(length, min_periods=1).mean()
    regime_z = zscore_clipped(regime_raw, length, clip)
    regime_norm = sigmoid(regime_z, 1.5)
    
    # ── Component 4: Breadth Flow ───────────────────────────────────────
    breadth_ma = breadth_series.rolling(length, min_periods=1).mean()
    breadth_ratio = breadth_series / breadth_ma.replace(0, 1)
    breadth_z = zscore_clipped(breadth_ratio - 1, length, clip)
    flow_norm = sigmoid(breadth_z, 1.5)
    
    # ── Component 5: Entropy (Market Disorder) ──────────────────────────
    # High entropy → market is confused/choppy → bearish for trend-followers.
    # Low entropy → market is orderly/trending → signal is more reliable.
    # We encode: low entropy = positive (bullish for confidence),
    #            high entropy = negative (bearish for confidence).
    returns_arr = pct_change.values
    entropy_vals = rolling_entropy(returns_arr, window=60, n_bins=15)
    # Map [0,1] entropy → [-1,1] where low entropy = +1 (ordered) 
    entropy_signal = 1.0 - 2.0 * entropy_vals
    entropy_norm = pd.Series(entropy_signal)
    
    # ── Component 6: Persistence (Hurst deviation from random walk) ─────
    # H > 0.5 → trending → positive signal (momentum will persist)
    # H < 0.5 → mean-reverting → negative signal (current move will reverse)
    # H = 0.5 → random walk → zero signal
    hurst_vals = rolling_hurst(nifty, window=120, step=5)
    hurst_signal = 2.0 * (hurst_vals - 0.5)  # Map [0,1] → [-1,1]
    hurst_norm = pd.Series(np.clip(hurst_signal, -1, 1))
    
    # ── Inverse-Variance Weighting ──────────────────────────────────────
    # Each component's weight = 1/variance. Stable signals get upweighted.
    # This is the signal-processing analog of minimum-variance portfolio.
    components = {
        'momentum': momentum_norm,
        'structure': structure_norm,
        'regime': regime_norm,
        'flow': flow_norm,
        'entropy': entropy_norm,
        'persistence': hurst_norm,
    }
    
    # Compute rolling variance of each component (last 60 observations)
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
    
    # Weighted combination
    msf_raw = sum(weights[name] * comp for name, comp in components.items())
    msf_spread = msf_raw * 10
    
    result['msf_spread'] = msf_spread
    result['momentum'] = momentum_norm * 10
    result['structure'] = structure_norm * 10
    result['regime'] = regime_norm * 10
    result['flow'] = flow_norm * 10
    result['entropy_component'] = entropy_norm * 10
    result['persistence_component'] = hurst_norm * 10
    
    # Log component weights for transparency
    weight_str = ', '.join(f"{k}={v:.2f}" for k, v in weights.items())
    logging.info(f"v2.0 MSF calculated in {time.time() - start_time:.2f}s [{weight_str}]")
    return result

# ══════════════════════════════════════════════════════════════════════════════
# SIMILAR PERIODS FINDER
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def find_similar_periods(df, top_n=10, recency_weight=0.1):
    """
    v2.0: Physics-Informed Similar Period Finder.
    
    Upgrades over v1.x:
    1. Rich feature vector: mood, volatility, momentum, Hurst, entropy, NIFTY ROC
    2. Mahalanobis distance (covariance-aware — correlated features don't double-count)
    3. Trajectory matching via cosine similarity on recent mood path shape
    4. Exponential recency decay instead of linear
    
    The intuition: two periods are "similar" not just when they have the same
    mood score, but when the entire market state (volatility regime, trend
    persistence, disorder level, and recent trajectory shape) matches.
    """
    if df.empty or 'Mood_Score' not in df.columns:
        return []
    
    latest = df.iloc[-1]
    current_mood = latest['Mood_Score']
    current_volatility = latest['Mood_Volatility']
    n = len(df)
    
    historical = df.iloc[:-30].copy() if n > 30 else df.iloc[:-1].copy()
    if historical.empty or len(historical) < 5:
        return []
    
    # ── Build Feature Vectors ───────────────────────────────────────────
    # Use all available v2.0 enriched columns, fallback gracefully
    
    # Momentum: 14-day rate of change of NIFTY
    nifty_roc = df['NIFTY'].pct_change(14).fillna(0).values
    
    feature_names = ['Mood_Score', 'Mood_Volatility']
    current_features = [current_mood, current_volatility]
    hist_features_list = [
        historical['Mood_Score'].values,
        historical['Mood_Volatility'].values,
    ]
    
    # Add Hurst if available
    if 'Hurst' in df.columns:
        feature_names.append('Hurst')
        current_features.append(latest['Hurst'])
        hist_features_list.append(historical['Hurst'].values)
    
    # Add Market Entropy if available
    if 'Market_Entropy' in df.columns:
        feature_names.append('Market_Entropy')
        current_features.append(latest['Market_Entropy'])
        hist_features_list.append(historical['Market_Entropy'].values)
    
    # Add NIFTY momentum
    feature_names.append('NIFTY_ROC')
    current_features.append(nifty_roc[-1])
    hist_roc = nifty_roc[:len(historical)]
    if len(hist_roc) < len(historical):
        hist_roc = np.pad(hist_roc, (len(historical) - len(hist_roc), 0), constant_values=0)
    hist_features_list.append(hist_roc[:len(historical)])
    
    # Add Fisher Info if available
    if 'Fisher_Info' in df.columns:
        feature_names.append('Fisher_Info')
        current_features.append(latest['Fisher_Info'])
        hist_features_list.append(historical['Fisher_Info'].values)
    
    current_vec = np.array(current_features, dtype=np.float64)
    hist_matrix = np.column_stack(hist_features_list)
    
    # Replace NaN/Inf with column medians for robustness
    for col in range(hist_matrix.shape[1]):
        col_data = hist_matrix[:, col]
        valid = np.isfinite(col_data)
        if valid.any():
            median_val = np.median(col_data[valid])
            hist_matrix[~valid, col] = median_val
        else:
            hist_matrix[:, col] = 0.0
    
    current_vec = np.where(np.isfinite(current_vec), current_vec, 0.0)
    
    # ── Mahalanobis Distance ────────────────────────────────────────────
    cov_matrix = np.cov(hist_matrix, rowvar=False)
    
    # Ensure cov_matrix is well-conditioned
    if cov_matrix.ndim < 2:
        cov_matrix = np.array([[max(cov_matrix, 1e-6)]])
    
    maha_distances = mahalanobis_distance_batch(hist_matrix, current_vec, cov_matrix)
    
    # Normalize to [0, 1] similarity (inverse distance)
    max_dist = maha_distances.max() if maha_distances.max() > 0 else 1.0
    maha_similarity = 1.0 - (maha_distances / max_dist)
    
    # ── Trajectory Similarity ───────────────────────────────────────────
    # Compare the shape of the mood score path over the last 20 days
    # using cosine similarity. This captures "are we in a similar trajectory?"
    trajectory_window = 20
    trajectory_sim = np.zeros(len(historical))
    
    if n > trajectory_window:
        current_trajectory = df['Mood_Score'].values[-trajectory_window:]
        # Detrend: subtract linear fit to focus on shape, not level
        ct_detrended = current_trajectory - np.linspace(current_trajectory[0], current_trajectory[-1], trajectory_window)
        
        hist_indices = historical.index
        for j, idx in enumerate(hist_indices):
            pos = df.index.get_loc(idx)
            if pos >= trajectory_window:
                hist_traj = df['Mood_Score'].values[pos - trajectory_window:pos]
                ht_detrended = hist_traj - np.linspace(hist_traj[0], hist_traj[-1], trajectory_window)
                trajectory_sim[j] = (cosine_similarity(ct_detrended, ht_detrended) + 1) / 2  # Map [-1,1] → [0,1]
    
    # ── Recency (Exponential Decay) ─────────────────────────────────────
    days_since = (latest['DATE'] - historical['DATE']).dt.days.values.astype(float)
    recency_half_life = 365.0  # 1-year half-life
    recency_bonus = np.exp(-np.log(2) * days_since / recency_half_life) * recency_weight
    
    # ── Combined Similarity ─────────────────────────────────────────────
    # Mahalanobis (50%) + Trajectory (35%) + Recency (15%)
    combined = 0.50 * maha_similarity + 0.35 * trajectory_sim + 0.15 * recency_bonus / max(recency_bonus.max(), 1e-6)
    
    historical = historical.copy()
    historical['similarity'] = combined
    top_similar = historical.nlargest(top_n, 'similarity')
    
    results = []
    for _, row in top_similar.iterrows():
        results.append({
            'date': row['DATE'].strftime('%Y-%m-%d'),
            'similarity': row['similarity'],
            'mood_score': row['Mood_Score'],
            'mood': row['Mood'],
            'mood_volatility': row['Mood_Volatility'],
            'nifty': row['NIFTY'] if 'NIFTY' in row else 0
        })
    
    return results

# ══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

def main():
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
            st.caption("Select which columns feed into the Mood Score model. Defaults are pre-selected.")
            selected_predictors = st.multiselect(
                "Predictor Columns",
                options=DEPENDENT_VARS,
                default=DEPENDENT_VARS,
                label_visibility="collapsed",
                help="These columns are used as dependent variables for PE & EY correlation-weighted mood scoring."
            )
            if not selected_predictors:
                st.warning("⚠️ Select at least one predictor column.")
                selected_predictors = DEPENDENT_VARS  # fallback to defaults
            
            if set(selected_predictors) != set(DEPENDENT_VARS):
                st.info(f"Using {len(selected_predictors)}/{len(DEPENDENT_VARS)} predictors")
        
        # Store in session state for downstream use
        st.session_state['selected_predictors'] = tuple(selected_predictors)
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class='info-box'>
            <p style='font-size: 0.8rem; margin: 0; color: var(--text-muted); line-height: 1.5;'>
                <strong>Version:</strong> {VERSION}<br>
                <strong>Engine:</strong> OU + Kalman + Entropy<br>
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
            <div class="tagline">Ornstein-Uhlenbeck · Kalman · Entropy · Hurst · Fisher | Quantitative Market Physics</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Separator between header and cards
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DATA LOADING
    # ═══════════════════════════════════════════════════════════════════════════
    with st.spinner("Loading market data..."):
        raw_df = load_data()
    
    if raw_df is None:
        st.stop()
    
    with st.spinner("Calculating mood scores..."):
        selected_preds = st.session_state.get('selected_predictors', tuple(DEPENDENT_VARS))
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
        msf_color = '#06b6d4'  # Cyan to match trace
        st.markdown(f"""
        <div class="metric-card {msf_class}">
            <h4>MSF Spread</h4>
            <h2 style="color: {msf_color};">{msf_spread:+.2f}</h2>
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
    
    # Mood Score Line
    fig.add_trace(
        go.Scattergl(
            x=df['DATE'],
            y=df['Mood_Score'],
            mode='lines',
            name='Mood Score',
            line=dict(color='#FFC300', width=2),
            hovertemplate='<b>%{x|%d %b %Y}</b><br>Mood: %{y:.2f}<extra></extra>',
        ),
        row=1, col=1
    )
    
    # Only zero line reference
    fig.add_hline(y=0, line_color='#757575', line_width=1, line_dash='dash', row=1, col=1)
    
    # Current value annotation
    last_point = df.iloc[-1]
    fig.add_annotation(
        x=last_point['DATE'],
        y=last_point['Mood_Score'],
        text=f"<b>{last_point['Mood_Score']:.1f}</b>",
        showarrow=True,
        arrowhead=2,
        arrowcolor='#FFC300',
        ax=40,
        ay=0,
        bgcolor='#1A1A1A',
        bordercolor='#FFC300',
        borderwidth=1,
        font=dict(color='#FFC300', size=11),
        row=1, col=1
    )
    
    # ─────────────────────────────────────────────────────────────────────────
    # ROW 2: MSF SPREAD INDICATOR (Oscillator Pane) - CYAN
    # ─────────────────────────────────────────────────────────────────────────
    
    # MSF Spread Line
    msf_values = msf_filtered['msf_spread'].values
    
    fig.add_trace(
        go.Scattergl(
            x=df['DATE'],
            y=msf_values,
            mode='lines',
            name='MSF Spread',
            line=dict(color='#06b6d4', width=2),
            hovertemplate='<b>%{x|%d %b %Y}</b><br>MSF: %{y:.2f}<extra></extra>',
        ),
        row=2, col=1
    )
    
    # Only zero line reference
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
    
    # Add red triangles at y=5 (top, inverted)
    if red_signals:
        fig.add_trace(
            go.Scatter(
                x=[df['DATE'].iloc[i] for i in red_signals],
                y=[5] * len(red_signals),
                mode='markers',
                name='Bearish Signal',
                marker=dict(
                    symbol='triangle-down',
                    size=8,
                    color='#ef4444',
                    line=dict(color='#ef4444', width=1)
                ),
                hoverinfo='skip',
                showlegend=False
            ),
            row=2, col=1
        )
    
    # Add green triangles at y=-5 (bottom)
    if green_signals:
        fig.add_trace(
            go.Scatter(
                x=[df['DATE'].iloc[i] for i in green_signals],
                y=[-5] * len(green_signals),
                mode='markers',
                name='Bullish Signal',
                marker=dict(
                    symbol='triangle-up',
                    size=8,
                    color='#10b981',
                    line=dict(color='#10b981', width=1)
                ),
                hoverinfo='skip',
                showlegend=False
            ),
            row=2, col=1
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # LAYOUT
    # ─────────────────────────────────────────────────────────────────────────
    
    fig.update_layout(
        height=750,
        template='plotly_dark',
        plot_bgcolor='#1A1A1A',
        paper_bgcolor='#1A1A1A',
        font=dict(color='#EAEAEA', family='Inter'),
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.05,  # Moved up to avoid toolbar overlap
            xanchor='right',
            x=1,
            font=dict(size=11)
        ),
        margin=dict(l=60, r=20, t=80, b=40),  # Increased top margin
        xaxis2=dict(
            showgrid=True,
            gridcolor='#2A2A2A',
            type='date'
        ),
        yaxis=dict(
            title=dict(text='Mood Score', font=dict(size=11, color='#888888')),
            showgrid=True,
            gridcolor='#2A2A2A',
            zeroline=False,
            autorange='reversed'
        ),
        yaxis2=dict(
            title=dict(text='MSF Spread', font=dict(size=11, color='#888888')),
            showgrid=True,
            gridcolor='#2A2A2A',
            zeroline=False
        )
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
    fig.update_xaxes(showgrid=True, gridcolor='#2A2A2A', row=2, col=1)
    
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

# ══════════════════════════════════════════════════════════════════════════════
# SIMILAR PERIODS VIEW
# ══════════════════════════════════════════════════════════════════════════════

def render_similar_periods(mood_df):
    """Render similar historical periods analysis."""
    
    st.markdown("""
        <div style="margin-bottom: 1rem;">
            <h3 style="color: #FFC300; margin: 0;">🔍 Similar Historical Periods</h3>
            <p style="color: #888888; font-size: 0.85rem; margin: 0;">AI-matched periods based on mood score and volatility patterns</p>
        </div>
    """, unsafe_allow_html=True)
    
    similar_periods = find_similar_periods(mood_df)
    
    if not similar_periods:
        st.warning("Not enough historical data to find similar periods.")
        return
    
    # Display as cards
    cols = st.columns(2)
    for i, period in enumerate(similar_periods[:10]):
        col = cols[i % 2]
        with col:
            similarity_pct = period['similarity'] * 100
            mood_val = period['mood_score']
            mood_class = 'bullish' if mood_val > 20 else 'bearish' if mood_val < -20 else 'neutral'
            
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
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CORRELATION ANALYSIS VIEW
# ══════════════════════════════════════════════════════════════════════════════

def render_correlation_analysis(raw_df):
    """Render correlation analysis between variables."""
    
    st.markdown("""
        <div style="margin-bottom: 1rem;">
            <h3 style="color: #FFC300; margin: 0;">📋 Correlation Analysis</h3>
            <p style="color: #888888; font-size: 0.85rem; margin: 0;">Variable relationships with PE and Earnings Yield anchors</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### PE Ratio Correlations")
        pe_corrs = calculate_anchor_correlations(raw_df, 'NIFTY50_PE', st.session_state.get('selected_predictors', tuple(DEPENDENT_VARS)))
        if not pe_corrs.empty:
            pe_corrs_display = pe_corrs.sort_values('correlation', key=abs, ascending=False)
            for _, row in pe_corrs_display.iterrows():
                corr_val = row['correlation']
                color = '#10b981' if corr_val > 0 else '#ef4444'
                bar_width = abs(corr_val) * 100
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem; padding: 0.5rem; background: #1A1A1A; border-radius: 8px;">
                    <span style="width: 120px; font-size: 0.8rem; color: #EAEAEA;">{row['variable']}</span>
                    <div style="flex: 1; height: 8px; background: #2A2A2A; border-radius: 4px; margin: 0 10px;">
                        <div style="width: {bar_width}%; height: 100%; background: {color}; border-radius: 4px;"></div>
                    </div>
                    <span style="width: 60px; text-align: right; font-size: 0.8rem; color: {color};">{corr_val:.2f}</span>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### Earnings Yield Correlations")
        ey_corrs = calculate_anchor_correlations(raw_df, 'NIFTY50_EY', st.session_state.get('selected_predictors', tuple(DEPENDENT_VARS)))
        if not ey_corrs.empty:
            ey_corrs_display = ey_corrs.sort_values('correlation', key=abs, ascending=False)
            for _, row in ey_corrs_display.iterrows():
                corr_val = row['correlation']
                color = '#10b981' if corr_val > 0 else '#ef4444'
                bar_width = abs(corr_val) * 100
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem; padding: 0.5rem; background: #1A1A1A; border-radius: 8px;">
                    <span style="width: 120px; font-size: 0.8rem; color: #EAEAEA;">{row['variable']}</span>
                    <div style="flex: 1; height: 8px; background: #2A2A2A; border-radius: 4px; margin: 0 10px;">
                        <div style="width: {bar_width}%; height: 100%; background: {color}; border-radius: 4px;"></div>
                    </div>
                    <span style="width: 60px; text-align: right; font-size: 0.8rem; color: {color};">{corr_val:.2f}</span>
                </div>
                """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# RUN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
