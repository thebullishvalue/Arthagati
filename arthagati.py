# -*- coding: utf-8 -*-
"""
ARTHAGATI (अर्थगति) - Market Sentiment Analysis | An @thebullishvalue Product
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Quantitative market mood analysis with MSF-enhanced indicators.
TradingView-style charting with institutional-grade analytics.
"""

import logging
import os
import time
from datetime import datetime
from io import StringIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytz
import requests
import streamlit as st
from plotly.subplots import make_subplots

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="ARTHAGATI | Market Sentiment Analysis",
    layout="wide",
    page_icon="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgd2lkdGg9IjI0IiBoZWlnaHQ9IjI0Ij48cGF0aCBmaWxsPSJub25lIiBzdHJva2U9IiNmZmNjMDAwIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS13aWR0aD0iMiIgZD0iTTEwIDIwYTUuMzU1IDUuMzU1IDAgMSAxIDAtMTAuNzEgNS4zNTUgNS4zNTUgMCAxIDEgMCAxMC43MXptLTUuMzU1LTUuMzU1djJsNS4zNTUgNEg0LjY0Nkw3IDIydnptOC05LjM3MXY2bDUtNEg5LjY0NlpNNCA5LjM3MXY2bDUtNEg3LjM3MloiLz48L3N2Zz4=",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════════════
# IDENTITY
# ══════════════════════════════════════════════════════════════════════════════

VERSION      = "v2.6.0"
PRODUCT_NAME = "Arthagati"
COMPANY      = "@thebullishvalue"

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

C_PRIMARY = '#FFB000'
C_GREEN   = '#00FF41'
C_RED     = '#FF3333'
C_AMBER   = '#FFB000'
C_CYAN    = '#00D4FF'
C_MUTED   = '#666666'
C_BG_CARD = '#141414'
C_BG_GRID = '#1E1E1E'
C_TEXT    = '#E0E0E0'

# ══════════════════════════════════════════════════════════════════════════════
# MODEL HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

# Correlation engine
CORR_HALF_LIFE  = 504    # ~2 trading years; exponential recency weight for Spearman
PCT_HALF_LIFE   = 252    # ~1 trading year;  recency weight for adaptive ECDF
MOOD_SCALE      = 30.0   # maps OU-normalised signal → mood score
KALMAN_CI_Z     = 1.96   # Kalman confidence band (≈95%)
KALMAN_HALF_LIFE = 126   # Kalman fading memory half-life (trading days, independent of PCT)
DATA_TTL        = 3600   # Streamlit cache TTL for the Google Sheets fetch (seconds)

# Walk-forward correlation rebalancing (eliminates look-ahead bias)
CORR_MIN_WARMUP       = 252   # minimum observations before first correlation checkpoint
CORR_REBALANCE_PERIOD = 63    # expanding-window rebalance interval (≈quarterly)

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
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');
    
    :root {
        --primary-color: #FFB000;
        --primary-rgb: 255, 176, 0;
        --background-color: #0A0A0A;
        --secondary-background-color: #141414;
        --bg-card: #141414;
        --bg-elevated: #1E1E1E;
        --text-primary: #E0E0E0;
        --text-secondary: #B0B0B0;
        --text-muted: #666666;
        --border-color: #252525;
        --border-light: #353535;
        --success-green: #00FF41;
        --danger-red: #FF3333;
        --warning-amber: #FFB000;
        --info-cyan: #00D4FF;
        --neutral: #666666;
        
        --font-display: 'IBM Plex Mono', monospace;
        --font-body: 'IBM Plex Sans', sans-serif;
        
        --space-xs: 0.25rem;
        --space-sm: 0.5rem;
        --space-md: 1rem;
        --space-lg: 1.5rem;
        --space-xl: 2rem;
    }
    
    * { font-family: var(--font-body), -apple-system, BlinkMacSystemFont, sans-serif; }
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
    
    .metric-card {
        background-color: var(--bg-card);
        padding: 1.25rem;
        border-radius: 4px;
        border: 1px solid var(--border-color);
        box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.08);
        margin-bottom: 0.5rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        cursor: pointer;
    }
    
    .metric-card:hover { box-shadow: 0 8px 30px rgba(0,0,0,0.3); border-color: var(--border-light); }
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
        cursor: pointer;
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
    .stButton>button:hover { box-shadow: 0 0 25px rgba(var(--primary-rgb), 0.6); background: var(--primary-color); color: var(--bg-card); transform: translateY(-2px); }
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

    /* ── Retro Broker Terminal Effects ─────────────────────────────────── */
    .scanlines {
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        pointer-events: none;
        z-index: 999998;
        background: repeating-linear-gradient(
            0deg,
            transparent,
            transparent 2px,
            rgba(0, 0, 0, 0.15) 2px,
            rgba(0, 0, 0, 0.15) 4px
        );
        opacity: 0.03;
    }
    
    .crt-glow {
        text-shadow: 0 0 5px var(--primary-color), 0 0 10px var(--primary-color);
    }
    
    .terminal-corner {
        position: relative;
    }
    .terminal-corner::before {
        content: '';
        position: absolute;
        top: -1px; left: -1px;
        width: 8px; height: 8px;
        border-top: 2px solid var(--primary-color);
        border-left: 2px solid var(--primary-color);
    }
    .terminal-corner::after {
        content: '';
        position: absolute;
        bottom: -1px; right: -1px;
        width: 8px; height: 8px;
        border-bottom: 2px solid var(--primary-color);
        border-right: 2px solid var(--primary-color);
    }

    /* ── Responsive Container ─────────────────────────────────── */
    .block-container {
        max-width: min(95%, 1400px) !important;
        margin: 0 auto !important;
    }
    
    @media (max-width: 768px) {
        .block-container {
            max-width: 100% !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        .premium-header {
            padding: 1rem !important;
        }
        .metric-card {
            padding: 1rem !important;
        }
    }

    /* ── Card Animations ───────────────────────────────────── */
    @keyframes cardReveal {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .metric-card {
        animation: cardReveal 0.4s ease-out forwards;
        opacity: 0;
    }
    .metric-card:nth-child(1) { animation-delay: 0.1s; }
    .metric-card:nth-child(2) { animation-delay: 0.2s; }
    .metric-card:nth-child(3) { animation-delay: 0.3s; }
    .metric-card:nth-child(4) { animation-delay: 0.4s; }

    .magnetic-hover {
        transition: transform 0.2s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.3s ease;
    }
    .magnetic-hover:hover {
        transform: translateY(-2px) scale(1.01);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.4);
    }

    /* ── Themed loading state ─────────────────────────────────────── */
    @keyframes pulse-glow {
        0%, 100% { opacity: 0.6; }
        50%       { opacity: 1.0; }
    }
    .loading-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-left: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin: 0.75rem 0;
        position: relative;
        overflow: hidden;
    }
    .loading-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(circle at 0% 50%, rgba(var(--primary-rgb), 0.06) 0%, transparent 60%);
        pointer-events: none;
    }
    .loading-label {
        font-size: 0.85rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: 0.3px;
        position: relative;
    }
    .loading-sub {
        font-size: 0.72rem;
        color: var(--text-muted);
        margin-top: 0.2rem;
        font-weight: 400;
        position: relative;
        letter-spacing: 0.2px;
    }
    .loading-dot {
        display: inline-block;
        width: 5px; height: 5px;
        border-radius: 50%;
        background: var(--primary-color);
        animation: pulse-glow 1.2s ease-in-out infinite;
        margin-right: 0.6rem;
        vertical-align: middle;
        position: relative;
        top: -1px;
    }

</style>
"""

st.markdown(_DESIGN_CSS, unsafe_allow_html=True)
st.markdown('<div class="scanlines"></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _progress_bar(slot, pct: int, label: str, sub: str = "") -> None:
    """
    Render a themed progress card into an st.empty() slot.

    Call with increasing pct values as each computation step completes.
    Call slot.empty() when all steps are done.

        slot = st.empty()
        _progress_bar(slot, 10, "Fetching data", "Google Sheets")
        df = load_data()
        _progress_bar(slot, 70, "Running engine", "OU · Kalman")
        mood_df = calculate_historical_mood(df)
        _progress_bar(slot, 100, "Complete")
        time.sleep(0.25)
        slot.empty()
    """
    bar_color = C_GREEN if pct == 100 else C_PRIMARY
    slot.markdown(f"""
    <div class="loading-card">
        <div class="loading-label">
            <span class="loading-dot"></span>{label}
        </div>
        {"" if not sub else f'<div class="loading-sub">{sub}</div>'}
        <div style="margin-top: 0.65rem; height: 3px; background: var(--border-color); border-radius: 2px; overflow: hidden;">
            <div style="width: {pct}%; height: 100%;
                        background: linear-gradient(90deg, {bar_color}, {C_AMBER});
                        border-radius: 2px; transition: width 0.3s ease;">
            </div>
        </div>
        <div style="text-align: right; font-size: 0.65rem; color: #555; margin-top: 0.25rem; font-family: monospace;">
            {pct}%
        </div>
    </div>
    """, unsafe_allow_html=True)

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

def adaptive_percentile(series, half_life=252):
    """
    Exponential-decay-weighted empirical CDF — O(N log N) via sorted-insert.

    For each time t, the percentile of x_t is:
        P(t) = Σ_{i≤t} w_i · 𝟙(x_i ≤ x_t) / Σ_{i≤t} w_i
    where w_i = exp(-λ·(t-i)), λ = ln(2)/half_life.

    Implementation: maintain a sorted array of observed values with their
    insertion times. At each step, binary-search for x_t's rank position,
    compute the cumulative weighted CDF using vectorised decay on the
    sorted array. The search is O(log N), but list insertion is O(N).
    Total time complexity: O(N²).

    Greenwald & Khanna (2001) motivates the streaming quantile approach;
    here the sorted-insert + searchsorted is exact.

    Used in: calculate_historical_mood (Layer 3)
    """
    from bisect import bisect_right

    values = np.asarray(series, dtype=np.float64)
    n = len(values)
    if n == 0:
        return np.array([])

    lam = np.log(2) / max(half_life, 1)
    valid = np.isfinite(values)

    if not np.any(valid):
        return np.full(n, 0.5)

    result = np.full(n, np.nan)

    # Maintain parallel sorted arrays: sorted_vals (for bisect) and
    # sorted_times (insertion time for each value, same order).
    sorted_vals = []    # sorted by value
    sorted_times = []   # insertion time corresponding to each sorted value
    total_weight = 0.0  # running sum of all weights (decayed each step)
    decay_factor = np.exp(-lam)  # multiplicative decay per step

    for t in range(n):
        # Decay all existing weights by one step (equivalent to aging everything)
        total_weight *= decay_factor

        if not valid[t]:
            continue

        v = values[t]
        w_new = 1.0  # current observation always has weight 1.0 (most recent)

        # Insert into sorted order
        pos = bisect_right(sorted_vals, v)
        sorted_vals.insert(pos, v)
        sorted_times.insert(pos, t)
        total_weight += w_new

        if total_weight < 1e-12:
            continue

        # Compute weighted CDF: sum of weights for all values ≤ v
        # All values in sorted_vals[:pos+1] have value ≤ v (side='right')
        # Their weights are exp(-λ·(t - insertion_time))
        times_leq = np.array(sorted_times[:pos + 1], dtype=np.float64)
        w_leq = np.exp(-lam * (t - times_leq))
        result[t] = np.sum(w_leq) / total_weight

    # Convert lists to arrays for the final cleanup
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
        return np.array([]), np.array([]), np.array([])
    
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
            '  export ARTHAGATI_SHEET_ID="1po7z42n3dYIQGAvn0D1-a4pmyxpnGPQ13TrNi3DB5_c"\n'
            '  export ARTHAGATI_SHEET_GID="1938234952"'
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
                logging.warning(
                    f"Google Sheets request timed out (attempt {attempt + 1}/{max_retries}). "
                    f"Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                logging.error(f"Google Sheets request failed after {max_retries} attempts: {e}")
        except requests.exceptions.RequestException as e:
            last_exception = e
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt * 2
                logging.warning(
                    f"Google Sheets request failed (attempt {attempt + 1}/{max_retries}). "
                    f"Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                logging.error(f"Google Sheets request failed after {max_retries} attempts: {e}")

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
            logging.warning(
                "Schema drift — %d expected column(s) absent from sheet: %s. "
                "Predictor set will be built from columns that are actually present.",
                len(missing), missing,
            )

        df['DATE'] = pd.to_datetime(df['DATE'], format='%m/%d/%Y', errors='coerce')

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
                logging.info("NIFTY50_EY absent or constant — derived from PE (EY = 1/PE × 100).")

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
        logging.info(
            "Data loaded — %d rows × %d columns | %s | %.2fs",
            len(df), len(df.columns), date_range, elapsed,
        )
        return df

    except Exception as exc:
        logging.error("Data load failed — pipeline halted. Cause: %s", exc)
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

@st.cache_data(max_entries=5, show_spinner=False)
def calculate_historical_mood(df, dependent_vars=None):
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
        logging.error(
            "Mood engine aborted — required anchor columns missing. "
            "Sheet must contain DATE, NIFTY50_PE, and NIFTY50_EY."
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

    # Pre-compute returns for expanding entropy
    var_returns_all = {}
    for var in vars_to_check:
        vals = df[var].values
        rets = np.empty(len(vals))
        rets[0] = np.nan
        with np.errstate(divide='ignore', invalid='ignore'):
            rets[1:] = np.where(np.abs(vals[:-1]) > 1e-12, np.diff(vals) / np.abs(vals[:-1]), 0.0)
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

        cp_n = cp + 1
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
    moods = np.where(mood_scores > 60, 'Very Bullish',
            np.where(mood_scores > 20, 'Bullish',
            np.where(mood_scores > -20, 'Neutral',
            np.where(mood_scores > -60, 'Bearish', 'Very Bearish'))))

    # ── Diagnostics (output-only — do NOT modify scores) ───────────────
    nifty_returns = df['NIFTY'].pct_change().fillna(0).values
    hurst_vals = rolling_hurst(df['NIFTY'].values, window=90, step=5)
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
    })

    logging.info(
        "Mood engine complete — %d rows in %.2fs | "
        "OU: θ=%.3f  μ=%.2f  t½=%.0fd | "
        "Diagnostics: Hurst=%.2f  Entropy=%.2f  Regime=%s | "
        "Walk-forward checkpoints: %d",
        n, time.time() - start_time,
        theta, mu, ou_half_life,
        hurst_vals[-1], entropy_vals[-1], regime_labels[-1],
        len(checkpoints),
    )
    return result_df

# ══════════════════════════════════════════════════════════════════════════════
# MSF-ENHANCED SPREAD INDICATOR
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(max_entries=5, show_spinner=False)
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
        logging.error("MSF Spread aborted — received an empty DataFrame; no rows to process.")
        return result
    
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
    
    weight_str = '  '.join(f"{k}={v:.0%}" for k, v in weights.items())
    logging.info(
        "MSF Spread complete — %.2fs | Inverse-variance weights: %s",
        time.time() - start_time, weight_str,
    )
    return result

# ══════════════════════════════════════════════════════════════════════════════
# SIMILAR PERIODS FINDER
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(max_entries=5, show_spinner=False)
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
        # Least-squares linear detrend (minimises residual variance, unlike endpoint
        # anchoring which distorts on V-shaped or reversal trajectories).
        _traj_x = np.arange(TRAJ_WINDOW, dtype=np.float64)
        _traj_xm = _traj_x - _traj_x.mean()
        _traj_xvar = np.sum(_traj_xm ** 2)

        def _ls_detrend(traj):
            if _traj_xvar < 1e-12:
                return traj - traj.mean()
            slope = np.sum(_traj_xm * (traj - traj.mean())) / _traj_xvar
            return traj - (traj.mean() + slope * _traj_xm)

        current_traj = df['Mood_Score'].values[-TRAJ_WINDOW:]
        ct_detrended = _ls_detrend(current_traj)

        for j, idx in enumerate(historical.index):
            pos = df.index.get_loc(idx)
            if pos >= TRAJ_WINDOW:
                hist_traj = df['Mood_Score'].values[pos - TRAJ_WINDOW:pos]
                ht_detrended = _ls_detrend(hist_traj)
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
# LANDING PAGE
# ══════════════════════════════════════════════════════════════════════════════

def render_landing_page() -> None:
    """Informational landing page — shown on first load; Run Analysis is in the sidebar."""

    # ── Main header ──────────────────────────────────────────────────
    st.markdown("""
    <div class="premium-header">
        <h1>ARTHAGATI <span style="color: var(--primary-color);">:</span> Market Sentiment Analysis</h1>
        <div class="tagline">Ornstein-Uhlenbeck · Kalman · Decay-Spearman · Adaptive Percentiles | Quantitative Market Physics</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Feature cards — the 3 analysis views ────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class='metric-card primary' style='min-height: 280px;'>
            <h3 style='color: var(--primary-color); margin-bottom: 1rem;'><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--primary-color)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block;vertical-align:middle;margin-right:6px;"><path d="m3 3 18 18"/><path d="m18 9-5 5-4-4-3 3"/></svg>Historical Mood</h3>
            <p style='color: var(--text-muted); font-size: 0.9rem; line-height: 1.6;'>
                Full sentiment timeline with OU forward projection, Kalman confidence bands,
                and regime transition markers on a TradingView-style chart.
            </p>
            <br>
            <p style='color: var(--text-secondary); font-size: 0.85rem;'>
                <strong>Features:</strong><br>
                • Mood Score −100 → +100<br>
                • MSF Spread confirmation<br>
                • 90-day OU mean-reversion path<br>
                • Hurst · Entropy · Regime diagnostics
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='metric-card success' style='min-height: 280px;'>
            <h3 style='color: var(--success-green); margin-bottom: 1rem;'><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--success-green)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block;vertical-align:middle;margin-right:6px;"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>Similar Periods</h3>
            <p style='color: var(--text-muted); font-size: 0.9rem; line-height: 1.6;'>
                Historical analog matching against the full dataset with forward-return
                outcomes, aggregate win-rates, and a backtest scatter.
            </p>
            <br>
            <p style='color: var(--text-secondary); font-size: 0.85rem;'>
                <strong>Features:</strong><br>
                • Mahalanobis state matching (55%)<br>
                • Trajectory shape similarity (35%)<br>
                • Recency decay (10%)<br>
                • 30 / 60 / 90-day forward returns
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='metric-card info' style='min-height: 280px;'>
            <h3 style='color: var(--info-cyan); margin-bottom: 1rem;'><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--info-cyan)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block;vertical-align:middle;margin-right:6px;"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/><path d="M16 13H8"/><path d="M16 17H8"/><path d="M10 9H8"/></svg>Correlation Analysis</h3>
            <p style='color: var(--text-muted); font-size: 0.9rem; line-height: 1.6;'>
                Full transparency into which variables drive the mood score and which
                are noise, ranked by the engine's own quality formula.
            </p>
            <br>
            <p style='color: var(--text-secondary); font-size: 0.85rem;'>
                <strong>Features:</strong><br>
                • PE &amp; EY correlation bars<br>
                • Shannon entropy quality score<br>
                • Keep / Useful / Weak ranking<br>
                • Dynamic from live sheet columns
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Analysis Methodology ─────────────────────────────────────────
    st.markdown("""<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--primary-color)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block;vertical-align:middle;margin-right:8px;"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg><span style="color:var(--primary-color);font-weight:600;">Analysis Methodology</span>""", unsafe_allow_html=True)

    col_m1, col_m2, col_m3 = st.columns(3)

    with col_m1:
        st.markdown("""
        <div class='signal-card bullish' style='padding: 1.5rem;'>
            <h4 style='color: var(--success-green); margin-bottom: 1rem;'>Mood Engine — 5 Layers</h4>
            <p style='color: var(--text-muted); font-size: 0.85rem; line-height: 1.7;'>
                Physics-informed scoring pipeline:
            </p>
            <ul style='color: var(--text-secondary); font-size: 0.85rem; line-height: 1.8; margin-top: 0.5rem;'>
                <li><strong>Decay-Spearman</strong> correlations (504d HL)</li>
                <li><strong>Entropy weighting</strong> — noisy vars suppressed</li>
                <li><strong>Adaptive percentiles</strong> — decay-weighted CDF</li>
                <li><strong>OU normalisation</strong> → [−100, +100]</li>
                <li><strong>Kalman smoothing</strong> + ±1.96σ band</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_m2:
        st.markdown("""
        <div class='signal-card bearish' style='padding: 1.5rem;'>
            <h4 style='color: var(--danger-red); margin-bottom: 1rem;'>MSF Spread — Confirmation</h4>
            <p style='color: var(--text-muted); font-size: 0.85rem; line-height: 1.7;'>
                Four-component oscillator −10 → +10:
            </p>
            <ul style='color: var(--text-secondary); font-size: 0.85rem; line-height: 1.8; margin-top: 0.5rem;'>
                <li><strong>Momentum</strong> — NIFTY ROC z-score (14d)</li>
                <li><strong>Structure</strong> — mood trend divergence</li>
                <li><strong>Flow</strong> — breadth participation</li>
                <li><strong>Regime</strong> — adaptive directional count</li>
                <li><strong>Weights</strong> — inverse-variance (Markowitz)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_m3:
        st.markdown("""
        <div class='signal-card neutral' style='padding: 1.5rem;'>
            <h4 style='color: var(--neutral); margin-bottom: 1rem;'>Regime Detection</h4>
            <p style='color: var(--text-muted); font-size: 0.85rem; line-height: 1.7;'>
                Hurst × Entropy quadrant classification:
            </p>
            <ul style='color: var(--text-secondary); font-size: 0.85rem; line-height: 1.8; margin-top: 0.5rem;'>
                <li><strong>Trending</strong> — momentum strategies favoured</li>
                <li><strong>Volatile Trend</strong> — directional with swings</li>
                <li><strong>Mean-Reverting</strong> — contrarian strategies</li>
                <li><strong>Choppy</strong> — reduce size, avoid</li>
                <li><strong>Output</strong> — scales MSF weights + OU horizon</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Mood Score Interpretation ────────────────────────────────────
    st.markdown("""<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--primary-color)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block;vertical-align:middle;margin-right:8px;"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg><span style="color:var(--primary-color);font-weight:600;">Mood Score Interpretation</span>""", unsafe_allow_html=True)

    col_s1, col_s2, col_s3 = st.columns(3)

    with col_s1:
        st.markdown("""
        <div style='background: rgba(16,185,129,0.1); border: 1px solid var(--success-green);
                    border-radius: 12px; padding: 1.25rem;'>
            <h4 style='color: var(--success-green); margin-bottom: 0.75rem;'><svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--success-green)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block;vertical-align:middle;margin-right:6px;"><circle cx="12" cy="12" r="10"/></svg>Bullish Zone</h4>
            <p style='color: var(--text-muted); font-size: 0.85rem;'>Score &gt; +20</p>
            <p style='color: var(--text-secondary); font-size: 0.85rem; margin-top: 0.5rem;'>
                Positive sentiment. Trend-following strategies favoured.
                At extremes (&gt;+60, Euphoric) mean-reversion risk rises sharply.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_s2:
        st.markdown("""
        <div style='background: rgba(136,136,136,0.1); border: 1px solid var(--neutral);
                    border-radius: 12px; padding: 1.25rem;'>
            <h4 style='color: var(--neutral); margin-bottom: 0.75rem;'><svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--neutral)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block;vertical-align:middle;margin-right:6px;"><circle cx="12" cy="12" r="10"/></svg>Neutral Zone</h4>
            <p style='color: var(--text-muted); font-size: 0.85rem;'>Score −20 to +20</p>
            <p style='color: var(--text-secondary); font-size: 0.85rem; margin-top: 0.5rem;'>
                No strong directional bias. Await macro confirmation or use
                MSF Spread and Similar Periods for additional context.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_s3:
        st.markdown("""
        <div style='background: rgba(239,68,68,0.1); border: 1px solid var(--danger-red);
                    border-radius: 12px; padding: 1.25rem;'>
            <h4 style='color: var(--danger-red); margin-bottom: 0.75rem;'><svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--danger-red)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block;vertical-align:middle;margin-right:6px;"><circle cx="12" cy="12" r="10" stroke="var(--danger-red)"/><line x1="8" y1="12" x2="16" y2="12"/></svg>Bearish Zone</h4>
            <p style='color: var(--text-muted); font-size: 0.85rem;'>Score &lt; −20</p>
            <p style='color: var(--text-secondary); font-size: 0.85rem; margin-top: 0.5rem;'>
                Negative sentiment. Defensive positioning warranted.
                At extremes (&lt;−60, Capitulation) contrarian signals may emerge.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── System Coverage ──────────────────────────────────────────────
    st.markdown("""<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--primary-color)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block;vertical-align:middle;margin-right:8px;"><path d="M2 12a10 10 0 0 1 20 0"/><path d="M12 2v20"/><path d="m4.93 4.93 14.14 14.14"/><path d="m19.07 4.93-14.14 14.14"/></svg><span style="color:var(--primary-color);font-weight:600;">System Coverage</span>""", unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown('<div class="metric-card neutral"><h4>Score Anchors</h4><h2>2</h2><div class="sub-metric">PE · Earnings Yield</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card neutral"><h4>Predictors</h4><h2>{len(DEPENDENT_VARS)}</h2><div class="sub-metric">Macro + Breadth vars</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card neutral"><h4>Math Primitives</h4><h2>12</h2><div class="sub-metric">Pure NumPy functions</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card neutral"><h4>OU Projection</h4><h2>90d</h2><div class="sub-metric">Forward reversion path</div></div>', unsafe_allow_html=True)
    with c5:
        st.markdown('<div class="metric-card neutral"><h4>Analog Returns</h4><h2>3</h2><div class="sub-metric">30 · 60 · 90 day</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Getting Started ──────────────────────────────────────────────
    st.markdown("""
    <div class='info-box'>
        <h4><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--primary-color)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block;vertical-align:middle;margin-right:6px;"><path d="m5 12 7-7 7 7"/><path d="M12 19V5"/></svg>Getting Started</h4>
        <p style='color: var(--text-muted); line-height: 1.7;'>
            Click <strong>▶ Run Analysis</strong> in the sidebar to fetch live data from Google Sheets
            and run the full 5-layer sentiment pipeline. Once loaded, use the sidebar to switch
            between <em>Historical Mood</em>, <em>Similar Periods</em>, and <em>Correlation Analysis</em>
            views — or tune the active predictor set in <em>Model Configuration</em> and click
            <strong>Apply</strong> to recompute with your custom variable selection.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ═══════════════════════════════════════════════════════════════════════════
    # SESSION STATE INIT — explicit defaults on every cold start
    # ═══════════════════════════════════════════════════════════════════════════
    st.session_state.setdefault('analysis_started', False)
    st.session_state.setdefault('active_predictors', None)

    # ═══════════════════════════════════════════════════════════════════════════
    # SIDEBAR — content differs between landing and analysis states
    # ═══════════════════════════════════════════════════════════════════════════
    analysis_started = st.session_state['analysis_started']

    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; margin-bottom: 1rem;">
            <div style="font-size: 1.75rem; font-weight: 800; color: var(--primary-color);">ARTHAGATI</div>
            <div style="color: var(--text-muted); font-size: 0.75rem; margin-top: 0.25rem;">अर्थगति | Market Sentiment</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        if not analysis_started:
            # ── Landing sidebar: Run Analysis is the only action ──────
            st.markdown('<div class="sidebar-title">▶ Start</div>', unsafe_allow_html=True)
            if st.button("▶  Run Analysis", use_container_width=True, type="primary"):
                st.session_state['analysis_started'] = True
                st.rerun()
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

    # ── Landing page in main area ─────────────────────────────────────
    if not analysis_started:
        render_landing_page()
        return

    # ── Past this point: analysis is running ─────────────────────────

    # ═══════════════════════════════════════════════════════════════════════════
    # LOAD DATA FIRST — needed to populate dynamic predictor options
    # ═══════════════════════════════════════════════════════════════════════════
    _prog = st.empty()
    _progress_bar(_prog, 5, "Fetching market data", "Google Sheets · gviz API · CSV decode")
    raw_df = load_data()

    if raw_df is None:
        _prog.empty()
        st.stop()

    available_predictors = [
        col for col in raw_df.columns
        if col not in NON_PREDICTOR_COLS and pd.api.types.is_numeric_dtype(raw_df[col])
    ]

    # Initialize or validate session-state predictors against actual columns
    current_preds = st.session_state.get('active_predictors')
    if not current_preds:
        st.session_state['active_predictors'] = tuple(available_predictors)
    else:
        valid = tuple(p for p in current_preds if p in available_predictors)
        st.session_state['active_predictors'] = valid if valid else tuple(available_predictors)

    # ═══════════════════════════════════════════════════════════════════════════
    # SIDEBAR — analysis controls
    # ═══════════════════════════════════════════════════════════════════════════
    with st.sidebar:
        view_mode = st.radio(
            "View Mode",
            ["Historical Mood", "Similar Periods", "Correlation Analysis"],
            label_visibility="collapsed"
        )
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-title"><svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--primary-color)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block;vertical-align:middle;margin-right:4px;"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>Controls</div>', unsafe_allow_html=True)
        if st.button("Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # ── Model Configuration ──
        st.markdown('<div class="sidebar-title"><svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--primary-color)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block;vertical-align:middle;margin-right:4px;"><path d="M12 2a8 8 0 0 0-8 8c0 1.892.783 3.63 2.046 4.912"/><path d="M12 18a8 8 0 0 0 8-8c0-1.892-.783-3.63-2.046-4.912"/><path d="M12 22a8 8 0 0 0 8-8c0-1.892-.783-3.63-2.046-4.912"/><path d="M12 2a8 8 0 0 0-8 8c0 1.892.783 3.63 2.046 4.912"/></svg>Model Configuration</div>', unsafe_allow_html=True)

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
                st.warning("Select at least one predictor.")
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
    
    # ── Data Staleness Check ────────────────────────────────────────────
    latest_date = raw_df['DATE'].max()
    ist_tz = pytz.timezone('Asia/Kolkata')
    today_ist = datetime.now(ist_tz).date()
    data_age_days = (pd.Timestamp(today_ist) - latest_date).days
    
    # >3 days gap (accounts for weekends: Fri data on Mon = 3 days, fine)
    if data_age_days > 3:
        st.markdown(f"""
        <div style="background: rgba(239,68,68,0.1); border: 1px solid var(--danger-red); border-radius: 10px; 
                    padding: 0.75rem 1.25rem; margin-bottom: 1rem; display: flex; align-items: center; gap: 12px;">
            <span style="font-size: 1.4rem;"><svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--warning-amber)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21.33h8a2 2 0 0 0 1.92-1.45L12 15"/><path d="M12 9v4"/><path d="M12 17h.01"/></svg></span>
            <div>
                <span style="color: var(--danger-red); font-weight: 700;">Stale Data</span>
                <span style="color: #888; font-size: 0.85rem;"> — Last data point is <b>{latest_date.strftime('%d %b %Y')}</b> ({data_age_days} days ago). 
                Scores reflect the last available data, not current market state. Update your Google Sheet.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    _progress_bar(_prog, 40, "Computing correlations", "Decay-weighted Spearman · PE & EY anchors")
    selected_preds = st.session_state.get('active_predictors', tuple(available_predictors))

    _progress_bar(_prog, 65, "Running sentiment engine", "OU normalization · Kalman smoothing · 5-layer pipeline")
    mood_df = calculate_historical_mood(raw_df, dependent_vars=selected_preds)

    if mood_df.empty:
        _prog.empty()
        st.error("Failed to calculate mood scores.")
        st.stop()

    _progress_bar(_prog, 88, "Computing MSF spread", "Momentum · Structure · Regime · Flow · inverse-variance weights")
    msf_df = calculate_msf_spread(mood_df)
    mood_df['MSF_Spread'] = msf_df['msf_spread'].values if not msf_df.empty else 0

    _progress_bar(_prog, 100, "Ready", "All systems nominal")
    time.sleep(0.25)
    _prog.empty()
    
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
    
    if view_mode == "Historical Mood":
        render_historical_mood(mood_df, msf_df)
    elif view_mode == "Similar Periods":
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
            <h3 style="color: var(--primary-color); margin: 0;"><svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--primary-color)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block;vertical-align:middle;margin-right:8px;"><path d="m3 3 18 18"/><path d="m18 9-5 5-4-4-3 3"/></svg>Market Mood Terminal</h3>
            <p style="color: var(--text-muted); font-size: 0.85rem; margin: 0;">TradingView-Style Analysis • Mood Score + MSF Spread Indicator</p>
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
    
    # ── Dynamic Y-Bounds for Mood Pane ──────────────────────────────────
    # Compute actual data extent (mood scores, confidence bands, OU projection)
    # so the chart scales tightly to visible data instead of fixed ±100.
    _y_candidates = [df['Mood_Score'].values]
    if 'Confidence_Upper' in df.columns:
        _y_candidates.append(df['Confidence_Upper'].values)
    if 'Confidence_Lower' in df.columns:
        _y_candidates.append(df['Confidence_Lower'].values)
    _y_candidates.append(proj_mood)
    _y_all = np.concatenate([c[np.isfinite(c)] for c in _y_candidates])
    _y_min = float(_y_all.min()) if len(_y_all) > 0 else -100
    _y_max = float(_y_all.max()) if len(_y_all) > 0 else 100
    _y_pad = max((_y_max - _y_min) * 0.08, 2.0)  # 8% padding, minimum 2 pts
    mood_y_lo = _y_min - _y_pad
    mood_y_hi = _y_max + _y_pad

    # ── Regime Transition Markers (WebGL — avoids SVG DOM bloat) ────────
    # Group transitions by colour, build interleaved x/y arrays with None
    # separators, render as single Scattergl traces per colour.
    if 'Regime' in df.columns:
        regimes = df['Regime'].values
        dates   = df['DATE'].values
        transition_groups = {}  # color → (x_list, y_list)
        for i in range(1, len(regimes)):
            if regimes[i] != regimes[i - 1] and regimes[i] != 'Unknown':
                color = REGIME_STYLES.get(regimes[i], (C_MUTED, 'neutral'))[0]
                if color not in transition_groups:
                    transition_groups[color] = ([], [])
                xg, yg = transition_groups[color]
                xg.extend([dates[i], dates[i], None])
                yg.extend([mood_y_lo, mood_y_hi, None])

        for color, (xg, yg) in transition_groups.items():
            fig.add_trace(
                go.Scattergl(
                    x=xg, y=yg,
                    mode='lines',
                    line=dict(color=color, width=1, dash='dot'),
                    opacity=0.5,
                    showlegend=False,
                    hoverinfo='skip',
                ),
                row=1, col=1
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
    mood_series = df['Mood_Score']
    msf_series = pd.Series(msf_values, index=df.index)
    
    # Local extrema over 'lookback' window (vectorised)
    roll_mood_min = mood_series.rolling(lookback + 1, min_periods=1).min()
    roll_mood_max = mood_series.rolling(lookback + 1, min_periods=1).max()
    roll_msf_min = msf_series.rolling(lookback + 1, min_periods=1).min()
    roll_msf_max = msf_series.rolling(lookback + 1, min_periods=1).max()
    
    # Previous windows' extrema using shift
    prev_mood_min, prev_msf_min = roll_mood_min.shift(lookback), roll_msf_min.shift(lookback)
    prev_mood_max, prev_msf_max = roll_mood_max.shift(lookback), roll_msf_max.shift(lookback)
    
    # Vectorised boolean masks for conditions
    bearish_mask = (mood_series == roll_mood_min) & (mood_series < prev_mood_min) & (roll_msf_min > prev_msf_min)
    bullish_mask = (mood_series == roll_mood_max) & (mood_series > prev_mood_max) & (roll_msf_max < prev_msf_max)
    
    # Enforce index boundaries (equivalent to i > lookback*2 and i < len(df) - 1)
    valid_indices = np.zeros(len(df), dtype=bool)
    valid_indices[lookback * 2 : len(df) - 1] = True
    
    red_signals = np.where(bearish_mask & valid_indices)[0]
    green_signals = np.where(bullish_mask & valid_indices)[0]
    
    # Red triangles at y=5 (bearish divergence, top)
    if len(red_signals) > 0:
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
    if len(green_signals) > 0:
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
            showgrid=True, gridcolor=C_BG_GRID, zeroline=False,
            range=[mood_y_hi, mood_y_lo],  # reversed: high at bottom, low at top
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
            <h4 style="color: var(--info-cyan); margin: 0;">MSF Component Breakdown</h4>
            <p style="color: #888; font-size: 0.8rem; margin: 0;">Current contribution of each component to the MSF Spread reading · Weights are inverse-variance (auto-calibrated)</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Get latest MSF component values
    msf_latest_idx = min(len(msf_filtered) - 1, len(df) - 1)
    if msf_latest_idx >= 0 and not msf_filtered.empty:
        comp_names = ['momentum', 'structure', 'regime', 'flow']
        comp_labels = ['Momentum', 'Structure', 'Regime', 'Flow']
        comp_colors = ['var(--warning-amber)', '#a78bfa', 'var(--success-green)', 'var(--info-cyan)']
        comp_icons = ['<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m5 12 7-7 7 7"/><path d="M12 19V5"/></svg>', '<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="16" width="20" height="8" rx="2"/><rect x="4" y="8" width="20" height="8" rx="2"/></svg>', '<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg>', '<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M2 6c0 1.1.9 2 2 2s2-.9 2-2"/><path d="M2 12c0 1.1.9 2 2 2s2-.9 2-2"/><path d="M2 18c0 1.1.9 2 2 2s2-.9 2-2"/><path d="M22 20c-2.5 0-4.5-2.1-4.5-5"/><path d="M12 14c-2.5 0-4.5-2.1-4.5-5"/><path d="M12 8c-2.5 0-4.5-2.1-4.5-5"/></svg>']
        
        c_cols = st.columns(4)
        for j, (name, label, color, icon) in enumerate(zip(comp_names, comp_labels, comp_colors, comp_icons)):
            val = msf_filtered[name].iloc[msf_latest_idx] if name in msf_filtered.columns else 0
            # Compute period average for context
            period_val = msf_filtered[name].mean() if name in msf_filtered.columns else 0
            
            bar_pct = (val + 10) / 20 * 100  # Map [-10, +10] → [0%, 100%]
            bar_pct = max(0, min(100, bar_pct))
            
            with c_cols[j]:
                st.markdown(f"""
                <div style="background: var(--bg-card); border-radius: 10px; padding: 0.75rem; border: 1px solid var(--border-color);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                        <span style="font-size: 0.75rem; color: #888; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">{icon} {label}</span>
                        <span style="font-size: 1.1rem; font-weight: 700; color: {color};">{val:+.1f}</span>
                    </div>
                    <div style="height: 6px; background: var(--border-color); border-radius: 3px; position: relative;">
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
            <h3 style="color: var(--primary-color); margin: 0;"><svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--primary-color)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block;vertical-align:middle;margin-right:8px;"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>Similar Historical Periods</h3>
            <p style="color: var(--text-muted); font-size: 0.85rem; margin: 0;">Mahalanobis + trajectory matching · Forward NIFTY returns from each analog</p>
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
            ret_color = 'var(--success-green)' if median_ret > 0 else 'var(--danger-red)'
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
                    fwd_color = 'var(--success-green)' if val > 0 else 'var(--danger-red)'
                    fwd_badges += f'<span style="font-size:0.7rem; color:{fwd_color}; margin-left:8px;">+{horizon}d: <b>{val:+.1f}%</b></span>'
                else:
                    fwd_badges += f'<span style="font-size:0.7rem; color:#555; margin-left:8px;">+{horizon}d: —</span>'
            
            st.markdown(f"""
            <div class="signal-card {mood_class}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <span style="font-weight: 700; color: var(--text-primary);">{period['date']}</span>
                    <span class="status-badge {mood_class}">{period['mood']}</span>
                </div>
                <div style="display: flex; justify-content: space-between; color: var(--text-muted); font-size: 0.85rem;">
                    <span>Similarity: <b style="color: var(--primary-color);">{similarity_pct:.1f}%</b></span>
                    <span>Mood: <b>{mood_val:.1f}</b></span>
                    <span>NIFTY: <b>{period['nifty']:,.0f}</b></span>
                </div>
                <div style="margin-top: 0.4rem; padding-top: 0.4rem; border-top: 1px solid var(--border-color);">
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
            <h3 style="color: var(--primary-color); margin: 0;"><svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--primary-color)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block;vertical-align:middle;margin-right:8px;"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg>Backtest: Mood Score vs Forward NIFTY Return</h3>
            <p style="color: var(--text-muted); font-size: 0.85rem; margin: 0;">
                Does today's mood score predict tomorrow's market? Each dot = one historical day.
                If there's a relationship, the scatter should show a pattern.
            </p>
            <p style="color: var(--danger-red); font-size: 0.75rem; margin-top: 0.5rem; font-weight: 600; padding: 6px; background: rgba(239, 68, 68, 0.1); border-radius: 4px;">
                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--danger-red)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block;vertical-align:middle;margin-right:4px;"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21.33h8a2 2 0 0 0 1.92-1.45L12 15"/><path d="M12 9v4"/><path d="M12 17h.01"/></svg>Note: This view represents a Hindsight Regime Fit. Historical points are evaluated using parameters learned from today's active correlation regime.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Compute mood_score at T vs NIFTY return at T+30 for all historical points
    n = len(mood_df)
    horizon = BACKTEST_HORIZON
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
            from scipy.stats import spearmanr as _spearmanr

            # 70/30 train/test split (chronological — no shuffling)
            split_idx = int(len(bt_mood_clean) * 0.7)
            train_m, train_r = bt_mood_clean[:split_idx], bt_fwd_clean[:split_idx]
            test_m, test_r = bt_mood_clean[split_idx:], bt_fwd_clean[split_idx:]

            # In-sample correlations (train)
            bt_pearson = np.corrcoef(train_m, train_r)[0, 1] if len(train_m) > 2 else 0
            bt_spearman, _ = _spearmanr(train_m, train_r)
            if not np.isfinite(bt_spearman):
                bt_spearman = 0.0

            # Out-of-sample correlations (test)
            oos_pearson = np.corrcoef(test_m, test_r)[0, 1] if len(test_m) > 2 else 0
            oos_spearman, _ = _spearmanr(test_m, test_r) if len(test_m) > 2 else (0.0, 1.0)
            if not np.isfinite(oos_spearman):
                oos_spearman = 0.0

            colors = np.where(bt_mood_clean > 0, C_GREEN, C_RED)

            fig_bt = go.Figure()

            # Train points (circles)
            fig_bt.add_trace(go.Scattergl(
                x=train_m, y=train_r,
                mode='markers',
                marker=dict(size=4, color=np.where(train_m > 0, C_GREEN, C_RED), opacity=0.4),
                hovertemplate='Mood: %{x:.1f}<br>+30d Return: %{y:.1f}%<extra></extra>',
                name=f'Train (70%, n={len(train_m)})',
            ))
            # Test points (diamonds, brighter)
            fig_bt.add_trace(go.Scattergl(
                x=test_m, y=test_r,
                mode='markers',
                marker=dict(size=5, color=np.where(test_m > 0, C_GREEN, C_RED), opacity=0.8, symbol='diamond'),
                hovertemplate='Mood: %{x:.1f}<br>+30d Return: %{y:.1f}%<extra></extra>',
                name=f'Test (30%, n={len(test_m)})',
            ))

            if len(train_m) > 10:
                x_range = np.linspace(bt_mood_clean.min(), bt_mood_clean.max(), 50)

                # Linear fit on TRAIN data only
                z1 = np.polyfit(train_m, train_r, 1)
                fig_bt.add_trace(go.Scatter(
                    x=x_range, y=z1[0] * x_range + z1[1],
                    mode='lines', line=dict(color=C_PRIMARY, width=2, dash='dash'),
                    name=f'Linear (train ρ={bt_pearson:.2f}, test ρ={oos_pearson:.2f})', showlegend=True,
                ))

                # Quadratic fit on TRAIN data only
                z2 = np.polyfit(train_m, train_r, 2)
                fig_bt.add_trace(go.Scatter(
                    x=x_range, y=z2[0] * x_range ** 2 + z2[1] * x_range + z2[2],
                    mode='lines', line=dict(color=C_CYAN, width=2, dash='dot'),
                    name=f'Quadratic (train ρ_s={bt_spearman:.2f}, test ρ_s={oos_spearman:.2f})', showlegend=True,
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
                    font=dict(size=10),
                ),
            )

            st.plotly_chart(fig_bt, config={'displayModeBar': False})

            # Interpretation — report both in-sample and out-of-sample
            oos_stronger = oos_spearman if abs(oos_spearman) > abs(oos_pearson) else oos_pearson

            if abs(oos_stronger) > 0.3:
                strength = 'strong' if abs(oos_stronger) > 0.5 else 'moderate'
                direction = 'positive' if oos_stronger > 0 else 'negative'
                st.markdown(f"""
                <div class="info-box">
                    <b>Out-of-sample (30%): Pearson {oos_pearson:.2f} · Spearman {oos_spearman:.2f}</b> — {strength} {direction} relationship holds on unseen data.<br>
                    <span style="color:#666;">In-sample (70%): Pearson {bt_pearson:.2f} · Spearman {bt_spearman:.2f}</span><br>
                    {'Higher mood scores have historically been followed by positive NIFTY returns.' if oos_stronger > 0 else 'Higher mood scores have historically been followed by negative NIFTY returns (contrarian signal).'}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="info-box">
                    <b>Out-of-sample (30%): Pearson {oos_pearson:.2f} · Spearman {oos_spearman:.2f}</b> — weak out-of-sample relationship at 30-day horizon.<br>
                    <span style="color:#666;">In-sample (70%): Pearson {bt_pearson:.2f} · Spearman {bt_spearman:.2f}</span><br>
                    The mood score's predictive power may be non-linear (check the quadratic curve) or work better at different horizons.
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
            <h3 style="color: var(--primary-color); margin: 0;"><svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--primary-color)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block;vertical-align:middle;margin-right:8px;"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/><path d="M16 13H8"/><path d="M16 17H8"/><path d="M10 9H8"/></svg>Correlation & Predictor Analysis</h3>
            <p style="color: var(--text-muted); font-size: 0.85rem; margin: 0;">Decay-weighted Spearman correlations with PE and EY anchors · Predictor quality assessment</p>
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
        <div class="info-box" style="border-left: 4px solid var(--danger-red);">
            <h4 style="color: var(--danger-red);"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--danger-red)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block;vertical-align:middle;margin-right:6px;"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21.33h8a2 2 0 0 0 1.92-1.45L12 15"/><path d="M12 9v4"/><path d="M12 17h.01"/></svg>Data Quality Issue</h4>
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
                st.caption(f"{anchor_col} has insufficient data variance — correlations may be unreliable.")
            
            corrs = calculate_anchor_correlations(raw_df, anchor_col, active_preds)
            if corrs.empty:
                st.caption("No correlations computed. Check data source.")
                return corrs
            
            corrs_display = corrs.sort_values('correlation', key=abs, ascending=False)
            for _, row in corrs_display.iterrows():
                corr_val = row['correlation']
                color = 'var(--success-green)' if corr_val > 0 else 'var(--danger-red)'
                bar_width = abs(corr_val) * 100
                strength_dot = '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 24 24" fill="var(--success-green)"><circle cx="12" cy="12" r="10"/></svg>' if abs(corr_val) >= 0.5 else '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 24 24" fill="var(--warning-amber)"><circle cx="12" cy="12" r="10"/></svg>' if abs(corr_val) >= 0.3 else '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 24 24" fill="var(--text-muted)"><circle cx="12" cy="12" r="10"/></svg>'
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem; padding: 0.5rem; background: var(--bg-card); border-radius: 8px;">
                    <span style="width: 14px; font-size: 0.6rem;">{strength_dot}</span>
                    <span style="width: 130px; font-size: 0.8rem; color: var(--text-primary);">{row['variable']}</span>
                    <div style="flex: 1; height: 8px; background: var(--border-color); border-radius: 4px; margin: 0 10px;">
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
            <h3 style="color: var(--primary-color); margin: 0;"><svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--primary-color)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block;vertical-align:middle;margin-right:8px;"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>Predictor Quality Assessment</h3>
            <p style="color: var(--text-muted); font-size: 0.85rem; margin: 0;">
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
                rec_color = 'var(--success-green)'
            elif row['quality'] >= max_quality * 0.2 and row['coverage'] > 30:
                rec = '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 24 24" fill="var(--warning-amber)" style="display:inline-block;vertical-align:middle;margin-right:4px;"><circle cx="12" cy="12" r="10"/></svg>USEFUL'
                rec_color = 'var(--warning-amber)'
            elif row['coverage'] < 10:
                rec = '❌ NO DATA'
                rec_color = 'var(--danger-red)'
            else:
                rec = '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 24 24" fill="var(--text-muted)" style="display:inline-block;vertical-align:middle;margin-right:4px;"><circle cx="12" cy="12" r="10"/></svg>WEAK'
                rec_color = 'var(--text-muted)'
            
            active_badge = '● Active' if row['active'] else '○ Inactive'
            active_color = 'var(--primary-color)' if row['active'] else '#555555'
            
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin-bottom: 0.4rem; padding: 0.6rem 0.75rem; background: var(--bg-card); border-radius: 8px; border: 1px solid {'var(--border-color)' if row['active'] else 'var(--bg-card)'};">
                <span style="width: 24px; font-size: 0.75rem; color: #555; font-weight: 700;">{rank}</span>
                <span style="width: 140px; font-size: 0.8rem; color: var(--text-primary); font-weight: 600;">{row['variable']}</span>
                <div style="flex: 1; height: 6px; background: var(--border-color); border-radius: 3px; margin: 0 12px;">
                    <div style="width: {bar_pct:.0f}%; height: 100%; background: linear-gradient(90deg, var(--primary-color), var(--warning-amber)); border-radius: 3px;"></div>
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
                <b style="color: var(--success-green);">✅ {keep_count} strong</b> predictors (high correlation × low entropy) ·
                <b style="color: var(--warning-amber);"><svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 24 24" fill="var(--warning-amber)" style="display:inline-block;vertical-align:middle;margin-right:4px;"><circle cx="12" cy="12" r="10"/></svg>{useful_count} useful</b> (moderate signal) ·
                <b style="color: #888;"><svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 24 24" fill="var(--text-muted)" style="display:inline-block;vertical-align:middle;margin-right:4px;"><circle cx="12" cy="12" r="10"/></svg>{weak_count} weak</b> (low signal or noisy)<br>
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

main()
