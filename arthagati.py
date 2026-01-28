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

VERSION = "v1.2.0"
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
def calculate_anchor_correlations(df, anchor):
    """Calculate correlations between anchor and dependent variables."""
    cols_to_check = [col for col in DEPENDENT_VARS if col in df.columns]
    
    if anchor not in df.columns or not cols_to_check:
        return pd.DataFrame(columns=['variable', 'correlation', 'strength', 'type'])
    
    analysis_df = df[[anchor] + cols_to_check].select_dtypes(include=[np.number])
    
    if anchor not in analysis_df.columns:
        return pd.DataFrame(columns=['variable', 'correlation', 'strength', 'type'])

    corr_matrix = analysis_df.corr(method='pearson')
    anchor_corrs = corr_matrix[anchor].drop(anchor, errors='ignore')
    
    correlations = []
    for var, corr in anchor_corrs.items():
        if pd.isna(corr):
            corr = 0.0
            
        strength = ('Strong' if abs(corr) >= 0.7 else 
                   'Moderate' if abs(corr) >= 0.5 else 
                   'Weak' if abs(corr) >= 0.3 else 'Very weak')
        
        correlations.append({
            'variable': var,
            'correlation': corr,
            'strength': strength,
            'type': 'positive' if corr > 0 else 'negative'
        })
    
    return pd.DataFrame(correlations)

@st.cache_data
def calculate_historical_mood(df):
    """Calculate historical mood scores."""
    start_time = time.time()
    if 'DATE' not in df.columns or 'NIFTY50_PE' not in df.columns or 'NIFTY50_EY' not in df.columns:
        logging.error("Required columns missing.")
        return pd.DataFrame(columns=['DATE', 'Mood_Score', 'Mood', 'Smoothed_Mood_Score', 'Mood_Volatility'])
    
    pe_corrs = calculate_anchor_correlations(df, 'NIFTY50_PE')
    ey_corrs = calculate_anchor_correlations(df, 'NIFTY50_EY')
    
    pe_weights = {row['variable']: abs(row['correlation']) for _, row in pe_corrs.iterrows()}
    ey_weights = {row['variable']: abs(row['correlation']) for _, row in ey_corrs.iterrows()}
    
    pe_total_weight = max(sum(pe_weights.values()), 1e-10)
    ey_total_weight = max(sum(ey_weights.values()), 1e-10)
    
    pe_weights = {k: v/pe_total_weight for k, v in pe_weights.items()}
    ey_weights = {k: v/ey_total_weight for k, v in ey_weights.items()}
    
    pe_percentiles = df['NIFTY50_PE'].expanding().rank(pct=True, method='max').values
    ey_percentiles = df['NIFTY50_EY'].expanding().rank(pct=True, method='max').values
    
    pe_base = -1 + 2 * (1 - pe_percentiles)
    ey_base = -1 + 2 * ey_percentiles
    
    pe_adjustments = np.zeros(len(df))
    ey_adjustments = np.zeros(len(df))
    
    vars_to_process = [col for col in DEPENDENT_VARS if col in df.columns]
    
    for var in vars_to_process:
        var_percentiles = df[var].expanding().rank(pct=True, method='max').values
        
        if var in pe_weights:
            pe_type = pe_corrs.loc[pe_corrs['variable'] == var, 'type'].iloc[0] if len(pe_corrs.loc[pe_corrs['variable'] == var]) > 0 else 'positive'
            weight = pe_weights.get(var, 0)
            if pe_type == 'positive':
                pe_adjustments += weight * (1 - var_percentiles)
            elif pe_type == 'negative':
                pe_adjustments -= weight * (1 - var_percentiles)
        
        if var in ey_weights:
            ey_type = ey_corrs.loc[ey_corrs['variable'] == var, 'type'].iloc[0] if len(ey_corrs.loc[ey_corrs['variable'] == var]) > 0 else 'positive'
            weight = ey_weights.get(var, 0)
            if ey_type == 'positive':
                ey_adjustments += weight * var_percentiles
            elif ey_type == 'negative':
                ey_adjustments -= weight * var_percentiles
    
    pe_scores = 0.5 * pe_base + 0.5 * pe_adjustments
    ey_scores = 0.5 * ey_base + 0.5 * ey_adjustments
    pe_scores = np.clip(pe_scores, -1, 1)
    ey_scores = np.clip(ey_scores, -1, 1)
    
    pe_corr_strength = sum(abs(row['correlation']) for _, row in pe_corrs.iterrows())
    ey_corr_strength = sum(abs(row['correlation']) for _, row in ey_corrs.iterrows())
    total_strength = pe_corr_strength + ey_corr_strength or 1
    pe_weight = pe_corr_strength / total_strength
    ey_weight = ey_corr_strength / total_strength
    
    raw_mood_scores = pe_weight * pe_scores + ey_weight * ey_scores
    
    mean_score = np.mean(raw_mood_scores)
    std_score = np.std(raw_mood_scores) or 1
    mood_scores = (raw_mood_scores - mean_score) / std_score * 30
    mood_scores = np.clip(mood_scores, -100, 100)
    
    mood_series = pd.Series(mood_scores)
    smoothed_mood_scores = mood_series.rolling(window=7, min_periods=1).mean()
    mood_volatility = mood_series.rolling(window=30, min_periods=1).std().fillna(0)
    
    # Classify mood using vectorized approach
    moods = np.where(mood_scores > 60, 'Very Bullish',
            np.where(mood_scores > 20, 'Bullish',
            np.where(mood_scores > -20, 'Neutral',
            np.where(mood_scores > -60, 'Bearish', 'Very Bearish'))))
    
    result_df = pd.DataFrame({
        'DATE': df['DATE'].values,
        'Mood_Score': mood_scores,
        'Mood': moods,
        'Smoothed_Mood_Score': smoothed_mood_scores.values,
        'Mood_Volatility': mood_volatility.values,
        'NIFTY': df['NIFTY'].values,
        'AD_RATIO': df['AD_RATIO'].values if 'AD_RATIO' in df.columns else 1.0
    })
    
    logging.info(f"Historical mood calculated in {time.time() - start_time:.2f} seconds.")
    return result_df

# ══════════════════════════════════════════════════════════════════════════════
# MSF-ENHANCED SPREAD INDICATOR
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def calculate_msf_spread(df, mood_col='Mood_Score', nifty_col='NIFTY', breadth_col='AD_RATIO'):
    """
    MSF-Enhanced Spread Indicator.
    Combines momentum, structure, regime, and flow components.
    """
    start_time = time.time()
    
    length = 20
    roc_len = 14
    clip = 3.0
    
    result = pd.DataFrame(index=df.index)
    
    mood = df[mood_col].values if mood_col in df.columns else np.zeros(len(df))
    nifty = df[nifty_col].values if nifty_col in df.columns else mood
    breadth = df[breadth_col].values if breadth_col in df.columns else np.ones(len(df))
    
    mood_series = pd.Series(mood, index=df.index)
    nifty_series = pd.Series(nifty, index=df.index)
    breadth_series = pd.Series(breadth, index=df.index)
    
    if len(mood) == 0:
        logging.error("Empty data for MSF Spread calculation.")
        return result
    
    # Component 1: Momentum (ROC z-score of NIFTY)
    roc_raw = nifty_series.pct_change(roc_len, fill_method=None)
    roc_z = zscore_clipped(roc_raw, length, clip)
    momentum_norm = sigmoid(roc_z, 1.5)
    
    # Component 2: Structure (Mood trend)
    trend_fast = mood_series.rolling(5, min_periods=1).mean()
    trend_slow = mood_series.rolling(length, min_periods=1).mean()
    trend_diff_z = zscore_clipped(trend_fast - trend_slow, length, clip)
    mood_accel_raw = mood_series.diff(5).diff(5)
    mood_accel_z = zscore_clipped(mood_accel_raw, length, clip)
    structure_z = (trend_diff_z + mood_accel_z) / np.sqrt(2.0)
    structure_norm = sigmoid(structure_z, 1.5)
    
    # Component 3: Regime Count
    pct_change = nifty_series.pct_change(fill_method=None)
    threshold = 0.0033
    regime_signals = np.select(
        [pct_change > threshold, pct_change < -threshold],
        [1, -1],
        default=0
    )
    regime_count = pd.Series(regime_signals, index=df.index).cumsum()
    regime_raw = regime_count - regime_count.rolling(length, min_periods=1).mean()
    regime_z = zscore_clipped(regime_raw, length, clip)
    regime_norm = sigmoid(regime_z, 1.5)
    
    # Component 4: Breadth Flow
    breadth_ma = breadth_series.rolling(length, min_periods=1).mean()
    breadth_ratio = breadth_series / breadth_ma.replace(0, 1)
    breadth_z = zscore_clipped(breadth_ratio - 1, length, clip)
    flow_norm = sigmoid(breadth_z, 1.5)
    
    # Combine: Momentum(30%) + Structure(25%) + Regime(25%) + Flow(20%)
    msf_raw = (
        0.30 * momentum_norm +
        0.25 * structure_norm +
        0.25 * regime_norm +
        0.20 * flow_norm
    )
    
    msf_spread = msf_raw * 10
    
    result['msf_spread'] = msf_spread
    result['momentum'] = momentum_norm * 10
    result['structure'] = structure_norm * 10
    result['regime'] = regime_norm * 10
    result['flow'] = flow_norm * 10
    
    logging.info(f"MSF Spread calculated in {time.time() - start_time:.2f} seconds.")
    return result

# ══════════════════════════════════════════════════════════════════════════════
# SIMILAR PERIODS FINDER
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def find_similar_periods(df, top_n=10, recency_weight=0.1):
    """Find historically similar market periods."""
    if df.empty or 'Mood_Score' not in df.columns:
        return []
    
    latest = df.iloc[-1]
    current_mood = latest['Mood_Score']
    current_volatility = latest['Mood_Volatility']
    
    historical = df.iloc[:-30].copy() if len(df) > 30 else df.iloc[:-1].copy()
    if historical.empty:
        return []
    
    historical['mood_diff'] = abs(historical['Mood_Score'] - current_mood)
    historical['vol_diff'] = abs(historical['Mood_Volatility'] - current_volatility)
    
    max_mood_diff = historical['mood_diff'].max() or 1
    max_vol_diff = historical['vol_diff'].max() or 1
    
    historical['mood_sim'] = 1 - (historical['mood_diff'] / max_mood_diff)
    historical['vol_sim'] = 1 - (historical['vol_diff'] / max_vol_diff)
    
    days_since = (latest['DATE'] - historical['DATE']).dt.days
    max_days = days_since.max() or 1
    historical['recency_bonus'] = recency_weight * (1 - days_since / max_days)
    
    historical['similarity'] = 0.6 * historical['mood_sim'] + 0.3 * historical['vol_sim'] + historical['recency_bonus']
    
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
        
        st.markdown('<p class="sidebar-title">⚙️ Controls</p>', unsafe_allow_html=True)
        
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown('<p class="sidebar-title">📊 View Mode</p>', unsafe_allow_html=True)
        view_mode = st.radio(
            "Select View",
            ["📈 Historical Mood", "🔍 Similar Periods", "📋 Correlation Analysis"],
            label_visibility="collapsed"
        )
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem 0; color: #888888; font-size: 0.7rem;">
            <div style="color: #FFC300; font-weight: 600;">{COMPANY}</div>
            <div style="margin-top: 0.25rem;">{PRODUCT_NAME} {VERSION}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # HEADER
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown(f"""
        <div class="premium-header">
            <h1>ARTHAGATI : Market Sentiment Analysis</h1>
            <div class="tagline">Quantitative Market Mood & MSF-Enhanced Indicators</div>
        </div>
    """, unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DATA LOADING
    # ═══════════════════════════════════════════════════════════════════════════
    with st.spinner("Loading market data..."):
        raw_df = load_data()
    
    if raw_df is None:
        st.stop()
    
    with st.spinner("Calculating mood scores..."):
        mood_df = calculate_historical_mood(raw_df)
    
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
        line=dict(color="#3A3A3A", width=1)
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
        pe_corrs = calculate_anchor_correlations(raw_df, 'NIFTY50_PE')
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
        ey_corrs = calculate_anchor_correlations(raw_df, 'NIFTY50_EY')
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
