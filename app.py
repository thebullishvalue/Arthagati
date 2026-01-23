"""
ARTHAGATI (अर्थगति) - Sentiment Intelligence | A Hemrek Capital Product
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Quantitative market mood analysis using PE/EY correlation metrics.
Historical sentiment tracking with similar period detection.
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
VERSION = "v1.1.0"
PRODUCT_NAME = "Arthagati"
COMPANY = "Hemrek Capital"

# --- GOOGLE SHEETS CONFIGURATION ---
# Replace this with your Google Sheet ID (found in the URL between /d/ and /edit)
SHEET_ID = "1po7z42n3dYIQGAvn0D1-a4pmyxpnGPQ13TrNi3DB5_c" 
# The GID determines which tab to fetch. '0' is usually the first tab.
SHEET_GID = "0" 

# Constants
EXPECTED_COLUMNS = [
    'DATE', 'NIFTY', 'AD_RATIO', 'REL_AD_RATIO', 'REL_BREADTH', 'BREADTH', 'COUNT', 'NIFTY50_PE', 'NIFTY50_EY',
    'NIFTY50_DY', 'NIFTY50_PB', 'IN10Y', 'IN02Y', 'IN30Y', 'INIRYY', 'REPO',
    'CRR', 'US02Y', 'US10Y', 'US30Y', 'US_FED', 'PE_DEV', 'EY_DEV'
]
DEPENDENT_VARS = [
    'AD_RATIO', 'REL_AD_RATIO', 'REL_BREADTH','BREADTH', 'COUNT', 'IN10Y', 'IN02Y', 'IN30Y',
    'INIRYY', 'REPO', 'CRR', 'US02Y', 'US30Y', 'US10Y', 'US_FED', 'NIFTY50_DY',
    'NIFTY50_PB', 'PE_DEV', 'EY_DEV'
]

# Streamlit page configuration
st.set_page_config(
    page_title="ARTHAGATI | Sentiment Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Premium Professional CSS (Hemrek Capital Design System) ---
def load_css():
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
            --success-dark: #059669;
            --danger-red: #ef4444;
            --danger-dark: #dc2626;
            --warning-amber: #f59e0b;
            --info-cyan: #06b6d4;
            
            --neutral: #888888;
        }
        
        * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        .main, [data-testid="stSidebar"] {
            background-color: var(--background-color);
            color: var(--text-primary);
        }
        
        .stApp > header {
            background-color: transparent;
        }
        
        #MainMenu {visibility: hidden;} footer {visibility: hidden;}
        
        .block-container {
            padding-top: 3.5rem;
            max-width: 90%; 
            padding-left: 2rem; 
            padding-right: 2rem;
        }
        
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
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: radial-gradient(circle at 20% 50%, rgba(var(--primary-rgb),0.08) 0%, transparent 50%);
            pointer-events: none;
        }
        
        .premium-header h1 {
            margin: 0;
            font-size: 2rem;
            font-weight: 700;
            color: var(--text-primary);
            letter-spacing: -0.50px;
            position: relative;
        }
        
        .premium-header .tagline {
            color: var(--text-muted);
            font-size: 0.9rem;
            margin-top: 0.25rem;
            font-weight: 400;
            position: relative;
        }
        
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
            height: 100%;
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(0,0,0,0.3);
            border-color: var(--border-light);
        }
        
        .metric-card h4 {
            color: var(--text-muted);
            font-size: 0.75rem;
            margin-bottom: 0.5rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .metric-card h2 {
            color: var(--text-primary);
            font-size: 1.75rem;
            font-weight: 700;
            margin: 0;
            line-height: 1;
        }
        
        .metric-card .sub-metric {
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-top: 0.5rem;
            font-weight: 500;
        }
        
        .metric-card.success h2 { color: var(--success-green); }
        .metric-card.danger h2 { color: var(--danger-red); }
        .metric-card.warning h2 { color: var(--warning-amber); }
        .metric-card.info h2 { color: var(--info-cyan); }
        .metric-card.neutral h2 { color: var(--neutral); }
        .metric-card.primary h2 { color: var(--primary-color); }
        
        /* Buttons */
        .stButton>button {
            border: 2px solid var(--primary-color);
            background: transparent;
            color: var(--primary-color);
            font-weight: 700;
            border-radius: 12px;
            padding: 0.75rem 2rem;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .stButton>button:hover {
            box-shadow: 0 0 25px rgba(var(--primary-rgb), 0.6);
            background: var(--primary-color);
            color: #1A1A1A;
            transform: translateY(-2px);
        }
        
        .stButton>button:active {
            transform: translateY(0);
        }

        /* Table Styling */
        .stMarkdown table {
            width: 100%;
            border-collapse: collapse;
            background: var(--bg-card);
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid var(--border-color);
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        }
        
        .stMarkdown table th,
        .stMarkdown table td {
            text-align: left !important;
            padding: 12px 10px;
            border-bottom: 1px solid var(--border-color);
        }
        
        .stMarkdown table th {
            background-color: var(--bg-elevated);
            font-size: 0.9rem;
            letter-spacing: 0.5px;
            color: var(--text-primary);
        }
        
        .stMarkdown table tr:last-child td {
            border-bottom: none;
        }
        
        .stMarkdown table tr:hover {
            background-color: var(--bg-elevated);
        }

        .green-highlight { color: var(--success-green); font-weight: bold; }
        .magenta-highlight { color: var(--warning-amber); font-weight: bold; }
        .blue-highlight { color: var(--danger-red); font-weight: bold; }

        .section-divider {
            height: 1px;
            background: linear-gradient(90deg, transparent 0%, var(--border-color) 50%, transparent 100%);
            margin: 1.5rem 0;
        }
        
        .sidebar-title { 
            font-size: 0.75rem; 
            font-weight: 700; 
            color: var(--primary-color); 
            text-transform: uppercase; 
            letter-spacing: 1px; 
            margin-bottom: 0.75rem; 
        }
        
        [data-testid="stSidebar"] { 
            background: var(--secondary-background-color); 
            border-right: 1px solid var(--border-color); 
        }
        
        .info-box { 
            background: var(--secondary-background-color); 
            border: 1px solid var(--border-color); 
            padding: 1.25rem; 
            border-radius: 12px; 
            margin: 0.5rem 0; 
            box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.08); 
        }
        .info-box h4 { color: var(--primary-color); margin: 0 0 0.5rem 0; font-size: 1rem; font-weight: 700; }
        .info-box p { color: var(--text-muted); margin: 0; font-size: 0.9rem; line-height: 1.6; }
        
        .stTabs [data-baseweb="tab-list"] { gap: 24px; background: transparent; }
        .stTabs [data-baseweb="tab"] { color: var(--text-muted); border-bottom: 2px solid transparent; transition: color 0.3s, border-bottom 0.3s; background: transparent; font-weight: 600; }
        .stTabs [aria-selected="true"] { color: var(--primary-color); border-bottom: 2px solid var(--primary-color); background: transparent !important; }
        
        .stPlotlyChart { border-radius: 12px; background-color: var(--secondary-background-color); padding: 10px; border: 1px solid var(--border-color); box-shadow: 0 0 25px rgba(var(--primary-rgb), 0.1); }
        .stDataFrame { border-radius: 12px; background-color: var(--secondary-background-color); border: 1px solid var(--border-color); }
        
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: var(--background-color); }
        ::-webkit-scrollbar-thumb { background: var(--border-color); border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--border-light); }
    </style>
    """, unsafe_allow_html=True)

load_css()

# --- Sidebar Controls (Nirnay-style) ---
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0; margin-bottom: 1rem;">
        <div style="font-size: 1.75rem; font-weight: 800; color: #FFC300;">ARTHAGATI</div>
        <div style="color: #888888; font-size: 0.75rem; margin-top: 0.25rem;">अर्थगति | Sentiment Intelligence</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-title">🔄 Data Controls</div>', unsafe_allow_html=True)
    
    # Clear cache button (forces fresh fetch on next load)
    if st.button("REFRESH DATA", help="Clear cached data and fetch fresh from Google Sheets"):
        st.cache_data.clear()
        st.toast("Data cache cleared!", icon="🔄")
        st.rerun()
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class='info-box'>
        <p style='font-size: 0.8rem; margin: 0; color: var(--text-muted); line-height: 1.5;'>
            <strong>Version:</strong> {VERSION}<br>
            <strong>Engine:</strong> PE/EY Correlation<br>
            <strong>Data:</strong> Auto-refresh hourly
        </p>
    </div>
    """, unsafe_allow_html=True)

# Title with Premium Header
st.markdown("""
<div class="premium-header">
    <h1>ARTHAGATI : Sentiment Intelligence</h1>
    <div class="tagline">Quantitative Market Mood & Correlation Analysis</div>
</div>
""", unsafe_allow_html=True)

# Tabs (reordered with Sentiment Dashboard first)
tab1, tab2, tab3 = st.tabs(["**📊 Sentiment Dashboard**", "**🔍 Similar Periods**", "**📋 Mood Score Table**"])

# Spread Indicator Function
@st.cache_data
def calculate_spread_indicator(df, mood_col='Mood_Score'):
    """
    Calculate Spread Indicator using Mood_Score for diff and moving averages.
    Returns negated diff, single gray trace color, and background colors.
    """
    start_time = time.time()
    # Inputs
    look = 50
    short = 90
    long = 200
    spreadup = 1.3
    spreadown = -1.5
    thres = 0.33
    
    result = pd.DataFrame(index=df.index)
    
    # Get Mood_Score
    close = df[mood_col].values
    if len(close) == 0:
        logging.error("Empty Mood_Score data.")
        return result
    
    # Calculate return_1
    close_prev = np.roll(close, 1)
    close_prev[0] = close[0]
    return_1 = (close - close_prev) / np.maximum(close_prev, 1e-10) * 100
    
    # Vectorized count calculation
    count = np.zeros(len(close))
    count[1:] = np.where(return_1[1:] > thres, 1, np.where(return_1[1:] < -thres, -1, 0))
    count = np.cumsum(count)
    
    # Calculate diff and negate it
    sma_count = pd.Series(count).rolling(window=look, min_periods=1).mean().values
    diff = -(count - sma_count)
    
    # Moving Averages
    close_series = pd.Series(close, index=df.index)
    ma90 = ta.sma(close_series, length=short).values
    ma200 = ta.sma(close_series, length=long).values
    
    # Spread calculations
    spread90 = (ma90 - close) * 100 / np.maximum(ma90, 1e-10)
    spread200 = (ma200 - close) * 100 / np.maximum(ma200, 1e-10)
    
    # Trace color - Single gray
    color_gray = 'rgba(136, 136, 136, 0.5)' # Neutral grey
    trace_colors = [color_gray] * len(close)
    
    # Background colors - Matched to success/danger variables
    bg_green = 'rgba(16, 185, 129, 0.15)' # success-green low opacity (slightly higher for visibility)
    bg_red = 'rgba(239, 68, 68, 0.15)'     # danger-red low opacity
    
    # We use 'None' string to represent no color to avoid issues with numpy object arrays and nulls
    bg_colors = np.where((spread90 > spreadup) & (spread200 > spreadup), bg_green,
                         np.where((spread90 < spreadown) & (spread200 < spreadown), bg_red, 'None'))
    
    result['diff'] = diff
    result['trace_color'] = trace_colors
    result['bg_color'] = bg_colors
    logging.info(f"Spread Indicator calculated in {time.time() - start_time:.2f} seconds.")
    return result

# Data Loading
@st.cache_data(ttl=3600, show_spinner="Loading market data from Google Sheets...")
def load_data():
    start_time = time.time()
    try:
        # Construct the Export URL for CSV - using gviz/tq endpoint for better reliability
        url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&gid={SHEET_GID}"
        
        # FIX: Read all columns as string first to avoid "Columns (0) have mixed types" warning
        # This prevents Pandas from guessing types before we are ready
        df = pd.read_csv(url, usecols=lambda x: x in EXPECTED_COLUMNS, dtype=str)
        
        if not any(col in df.columns for col in EXPECTED_COLUMNS):
            raise ValueError("None of the expected columns found in the Sheet.")
        
        missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]
        if missing_columns:
            logging.warning(f"Missing columns: {missing_columns}. Setting to 0.0.")
            for col in missing_columns:
                df[col] = "0.0" # Set as string "0.0" since we are in string mode
        
        # FIX: Explicitly specify format='%m/%d/%Y' for speed and correctness.
        # The user reported Google Sheets changed format to MM/DD/YYYY. 
        # Without this, Pandas falls back to 'dateutil' which causes 503 timeouts on large sheets.
        df['DATE'] = pd.to_datetime(df['DATE'], format='%m/%d/%Y', errors='coerce')
        
        # Convert numeric columns safely
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
        st.error(f"Failed to load data. Ensure the Google Sheet is 'Public' (Anyone with link) and the ID is correct. Error: {str(e)}")
        return None

# OPTIMIZED: Vectorized Anchor Correlations
@st.cache_data
def calculate_anchor_correlations(df, anchor):
    # Filter for columns that actually exist in the dataframe
    cols_to_check = [col for col in DEPENDENT_VARS if col in df.columns]
    
    if anchor not in df.columns or not cols_to_check:
        return pd.DataFrame(columns=['variable', 'correlation', 'strength', 'type'])
    
    # Calculate all correlations at once using optimized Pandas vectorization
    # Select only numeric data for correlation to avoid errors
    analysis_df = df[[anchor] + cols_to_check].select_dtypes(include=[np.number])
    
    if anchor not in analysis_df.columns:
        return pd.DataFrame(columns=['variable', 'correlation', 'strength', 'type'])

    # Compute correlation matrix
    corr_matrix = analysis_df.corr(method='pearson')
    
    # Extract correlations for the anchor variable, dropping the anchor itself
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
            'correlation': round(corr, 2),
            'strength': strength,
            'type': 'positive' if corr > 0 else 'negative'
        })
    
    result_df = pd.DataFrame(correlations)
    if not result_df.empty:
        return result_df.sort_values(by='correlation', key=abs, ascending=False)
    return result_df

# OPTIMIZED: Historical Mood Calculation
@st.cache_data
def calculate_historical_mood(df):
    start_time = time.time()
    if 'DATE' not in df.columns or 'NIFTY50_PE' not in df.columns or 'NIFTY50_EY' not in df.columns:
        logging.error("Required columns missing.")
        return pd.DataFrame(columns=['DATE', 'Mood_Score', 'Mood', 'Smoothed_Mood_Score', 'Mood_Volatility'])
    
    # 1. Correlations (Now Vectorized)
    pe_corrs = calculate_anchor_correlations(df, 'NIFTY50_PE')
    ey_corrs = calculate_anchor_correlations(df, 'NIFTY50_EY')
    
    pe_weights = {row['variable']: abs(row['correlation']) for _, row in pe_corrs.iterrows()}
    ey_weights = {row['variable']: abs(row['correlation']) for _, row in ey_corrs.iterrows()}
    
    pe_total_weight = max(sum(pe_weights.values()), 1e-10)
    ey_total_weight = max(sum(ey_weights.values()), 1e-10)
    
    pe_weights = {k: v/pe_total_weight for k, v in pe_weights.items()}
    ey_weights = {k: v/ey_total_weight for k, v in ey_weights.items()}
    
    # 2. Percentiles (Massively Optimized)
    # Replaced O(N^2) loop with O(N) vectorized expanding rank
    # expanding().rank(pct=True) is equivalent to mean(history <= current)
    pe_percentiles = df['NIFTY50_PE'].expanding().rank(pct=True, method='max').values
    ey_percentiles = df['NIFTY50_EY'].expanding().rank(pct=True, method='max').values
    
    pe_base = -1 + 2 * (1 - pe_percentiles)
    ey_base = -1 + 2 * ey_percentiles
    
    pe_adjustments = np.zeros(len(df))
    ey_adjustments = np.zeros(len(df))
    
    # 3. Adjustments (Optimized loop)
    # We still iterate through vars, but the percentile calc inside is now vectorized
    vars_to_process = [col for col in DEPENDENT_VARS if col in df.columns]
    
    for var in vars_to_process:
        # Vectorized percentile calculation for the dependent variable
        var_percentiles = df[var].expanding().rank(pct=True, method='max').values
        
        # PE Adjustments
        if var in pe_weights:
            pe_type = pe_corrs.loc[pe_corrs['variable'] == var, 'type'].iloc[0]
            weight = pe_weights.get(var, 0)
            if pe_type == 'positive':
                pe_adjustments += weight * (1 - var_percentiles)
            elif pe_type == 'negative':
                pe_adjustments -= weight * (1 - var_percentiles)
        
        # EY Adjustments
        if var in ey_weights:
            ey_type = ey_corrs.loc[ey_corrs['variable'] == var, 'type'].iloc[0]
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
    
    conditions = [
        mood_scores > 60,
        mood_scores > 20,
        mood_scores > -20,
        mood_scores > -60,
        True
    ]
    choices = ['Very Bullish', 'Bullish', 'Neutral', 'Bearish', 'Very Bearish']
    mood_categories = np.select(conditions, choices, default='Neutral')
    
    result = pd.DataFrame({
        'DATE': df['DATE'],
        'Mood_Score': mood_scores,
        'Smoothed_Mood_Score': smoothed_mood_scores,
        'Mood_Volatility': mood_volatility,
        'Mood': mood_categories,
        'NIFTY': df['NIFTY'],
        'BREADTH': df['BREADTH'],
        'COUNT': df['COUNT'],
        'NIFTY50_PE': df['NIFTY50_PE'],
        'PE_DEV': df['PE_DEV'],
        'EY_DEV': df['EY_DEV']
    })
    logging.info(f"Historical Mood calculated in {time.time() - start_time:.2f} seconds.")
    return result

# Similar Periods
@st.cache_data
def find_similar_periods(historical_mood_df):
    start_time = time.time()
    if len(historical_mood_df) < 30:
        return []
    
    current_data = historical_mood_df.iloc[-1]
    if 'Mood_Score' not in historical_mood_df.columns or 'Mood_Volatility' not in historical_mood_df.columns:
        return []
    
    mood_range = historical_mood_df['Mood_Score'].max() - historical_mood_df['Mood_Score'].min() or 1
    volatility_range = historical_mood_df['Mood_Volatility'].max() - historical_mood_df['Mood_Volatility'].min() or 1
    mood_min = historical_mood_df['Mood_Score'].min()
    volatility_min = historical_mood_df['Mood_Volatility'].min()
    
    mood_weight = 0.85
    volatility_weight = 0.15
    
    similarity_scores = []
    current_mood = current_data['Mood_Score']
    current_volatility = current_data['Mood_Volatility']
    
    mood_scaled = (historical_mood_df['Mood_Score'] - mood_min) / mood_range
    volatility_scaled = (historical_mood_df['Mood_Volatility'] - volatility_min) / volatility_range
    current_mood_scaled = (current_mood - mood_min) / mood_range
    current_volatility_scaled = (current_volatility - volatility_min) / volatility_range
    
    mood_diff = (current_mood_scaled - mood_scaled) ** 2
    volatility_diff = (current_volatility_scaled - volatility_scaled) ** 2
    weighted_distance = mood_weight * mood_diff + volatility_weight * volatility_diff
    similarity = 1 / (1 + np.sqrt(weighted_distance))
    
    for i, sim in enumerate(similarity):
        result_dict = {
            'date': historical_mood_df['DATE'].iloc[i].strftime('%Y-%m-%d'),
            'similarity': sim,
            'mood_score': historical_mood_df['Mood_Score'].iloc[i],
            'mood_volatility': historical_mood_df['Mood_Volatility'].iloc[i],
            'mood': historical_mood_df['Mood'].iloc[i],
            'nifty': historical_mood_df.get('NIFTY', pd.Series([None] * len(historical_mood_df))).iloc[i],
            'pe_dev': historical_mood_df.get('PE_DEV', pd.Series([0.0] * len(historical_mood_df))).iloc[i],
            'ey_dev': historical_mood_df.get('EY_DEV', pd.Series([0.0] * len(historical_mood_df))).iloc[i]
        }
        similarity_scores.append(result_dict)
    
    result = sorted(similarity_scores, key=lambda x: x['similarity'], reverse=True)[:10]
    logging.info(f"Similar Periods calculated in {time.time() - start_time:.2f} seconds.")
    return result

# Download Data
def create_download_data(historical_mood_df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        export_df = historical_mood_df[['DATE', 'Mood_Score', 'Mood_Volatility']].copy()
        export_df['Mood_Score'] = export_df['Mood_Score'].round(2)
        export_df['Mood_Volatility'] = export_df['Mood_Volatility'].round(2)
        export_df.to_excel(writer, sheet_name='Mood Analysis', index=False)
    
    return output.getvalue()

# --- RENDER HELPER FOR TABLES ---
def render_styled_html(df):
    """Applies formatting and renders HTML using Pandas Styler matching quo.py style"""
    styler = df.style
    # Apply the CSS class for table styling defined in load_css
    styler = styler.set_table_attributes('class="stMarkdown table"').hide(axis="index")
    return styler.to_html(escape=False)

# Load data
df = load_data()
if df is None:
    st.stop()

# Calculate historical mood and similar periods
try:
    with st.spinner("Calculating market mood..."):
        historical_mood_df = calculate_historical_mood(df)
    
    with st.spinner("Finding similar periods..."):
        similar_periods = find_similar_periods(historical_mood_df)
except Exception as e:
    st.error(f"Error in calculations: {str(e)}")
    logging.error(f"Error in calculations: {str(e)}")
    st.stop()

# Cards for Current Mood and Date
if not historical_mood_df.empty:
    latest_data = historical_mood_df.iloc[-1]
    mood_score = latest_data['Mood_Score']
    
    # Determine card style based on mood
    if mood_score > 60:
        card_class = "success"
    elif mood_score > 20:
        card_class = "warning" # Using Amber for mild bullish to distinguish from strong green
    elif mood_score < -60:
        card_class = "danger"
    elif mood_score < -20:
        card_class = "neutral" # Using Gray for bearish bias
    else:
        card_class = "info"
        
    col1, col2 = st.columns(2)
    
    with col1:
        # Forced Gold color for value to match Date Card
        st.markdown(f"""
        <div class="metric-card {card_class}">
            <h4>Current Mood Score</h4>
            <h2 style="color: #FFC300;">{latest_data['Mood_Score']:.2f}</h2>
            <div class="sub-metric">{latest_data['Mood']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card primary">
            <h4>Analysis Date</h4>
            <h2>{}</h2>
            <div class="sub-metric">Latest Data Point</div>
        </div>
        """.format(latest_data['DATE'].strftime('%Y-%m-%d')), unsafe_allow_html=True)

# Historical Mood Tab
with tab1:
    if not historical_mood_df.empty:
        start_time = time.time()
        # Use all data
        df = historical_mood_df.copy()
        
        # Calculate Spread Indicator
        indicator_df = calculate_spread_indicator(df)
        if indicator_df.empty or 'diff' not in indicator_df.columns:
            st.error("Failed to calculate Spread Indicator.")
            st.stop()
        
        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
            subplot_titles=("Mood Score", "Spread Indicator")
        )
        
        # Mood Score Trace (Row 1) - USING WEBGL for performance
        fig.add_trace(
            go.Scattergl(
                x=df['DATE'],
                y=df['Mood_Score'],
                mode='lines',
                name='Mood Score',
                line=dict(color='#06b6d4', width=2), # Info Cyan
                hovertemplate='<b>%{x|%d %b %Y}</b><br>Mood: %{customdata}<br>Score: %{y:.2f}<extra></extra>',
                customdata=df['Mood']
            ),
            row=1,
            col=1
        )
        
        # Optimized Background Highlighting (Row 2) - USING MERGED SHAPES
        # Instead of 1000s of bars/shapes, we merge consecutive days into single blocks.
        bg_changes = []
        prev_color = 'None'
        start_idx = 0
        
        # Convert to numpy/list for fast iteration
        bg_colors_list = indicator_df['bg_color'].values
        dates_list = df['DATE'].values
        
        for i in range(len(df)):
            current_color = bg_colors_list[i]
            
            # Detect change
            if current_color != prev_color:
                # If the PREVIOUS block was a valid color (not 'None'), save it
                if prev_color != 'None':
                    bg_changes.append({
                        'color': prev_color,
                        'start': dates_list[start_idx],
                        'end': dates_list[i] # Ends at the start of new color
                    })
                # Reset for new block
                start_idx = i
                prev_color = current_color
        
        # Handle the final block
        if prev_color != 'None' and start_idx < len(df):
            bg_changes.append({
                'color': prev_color,
                'start': dates_list[start_idx],
                'end': dates_list[-1]
            })
        
        # Add shapes to layout (much lighter than individual traces)
        shapes = []
        for change in bg_changes:
            # We must use 'xref' and 'yref' specific to the subplot
            # subplot row 2 col 1 usually maps to x2, y2
            shapes.append(dict(
                type="rect",
                xref="x2", 
                yref="y2",
                x0=change['start'],
                y0=-20,
                x1=change['end'],
                y1=20,
                fillcolor=change['color'],
                opacity=1, # Opacity is handled in the RGBA string
                layer="below",
                line_width=0
            ))
        
        # Spread Indicator Trace (Row 2) - USING WEBGL
        fig.add_trace(
            go.Scattergl(
                x=df['DATE'],
                y=indicator_df['diff'],
                mode='lines',
                name='Spread Indicator',
                line=dict(color='#888888', width=2), # Neutral color
                connectgaps=False,
                hovertemplate='<b>%{x|%d %b %Y}</b><br>Spread: %{y:.2f}<extra></extra>',
                showlegend=True
            ),
            row=2,
            col=1
        )
        
        # Row 2 Oscillator Bounds (Spread Indicator)
        # Using shapes for lines is better for performance than adding traces
        # But for simple horizontal lines, add_hline is fine as it uses shapes internally
        fig.add_hline(y=0, line_color='#757575', line_width=1, annotation_text="Zero", annotation_position="top left", annotation_font_size=10, row=2, col=1)
        fig.add_hline(y=10, line_color='#ef4444', line_width=1, annotation_text="Upper", annotation_position="top left", annotation_font_size=10, row=2, col=1)
        fig.add_hline(y=-10, line_color='#10b981', line_width=1, annotation_text="Lower", annotation_position="bottom left", annotation_font_size=10, row=2, col=1)
        fig.add_hline(y=7, line_color='#ef4444', line_dash="dot", line_width=1, annotation_text="Mid Upper", annotation_position="top left", annotation_font_size=10, row=2, col=1)
        fig.add_hline(y=-7, line_color='#10b981', line_dash="dot", line_width=1, annotation_text="Mid Lower", annotation_position="bottom left", annotation_font_size=10, row=2, col=1)
        
        # Neutral Line (Row 1)
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="#757575",
            line_width=1,
            annotation_text="Neutral",
            annotation_position="top right",
            annotation_font_size=10,
            annotation_font_color="#BDBDBD",
            row=1,
            col=1
        )
        
        # Last Point Annotation (Row 1)
        last_point = df.iloc[-1]
        fig.add_annotation(
            x=last_point['DATE'],
            y=last_point['Mood_Score'],
            text=f"Current: {last_point['Mood_Score']:.2f}",
            showarrow=True,
            arrowhead=1,
            ax=-40,
            ay=-30,
            bgcolor="#212121",
            bordercolor="#FFC300",
            font_color="#FAFAFA",
            row=1,
            col=1
        )
        
        # Update Layout for Dark Theme with Premium Colors
        fig.update_layout(
            height=800,
            template="plotly_dark",
            plot_bgcolor='#1A1A1A',
            paper_bgcolor='#1A1A1A',
            font_color='#EAEAEA',
            hovermode="x unified",
            showlegend=True,
            shapes=shapes, # Add the optimized background shapes here
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font_color="#E0E0E0"),
            margin=dict(l=20, r=20, t=60, b=20),
            xaxis=dict(
                type="date",
                showgrid=True,
                gridcolor='#2A2A2A',
                showspikes=True,
                spikemode="toaxis+across",
                spikesnap="cursor",
                spikecolor="#FFC300",
                spikethickness=1
            ),
            xaxis2=dict(
                type="date",
                showgrid=True,
                gridcolor='#2A2A2A',
                showspikes=True,
                spikemode="toaxis+across",
                spikesnap="cursor",
                spikecolor="#FFC300",
                spikethickness=1
            ),
            yaxis=dict(
                title="Mood Score",
                autorange="reversed",
                showgrid=True,
                gridcolor='#2A2A2A',
                zeroline=False,
                showspikes=True,
                spikemode="toaxis+across",
                spikesnap="data",
                spikecolor="#FFC300",
                spikethickness=1
            ),
            yaxis2=dict(
                title="Spread Indicator",
                showgrid=True,
                gridcolor='#2A2A2A',
                zeroline=False,
                range=[-20, 20],
                showspikes=True,
                spikemode="toaxis+across",
                spikesnap="data",
                spikecolor="#FFC300",
                spikethickness=1
            )
        )
        
        # Display Chart
        with st.spinner("Rendering chart..."):
            st.plotly_chart(fig, config={
                'displayModeBar': True,
                'scrollZoom': True,
                'modeBarButtonsToAdd': [
                    'drawline',
                    'drawopenpath',
                    'drawcircle',
                    'drawrect',
                    'eraseshape',
                    'zoom2d', 'pan2d', 'autoScale2d', 'resetScale2d'
                ]
            }) 

        logging.info(f"Chart rendered in {time.time() - start_time:.2f} seconds.")
        
# Similar Periods Tab
with tab2:
    st.subheader("Similar Historical Market Periods")
    st.write("Most similar periods to current conditions based on Mood Score and Volatility.")
    
    if similar_periods:
        similar_df = pd.DataFrame(similar_periods)
        
        # Apply Strict 2-decimal formatting
        similar_df['similarity'] = (similar_df['similarity'] * 100).apply(lambda x: f"{x:.2f}")
        similar_df['mood_score'] = similar_df['mood_score'].apply(lambda x: f"{x:.2f}")
        similar_df['mood_volatility'] = similar_df['mood_volatility'].apply(lambda x: f"{x:.2f}")
        similar_df['pe_dev'] = similar_df['pe_dev'].apply(lambda x: f"{x:.2f}")
        similar_df['ey_dev'] = similar_df['ey_dev'].apply(lambda x: f"{x:.2f}")
        
        # Ensure Nifty is also formatted if present
        if 'nifty' in similar_df.columns:
             similar_df['nifty'] = similar_df['nifty'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "0.00")

        def highlight_mood(row):
            if row['mood'] == 'Very Bullish':
                return f'<span class="green-highlight">{row["mood"]}</span>'
            elif row['mood'] == 'Neutral':
                return f'<span class="magenta-highlight">{row["mood"]}</span>'
            elif row['mood'] == 'Very Bearish':
                return f'<span class="blue-highlight">{row["mood"]}</span>'
            else:
                return row['mood']
        
        display_df = similar_df.copy()
        display_df['mood'] = display_df.apply(highlight_mood, axis=1)
        
        display_cols = ['date', 'mood_score', 'mood_volatility', 'mood', 'similarity', 'nifty', 'pe_dev', 'ey_dev']
        col_names = ['Date', 'Mood Score', 'Mood Volatility', 'Mood', 'Similarity (%)', 'NIFTY', 'P/E Deviation', 'E/Y Deviation']
        
        # Use the new render_styled_html function which applies the class "stMarkdown table"
        html_table = render_styled_html(display_df[display_cols].rename(columns=dict(zip(display_cols, col_names))))
        st.markdown(html_table, unsafe_allow_html=True)
    else:
        st.write("No similar periods found or insufficient data.")

    # --- DOWNLOAD UTILITY (MOVED FROM TAB 1) ---
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.subheader("Download Data")
    st.write("Export mood analysis data:")
    if st.button("📥 Download CSV Report"):
        with st.spinner("Preparing download..."):
            excel_data = create_download_data(historical_mood_df)
            st.download_button(
                label="⬇️ Download Excel",
                data=excel_data,
                file_name=f"mood_analysis_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    # --- END DOWNLOAD UTILITY ---


# Mood Score Table Tab
with tab3:
    st.subheader("Mood Score Data")
    st.write("Tabular view of historical market mood data.")
    
    if not historical_mood_df.empty:
        timeframe = st.selectbox("Select Timeframe for Table", ["1 Month", "3 Months", "6 Months", "1 Year", "All"], index=4, key="table_timeframe")
        
        max_date = historical_mood_df['DATE'].max()
        if timeframe != "All":
            days = {"1 Month": 30, "3 Months": 90, "6 Months": 180, "1 Year": 365}[timeframe]
            min_date = max_date - pd.Timedelta(days=days)
            filtered_df = historical_mood_df[historical_mood_df['DATE'] >= min_date].copy()
        else:
            filtered_df = historical_mood_df.copy()
        
        table_cols = ['DATE', 'BREADTH', 'COUNT', 'NIFTY50_PE', 'PE_DEV', 'EY_DEV', 'Mood_Score']
        display_cols = ['Date', 'Breadth', 'Count', 'NIFTY50 P/E', 'P/E Deviation', 'E/Y Deviation', 'Mood Score']
        
        table_df = filtered_df[table_cols].copy()
        table_df['DATE'] = table_df['DATE'].dt.strftime('%Y-%m-%d')

        # Format numeric columns with strictly 2 decimal places
        numeric_cols_to_format = ['BREADTH', 'COUNT', 'NIFTY50_PE', 'PE_DEV', 'EY_DEV']
        for col in numeric_cols_to_format:
             if col in table_df.columns:
                 table_df[col] = table_df[col].apply(lambda x: f"{x:.2f}")
        
        def highlight_mood_score(row):
            score = row['Mood_Score']
            formatted_score = f"{score:.2f}"
            if score > 60:
                return f'<span class="green-highlight">{formatted_score}</span>'
            elif score > -20 and score <= 20:
                return f'<span class="magenta-highlight">{formatted_score}</span>'
            elif score <= -60:
                return f'<span class="blue-highlight">{formatted_score}</span>'
            else:
                return formatted_score
        
        # Apply highlighting/formatting to Mood_Score separately
        table_df['Mood_Score'] = table_df.apply(highlight_mood_score, axis=1)
        
        html_table = render_styled_html(table_df.rename(columns=dict(zip(table_cols, display_cols))))
        st.markdown(html_table, unsafe_allow_html=True)
    else:
        st.write("No mood score data available.")

# --- FOOTER ---
# Dynamic footer with IST time
def render_footer():
    from datetime import timezone
    utc_now = datetime.now(timezone.utc)
    ist_now = utc_now + timedelta(hours=5, minutes=30)
    current_time_ist = ist_now.strftime("%Y-%m-%d %H:%M:%S IST")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.caption(f"© 2026 {PRODUCT_NAME} | {COMPANY} | {VERSION} | {current_time_ist}")

render_footer()
