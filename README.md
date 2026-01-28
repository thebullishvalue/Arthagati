# ARTHAGATI (अर्थगति) - Market Sentiment Analysis

**A Hemrek Capital Product | v1.2.0**

Quantitative market mood analysis with MSF-enhanced indicators and TradingView-style charting.

## Features

### 📈 Historical Mood Terminal (TradingView Style)
- **Main Chart**: Mood Score (Yellow line) with zero reference
- **Indicator Pane**: MSF Spread oscillator (Cyan line) with auto-scaling
- **Divergence Signals**: 
  - 🔻 Red triangles (y=5) = Mood lower low + MSF higher low
  - 🔺 Green triangles (y=-5) = Mood higher high + MSF lower high
- **Timeframe Selector**: 1W, 1M, 3M, 6M, YTD, 1Y, 2Y, 5Y, MAX
- **Period Summary**: High, Low, Average metrics for selected timeframe

### 📊 MSF-Enhanced Spread Indicator
Four-component oscillator inspired by Nirnay's MSF logic:

| Component | Weight | Source | Description |
|-----------|--------|--------|-------------|
| Momentum | 30% | NIFTY | ROC z-score normalized |
| Structure | 25% | Mood | Trend acceleration |
| Regime | 25% | NIFTY | Price persistence |
| Flow | 20% | AD_RATIO | Breadth participation |

### Divergence Detection
- **Red Signal (y=5)**: Mood Score making lower lows while MSF making higher lows
- **Green Signal (y=-5)**: Mood Score making higher highs while MSF making lower highs

**Signal Interpretation:**
- Red triangles at top suggest caution
- Green triangles at bottom suggest opportunity
- Auto-scaling Y-axis for both charts

### 🔍 Similar Periods Analysis
- AI-matched historical periods based on mood score and volatility
- Similarity scoring with recency weighting
- Card-based display with mood classification

### 📋 Correlation Analysis
- PE Ratio correlations with dependent variables
- Earnings Yield correlations
- Visual bar charts with strength indicators

## Installation

```bash
pip install -r requirements.txt
streamlit run arthagati.py
```

## Requirements
- Python 3.10+
- streamlit
- pandas
- numpy
- plotly
- pandas_ta
- pytz

## Data Source
Google Sheets with market breadth and valuation data.

Required columns:
- DATE, NIFTY, AD_RATIO, NIFTY50_PE, NIFTY50_EY
- Plus: Breadth metrics, bond yields, valuation ratios

## Hemrek Capital Design System
- Golden accent theme (#FFC300)
- Dark mode interface (Nirnay-grade)
- Consistent with NIRNAY, AARAMBH, PRAGYAM, SWING, SAMHITA

## Version History
- v1.2.0: Complete redesign with TradingView-style charts, divergence signals, timeframe selector, Nirnay UI/UX
- v1.1.0: Initial MSF integration
- v1.0.0: Initial release
