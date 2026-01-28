# ARTHAGATI (अर्थगति) - Market Sentiment Analysis

**A Hemrek Capital Product | v1.2.0**

Quantitative market mood analysis with MSF-enhanced indicators and TradingView-style charting.

## Features

### 📈 Historical Mood Terminal (TradingView Style)
- **Main Chart**: Mood Score with smoothed overlay (inverted axis for intuitive reading)
- **Indicator Pane**: MSF Spread oscillator with overbought/oversold zones
- **Timeframe Selector**: 1W, 1M, 3M, 6M, YTD, 1Y, 2Y, 5Y, MAX
- **Zone Bands**: Visual highlighting of bullish/bearish/neutral zones
- **Period Summary**: High, Low, Average metrics for selected timeframe

### 📊 MSF-Enhanced Spread Indicator
Four-component oscillator inspired by Nirnay's MSF logic:

| Component | Weight | Source | Description |
|-----------|--------|--------|-------------|
| Momentum | 30% | NIFTY | ROC z-score normalized |
| Structure | 25% | Mood | Trend acceleration |
| Regime | 25% | NIFTY | Price persistence |
| Flow | 20% | AD_RATIO | Breadth participation |

**Signal Interpretation:**
- `> +5`: Overbought (caution)
- `+2 to +5`: Bullish momentum
- `-2 to +2`: Neutral
- `-5 to -2`: Bearish momentum
- `< -5`: Oversold (potential bounce)

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
- v1.2.0: Complete redesign with TradingView-style charts, timeframe selector, Nirnay UI/UX
- v1.1.0: Initial MSF integration
- v1.0.0: Initial release
