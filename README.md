# ARTHAGATI (अर्थगति) - Sentiment Intelligence

**A Hemrek Capital Product**

Quantitative market mood analysis using PE/EY correlation metrics. Historical sentiment tracking with similar period detection for market timing insights.

## Features

- **Mood Score Engine**: Weighted PE/EY correlation-based sentiment scoring (-100 to +100)
- **Spread Indicator**: Momentum-based market timing with MA crossover analysis
- **Similar Periods**: Historical analog detection using mood similarity scoring
- **Regime Classification**: Very Bullish → Bullish → Neutral → Bearish → Very Bearish

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Requirements

- Python 3.10+
- streamlit
- pandas
- numpy
- plotly
- pandas_ta

## Data Source

Connects to Google Sheets for live market data. Configure `SHEET_ID` and `SHEET_GID` in the app for your data source.

## Version

v1.1.0 - Hemrek Capital Design System
