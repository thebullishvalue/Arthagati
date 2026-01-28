# ARTHAGATI (अर्थगति) - Market Sentiment Analysis

**A Hemrek Capital Product | v1.2.0**

Quantitative market mood analysis with MSF-enhanced spread indicators.

## Features

### Market Mood Analysis
- **Mood Score**: Composite sentiment indicator (-100 to +100)
- **Historical Tracking**: Full time-series visualization
- **Similar Periods**: AI-matched historical analogues

### MSF-Enhanced Spread Indicator (NEW in v1.2.0)
Replaces the original spread indicator with MSF (Momentum Structure Flow) inspired components:

#### Components
1. **Momentum (30%)**: ROC z-score of NIFTY - price momentum
2. **Structure (25%)**: Mood trend acceleration and divergence
3. **Regime (25%)**: Trend persistence counting
4. **Flow (20%)**: AD_RATIO breadth as participation proxy

#### Signal Interpretation
| MSF Spread | Zone | Interpretation |
|------------|------|----------------|
| > +7 | Overbought | Caution - potential reversal |
| +3 to +7 | Bullish | Strong upward momentum |
| -3 to +3 | Neutral | No clear directional bias |
| -7 to -3 | Bearish | Downward momentum |
| < -7 | Oversold | Caution - potential bounce |

### Correlation Analysis
- Anchor variable correlations (PE, EY)
- Dependent variable impact analysis
- Strength classification (Strong/Moderate/Weak)

## Data Source
- Google Sheets with market breadth and valuation data
- Required columns: DATE, NIFTY, AD_RATIO, NIFTY50_PE, NIFTY50_EY, etc.

## Installation

```bash
pip install -r requirements.txt
streamlit run arthagati.py
```

## Requirements
- streamlit
- pandas
- numpy
- plotly
- pandas_ta

## Hemrek Capital Design System
- Golden accent theme (#FFC300)
- Dark mode interface
- Consistent with NIRNAY, AARAMBH, PRAGYAM, SWING, SAMHITA

## Version History
- v1.2.0: MSF-enhanced spread indicator with 4 components
- v1.1.0: Hemrek design system, performance optimizations
- v1.0.0: Initial release