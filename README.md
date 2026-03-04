# ARTHAGATI (अर्थगति) v2.0

**Market Sentiment Analysis Engine** · A Hemrek Capital Product

Quantitative market mood scoring with physics-informed mathematics. Built on Ornstein-Uhlenbeck normalization, Kalman filtering, and decay-weighted Spearman correlations.

---

## What It Does

Arthagati answers one question: **"What is the market's current sentiment state, and how confident should I be in that reading?"**

It ingests macro, breadth, and valuation data from a Google Sheet and produces:

- **Mood Score** (−100 to +100) — a correlation-weighted composite of 21 market variables anchored to PE and Earnings Yield
- **MSF Spread** (−10 to +10) — a momentum/structure/regime/flow oscillator for confirmation
- **Similar Historical Periods** — AI-matched analogs using Mahalanobis distance + trajectory shape
- **Predictor Quality Assessment** — transparency into which variables drive the score and which are noise

---

## Architecture (v2.0)

### Mood Score Pipeline — 5 Layers

```
Google Sheet Data
       │
       ▼
Layer 1: Adaptive Correlations
       │  Exponential-decay-weighted Spearman rank correlation
       │  Half-life: 504 days. Recent regime structure matters more.
       │  Spearman (not Pearson): robust to outliers, captures nonlinear monotonic relationships.
       ▼
Layer 2: Information-Theoretic Weighting
       │  weight = |correlation| × (1 − Shannon_entropy_of_variable)
       │  High-entropy (noisy) variables suppressed. Structured signals amplified.
       ▼
Layer 3: Adaptive Percentiles
       │  Decay-weighted empirical CDF (half-life: 252 days)
       │  "Where is PE today vs recent history?" — not vs all-time.
       ▼
Layer 4: Ornstein-Uhlenbeck Normalization
       │  Models mood as mean-reverting diffusion: dx = θ(μ−x)dt + σdW
       │  Normalizes by stationary std: σ/√(2θ) × 30 → [-100, +100]
       │  Diagnostics: half-life = ln(2)/θ days
       ▼
Layer 5: Kalman Smoothing
       │  1D Kalman filter with auto-estimated noise parameters
       │  High noise → more smoothing, low noise → tracks signal
       ▼
  Mood Score + Diagnostics (Hurst, Entropy, OU params)
```

### MSF Spread — Confirmation Oscillator

| Component | What It Measures | v2.0 Upgrade |
|-----------|-----------------|--------------|
| Momentum | NIFTY rate-of-change z-score | — |
| Structure | Mood trend divergence + acceleration | — |
| Regime | Directional move count | Adaptive threshold (scales with local vol, was: fixed 0.0033) |
| Flow | Breadth participation divergence | — |
| **Weighting** | Component allocation | **Inverse-variance** (was: fixed 30/25/25/20) |

### Similar Periods — Matching Engine

| Component | Weight | Method |
|-----------|--------|--------|
| State match | 55% | Mahalanobis distance on 5-feature vector (mood, vol, momentum, Hurst, entropy) |
| Trajectory | 35% | Cosine similarity on detrended 20-day mood path shape |
| Recency | 10% | Exponential decay (365-day half-life) |

---

## Data Schema

### Input Columns (Google Sheet)

| Column | Description |
|--------|-------------|
| DATE | Date in MM/DD/YYYY format |
| NIFTY | NIFTY 50 index level |
| AD_RATIO | Advance-Decline ratio |
| REL_AD_RATIO | Relative AD ratio |
| REL_BREADTH | Relative breadth |
| BREADTH | Market breadth |
| COUNT | Stock count |
| NIFTY50_PE | NIFTY 50 Price-to-Earnings ratio |
| NIFTY50_EY | NIFTY 50 Earnings Yield (auto-derived from 1/PE × 100 if missing) |
| NIFTY50_DY | NIFTY 50 Dividend Yield |
| NIFTY50_PB | NIFTY 50 Price-to-Book |
| IN10Y, IN02Y, IN30Y | India government bond yields (10Y, 2Y, 30Y) |
| INIRYY | India inflation rate YoY |
| REPO | RBI repo rate |
| CRR | Cash Reserve Ratio |
| US02Y, US10Y, US30Y | US Treasury yields |
| US_FED | US Federal Funds Rate |
| PE_DEV | PE deviation from historical mean |
| EY_DEV | EY deviation from historical mean |

### Derived Columns (computed in-app)

| Column | Formula | Why |
|--------|---------|-----|
| IN_TERM_SPREAD | IN10Y − IN02Y | India yield curve slope. Negative = inverted = recession signal. |
| US_TERM_SPREAD | US10Y − US02Y | US yield curve slope. Every US recession since 1960 was preceded by inversion. |
| NIFTY50_EY (if missing) | (1 / NIFTY50_PE) × 100 | Auto-derived if sheet data is empty or constant. |

---

## Model Configuration

### Predictor Selection (Sidebar → Model Configuration)

Users can customize which of the 21 predictor columns feed into the mood engine.

**Operational design:** The multiselect uses a **staging → commit** pattern to prevent continuous recomputation:

1. User adjusts predictors in the multiselect freely (no computation)
2. Pending changes shown: "+2 added, −1 removed"
3. User clicks **"✅ Apply Configuration"** to commit
4. Only then does the engine recompute with the new predictor set
5. Cache is cleared on apply to force fresh computation

This prevents the recomputation loop that would occur if every multiselect click triggered the engine.

### Predictor Quality Assessment (Correlation Analysis tab)

The correlation tab now includes a ranked assessment of all predictors:

- **Quality score** = |avg_correlation| × (1 − entropy) — same formula the engine uses internally
- **Recommendations**: ✅ KEEP (strong signal), 🟡 USEFUL (moderate), ⚪ WEAK (low signal or noisy), ❌ NO DATA
- **Transparency**: shows |ρ| (correlation strength) and H (Shannon entropy) for each variable

---

## Mathematical Primitives

11 mathematical functions, each with exactly one purpose:

| Function | Used In | Purpose |
|----------|---------|---------|
| `exponential_decay_weights` | Correlations | Recency weighting |
| `weighted_spearman` | Correlations | Robust rank correlation with decay |
| `shannon_entropy` | Variable weighting | Penalize noisy predictors |
| `adaptive_percentile` | Mood scoring | Decay-weighted historical positioning |
| `ornstein_uhlenbeck_estimate` | Normalization | Physics-based scaling + half-life |
| `kalman_filter_1d` | Smoothing | Adaptive noise filtering |
| `rolling_hurst` | Diagnostics | Trending vs mean-reverting character |
| `rolling_entropy` | Diagnostics | Market disorder measurement |
| `mahalanobis_distance_batch` | Similar periods | Covariance-aware state matching |
| `cosine_similarity` | Similar periods | Trajectory shape matching |
| `_hurst_rs` | Internal | Rescaled Range Hurst estimation |

All functions are **pure numpy** — zero additional dependencies beyond the base requirements.

---

## v2.1 Features

### OU Forward Projection

The chart now shows a **dotted line** extending 90 days beyond the current data point. This is the Ornstein-Uhlenbeck expected reversion path:

```
E[mood(t+n)] = μ + (mood_current − μ) · exp(−θ·n)
```

The projection shows where the mood score is mathematically expected to converge. The `EQ` label marks the equilibrium level with the OU half-life in days.

### Kalman Confidence Bands

A translucent band surrounds the mood score line showing ±1.96 standard deviations of the Kalman filter's estimate variance (~95% confidence interval). When the band is **tight**, the reading is confident. When it's **wide**, the system is uncertain. A mood score of +40 with tight bands means something very different from +40 with wide bands.

### Data Staleness Warning

If the most recent data point is more than 3 days old (accounting for weekends), a red banner appears warning that scores reflect stale data and the Google Sheet needs updating.

### MSF Component Decomposition

Below the period summary, a breakdown shows each MSF component's current contribution and period average:
- **Momentum** — NIFTY rate-of-change z-score
- **Structure** — Mood trend divergence + acceleration
- **Regime** — Adaptive-threshold directional count
- **Flow** — Breadth participation divergence

### Forward Returns in Similar Periods

Each similar period card now shows what happened to NIFTY 30, 60, and 90 days later. Aggregate summary cards show median returns and win rates across all analogs.

### Backtest Scatter Plot

The Similar Periods tab includes a scatter plot of mood score at time T vs NIFTY return at T+30 days for all historical data. Shows the linear correlation (ρ) with a regression line and interpretation text.

### Regime Transition Detection

Uses Hurst exponent × entropy to classify the market into 4 regimes:

| Regime | Hurst | Entropy | Trading Implication |
|--------|-------|---------|-------------------|
| Trending | > 0.5 | Low | Momentum strategies work |
| Volatile Trend | > 0.5 | High | Directional with large swings |
| Mean-Reverting | < 0.5 | Low | Contrarian/range strategies work |
| Choppy | < 0.5 | High | Hardest to trade — reduce size |

Regime transitions are marked as vertical dotted lines on the mood chart. The current regime is displayed as a diagnostic card in the top metrics row.

---

## Requirements

```
streamlit
pandas
numpy
plotly
pandas_ta
pytz
```

## Setup

1. Ensure your Google Sheet (ID in `SHEET_ID`) is set to "Anyone with the link can view"
2. Populate columns per the schema above
3. Run: `streamlit run arthagati.py`

## Configuration

| Constant | Default | Description |
|----------|---------|-------------|
| `SHEET_ID` | (set in code) | Google Sheet ID |
| `SHEET_GID` | "0" | Sheet tab GID |

---

## Version History

| Version | Changes |
|---------|---------|
| v1.2.0 | Original release: Pearson correlations, expanding percentiles, fixed MSF weights |
| v2.0.0 | Decay-Spearman correlations, adaptive percentiles, OU normalization, Kalman smoothing, inverse-variance MSF, Mahalanobis similarity, predictor quality assessment, apply-button config, EY auto-derivation, yield term spreads |
| v2.1.0 | OU forward projection (90d dotted line), Kalman confidence bands (±1.96σ), data staleness warning, MSF component decomposition, forward returns in similar periods (30/60/90d NIFTY), backtest scatter (mood vs +30d return), regime transition detection (Hurst×Entropy quadrant model), diagnostic cards row |

---

*© 2026 Arthagati · Hemrek Capital*
