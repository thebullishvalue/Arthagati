# ARTHAGATI (अर्थगति) · v2.8.0

**Market Sentiment Analysis Engine** — An @thebullishvalue Product

> Quantitative market-mood scoring built on physics-informed mathematics:
> Ornstein-Uhlenbeck mean-reversion, Kalman filtering with burn-in bootstrap,
> walk-forward correlations, Ledoit-Wolf covariance shrinkage, and a
> post-engine ensemble calibrator (Intelligence Mode) driven by Optuna TPE
> over walk-forward folds.

---

## Table of Contents

- [What It Does](#what-it-does)
- [System Architecture](#system-architecture)
  - [Mood Score Pipeline](#mood-score-pipeline)
  - [MSF Spread Oscillator](#msf-spread-oscillator)
  - [WaveTrend (LazyBear · Mood-driven)](#wavetrend-lazybear--mood-driven)
  - [Intelligence Mode — Post-Engine Ensemble](#intelligence-mode--post-engine-ensemble)
  - [Similar Periods Engine](#similar-periods-engine)
  - [Regime Detection](#regime-detection)
- [Mathematical Primitives](#mathematical-primitives)
- [Data Schema](#data-schema)
- [Configuration](#configuration)
- [Key Features](#key-features)
- [Setup](#setup)
- [Version History](#version-history)

---

## What It Does

Arthagati answers one question: **"What is the market's current sentiment
state, how confident should I be in that reading, and what would a
walk-forward-calibrated ensemble of the engine's signals say about its
predictive worth?"**

It ingests macro, breadth, and valuation data from a Google Sheet and
produces six outputs:

| Output | Range | Description |
|--------|-------|-------------|
| **Mood Score** | −100 to +100 | Correlation-weighted composite anchored to PE and Earnings Yield |
| **MSF Spread** | −10 to +10 | Momentum / Structure / Flow / Regime confirmation oscillator |
| **WaveTrend** | (unbounded) | LazyBear oscillator on Mood Score with WT1/WT2 crossover signals |
| **Calibrated Conviction** | −100 to +100 | Optuna-tuned linear ensemble of engine-output features (Intelligence Mode) |
| **Similar Periods** | — | Historical analogs matched by Mahalanobis distance + trajectory shape, with forward returns at 5D / 20D / 60D / 90D |
| **Predictor Assessment** | — | Transparency into which variables drive the score and which are noise |

---

## System Architecture

### Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     DATA INGESTION LAYER                            │
│  Google Sheets (gviz API · env var coords) → CSV parse              │
│  Forward-fill NaN · Derive term spreads · Auto-derive EY from PE    │
└────────────────────────────┬────────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
┌─────────────────────────┐   ┌─────────────────────────────────────┐
│   MOOD SCORE PIPELINE   │   │      MSF SPREAD OSCILLATOR          │
│   (5-Layer Engine)      │   │   (4-Component, Inverse-Variance)   │
│                         │   │                                     │
│  L1: Walk-Fwd Corr      │   │  Momentum  → NIFTY ROC z-score      │
│  L2: Entropy Weighting  │   │  Structure → Mood trend divergence  │
│  L3: Adaptive Percentile│   │  Regime    → Adaptive dir. count    │
│  L4: OU Normalization   │   │  Flow      → Breadth divergence     │
│  L5: Kalman Smoothing   │   │                                     │
└────────────┬────────────┘   └──────────────┬──────────────────────┘
             │                               │
             ├───────────────────────────────┘
             ▼
┌─────────────────────────────────────────────────────────────────────┐
│   WAVETREND OSCILLATOR (LazyBear · Mood-Score-driven)               │
│   esa = EMA(Mood, 10) · ci = (Mood − esa) / (0.015 · EMA(|Δ|,10))   │
│   WT1 = EMA(ci, 21) · WT2 = SMA(WT1, 4) · crossover ▲ / ▼ signals   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
┌──────────────────────────┐  ┌────────────────────────────────────┐
│  INTELLIGENCE MODE       │  │   OUTPUT LAYER                     │
│  (post-engine ensemble)  │  │                                    │
│                          │  │   Mood Score · MSF Spread · WT     │
│  10-feature matrix F →   │──┤   Calibrated Conviction (if IM ON) │
│  Optuna TPE on weights w │  │   Diagnostics · Similar Periods    │
│  Walk-forward CV folds   │  │   Backtest · Correlation Analysis  │
│  Calibrated = F @ w      │  │                                    │
└──────────────────────────┘  └────────────────────────────────────┘
```

---

### Mood Score Pipeline

Five processing layers transform raw market data into a normalised sentiment score:

```
Raw Data ──► L1: Walk-Forward Correlations ──► L2: Entropy Weighting
                                                  │
                                                  ▼
             L5: Kalman Smoothing ◄── L4: OU Normalization ◄── L3: Adaptive Percentiles
                                                  │
                                                  ▼
                                         Mood Score [−100, +100]
                                         + Diagnostics
```

#### Layer 1 — Walk-Forward Correlations
- Exponential-decay-weighted Spearman rank correlation at quarterly checkpoints
- Half-life: `CORR_HALF_LIFE` = 504 days (~2 trading years)
- Expanding window eliminates look-ahead bias
- Weight blending across checkpoints (α ≈ 0.29, HL = 2) prevents discontinuous jumps

#### Layer 2 — Information-Theoretic Weighting
- `weight = |correlation| × (1 − Shannon_entropy)`
- Entropy bins via Freedman-Diaconis rule: `bin_width = 2·IQR·n^{-1/3}`
- Miller-Madow bias correction on entropy estimate
- Noisy/random variables suppressed; structured signals amplified

#### Layer 3 — Adaptive Percentiles
- Decay-weighted empirical CDF with sorted-insert + binary search: **O(N log N)**
- Half-life: `PCT_HALF_LIFE` = 252 days (~1 trading year)
- Answers: *"Where is PE today vs recent history?"* — not vs all-time

#### Layer 4 — Ornstein-Uhlenbeck Normalisation
- Models mood as mean-reverting diffusion: `dx = θ(μ − x)dt + σdW`
- Kendall-Marriott-Pope first-order bias correction on AR(1) coefficient
- Per-observation residual RSS (correct under expanding AR(1) coefficients)
- Normalises by stationary std: `(x − μ) / (σ/√2θ) × MOOD_SCALE` → **[−100, +100]**

#### Layer 5 — Kalman Smoothing
- 1D fading-memory Kalman filter (Sorenson-Sacks)
- Harvey (1990) burn-in bootstrap: first 50 obs calibrated from first stable window
- Confidence band: `tanh` soft-clip `±KALMAN_CI_Z × √variance` (~95% interval)
- Tight band = confident reading; wide band = system is uncertain

---

### MSF Spread Oscillator

Four-component confirmation oscillator, weighted by inverse-variance (Markowitz for signals):

| Component | Measures | Method |
|-----------|----------|--------|
| **Momentum** | NIFTY rate-of-change z-score | `MSF_ROC_LEN` = 14 days |
| **Structure** | Mood trend divergence + acceleration | Fast/slow trend + curvature |
| **Regime** | Directional count | Windowed `rolling(20).sum()` — prevents cumsum drift |
| **Flow** | Breadth participation divergence | Deviation from rolling mean |

**Reference bands**: ±5 (primary, solid) and ±3 (secondary, dotted).
**Divergence markers**: ▲ bullish at y=−4, ▼ bearish at y=+4 (just inside the primary bands).
**Weighting**: inverse-variance — stable components receive more weight, recalculated each run.

---

### WaveTrend (LazyBear · Mood-driven)

Faithful port of LazyBear's WaveTrend Oscillator, with `hlc3` replaced by `Mood_Score`:

```
ap  = Mood_Score
esa = ema(ap, 10)                  // Channel length (n1)
d   = ema(|ap − esa|, 10)           // Floored at 0.5 to prevent warmup blowup
ci  = (ap − esa) / (0.015 · d)
tci = ema(ci, 21)                  // Average length (n2)
wt1 = tci                          // Wave line
wt2 = sma(wt1, 4)                  // Signal line
```

| Element | Detail |
|---|---|
| **Source** | `Mood_Score` (engine output) |
| **OB / OS** | ±80 (primary, solid) · ±60 (secondary, dotted) |
| **Crossover signals** | ▲ green when WT1 crosses above WT2 (bullish) at y=+70 · ▼ red when WT2 crosses above WT1 (bearish) at y=−70 |
| **WT1 − WT2 area fill** | Cyan, transparent (zero-baselined) — magnitude of the wave/signal spread |
| **Y-axis convention** | Reversed (negative on top, positive on bottom) to match the Mood Score pane |

The first 31 bars are masked to let the EMA chain stabilise before the
oscillator is drawn.

---

### Intelligence Mode — Post-Engine Ensemble

A walk-forward Bayesian calibrator that runs **on the engine's output**
(not on its internal hyperparameters) — keeping per-trial cost in the
millisecond range while preserving the engine's mathematical guarantees.

**Feature matrix** (10 columns, standardised, built once per Run Analysis):

| Feature | Meaning |
|---|---|
| `mood`           | Raw Mood Score |
| `mood_smooth`    | Smoothed Mood Score (Kalman) |
| `mood_diverge`   | `mood − mood_smooth` |
| `mood_squared`   | `sign(mood) · mood² / 100` (asymmetric amplification) |
| `mood_sqrt`      | `sign(mood) · √|mood|` (asymmetric damping) |
| `msf_spread`     | MSF Spread composite |
| `msf_momentum`   | MSF momentum component |
| `msf_structure`  | MSF structure component |
| `msf_regime`     | MSF regime component |
| `msf_flow`       | MSF flow component |

**Optimiser**: Optuna TPE sampler with `MedianPruner` (warmup 8 trials, 1 step).
**Objective**: 0.65 × Validation IR + 0.35 × Train IR − L2 penalty on weights.
**Information Ratio**: mean Spearman / std Spearman across (folds × horizons).
**Horizons**: 5D, 20D, 60D, 90D — short-term predictability + medium-term carry.
**Cross-validation**: walk-forward expanding-window folds with a 5-day
embargo gap to prevent forward-return leakage at fold boundaries.

**Quality gate** before a profile is auto-activated:
- `Val IR > 0` (otherwise: *No Edge* — profile saved but inactive)
- `Stability = Val IR / Train IR ≥ 30%` (otherwise: *Overfit* — flagged, still saved)

**Persistence**: `profiles/active.json` (atomic write via `tempfile + os.replace`),
plus a timestamped archive `profiles/profile_<UTC>.json`. The Sidebar
Passport surfaces Profile · Trained on · Train IR · Val IR · Updated,
with a predictor-count mismatch warning when the saved profile was fit
on a different active predictor set.

**Calibrated Conviction**:

```
calibrated_conviction = tanh( (F @ w) / 3 ) · 100   ∈ [−100, +100]
```

Shown as a metric card in the diagnostics strip + as the centrepiece of
the Intelligence Center's *Calibration Impact* section. Computed across
the full history, so historical periods are calibrated under the same
ensemble as the latest signal.

---

### Similar Periods Engine

Three-part scoring to find historical analogs:

| Component | Weight | Method |
|-----------|--------|--------|
| **State Match** | 55% | Mahalanobis distance with Ledoit-Wolf OAS shrinkage on 5-feature vector |
| **Trajectory** | 35% | Cosine similarity on least-squares detrended 20-day mood path |
| **Recency** | 10% | Exponential decay (365-day half-life) |

Each match includes forward NIFTY returns at **5D, 20D, 60D, and 90D**.
A backtest scatter plots mood score at T vs NIFTY return at
T+`BACKTEST_HORIZON` (20 days) with 70/30 train/test split.

---

### Regime Detection

Hurst exponent × entropy classifies the market into four quadrants:

| Regime | Hurst | Entropy | Strategy Implication |
|--------|-------|---------|---------------------|
| **Trending** | > 0.5 | Low | Momentum strategies work |
| **Volatile Trend** | > 0.5 | High | Directional with large swings |
| **Mean-Reverting** | < 0.5 | Low | Contrarian / range strategies |
| **Choppy** | < 0.5 | High | Hardest to trade — reduce size |

Current regime displayed in diagnostic cards.

---

## Mathematical Primitives

Pure-NumPy functions with single callsites:

| Function | Layer | Purpose |
|----------|-------|---------|
| `exponential_decay_weights` | L1 | Recency weighting |
| `weighted_spearman` | L1 | Robust rank correlation with decay |
| `shannon_entropy` | L2 | Freedman-Diaconis bin-width entropy estimation |
| `adaptive_percentile` | L3 | O(N log N) sorted-insert decay-weighted CDF |
| `kalman_filter_1d` | L5 | Fading-memory filter with burn-in bootstrap |
| `rolling_hurst` | Diagnostics | DFA-1 with minimum 4-segment guard |
| `rolling_entropy` | Diagnostics | Market disorder measurement |
| `_ledoit_wolf_shrinkage` | Similar Periods | Analytical OAS covariance shrinkage |
| `mahalanobis_distance_batch` | Similar Periods | Shrinkage-regularised state matching |
| `cosine_similarity` | Similar Periods | Least-squares detrended trajectory matching |
| `detect_regime_transitions` | Diagnostics | Hurst × Entropy quadrant classification |
| `_calculate_wavetrend_impl` | WaveTrend | LazyBear oscillator on Mood Score |

Plus internal helpers: `_hurst_dfa` (DFA implementation), `sigmoid`
(overflow-safe normalisation), `rolling_mean_fast` (O(N) cumsum-based),
`zscore_clipped` (NaN-aware rolling z-score).

---

## Data Schema

### Source Columns (Google Sheet)

| Category | Columns |
|----------|---------|
| **Index** | `DATE`, `NIFTY` |
| **Valuation Anchors** | `NIFTY50_PE`, `NIFTY50_EY`, `NIFTY50_DY`, `NIFTY50_PB`, `PE_DEV`, `EY_DEV` |
| **Breadth** | `AD_RATIO`, `REL_AD_RATIO`, `REL_BREADTH`, `BREADTH`, `COUNT` |
| **India Macro** | `IN10Y`, `IN02Y`, `IN30Y`, `INIRYY`, `REPO`, `CRR` |
| **US Macro** | `US02Y`, `US10Y`, `US30Y`, `US_FED` |

### Derived Columns (computed in-app)

| Column | Formula | Purpose |
|--------|---------|---------|
| `IN_TERM_SPREAD` | `IN10Y − IN02Y` | India yield curve slope — inverted = recession signal |
| `US_TERM_SPREAD` | `US10Y − US02Y` | US yield curve slope — every US recession since 1960 preceded by inversion |
| `NIFTY50_EY` | `(1 / NIFTY50_PE) × 100` | Auto-derived if sheet column is empty or constant |
| `MSF_Spread`, `WT1`, `WT2` | (engine) | Indicator outputs added to the engine-output dataframe |

The app loads **all columns** present in the sheet. Any numeric column
beyond the four anchor keys (`DATE`, `NIFTY`, `NIFTY50_PE`, `NIFTY50_EY`)
is available as a selectable predictor.

---

## Configuration

### Environment Variables

The Google Sheet coordinates are configured via two environment variables:

```bash
export ARTHAGATI_SHEET_ID="<spreadsheet-id>"
export ARTHAGATI_SHEET_GID="<worksheet-gid>"
```

**Getting your Sheet coordinates:**
1. Open your Google Sheet
2. Copy the **SHEET_ID** from the URL: `docs.google.com/spreadsheets/d/<SHEET_ID>/edit...`
3. The **SHEET_GID** is the `gid` parameter in the URL (usually `0` for the first sheet)

**Sheet access:** the sheet must be set to **"Anyone with the link can view"**.
The gviz endpoint works without authentication.

### Hyperparameters

| Constant | Default | Purpose |
|----------|---------|---------|
| `DATA_TTL` | 3600s | Cache TTL for Sheets fetch |
| `CORR_HALF_LIFE` | 504d | Spearman recency weight decay |
| `PCT_HALF_LIFE` | 252d | Adaptive ECDF recency weight decay |
| `MOOD_SCALE` | 30.0 | OU signal → mood score scaling |
| `KALMAN_CI_Z` | 1.96 | Confidence band width (~95%) |
| `KALMAN_HALF_LIFE` | 126d | Kalman fading memory |
| `CORR_MIN_WARMUP` | 252 | Min observations before first checkpoint |
| `CORR_REBALANCE_PERIOD` | 63 | Expanding-window rebalance interval |
| `MSF_WINDOW` | 20 | MSF rolling window |
| `MSF_ROC_LEN` | 14 | NIFTY rate-of-change period |
| `MSF_ZSCORE_CLIP` | 3.0 | Z-score clipping threshold |
| `MSF_SCALE` | 10.0 | MSF output scaling |
| `MSF_OB_LEVEL_1 / _2` | ±5 / ±3 | MSF reference bands (primary / secondary) |
| `MSF_SIGNAL_Y` | 4 | MSF divergence-triangle y-coordinate magnitude |
| `WT_CHANNEL_LEN` | 10 | WaveTrend n1 (channel length) |
| `WT_AVERAGE_LEN` | 21 | WaveTrend n2 (average length) |
| `WT_SIGNAL_LEN` | 4 | WaveTrend signal-line SMA period |
| `WT_OB_LEVEL_1 / _2` | ±80 / ±60 | WaveTrend OB/OS bands |
| `CC_OB_LEVEL_1 / _2` | ±100 / ±80 | Calibrated Conviction reference bands |
| `SIMILAR_W_MAHA` | 0.55 | Mahalanobis distance weight |
| `SIMILAR_W_TRAJ` | 0.35 | Trajectory similarity weight |
| `SIMILAR_W_RECV` | 0.10 | Recency decay weight |
| `TRAJ_WINDOW` | 20 | Trajectory comparison window |
| `OU_PROJ_DAYS` | 90 | OU forward projection horizon |
| `BACKTEST_HORIZON` | 20 | Forward-return horizon for the backtest scatter |

### Predictor Selection

Sidebar → Model Configuration uses a **staging → commit** pattern:
1. Adjust predictors in multiselect (no recomputation)
2. Pending diff shown: `+2 added, −1 removed`
3. Click **Apply Configuration** to commit
4. Engine recomputes with new predictor set; cache + calibration cleared

---

## Key Features

### Engine-Output Session Cache
`mood_df` and `msf_df` are cached in `st.session_state` keyed by
`(row count, first date, last date, sorted predictor set)`. View
switches, timeframe button clicks, and expander toggles are
**O(150 ms)** — only data refresh / predictor changes / data-age >14d
trigger a full recompute.

### Cross-Session Profile Caching
The Intelligence Mode profile (`profiles/active.json`) is reused across
sessions as long as it remains *fresh*: same predictor count, data end
≤14 days newer than fit time, and profile age ≤14 days. Streamlit-Cloud
wake-ups don't pay the calibration cost if the profile on disk still
applies to the current data.

### OU Forward Projection
The mood chart extends a dotted line 90 days beyond the last data point
showing the Ornstein-Uhlenbeck expected reversion path:
`E[mood(t+n)] = μ + (mood_current − μ) · exp(−θ · n)`.

### Kalman Confidence Bands
A translucent band surrounds the mood score line showing ±1.96σ of the
filter's estimate variance. A mood of +40 with tight bands is
fundamentally different from +40 with wide bands.

### Divergence Signals
- **MSF Spread** — bullish (▲ at y=−4) and bearish (▼ at y=+4) divergence triangles, detected via 10-bar lookback extrema comparison
- **WaveTrend** — bullish (▲ at y=+70) and bearish (▼ at y=−70) WT1/WT2 crossover triangles

### Data Staleness Warning
If the most recent data point is more than 3 calendar days old, an amber
warning callout reports the gap and reminds the user that scores reflect
stale data.

### MSF Component Breakdown
Four horizontal bars show each component's current contribution vs
period average, with colours indicating direction.

### Backtest Scatter
Similar Periods view includes a chronological 70/30 train/test scatter
of Mood Score at T vs NIFTY return at T+`BACKTEST_HORIZON` (20 days),
with linear and quadratic fit lines and both Pearson/Spearman correlations
reported.

### Intelligence Center Dashboard
A read-only view (Run Analysis triggers the actual calibration) showing:
- **Calibration Diagnostics** strip — Train IR · Val IR · Stability · Quality
- **Calibration Impact** strip — Raw Mood · Calibrated Conviction · Net Shift · Direction
- **Feature Analysis** grid — per-feature card with weight + fANOVA importance + Bullish/Bearish/Neutral badge
- **Predictive Power Lift** table — per-horizon Spearman IR comparison: raw Mood vs Calibrated
- **Profile Provenance** table — run timestamp, predictors, CV setup, data window, schema version

---

## Setup

### Local

```bash
# 1. Set environment variables with your Sheet coordinates
export ARTHAGATI_SHEET_ID="<your-spreadsheet-id>"
export ARTHAGATI_SHEET_GID="<your-worksheet-gid>"

# 2. Make sure the sheet is "Anyone with the link can view"

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
streamlit run arthagati.py
```

### Streamlit Cloud

1. Push repo to GitHub
2. **App Settings → Environment Variables** — add `ARTHAGATI_SHEET_ID` and `ARTHAGATI_SHEET_GID`
3. Deploy

---

## Version History

| Version | Date | Summary |
|---------|------|---------|
| **v2.8.0** | 2026-05-28 | WaveTrend Oscillator (LazyBear · Mood-driven), Intelligence Mode (post-engine ensemble calibration via Optuna TPE + walk-forward CV), Calibrated Conviction metric, granular forward horizons (5D / 20D / 60D / 90D), MSF Spread reference bands at ±5/±3, structured run-summary console log |
| **v2.7.0** | 2026-04-15 | Obsidian Quant UI port: modular `ui/` package with `theme.css`, components, tabs; Sanskrit serif masthead; section headers with icon badges; analog/correlation/quality cards |
| **v2.6.0** | 2026-04-06 | Google Sheets Infrastructure Simplification: gviz API migration, OAuth removal, environment variable configuration, retry logic |
| **v2.5.0** | 2026-04-05 | Production Readiness & Code Cleanup: Dead function removal, unused return value elimination, type hint modernisation, version consistency |
| **v2.4.0** | — | Adversarial Audit Resolution: OU RSS fix, backward leakage removal, DFA segment guard, MSF regime artifact fix, O(N log N) adaptive percentiles, Kalman warm-up bootstrap, Freedman-Diaconis entropy bins, Ledoit-Wolf shrinkage, walk-forward weight blending, tanh confidence band soft-clip, least-squares trajectory detrend, 70/30 backtest split |
| **v2.3.0** | — | Walk-Forward Correlations & Bias Corrections: Expanding-window Spearman, percentile symmetry fix, DFA replacing R/S, Kendall-Marriott-Pope bias correction, dynamic y-axis |
| **v2.2.1** | — | UI Rendering & Memory Optimizations: WebGL regime transitions, bounded caching (`max_entries=5`) |
| **v2.2.0** | — | Performance Architecture Rewrite: C-level NumPy vectorisation, O(N) cumulative sums, memory-optimised 1D slice lookbacks, 99%+ execution time reduction |
| **v2.1.0** | — | Diagnostics & Forward Returns: OU projection, Kalman bands, forward returns, backtest scatter, regime detection, staleness warnings |
| **v2.0.0** | — | Physics-Informed Mathematics: OU normalisation, Mahalanobis similarity, inverse-variance MSF, Kalman smoothing, adaptive percentiles, decay-Spearman correlations |
| **v1.2.0** | — | Initial Release: Pearson correlations, expanding percentiles, fixed MSF weights |

---

*© 2026 Arthagati · @thebullishvalue*
