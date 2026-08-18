# ARTHAGATI (अर्थगति) · v2.10.0

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
  - [Signal Validation](#signal-validation)
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
produces five outputs:

| Output | Range | Description |
|--------|-------|-------------|
| **Mood Score** | −100 to +100 | Correlation-weighted composite anchored to PE and Earnings Yield |
| **MSF Spread** | −10 to +10 | Momentum / Structure / Flow / Regime confirmation oscillator |
| **WaveTrend** | (unbounded) | LazyBear oscillator on Mood Score with WT1/WT2 crossover signals |
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
│   WT1 = EMA(ci, 21) · WT2 = ALMA(WT1, 20) · crossover ▲ / ▼ signals │
└────────────────────────────┬────────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
┌──────────────────────────┐  ┌────────────────────────────────────┐
│  INTELLIGENCE MODE       │  │   OUTPUT LAYER                     │
│  (post-engine ensemble)  │  │                                    │
│                          │  │   Mood Score · MSF Spread · WT     │
│  7-feature matrix F →    │──┤   Calibrated Conviction (if IM ON) │
│  Optuna TPE on weights w │  │   Diagnostics · Similar Periods    │
│  Walk-forward CV + 25%   │  │   Backtest · Correlation Analysis  │
│  holdout + permutation   │  │                                    │
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
- **Strictly causal**: the statistics applied to segment *k* are estimated on
  data ending at checkpoint *k−1*. The score at time *t* is a function of
  data up to *t* and nothing after it, which a regression test asserts to
  1e-9 by perturbing future rows and checking the past does not move.
- Rows before `CORR_MIN_WARMUP` borrow the first checkpoint's statistics and
  are flagged `Is_Warmup`; every evaluation path excludes them
- Weight blending across checkpoints (α ≈ 0.29, HL = 2) prevents discontinuous jumps

#### Layer 2 — Information-Theoretic Weighting
- `weight = |correlation| × (1 − Shannon_entropy)`
- Entropy bins via Freedman-Diaconis rule: `bin_width = 2·IQR·n^{-1/3}`
- Miller-Madow bias correction on entropy estimate
- Noisy/random variables suppressed; structured signals amplified

#### Layer 3 — Adaptive Percentiles
- Decay-weighted empirical CDF over a Fenwick tree of value ranks: **O(N log N)**
- Writing `w_i = exp(-λt)·exp(λi)`, the `exp(-λt)` factor cancels between
  numerator and denominator, leaving a prefix-sum over rank
- Half-life: `PCT_HALF_LIFE` = 252 days (~1 trading year)
- Answers: *"Where is PE today vs recent history?"* — not vs all-time

#### Layer 4 — Ornstein-Uhlenbeck Normalisation
- Models mood as mean-reverting diffusion: `dx = θ(μ − x)dt + σdW`
- Kendall-Marriott-Pope first-order bias correction on AR(1) coefficient
- Per-observation residual RSS (correct under expanding AR(1) coefficients)
- Rescales by stationary std: `(x − μ) / (σ/√2θ) × MOOD_SCALE` → **[−100, +100]**
- **Scope note**: the input is already an expanding z-score, so the stationary
  std is ≈ 1 (measured 1.09) and this layer's effect on the *score* is a few
  percent — the −100…+100 range comes from `MOOD_SCALE`. Layer 4 earns its
  place through the diagnostics it produces (θ, μ, half-life, and the forward
  reversion projection), not through the rescaling.

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

**Weighting**: inverse-variance — stable components receive more weight —
with two guard rails:

- **Causal.** Variance is *expanding*, so the weight at time *t* uses only
  observations up to *t*. It was previously computed from the trailing 60
  rows and applied across all of history, which meant past MSF values shifted
  every time new data arrived (measured: up to 0.52 on a ±5 band).
- **Clamped.** Weights are bounded to `[0.10, 0.50]` and any component whose
  variance collapses is excluded and reported. A zero-variance component took
  `1/1e-6` inverse-variance, won ~100% of the weight, and — being identically
  zero after the z-score/sigmoid chain — flattened the composite to a
  constant (measured std 0.0001 against a healthy 1.95). A sheet missing its
  `AD_RATIO` column was enough to trigger it, silently. The view now shows a
  banner naming the dead component.

---

### WaveTrend (Mood-driven)

Port of the WaveTrend oscillator core, with `hlc3` replaced by `Mood_Score`:

```
ap  = Mood_Score
esa = ema(ap, 10)                       // Channel length (n1)
d   = ema(|ap − esa|, 10)
ci  = (ap − esa) / max(0.015 · d, 1e-6) // denominator floored
tci = ema(ci, 21)                       // Average length (n2)
wt1 = tci                               // Wave line
wt2 = alma(wt1, 20, 0.85, 6)            // Signal line
```

| Element | Detail |
|---|---|
| **Source** | `Mood_Score` (engine output) |
| **Signal line** | `ALMA(20, offset 0.85, sigma 6)` — matches `ta.alma` exactly |
| **OB / OS** | **Calibrated from the data** — see below |
| **Crossover signals** | ▲ green when WT1 crosses above WT2 · ▼ red when WT2 crosses above WT1 |
| **WT1 − WT2 area fill** | Cyan, transparent (zero-baselined) |
| **Y-axis convention** | Reversed (negative on top) to match the Mood Score pane |

**Why the bands are computed, not hardcoded.** The familiar ±80 / ±60 levels
assume `ci` built from `hlc3`. Driven by `Mood_Score` the oscillator has a
different scale: |wt1| empirically peaks near 70 and never reaches 80, so the
primary band was unreachable and roughly 40% of the pane was permanently
empty. The levels are now the 95th and 80th percentiles of |wt1| over the
**full history** — stable across timeframe switches, and slow-moving as data
arrives. `WT_OB_LEVEL_1 / _2` remain as fallbacks for short series.

The first `n1 + n2 = 31` bars are masked while the EMA chain stabilises.

### Signal Validation

Measures whether the Mood Score carries out-of-sample predictive power.
Nothing is fitted, so there is nothing to overfit.

| Element | Detail |
|---|---|
| **Holdout** | final 25% of history, scored once |
| **Statistic** | mean Spearman rho over 6 contiguous blocks x horizons |
| **Null** | 200 circular shifts of the signal against the same returns |
| **Baseline** | `−PE` — "cheap is good", no engine at all |
| **Power floor** | verdict only at horizons with ≥10 independent forward windows |

Horizons longer than the holdout can support are reported as **descriptive**
and marked `*` in the view — never folded into the verdict. On 20 years of
NIFTY, a 1,246-row holdout supports +20D and +60D; +125D and +250D are shown
but not validated.

**Result on the reference sheet** (NIFTY, 2006–2026, holdout 2021–2026):

```
verdict        Edge Confirmed
holdout rho    +0.538   (p = 0.005, 200 permutations)
−PE baseline   +0.532   margin +0.006
by horizon     +20D +0.42   +60D +0.66   +125D +0.47*  +250D +0.64*
```

**Read the margin.** The edge is real and significant, and it is almost
entirely the PE anchor. An ablation across the whole pipeline:

| Signal | dev rho | holdout rho |
|---|---|---|
| `−PE` level, no engine | +0.467 | **+0.549** |
| PE percentile only (L3) | +0.326 | +0.543 |
| PE+EY percentile base, no predictors | +0.327 | +0.543 |
| Full engine, selected 4 predictors | +0.334 | +0.544 |
| Full engine, all 37 eligible | +0.238 | +0.555 |
| Full engine, breadth only | +0.141 | +0.434 |

Every configuration lands between +0.53 and +0.55. The five layers do not add
rank information over inverting the PE ratio. What they do add is a bounded,
comparable score, a confidence band, an equilibrium and half-life, and regime
context — which is a reasonable product, stated honestly.

---

### Why Intelligence Mode was removed

v2.8.0 shipped a post-engine ensemble: Optuna tuned a linear combination of
engine outputs into a "Calibrated Conviction" signal. It was removed in
v2.10.0 because measurement showed it **reduces** the signal's out-of-sample
power, on every configuration tested:

| Predictor set | raw Mood Score | fitted ensemble | margin |
|---|---|---|---|
| selected 4 | +1.674 | +1.239 | **−0.436** |
| current 12 | +1.893 | −0.444 | **−2.337** |
| all 37 | +1.753 | +1.363 | **−0.390** |

The mechanism is visible in the weights. Only `mood` carries forward
information (holdout rho +0.31 at 90d); the four MSF components sit between
−0.03 and +0.01. Maximising an information ratio across CV folds rewards
whatever fits in-sample, so the search loaded on the technicals — in the
12-predictor run it assigned `mood` a weight of **−0.37**, inverting its one
useful input.

The quality gate rebuilt in v2.9.0 caught this every time and refused to
activate. A component whose own gate rejects it on every real configuration
is not a feature. The measurement apparatus it was built on — holdout,
embargo, permutation null — survives as `validation.py`, pointed at the
question the product actually needs answered.

### Similar Periods Engine

Three-part scoring to find historical analogs:

| Component | Weight | Method |
|-----------|--------|--------|
| **State Match** | 55% | Mahalanobis distance with Ledoit-Wolf OAS shrinkage on 5-feature vector |
| **Trajectory** | 35% | Cosine similarity on least-squares detrended 20-day mood path |
| **Recency** | 10% | Exponential decay (365-day half-life) |

**Minimum separation.** Analogs are selected greedily with a
`SIMILAR_MIN_SEPARATION` = 20 trading-day gap. Adjacent trading days describe
near-identical states, so an unconstrained top-10 routinely collapsed onto two
or three episodes — measured, five of ten inside a 32-row window — while the
UI quoted a median forward return and a hit rate over those ten rows as though
they were ten independent observations. The trailing 90 rows are also excluded,
since they cannot carry a full set of forward returns.

Each match includes forward NIFTY returns at **5D, 20D, 60D, and 90D**.

**Backtest scatter.** Mood Score at T vs NIFTY return at T+`BACKTEST_HORIZON`
(20 days), 70/30 chronological split with a **one-horizon embargo** between
the halves — without it the last 20 training points draw their labels from
inside the test window. Consecutive dots share almost all of their forward
window, so the effective sample is far smaller than the dot count and the
reported coefficients are more certain-looking than they are; the view says so.

### Regime Detection

Hurst exponent × entropy classifies the market into four quadrants:

| Regime | Hurst | Entropy | Strategy Implication |
|--------|-------|---------|---------------------|
| **Trending** | high | low | Momentum strategies work |
| **Volatile Trend** | high | high | Directional with large swings |
| **Mean-Reverting** | low | low | Contrarian / range strategies |
| **Choppy** | low | high | Hardest to trade — reduce size |

**Thresholds are relative and causal.** Both axes are split at their own
*expanding median*, not at a fixed constant:

- The theoretical `H = 0.5` random-walk boundary does not apply. Hurst is
  measured on the mood score — a smoothed composite of percentiles — where
  ~84% of observations sit above 0.5 and the upper quartile pins to the 0.99
  clip. A 0.5 split assigned nearly everything to "trending" and the four
  quadrants collapsed to one (measured: 887 / 864 / 136 / 113).
- The threshold was previously the median of the *whole* series, so a regime
  label depended on data from after the point it described. An expanding
  median uses only observations up to and including each row.

So **"Trending" means persistent relative to this series' own history**, not
`H > 0.5` in the absolute sense. Classification is withheld (`Unknown`) until
`REGIME_MIN_HISTORY` observations are available.

Regime is a **diagnostic only** — it never feeds the score, the MSF weights,
or the OU horizon.

## Mathematical Primitives

Pure-NumPy functions with single callsites:

| Function | Layer | Purpose |
|----------|-------|---------|
| `exponential_decay_weights` | L1 | Recency weighting |
| `weighted_spearman` | L1 | Robust rank correlation with decay |
| `shannon_entropy` | L2 | Freedman-Diaconis bin-width entropy estimation |
| `adaptive_percentile` | L3 | O(N log N) Fenwick-tree decay-weighted CDF |
| `kalman_filter_1d` | L5 | Fading-memory filter with burn-in bootstrap |
| `rolling_hurst` | Diagnostics | DFA-1 with minimum 4-segment guard |
| `rolling_entropy` | Diagnostics | Market disorder measurement |
| `_ledoit_wolf_shrinkage` | Similar Periods | Analytical OAS covariance shrinkage |
| `mahalanobis_distance_batch` | Similar Periods | Shrinkage-regularised state matching |
| `cosine_similarity` | Similar Periods | Least-squares detrended trajectory matching |
| `detect_regime_transitions` | Diagnostics | Hurst × Entropy quadrant classification |
| `_calculate_wavetrend_impl` | WaveTrend | WaveTrend oscillator on Mood Score |
| `wavetrend_bands` | WaveTrend | Empirically calibrated OB/OS levels |

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
| `CORR_MIN_WARMUP` | 252 | Warm-up length; earlier rows flagged `Is_Warmup` |
| `CORR_REBALANCE_PERIOD` | 63 | Expanding-window rebalance interval |
| `MSF_WINDOW` | 20 | MSF rolling window |
| `MSF_ROC_LEN` | 14 | NIFTY rate-of-change period |
| `MSF_ZSCORE_CLIP` | 3.0 | Z-score clipping threshold |
| `MSF_SCALE` | 10.0 | MSF output scaling |
| `MSF_OB_LEVEL_1 / _2` | ±5 / ±3 | MSF reference bands (primary / secondary) |
| `MSF_SIGNAL_Y` | 4 | MSF divergence-triangle y-coordinate magnitude |
| `WT_CHANNEL_LEN` | 10 | WaveTrend n1 (channel length) |
| `WT_AVERAGE_LEN` | 21 | WaveTrend n2 (average length) |
| `WT_SIGNAL_LEN` | 20 | WaveTrend ALMA signal-line period |
| `WT_OB_QUANTILE_1 / _2` | 0.95 / 0.80 | Quantiles of \|wt1\| used for the OB/OS bands |
| `WT_OB_LEVEL_1 / _2` | ±60 / ±40 | Fallback bands for short series |
| `CC_OB_LEVEL_1 / _2` | ±100 / ±80 | Calibrated Conviction reference bands |
| `SIMILAR_W_MAHA` | 0.55 | Mahalanobis distance weight |
| `SIMILAR_W_TRAJ` | 0.35 | Trajectory similarity weight |
| `SIMILAR_W_RECV` | 0.10 | Recency decay weight |
| `TRAJ_WINDOW` | 20 | Trajectory comparison window |
| `OU_PROJ_DAYS` | 90 | OU forward projection horizon |
| `BACKTEST_HORIZON` | 20 | Forward-return horizon for the backtest scatter |
| `SIMILAR_MIN_SEPARATION` | 20 | Minimum trading days between accepted analogs |
| `MSF_MIN_WEIGHT / _MAX_WEIGHT` | 0.10 / 0.50 | Inverse-variance weight clamp |
| `HOLDOUT_FRACTION` | 0.25 | Share of history withheld from the calibrator |
| `GATE_MIN_HOLDOUT_IR` | 0.25 | Minimum holdout effect size to activate |
| `GATE_MAX_P_VALUE` | 0.05 | Permutation-null threshold |
| `GATE_MIN_INDEPENDENT_WINDOWS` | 10 | Power floor before a verdict is issued |

### Predictor Selection

Sidebar → Model Configuration uses a **staging → commit** pattern:
1. Adjust predictors in multiselect (no recomputation)
2. Pending diff shown: `+2 added, −1 removed`
3. Click **Apply Configuration** to commit
4. Engine recomputes with new predictor set; cache + calibration cleared

The default set includes `IN_TERM_SPREAD` and `US_TERM_SPREAD` and excludes
the 2-year legs they are built from — per VISION §2-I, the spread carries the
orthogonal information and including both double-counts the curve. The
spreads were previously derived and then left out of `DEPENDENT_VARS`, so the
documented flagship feature was off by default while the raw yields it was
meant to replace were on.

### Testing

```bash
pip install pytest
pytest                 # fast suite
pytest -m slow         # statistical regressions (runs the engine repeatedly)
```

The suite pins the defects this release fixed rather than testing happy
paths: causality of the mood score, MSF and regime series; exactness of the
Fenwick percentile against a direct transcription of its definition; the
degenerate-component guard; analog separation; and — the headline —
that the calibration gate rejects data containing no edge.

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
≤14 days newer than fit time, profile age ≤14 days, **and a grade of
Quality OK** — a profile that failed the gate is never treated as fresh.

> **Multi-user caveat.** `active.json` is a single file shared by every
> session of a deployment. Set `ARTHAGATI_PROFILE_DIR` to isolate it. Reset
> to Defaults is session-scoped and does not delete the shared file, which it
> previously did — removing the calibrated profile for every concurrent user.

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
If the most recent data point is more than `STALE_DATA_DAYS` (4) calendar
days old, an amber callout reports the gap. The threshold clears a normal
weekend without firing.

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
| **v2.10.0** | 2026-08-18 | **Measured predictor selection; Intelligence Mode removed.** 65 columns → 4 by development-only selection with a single holdout scoring; the Optuna ensemble deleted after it reduced out-of-sample power on every configuration; `validation.py` + Signal Validation view; mood-score semantics corrected (valuation-contrarian, not sentiment); Hurst on increments; reachable classification bands. Verdict on the reference sheet: **Edge Confirmed, holdout rho +0.538, p = 0.005** |
| **v2.9.0** | 2026-08-18 | **Audit remediation.** Eliminated look-ahead in the mood, MSF and regime series; rebuilt the Intelligence Mode quality gate around a 25% holdout, a 90-day embargo and a permutation null (it previously graded pure noise "Quality OK"); O(N log N) percentiles; MSF degenerate-component guard; analog separation; `config.py` extraction; first test suite |
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
