# ARTHAGATI (अर्थगति) · v2.5.0

**Market Sentiment Analysis Engine** — A Hemrek Capital Product

Quantitative market mood scoring built on physics-informed mathematics:
Ornstein-Uhlenbeck mean-reversion, Kalman filtering with burn-in bootstrap,
walk-forward correlations, and Ledoit-Wolf covariance shrinkage.

---

## What It Does

Arthagati answers one question: **"What is the market's current sentiment state, and how confident should I be in that reading?"**

It ingests macro, breadth, and valuation data from a Google Sheet and produces:

| Output | Range | Description |
|--------|-------|-------------|
| **Mood Score** | −100 to +100 | Correlation-weighted composite of market variables anchored to PE and Earnings Yield |
| **MSF Spread** | −10 to +10 | Momentum / Structure / Flow / Regime oscillator for confirmation |
| **Similar Periods** | — | Historical analogs matched by Mahalanobis distance + trajectory shape |
| **Predictor Assessment** | — | Transparency into which variables drive the score and which are noise |

---

## Architecture

### Mood Score Pipeline — 5 Layers

```
Google Sheet Data
       │
       ▼
Layer 1: Walk-Forward Correlations
       │  Expanding-window decay-weighted Spearman at quarterly checkpoints
       │  Half-life: CORR_HALF_LIFE = 504 days (~2 trading years)
       │  Exponential weight blending across checkpoints (α ≈ 0.29, HL = 2)
       ▼
Layer 2: Information-Theoretic Weighting
       │  weight = |correlation| × (1 − Shannon_entropy)
       │  Entropy bins via Freedman-Diaconis rule (2·IQR·n^{-1/3})
       │  High-entropy (noisy) variables suppressed. Structured signals amplified.
       ▼
Layer 3: Adaptive Percentiles — O(N log N)
       │  Sorted-insert + binary search (np.searchsorted) on decay-weighted CDF
       │  Half-life: PCT_HALF_LIFE = 252 days
       │  "Where is PE today vs recent history?" — not vs all-time
       ▼
Layer 4: Ornstein-Uhlenbeck Normalization
       │  Models mood as mean-reverting diffusion: dx = θ(μ − x)dt + σdW
       │  Per-observation residual RSS (correct under expanding AR(1) coefficients)
       │  Kendall-Marriott-Pope first-order bias correction on AR(1) coefficient
       │  Normalizes by stationary std: (x − μ) / (σ/√2θ) × MOOD_SCALE → [−100, +100]
       ▼
Layer 5: Kalman Smoothing
       │  1D fading-memory Kalman filter (Sorenson-Sacks)
       │  Harvey (1990) burn-in bootstrap: first 50 obs calibrated from first stable window
       │  Confidence band: tanh soft-clip ±KALMAN_CI_Z × √variance (~95% interval)
       ▼
  Mood Score + Diagnostics (Hurst, Entropy, OU parameters)
```

### MSF Spread — Confirmation Oscillator

| Component | Measures | Method |
|-----------|----------|--------|
| Momentum  | NIFTY rate-of-change z-score (`MSF_ROC_LEN` = 14 days) | — |
| Structure | Mood trend divergence + acceleration | — |
| Regime    | Windowed directional count | `rolling(MSF_WINDOW).sum()` — prevents cumsum drift artifact |
| Flow      | Breadth participation divergence | — |
| **Weighting** | Component allocation | **Inverse-variance** (Markowitz for signals) |

### Similar Periods — Matching Engine

| Component | Weight | Method |
|-----------|--------|--------|
| State match | `SIMILAR_W_MAHA` = 55% | Mahalanobis distance (Ledoit-Wolf OAS shrinkage covariance) on 5-feature vector |
| Trajectory | `SIMILAR_W_TRAJ` = 35% | Cosine similarity on least-squares detrended `TRAJ_WINDOW` = 20-day mood path |
| Recency    | `SIMILAR_W_RECV` = 10% | Exponential decay (365-day half-life) |

---

## Data Schema

### Source Columns (Google Sheet)

| Column | Description |
|--------|-------------|
| `DATE` | Date in `MM/DD/YYYY` format |
| `NIFTY` | NIFTY 50 index level |
| `AD_RATIO` | Advance-Decline ratio |
| `REL_AD_RATIO` | Relative AD ratio |
| `REL_BREADTH` | Relative breadth |
| `BREADTH` | Market breadth |
| `COUNT` | Stock count |
| `NIFTY50_PE` | NIFTY 50 Price-to-Earnings ratio |
| `NIFTY50_EY` | NIFTY 50 Earnings Yield (auto-derived as `1/PE × 100` if absent or constant) |
| `NIFTY50_DY` | NIFTY 50 Dividend Yield |
| `NIFTY50_PB` | NIFTY 50 Price-to-Book |
| `IN10Y`, `IN02Y`, `IN30Y` | India government bond yields (10Y, 2Y, 30Y) |
| `INIRYY` | India inflation rate year-on-year |
| `REPO` | RBI repo rate |
| `CRR` | Cash Reserve Ratio |
| `US02Y`, `US10Y`, `US30Y` | US Treasury yields |
| `US_FED` | US Federal Funds Rate |
| `PE_DEV` | PE deviation from historical mean |
| `EY_DEV` | Earnings Yield deviation from historical mean |

### Derived Columns (computed in-app, not required in sheet)

| Column | Formula | Purpose |
|--------|---------|---------|
| `IN_TERM_SPREAD` | `IN10Y − IN02Y` | India yield curve slope. Negative = inverted = recession signal. |
| `US_TERM_SPREAD` | `US10Y − US02Y` | US yield curve slope. Every US recession since 1960 was preceded by inversion. |
| `NIFTY50_EY` | `(1 / NIFTY50_PE) × 100` | Auto-derived if sheet column is empty or constant. |

The app loads **all columns** present in the sheet — any column beyond `EXPECTED_COLUMNS` is automatically available as a predictor in the Predictor Configuration panel.

---

## Configuration

### Secrets (never in source)

Sheet credentials and document coordinates live exclusively in `.streamlit/secrets.toml` (local) or the Streamlit Cloud Secrets panel (deployed). See `.streamlit/secrets.toml.example` for the required structure.

```toml
[google_service_account]
# Full service account JSON fields (type, project_id, private_key, client_email, …)

[sheet]
id  = "<spreadsheet-id>"   # long alphanumeric string in the Sheet URL
gid = "<worksheet-gid>"    # numeric tab ID in the URL after #gid=
```

The service account email must have at least **Viewer** access to the sheet.

### Code Constants

| Constant | Default | Description |
|----------|---------|-------------|
| `DATA_TTL` | `3600` | Cache TTL for the Sheets fetch (seconds) |
| `CORR_HALF_LIFE` | `504` | Spearman recency weight half-life (days) |
| `PCT_HALF_LIFE` | `252` | Adaptive ECDF recency weight half-life (days) |
| `MOOD_SCALE` | `30.0` | OU-normalised signal → mood score scaling factor |
| `KALMAN_CI_Z` | `1.96` | Kalman confidence band width (~95%) |
| `OU_PROJ_DAYS` | `90` | OU forward projection horizon (calendar days) |
| `MSF_WINDOW` | `20` | MSF rolling window (bars) |
| `MSF_ROC_LEN` | `14` | NIFTY rate-of-change period (bars) |
| `BACKTEST_HORIZON` | `30` | Forward-return horizon for backtest scatter (trading days) |

### Predictor Selection (Sidebar → Model Configuration)

The multiselect uses a **staging → commit** pattern to prevent continuous recomputation:

1. Adjust the predictor set in the multiselect (no recomputation yet)
2. Pending diff is shown: `+2 added, −1 removed`
3. Click **✅ Apply Configuration** to commit
4. Only on apply does the engine recompute with the new predictor set and the cache clear

Available options are populated dynamically from the actual sheet columns — any column in the sheet (minus the four anchor/index columns `DATE`, `NIFTY`, `NIFTY50_PE`, `NIFTY50_EY`) appears as a selectable predictor.

---

## Mathematical Primitives

Twelve pure-NumPy functions — each with exactly one callsite and one purpose:

| Function | Used In | Purpose |
|----------|---------|---------|
| `exponential_decay_weights` | Correlations | Recency weighting |
| `weighted_spearman` | Correlations | Robust rank correlation with decay |
| `shannon_entropy` | Variable weighting | Freedman-Diaconis bin-width entropy estimation |
| `adaptive_percentile` | Mood scoring | O(N log N) sorted-insert decay-weighted CDF |
| `kalman_filter_1d` | Smoothing | Fading-memory filter with burn-in bootstrap |
| `rolling_hurst` | Diagnostics | DFA-1 with minimum 4-segment guard |
| `rolling_entropy` | Diagnostics | Market disorder measurement |
| `_ledoit_wolf_shrinkage` | Similar periods | Analytical OAS covariance shrinkage (Chen et al. 2010) |
| `mahalanobis_distance_batch` | Similar periods | Shrinkage-regularized state matching |
| `cosine_similarity` | Similar periods | Least-squares detrended trajectory shape matching |
| `detect_regime_transitions` | Diagnostics | Hurst × Entropy quadrant classification |
| `_hurst_dfa` | Internal | Detrended Fluctuation Analysis (Peng et al. 1994) |

---

## Key Features

### Engine Vectorization & Performance (v2.2.0)
The system relies entirely on compiled C-extensions under the hood, replacing all explicit Python expanding/rolling loops with $O(N)$ cumulative sums and array striding. The memory-optimized adaptive percentiles, exact average-tie ranking, and vectorized Ornstein-Uhlenbeck estimator reduce end-to-end execution time by over 99%.

### OU Forward Projection

The chart extends a dotted line `OU_PROJ_DAYS` = 90 days beyond the last data point. This is the Ornstein-Uhlenbeck expected reversion path:

```
E[mood(t+n)] = μ + (mood_current − μ) · exp(−θ · n)
```

The `EQ` label marks the equilibrium level with the OU half-life in days.

### Kalman Confidence Bands

A translucent band surrounds the mood score line showing `±KALMAN_CI_Z` standard deviations of the Kalman filter's estimate variance (~95% interval). A tight band means a confident reading; a wide band means the system is uncertain. A mood of +40 with tight bands is fundamentally different from +40 with wide bands.

### Regime Detection

Uses Hurst exponent × entropy to classify the market into four quadrants:

| Regime | Hurst | Entropy | Implication |
|--------|-------|---------|-------------|
| Trending | > 0.5 | Low | Momentum strategies work |
| Volatile Trend | > 0.5 | High | Directional with large swings |
| Mean-Reverting | < 0.5 | Low | Contrarian / range strategies work |
| Choppy | < 0.5 | High | Hardest to trade — reduce size |

Regime transitions are marked as vertical dotted lines on the mood chart. The current regime appears as a diagnostic card in the top metrics row.

### Similar Periods — Forward Returns

Each analog card shows what happened to NIFTY 30, 60, and 90 days after that historical state. Aggregate summary cards show median returns and win rates across all matched analogs.

### Backtest Scatter

The Similar Periods tab includes a scatter of mood score at T vs NIFTY return at T+`BACKTEST_HORIZON` = 30 days, for all historical data. Displays the Pearson ρ with a regression line and plain-English interpretation.

### Data Staleness Warning

If the most recent row is more than 3 calendar days old (accounting for weekends), a red banner warns that scores reflect stale data and the Google Sheet needs updating.

### MSF Component Decomposition

Below the period summary, a breakdown shows each MSF component's current contribution and period average. Component weights are recalculated each run via inverse-variance allocation.

---

## Setup

### Local

1. Create a Google Cloud service account with Sheets read-only scope and share the target sheet with its email (Viewer is enough)
2. Copy `.streamlit/secrets.toml.example` → `.streamlit/secrets.toml` and fill in the service account fields and sheet coordinates
3. Install dependencies: `pip install -r requirements.txt`
4. Run: `streamlit run arthagati.py`

### Streamlit Cloud (GitHub deployment)

1. Push the repo to GitHub — `secrets.toml` is gitignored and will not be committed
2. In Streamlit Cloud: **App Settings → Secrets** — paste the full TOML content from your local `secrets.toml`
3. Deploy — the app reads `st.secrets` identically in both environments; no code change needed

---

## Version History

| Version | Changes |
|---------|---------|
| v1.2.0 | Initial release: Pearson correlations, expanding percentiles, fixed MSF weights |
| v2.0.0 | Decay-Spearman correlations · Adaptive percentiles · OU normalization · Kalman smoothing · Inverse-variance MSF · Mahalanobis similarity · Predictor quality assessment · Apply-button config · EY auto-derivation · Yield term spreads |
| v2.1.0 | OU forward projection (90d) · Kalman confidence bands (±1.96σ) · Data staleness warning · MSF component decomposition · Forward returns in similar periods (30/60/90d) · Backtest scatter · Regime transition detection · Diagnostic cards row · Dynamic predictor options from sheet columns · Named constants for all hyperparameters |
| v2.2.0 | Performance Architecture Rewrite: Complete C-level NumPy vectorization of mathematical primitives · Replaced Python expanding/rolling loops with O(N) cumulative sums · Memory-optimized 1D slice lookbacks for adaptive percentiles (preventing O(N²) memory blowout) · Vectorized Ornstein-Uhlenbeck estimation, Kalman filter variances, and Mahalanobis similarity via array striding · 99%+ reduction in engine execution time |
| v2.5.0 | Production Readiness & Code Cleanup: Removed dead `ornstein_uhlenbeck_estimate()` function · Removed unused `kalman_gains` return value · Added type hints · Version consistency |
| v2.2.1 | UI Rendering & Memory Optimizations: Migrated regime transitions to WebGL (`go.Scattergl`) to prevent DOM bloat · Bounded Streamlit caching (`max_entries=5`) to prevent server RAM blowout |

---

*© 2026 Arthagati · Hemrek Capital*
