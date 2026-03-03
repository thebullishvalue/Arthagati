# ARTHAGATI v2.0 — Backend Architecture Vision
### Hemrek Capital | Architect's Working Document

---

## 1. SYSTEM PURPOSE (First Principles)

Arthagati answers one question: **"What is the market's current sentiment state, and how confident should I be in that reading?"**

It does this by:
- Taking **macro/breadth/valuation** inputs (PE, EY, yields, breadth, policy rates)
- Computing **correlations** between these variables and valuation anchors (PE, EY)
- Constructing a **weighted composite sentiment score** (Mood Score)
- Building a **multi-component oscillator** (MSF Spread) for momentum/structure confirmation
- Finding **historical analogs** with similar market states
- Showing **correlation structure** for transparency

Every mathematical choice must serve this pipeline. If a theory doesn't improve one of these 6 steps, it doesn't belong.

---

## 2. ALL CANDIDATE THEORIES (Full Table)

### A. Correlation Estimation

| Theory | What It Does | Relevance to Arthagati |
|--------|-------------|----------------------|
| Pearson correlation | Linear association, full sample | **Current v1.x** — assumes linearity & stationarity |
| Spearman rank correlation | Monotonic association, robust to outliers | ✅ Macro↔valuation relationships are monotonic |
| Exponential decay weighting | Recent data weighted more | ✅ Correlation structure is non-stationary |
| Mutual Information | Captures arbitrary nonlinear dependence | ⚠️ Overkill — macro↔PE relationships are monotonic, not U-shaped |
| Dynamic Conditional Correlation (DCC-GARCH) | Time-varying correlation model | ❌ Heavy, needs MLE, fragile with limited data |
| Copulas | Joint distribution modeling | ❌ Overkill for weight estimation |
| Kendall's tau | Concordance-based rank correlation | ⚠️ Similar to Spearman, slower to compute, no clear advantage |

### B. Variable Weighting

| Theory | What It Does | Relevance |
|--------|-------------|-----------|
| Correlation magnitude | Weight = |corr| | **Current v1.x** — treats noisy high-corr variables same as stable ones |
| Shannon entropy penalty | Weight = |corr| × (1 - entropy) | ✅ Suppresses noisy/random variables |
| Mutual Information weighting | Weight = MI(var, anchor) | ⚠️ More principled but Spearman + entropy achieves 90% of the benefit |
| LASSO/Ridge regularization | Penalized regression weights | ❌ Wrong framing — we're not predicting, we're constructing a composite |
| PCA loading weights | Variance-explained based | ❌ Loses interpretability, unclear sign convention |

### C. Historical Positioning (Percentiles)

| Theory | What It Does | Relevance |
|--------|-------------|-----------|
| Expanding rank | Percentile against all history | **Current v1.x** — 2005 data pollutes 2025 percentiles |
| Decay-weighted empirical CDF | Recent history weighted more | ✅ Market structure evolves; percentiles should adapt |
| Kernel Density Estimation | Full distribution estimate | ⚠️ More than we need — percentile is the sufficient statistic |
| Regime-conditional percentile | Percentile within current regime | ⚠️ Requires regime detection first — circular |

### D. Score Normalization

| Theory | What It Does | Relevance |
|--------|-------------|-----------|
| Global z-score | (x - μ_all) / σ_all × 30 | **Current v1.x** — adding 1 point shifts ALL history |
| Expanding z-score | Running mean/std | Better, but still treats history equally |
| Ornstein-Uhlenbeck | Mean-reverting diffusion model | ✅ **Mood IS mean-reverting.** OU gives natural units, equilibrium, reversion speed, and half-life |
| Quantile normalization | Rank → uniform → normal | ❌ Destroys magnitude information |

### E. Smoothing

| Theory | What It Does | Relevance |
|--------|-------------|-----------|
| Simple Moving Average | Fixed window average | **Current v1.x** — arbitrary window, uniform weights |
| EMA | Exponential moving average | Better than SMA, but still fixed bandwidth |
| Kalman filter (1D) | Adaptive state estimation | ✅ Auto-adjusts smoothing to signal-to-noise ratio |
| Savitzky-Golay | Polynomial smoothing | ⚠️ Preserves peaks but fixed bandwidth |
| Hodrick-Prescott | Trend-cycle decomposition | ❌ Macro-focused, endpoint instability |
| Wavelet denoising | Multi-scale decomposition | ❌ Overkill, hard to interpret |

### F. Oscillator Construction (MSF)

| Theory | What It Does | Relevance |
|--------|-------------|-----------|
| Fixed weights | Arbitrary allocation | **Current v1.x** — 30/25/25/20 with no basis |
| Inverse-variance weighting | Stable signals get more weight | ✅ Minimum-variance portfolio of signals (Markowitz for signals) |
| Fixed regime threshold | One threshold for all regimes | **Current v1.x** — 0.0033 is arbitrary |
| Adaptive threshold | Scale with local volatility | ✅ A move is "directional" if it exceeds local noise |
| Entropy/Hurst as components | Add disorder/persistence to oscillator | ❌ Dilutes the oscillator's purpose (momentum/structure alignment) |

### G. Diagnostics

| Theory | What It Does | Relevance |
|--------|-------------|-----------|
| Hurst exponent (R/S) | Trending (H>0.5) vs mean-reverting (H<0.5) | ✅ AS DIAGNOSTIC — tells user if mood is likely to persist or reverse |
| Shannon entropy (rolling) | Market disorder measure | ✅ AS DIAGNOSTIC — tells user if market is confused/choppy |
| OU half-life | ln(2)/θ — expected time to halve deviation | ✅ AS DIAGNOSTIC — "this extreme should normalize in ~X days" |
| Fisher Information | 1/variance — signal confidence | ❌ It's just inverse rolling variance. Not worth the naming overhead |
| Lyapunov exponents | Chaos detection | ❌ Unstable estimates, needs >10,000 points |
| Fractal dimension | Complexity of price path | ❌ Hurst already captures this (D = 2 - H) |

### H. Similar Period Matching

| Theory | What It Does | Relevance |
|--------|-------------|-----------|
| Manhattan distance (2 features) | Simple absolute difference | **Current v1.x** — ignores covariance, too few features |
| Mahalanobis distance | Covariance-aware distance | ✅ Correlated features don't double-count |
| Dynamic Time Warping (DTW) | Time-elastic sequence matching | ⚠️ More principled for trajectories but O(n²), heavy |
| Cosine similarity on trajectories | Shape matching (direction, not magnitude) | ✅ Lighter DTW approximation, captures "are we on a similar path?" |
| k-NN with rich features | Multi-feature nearest neighbor | Essentially what we're building with Mahalanobis |

### I. Data Engineering

| Theory | What It Does | Relevance |
|--------|-------------|-----------|
| Raw yields as inputs | IN10Y, IN02Y, US10Y, US02Y separately | **Current v1.x** — raw yields are correlated; spreads carry more info |
| Term spread extraction | 10Y − 2Y (yield curve slope) | ✅ Classic recession/expansion signal, orthogonal info vs raw yields |
| Real rate computation | Nominal yield − inflation | ⚠️ Useful but adds complexity |
| Sovereign spread | IN10Y − US10Y | ⚠️ Useful but adds complexity |

---

## 3. CURATION MATRIX

### Principle: **Every theory gets exactly ONE job. No theory appears in two layers.**

| Theory | VERDICT | Layer | Job |
|--------|---------|-------|-----|
| Exponential-decay Spearman | ✅ KEEP | Correlation | Replace static Pearson |
| Shannon entropy (variable weighting) | ✅ KEEP | Weighting | Penalize noisy variables |
| Adaptive percentile (decay ECDF) | ✅ KEEP | Positioning | Replace expanding rank |
| Ornstein-Uhlenbeck estimation | ✅ KEEP | Normalization | Physics-based scaling + half-life diagnostic |
| Kalman filter (1D) | ✅ KEEP | Smoothing | Adaptive noise filtering |
| Inverse-variance weighting | ✅ KEEP | MSF Oscillator | Replace fixed weights |
| Adaptive regime threshold | ✅ KEEP | MSF Oscillator | Replace fixed 0.0033 |
| Hurst exponent (rolling) | ✅ KEEP | Diagnostics ONLY | Output: trending vs reverting |
| Shannon entropy (rolling) | ✅ KEEP | Diagnostics ONLY | Output: market disorder |
| OU half-life | ✅ KEEP | Diagnostics ONLY | Output: expected normalization time |
| Mahalanobis distance | ✅ KEEP | Similar Periods | Covariance-aware matching |
| Cosine trajectory similarity | ✅ KEEP | Similar Periods | Path shape matching |
| Term spread extraction | ✅ KEEP | Data Engineering | Derive 10Y−2Y spreads |
| Entropy gate on mood score | ❌ CUT | — | Over-compresses useful signals during volatile periods |
| Hurst-adjusted classification | ❌ CUT | — | Hurst estimates are noisy; dynamic thresholds add instability |
| Entropy/Hurst as MSF components | ❌ CUT | — | Dilutes oscillator purpose (alignment detection) |
| Fisher Information column | ❌ CUT | — | Just 1/variance; not actionable |
| Mutual Information | ❌ CUT | — | Spearman + entropy achieves 90% of the benefit, simpler |
| DCC-GARCH | ❌ CUT | — | Too heavy for Streamlit app with ~1500 rows |
| Wavelet/Lyapunov/Fractal | ❌ CUT | — | Overkill, unstable estimates |

---

## 4. FINAL ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DATA INGESTION                                    │
│  Google Sheets → Clean → Derive Term Spreads (IN/US 10Y−2Y)        │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│              LAYER 1: ADAPTIVE CORRELATIONS                         │
│  Exponential-decay weighted Spearman rank correlation               │
│  Each variable → PE anchor correlation, EY anchor correlation       │
│  Half-life: ~504 days (2 trading years)                             │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│         LAYER 2: INFORMATION-THEORETIC WEIGHTING                    │
│  weight = |correlation| × (1 − normalized_entropy(variable))       │
│  Entropy computed on variable's returns distribution                │
│  Noisy/random variables suppressed, structured ones amplified       │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│           LAYER 3: ADAPTIVE PERCENTILES                             │
│  Decay-weighted empirical CDF (half-life: ~252 trading days)        │
│  "Where is PE today vs recent-ish history?" not vs all-time         │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│           LAYER 4: OU NORMALIZATION                                 │
│  Estimate Ornstein-Uhlenbeck: dx = θ(μ − x)dt + σdW               │
│  Normalize: score = (x − μ) / (σ/√2θ) × 30                        │
│  Diagnostics: half-life = ln(2)/θ, equilibrium = μ                  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│            LAYER 5: KALMAN SMOOTHING                                │
│  1D Kalman filter with auto-estimated noise parameters              │
│  High noise → more smoothing, low noise → tracks signal             │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│          OUTPUT: MOOD SCORE + DIAGNOSTICS                           │
│  Mood Score: [-100, +100] with fixed classification thresholds      │
│  Diagnostics (enriched columns, NOT score modifiers):               │
│    • Hurst exponent → trending vs mean-reverting character          │
│    • Market entropy → ordered vs disordered regime                  │
│    • OU half-life → expected days to normalize from current extreme  │
└─────────────────────────────────────────────────────────────────────┘
```

### MSF Spread (Parallel Pipeline)

```
┌─────────────────────────────────────────────┐
│  Component 1: Momentum (NIFTY ROC z-score)  │──┐
│  Component 2: Structure (Mood trend div.)   │──┤
│  Component 3: Regime (adaptive threshold)   │──┤ Inverse-Variance
│  Component 4: Flow (breadth divergence)     │──┤ Weighting
└─────────────────────────────────────────────┘  │
                                                  ▼
                                          MSF Spread [-10, +10]
```

### Similar Periods (Matching Engine)

```
Feature Vector = [mood, volatility, NIFTY_ROC, Hurst, entropy, term_spread]
                            │
              ┌─────────────┴──────────────┐
              ▼                            ▼
     Mahalanobis Distance       Cosine Trajectory Similarity
     (state matching: 55%)      (path shape matching: 35%)
              │                            │
              └──────────┬─────────────────┘
                         ▼
              Exponential Recency Decay (10%)
                         │
                         ▼
                Combined Similarity Score
```

---

## 5. WHAT v2.0 CUTS FROM THE v1.x → v2.0-draft OVERHAUL

| Feature from draft-v2.0 | Why it's cut |
|--------------------------|-------------|
| Entropy gate on mood scores | Compresses legitimate signals during volatile regimes. If the market IS chaotic, the mood score should reflect that chaos — not be silenced. |
| Hurst-adjusted classification thresholds | Hurst estimates are noisy (±0.1 variance). Using them to dynamically shift classification thresholds makes the system unstable. A "Bullish" reading flickering to "Neutral" because Hurst jittered is worse than fixed thresholds. |
| Entropy as MSF component | The MSF measures momentum↔sentiment alignment. Adding entropy dilutes this. Entropy belongs in diagnostics. |
| Hurst as MSF component | Same reasoning. Persistence is a meta-property of the signal, not a signal component. |
| Fisher Information | It's 1/rolling_variance. Computing it, naming it, and outputting it adds complexity for something the user can eyeball from the volatility column. |
| 14 primitive functions | Reduced to 11. Each one now has exactly one callsite and one purpose. |

---

## 6. KEY DESIGN DECISIONS

**Q: Why fixed mood classification thresholds (±20, ±60) instead of adaptive?**
A: Stability. When a user sees "Bullish" today and "Bullish" yesterday, those should mean the same thing. Adaptive thresholds mean the label "Bullish" changes meaning over time. The OU normalization already handles scale adaptation — the classification layer should be stable.

**Q: Why Hurst/entropy as diagnostics only?**
A: They inform the INTERPRETATION of the mood score, not its VALUE. "Mood is +45 (Bullish), Hurst is 0.62 (trending), entropy is low (ordered)" tells the user: "This bullish reading is in a trending, ordered market — trust it." If we embedded Hurst into the score, the user loses this interpretive layer.

**Q: Why inverse-variance for MSF but not for mood?**
A: The mood is a fundamentally different construction — it's a correlation-weighted composite of percentiles, not a portfolio of signals. The weights come from correlation strength × entropy quality, which is the right weighting for that layer. MSF components, being different technical indicators, ARE signal portfolios and benefit from variance-based allocation.

**Q: Why term spreads but not real rates or sovereign spreads?**
A: 10Y−2Y is the single most studied and validated macro signal in financial economics (every recession since 1960). It adds genuine orthogonal information vs raw yields. Real rates and sovereign spreads add marginal value but double the number of derived features.
