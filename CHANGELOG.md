# Changelog

All notable changes to this project will be documented in this file.

## [v2.4.0] - Adversarial Audit Resolution

### Fixed
- **OU Residual Sum of Squares (H2)**: Replaced algebraic expanding RSS formula (incorrect when expanding AR(1) coefficients vary per step) with per-observation residuals `e²_i = (y_i − a_i − b_i·x_i)²` accumulated via expanding mean. Sigma and half-life diagnostics are now mathematically correct.
- **Backward Information Leakage (M5)**: Removed `bfill()` from data imputation. Only `ffill()` is applied; early NaN values remain NaN and are handled by `np.isfinite()` guards in all math primitives. Prevents future data leaking into early observations.
- **DFA Segment Guard (M1)**: Increased minimum segment count from 1 to 4 per Peng et al. (1994) recommendation, preventing degenerate single-segment Hurst estimates.
- **MSF Regime Trend Artifact (M4)**: Replaced unbounded `cumsum()` with windowed `rolling(MSF_WINDOW).sum()` to prevent directional count drift that created false regime signals over long histories.
- **Rolling Entropy Off-by-One (L3)**: Fixed `sliding_window_view` scope and result index alignment so `result[i]` correctly corresponds to entropy of the window ending at index `i`.
- **Sigmoid Overflow (L4)**: Added input clipping (`±500`) before `np.exp()` to prevent overflow for extreme z-scores.
- **rolling_mean_fast NaN Semantics (L6)**: Returns `NaN` instead of `0.0` for all-NaN windows, preventing downstream consumers from treating missing data as zero.

### Changed
- **O(N log N) Adaptive Percentiles (H1)**: Replaced O(N²) inner loop with sorted-insert + `np.searchsorted` binary search. Maintains a sorted value array with insertion times; computes weighted CDF via vectorised decay on the sorted subset. Inspired by Greenwald & Khanna (2001) streaming quantile approach.
- **Kalman Warm-Up Bootstrap (H3)**: Early expanding variance estimates (first 50 observations) are bootstrapped from the first stable window per Harvey (1990), preventing poorly calibrated Kalman gains during warm-up.
- **Freedman-Diaconis Entropy Bins (M2)**: Shannon entropy bin count now uses Freedman-Diaconis rule (`bin_width = 2·IQR·n^{-1/3}`) instead of capped `sqrt(N)`, providing data-adaptive bin selection.
- **Ledoit-Wolf Covariance Shrinkage (M3)**: Mahalanobis distance now uses analytical Ledoit-Wolf shrinkage (OAS variant, Chen et al. 2010) instead of ad-hoc diagonal regularization. Always well-conditioned.
- **Walk-Forward Weight Blending (M6)**: Checkpoint correlation weights are exponentially blended with previous checkpoint (`α ≈ 0.29`, half-life = 2 checkpoints) to eliminate discontinuous jumps at segment boundaries.
- **Confidence Band Soft-Clip (L2)**: Replaced hard `np.clip(±100)` with `tanh(x/100)·100` so band width is preserved near score extremes, maintaining visual uncertainty information.
- **Least-Squares Trajectory Detrend (L5)**: Similar-period trajectory matching now uses least-squares linear detrend (minimises residual variance) instead of endpoint anchoring, which distorted V-shaped or reversal trajectories.
- **Backtest Train/Test Split (L1)**: Backtest scatter now shows 70/30 chronological train/test split with separate in-sample and out-of-sample Pearson and Spearman correlations. Fit lines are trained on the 70% only.

## [v2.3.0] - Walk-Forward Correlations & Bias Corrections

### Fixed
- **Look-Ahead Bias (C1/C2)**: Restructured Layer 1-2 to use expanding-window walk-forward correlations at quarterly checkpoints instead of full-sample.
- **Percentile Semantics (L1)**: Symmetric [-1,+1] adjustments for PE and EY anchors, fixing asymmetric bearish/bullish capacity.
- **Hurst Bias (M2)**: Replaced R/S with DFA-1 (Peng et al. 1994, Weron 2002).
- **OU AR(1) Bias (M1)**: Applied Kendall-Marriott-Pope first-order correction to expanding AR(1) coefficient.
- **Dynamic Y-Axis**: Mood chart scales to actual data bounds with 8% padding.

## [v2.2.1] - UI Rendering & Memory Optimizations

### Changed
- **WebGL Chart Rendering**: Grouped and migrated regime transition markers from individual SVG layout shapes (`add_vline`) to interleaved WebGL traces (`go.Scattergl`). This eliminates DOM bloat and restores smooth panning/zooming, especially on the 'MAX' timeframe.

### Fixed
- **Cache Memory Bloat**: Applied `max_entries=5` to all heavy `@st.cache_data` decorators. This strictly caps the server memory footprint, preventing RAM blowout when users rapidly toggle different predictor configurations.

## [v2.2.0] - Performance & Vectorization Architecture Rewrite

### Added
- **C-Level Vectorization Engine**: Migrated underlying mathematical primitives away from Python-level `for` loops into compiled C-extensions via NumPy `cumsum` and `sliding_window_view`.
- **O(N) Moving Averages & Variances**: Replaced heavy Pandas object instantiations (`.rolling().mean()`, `.expanding().std()`) with mathematically exact $O(N)$ NumPy cumulative sums, providing sub-millisecond execution.
- **Pure-NumPy Ranking**: Swapped Pandas `.rank()` inside the weighted Spearman calculation with a custom C-vectorized tie-averaging rank algorithm.

### Changed
- **Kalman Filter (Sorenson & Sacks)**: Implemented an exponential fading memory factor into the 1D Kalman filter to ensure variance predictions correctly discount non-stationary regimes.
- **Ornstein-Uhlenbeck Estimation**: Converted the $O(N^2)$ expanding-window OU estimation loop into a single-pass $O(N)$ vectorized algorithm. 
- **Trajectory Similarity**: Migrated the 20-day cosine similarity trajectory matching from explicit iteration into instantaneous matrix multiplications using array striding.
- **Regime Transition Detection**: Fully vectorized the Hurst and Entropy quadrant classification logic. 

### Fixed
- **Memory Blowout in Adaptive Percentiles**: Fixed an issue where 2D NumPy broadcasting created an $O(N^2)$ memory footprint leading to 40GB+ RAM allocations. Rewrote to use an $O(N)$ memory 1D slice lookback, reducing execution time of the Mood Engine from ~120s down to <2s.

## [v2.1.0] - Diagnostics & Forward Returns
- Added 90-day OU forward mean-reversion projection.
- Added ±1.96σ Kalman confidence bands.
- Added forward return outcomes (30, 60, 90-day) to similar historical periods.
- Added backtest scatter plot for NIFTY return validation.
- Added explicit data staleness warnings for the Google Sheets fetch.

## [v2.0.0] - Physics-Informed Mathematics
- Overhauled the sentiment engine to use Ornstein-Uhlenbeck stochastic processes.
- Implemented Mahalanobis distance for covariance-aware state matching.
- Replaced fixed thresholds with inverse-variance (Markowitz) weighted MSF components.