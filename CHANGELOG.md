# Changelog

All notable changes to this project will be documented in this file.

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