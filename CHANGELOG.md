# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [v2.10.0] — 2026-08-18

### Measured Predictor Selection · Intelligence Mode Removed

Everything in this release is a consequence of pointing the v2.9.0 measurement
apparatus at the real sheet (NIFTY, 4,985 rows, 2006–2026) instead of at
synthetic data.

#### Removed — Intelligence Mode

The post-engine Optuna ensemble is gone. It **reduced** the signal's
out-of-sample power on every configuration tested:

| Predictor set | raw Mood Score | fitted ensemble | margin |
|---|---|---|---|
| selected 4 | +1.674 | +1.239 | −0.436 |
| current 12 | +1.893 | −0.444 | −2.337 |
| all 37 | +1.753 | +1.363 | −0.390 |

Only `mood` carries forward information (holdout rho +0.31 at 90d); the four
MSF components sit between −0.03 and +0.01. Maximising an information ratio
across CV folds rewards in-sample fit, so the search loaded on the technicals —
in the 12-predictor run it weighted `mood` at **−0.37**, inverting its one
useful input. The v2.9.0 quality gate caught this every time and refused to
activate; a component whose own gate rejects it on every real configuration is
not a feature.

Deleted: `intelligence.py`, `ui/tabs/tab_intelligence.py`, the sidebar Model
Passport, the Calibrated Conviction card, `profiles/`, the `optuna` dependency,
and `tests/test_calibration.py`.

#### Added — `validation.py` + Signal Validation view

The measurement apparatus was the good part and survives, repointed from
"tune an ensemble" to "does this signal work". Holdout, embargo, permutation
null, power floor — no fitting.

- Horizons the holdout cannot support are reported as **descriptive** and
  excluded from the verdict, rather than the whole verdict being withheld.
  A 1,246-row holdout supports +20D/+60D; +125D/+250D are shown marked `*`.
- Baseline is `−PE` — the engine must beat "cheap is good with no engine".

**Result on the reference sheet**: `Edge Confirmed`, holdout rho **+0.538**,
**p = 0.005**. The margin over the `−PE` baseline is **+0.006**.

#### Changed — default predictors, by measurement

65 sheet columns → 37 eligible (dropping 26 NIFTY-derived columns and 2
duplicates) → 24 cluster representatives (|rho| ≥ 0.90) → greedy forward
selection on the **development window only**, holdout scored once at the end.

New defaults: **`SPREAD_02Y`, `US_TERM_SPREAD`, `CRR`, `US02Y`** — all rate
and liquidity variables.

```
development rho   current 12 +0.189  ->  selected 4 +0.334   (+77%)
holdout rho       current 12 +0.526  ->  selected 4 +0.544   (+3%)
```

Reported honestly: **most of the development gain did not transfer.** The
holdout spread across every set tested is +0.53 to +0.55 (breadth-only
excepted at +0.43), which is within noise for this sample.

- The breadth family ranked last in the univariate screen (+0.10 to +0.14
  against +0.29 for `SPREAD_02Y`) and is no longer a default.
- `IN_TERM_SPREAD`, promoted to a default in v2.9.0 on the strength of
  VISION §2-I, ranked **last of 37** at +0.094 and has been dropped. The
  design document's argument did not survive measurement.
- NIFTY-derived columns are withheld from the predictor multiselect entirely —
  selecting one makes the valuation score a function of the price it is then
  scored against.

#### Added — the ablation, in the README

| Signal | dev rho | holdout rho |
|---|---|---|
| `−PE`, no engine | +0.467 | **+0.549** |
| PE percentile only | +0.326 | +0.543 |
| PE+EY base, no predictors | +0.327 | +0.543 |
| Full engine, selected 4 | +0.334 | +0.544 |
| Full engine, all 37 | +0.238 | +0.555 |

The five-layer pipeline adds no rank information over inverting PE. It
contributes a bounded comparable score, a confidence band, an equilibrium and
half-life, and regime context. That is a reasonable product; it is now stated
rather than implied.

#### Fixed — from the real-data walkthrough

- **The interpretive layer was inverted.** The Mood Score correlates **−0.54**
  with the trailing 60-day return and **+0.22** with the forward 250-day
  return: it is a valuation-contrarian gauge. The UI called it sentiment and
  advised trend-following. In October 2008 it read **+21 to +39 ("Bullish")**
  while NIFTY fell 25%; through the 2020–21 melt-up it read **−36 to −15**.
  Both readings were correct; the guidance was backwards. Rewritten with the
  historical evidence stated inline.
- **Hurst was pinned at its clip** — 87.3% of rows at 0.99, because DFA on an
  integrated series returns H>1. Regime had collapsed to two quadrants
  (45%/45%). Now measured on mood *increments*: p5/p50/p95 0.23/0.48/0.79, and
  all four quadrants are populated (28/25/23/19%).
- **"Very Bearish" had never fired in twenty years** — not in the GFC, not in
  COVID. The score's realised 1st–99th percentile range is −49 to +56, so the
  ±60 outer band was unreachable. Bands are now ±20/±45 and all five classes
  occur (Very Bearish 2.8%, Very Bullish 3.4%).
- Per-horizon cells in the old lift table always rendered 0.000 — the IR floor
  of 4 samples exceeded the 3 blocks a single horizon has.

#### Known limitations

- The holdout is five years. Differences between predictor sets are within its
  resolution; treat the selected four as "no worse, much simpler", not as
  established superiority.
- The engine's strongest relationship is at 250 days, which 20 years of history
  cannot validate to this standard (5 independent windows against a floor of
  10). It is reported as descriptive.
- The sheet stores `NIFTY50_EY`, `COR. EY` and `EY_DEV` as percent strings
  (`"6.48%"`), which parse to all-NaN. EY is silently re-derived as `1/PE`;
  the other two are excluded.

---

## [v2.9.0] — 2026-08-18

### Audit Remediation

A full adversarial audit of v2.8.0 found that the system's three
self-validation surfaces — the backtest scatter, the Intelligence Mode
quality gate, and the Predictive Power Lift table — were each constructed so
that they could not fail, and that the mood and MSF series were not causal
despite documentation claiming otherwise. This release fixes the defects and
adds the regression tests that would have caught them.

#### Fixed — Correctness

- **Look-ahead in the mood score.** The walk-forward block applied
  correlation weights estimated on data through the END of the segment it was
  scoring, so a score at time *t* depended on up to 63 days of its own
  future. Measured: perturbing only rows after index 300 moved scores inside
  the untouched prefix by as much as **12.75 points**. Segment *k* now reads
  its statistics from checkpoint *k−1*; the regression asserts bit-equality
  to 1e-9. Rows before `CORR_MIN_WARMUP` borrow the first checkpoint's
  statistics and are flagged `Is_Warmup`, and every evaluation path excludes
  them.
- **Look-ahead in the MSF Spread.** Component weights were derived from the
  variance of the trailing 60 rows and applied across all of history, so
  past MSF values shifted whenever new data arrived (measured: up to 0.52 on
  a ±5 band). Variance is now expanding.
- **Look-ahead in the regime labels.** Both classification axes were split at
  the median of the *whole* series. Now split at their own expanding median.
- **MSF degenerate-component capture.** A component with no variance took
  `1/1e-6` inverse-variance, won ~100% of the weight, and — being identically
  zero after the z-score/sigmoid chain — flattened the composite to a
  constant (measured std 0.0001 against a healthy 1.95). A sheet missing its
  `AD_RATIO` column triggered it silently. Weights are clamped to
  `[0.10, 0.50]`; a dead component is excluded and reported in the UI.
- **Percentile overshoot.** The decayed ECDF could return 1.0000000000000002,
  which then flowed into Layer 3's `1 − 2·pct` mapping. Clipped to `[0, 1]`.
- **Zero-crossing return blow-ups.** Entropy was estimated on `diff / |prev|`.
  Both term spreads and `PE_DEV` / `EY_DEV` cross zero, and a near-zero
  denominator produced changes of several hundred ×, dominating the
  Freedman-Diaconis bin width. Now first differences.
- **`kalman_filter_1d` returned a 3-tuple on empty input** while annotated
  and unpacked as 2.
- **Backtest split had no embargo** — the last `horizon` training points drew
  their labels from inside the test window.
- **Coverage counted NaN as data.** `raw_df[var] != 0` is True for NaN, so an
  all-NaN column reported 100% coverage.

#### Changed — Intelligence Mode

The quality gate previously asked whether the optimised validation IR
exceeded zero, while Optuna maximised `0.65·val_IR + 0.35·train_IR` — testing
the objective against itself. On forward returns drawn from an independent
random walk it reported train IR 1.56, val IR 0.71, stability 45% and a
**Quality OK** badge, on five seeds out of five.

- **Holdout.** The final 25% of the series is withheld from the search, the
  CV folds, and the feature standardisation, and scored once afterwards. The
  holdout IR is now what the gate and the UI report.
- **Embargo = max(horizon) = 90 days**, was 5 against horizons up to 90 — so
  training labels reached 85 days into the validation window.
- **Permutation null.** 200 circular shifts of the signal against the same
  returns; the statistic is mean rank correlation, which is far more stable
  than the IR on a dozen correlated measurements.
- **Power floor.** Forward windows overlap, so a 600-row holdout carries ~6
  independent observations at a 90-day horizon — too few for any threshold to
  discriminate. Below `GATE_MIN_INDEPENDENT_WINDOWS` (10) the calibrator
  returns **Insufficient Data** and declines to grade rather than guessing.
- **Feature matrix reduced from 10 to 7 columns.** It carried five
  near-collinear transforms of the mood score (pairwise |ρ| 0.91–0.97,
  condition number 1.2 × 10¹⁷), making the per-feature weights and their
  Bullish/Bearish badges unidentifiable. Condition number is now ~10⁴.
  `l2_alpha` raised 0.001 → 0.05, and standardisation is expanding rather
  than full-sample.
- **Feature "importance"** was Optuna fANOVA over the objective — a property
  of the search surface, not of the feature. Now contribution share of the
  fitted coefficients.
- **Predictive Power Lift** scored raw Mood and Calibrated Conviction on the
  very folds the weights were fitted to, so a positive lift was guaranteed;
  on random-walk returns it still reported +0.92. Now scored on the holdout.
- **Only `Quality OK` activates.** `Overfit` profiles were previously saved
  *and* applied.
- **Profile import is validated** — feature names, numeric types and weight
  bounds — and the grade is recomputed rather than trusted. A weight of 5000
  used to import cleanly and produce a permanently saturated ±100 signal.
- **Stability is holdout/train**, was val/train — a ratio the objective
  actively pushed above 1 by weighting val at 0.65 against train at 0.35.
- **Row counts** summed overlapping expanding folds, reporting 5,140 "train
  rows" for a 2,200-row series. Now unique rows.
- Schema version 3; v2 profiles are rejected on load.

#### Changed — Performance

- **`adaptive_percentile` is O(N log N)**, was O(N²) despite the README and
  its own docstring claiming otherwise. Rewritten over a Fenwick tree of
  value ranks: the `exp(-λt)` factor cancels between numerator and
  denominator, leaving a prefix-sum. Verified bit-exact (4.5e-15) against a
  direct transcription of the definition. 4,000 rows: 436 ms → 37 ms. The
  full engine on 5,000 rows: ~30 s → **6.9 s**.

#### Changed — Views

- **Timeframe windows filter by date, not row count.** Calendar-day constants
  were passed to `.tail(n)`, stretching every window by ~1.4× — "1Y" returned
  365 rows spanning 510 calendar days, "5Y" spanned just over seven years.
- **WaveTrend OB/OS bands are calibrated from the data** (95th/80th
  percentile of |wt1| over the full history). The inherited ±80 was
  unreachable when the source is `Mood_Score` — |wt1| peaks near 70 — leaving
  ~40% of the pane permanently empty.
- **Analog separation.** Analogs are selected greedily with a 20-day minimum
  gap; five of ten previously fell inside a 32-row window while the UI quoted
  their median return as ten independent observations. The trailing 90 rows
  are excluded, and `recency_weight` is now the actual blend weight — it was
  scaled in and normalised straight back out, making it a no-op.
- **Predictor Quality scores every numeric column.** It read |ρ| from frames
  built only from the *active* set, so an inactive predictor always scored
  0.00 and was badged "Weak" — the panel meant to guide predictor selection
  could never recommend anything not already selected.
- **Signal precedence is stated.** Mood Score and Calibrated Conviction
  differ in sign on ~46% of days; the UI now says which governs and why.
- **The Intelligence Center no longer renders a stale profile as live.** A
  run rejected by the gate left the previous profile on screen while the
  metric strip hid the conviction card.
- Backtest labels said "+30d" while the horizon was 20.

#### Changed — Infrastructure

- **`config.py`** holds all schemas, hyperparameters and display constants.
  `ui/tabs/*` imported constants via `from arthagati import ...`; because
  Streamlit runs the entrypoint as `__main__`, that re-executed the whole
  script as a second module object — confirmed under a real `ScriptRunContext`
  to inject the theme CSS twice, and a hard error on the Streamlit 1.32 floor
  the requirements declared.
- **Cache clears are scoped** to this app's cached functions.
  `st.cache_data.clear()` is process-global and flushed every concurrent
  user's frames.
- **Reset to Defaults is session-scoped.** It deleted `active.json`, which is
  shared by every session of a deployment.
- **`ARTHAGATI_PROFILE_DIR`** relocates the profile store; archives are
  pruned to the most recent 20.
- `datetime.utcnow()` → timezone-aware `datetime.now(timezone.utc)`.
- Removed the dead `hyperparam_overrides` / `TUNABLE_HYPERPARAMS` /
  `get_default_hyperparams` machinery, whose docstrings described an
  architecture the same file repudiated 1,800 lines later. Also removed
  `run_calibration`, `list_profiles`'s unused callers, `_render_profile_grid`,
  `_stat_card`, ~15 unused imports and three unused parameters.
- A real spreadsheet ID was embedded in the sheet-fetch error message as an
  "example".
- Version strings unified — `config.VERSION` is the single source.

#### Added

- **Test suite** (`tests/`, 46 fast + slow-marked statistical regressions):
  causality of the mood, MSF and regime series; Fenwick percentile exactness
  against a reference implementation; degenerate-component guards; analog
  separation; profile-import validation; and the headline regression that the
  gate rejects data containing no edge.
- `Is_Warmup` column marking rows scored with borrowed statistics.
- `Insufficient Data` quality grade.
- Degraded-input banner naming any MSF component excluded for lack of signal.

#### Known limitations

- The gate is **deliberately conservative**: on synthetic data carrying a
  genuine but weak edge it activates on a minority of draws. Withholding a
  real marginal signal is the cheaper error.
- Overlapping forward windows are not corrected for in the backtest scatter's
  reported coefficients; the view says so rather than adjusting them.
- Layer 4's OU rescaling is near-identity on the score (stationary std ≈ 1.09
  measured) because its input is already an expanding z-score. It is retained
  for the diagnostics it produces, and the README now says so.

---

## [v2.8.0] — 2026-05-28

### Intelligence Mode + WaveTrend + Granular Horizons

Two new indicators (WaveTrend on Mood Score, and a calibrated post-engine ensemble called *Calibrated Conviction*), plus a wholesale shift to more granular forward-return horizons (5D / 20D / 60D / 90D). A new Intelligence Center view exposes the calibration's diagnostics, per-feature weights, and predictive-power lift. Sidebar Model Passport ports Nishkarsh's fidelity for status + import / export / reset. Heavy investment in caching: engine output is session-cached by input fingerprint, calibrated profiles persist on disk and survive Streamlit Cloud cold wakes when fresh.

#### Added
- **WaveTrend Oscillator (LazyBear)** — faithful port with `hlc3` replaced by `Mood_Score`. Channel length 10, average length 21, signal-line SMA period 4. OB/OS bands at ±80 / ±60. WT1/WT2 crossover triangles (▲ at y=+70 bullish, ▼ at y=−70 bearish). Cyan zero-baselined WT1−WT2 area fill. First 31 bars masked while EMA chain warms up; divisor `d` floored at 0.5 to prevent warmup blowup. Rendered as a third pane in the Historical Mood chart with reversed y-axis (negative on top).
- **Intelligence Mode — post-engine ensemble calibration**:
  - Builds a 10-column feature matrix `F` from engine output: `mood`, `mood_smooth`, `mood_diverge`, `mood_squared`, `mood_sqrt`, `msf_spread`, plus the four MSF components.
  - Optuna TPE sampler with `MedianPruner` searches the linear weight vector `w`. Per-trial cost is one `F @ w` plus a handful of Spearman correlations (microseconds), vs. the v1 prototype that re-ran the full engine per trial (~30-60s).
  - Objective: `0.65 · Val IR + 0.35 · Train IR − L2(w)` where IR = mean Spearman / std Spearman across folds × horizons.
  - Walk-forward CV with purged 5-day embargo across 5 folds × 4 horizons.
  - Quality gate: `Val IR > 0` (else *No Edge* — profile saved but inactive) and `Stability ≥ 30%` (else *Overfit* — flagged but saved).
  - Output: `calibrated_conviction = tanh((F @ w) / 3) · 100` ∈ [−100, +100], applied across the full history.
- **Calibrated Conviction** metric card in the diagnostics strip (when Intelligence Mode is ON).
- **Intelligence Center view** — read-only dashboard:
  - Calibration Diagnostics (4-card strip: Train IR · Val IR · Stability · Quality)
  - Calibration Impact (4-card strip: Raw Mood · Calibrated Conviction · Net Shift · Direction)
  - Feature Analysis grid — per-feature card combining linear weight + fANOVA importance + Bullish/Bearish/Neutral badge, ranked by importance
  - Predictive Power Lift table — per-horizon Spearman IR comparison: raw Mood vs Calibrated Conviction
  - Profile Provenance table — run timestamp, predictors, CV setup, data window, schema version
- **Sidebar Model Passport** (Nishkarsh fidelity port):
  - Intelligence Mode toggle
  - Status card (metric-card chrome with success / warning / neutral colour classes)
  - Trained on · Train IR · Val IR · Updated rows
  - Predictor-count mismatch warning
  - ↑ Import Profile / ↓ Export Profile / ↺ Reset to Defaults
- **MSF Spread reference bands** — solid (primary) + dotted (secondary) horizontal lines at ±5 / ±3. Divergence triangles moved from ±5 to ±4 so markers and primary band don't overlap.
- **MSF y-axis range lock** — ensures the ±5 bands stay in view regardless of how compressed the MSF signal is in a given window.
- **Granular forward horizons** — `BACKTEST_HORIZON` lowered from 30 → 20. Forward-return tiles in Similar Periods analog cards now show **5D / 20D / 60D / 90D** (was 30D / 60D / 90D). Median-return summary cards updated accordingly. `find_similar_periods` returns `fwd_5d`, `fwd_20d`, `fwd_60d`, `fwd_90d`.
- **Engine-output session cache** — `mood_df` + `msf_df` cached in `st.session_state` keyed by `(row count, first date, last date, sorted predictor set)`. View switches and timeframe button clicks return in ~150ms instead of re-running the 30s engine.
- **Cross-session profile freshness check** — profile reused across Streamlit Cloud cold wakes when (a) predictor count matches, (b) data end ≤14 days newer than profile fit time, and (c) profile age ≤14 days. Quality-gate failures are *not* counted as fresh.
- **Structured pipeline summary box** in the terminal console — view mode, Intelligence on/off, predictor count, latest Mood / MSF / Calibrated Conviction, regime, OU half-life, Hurst, market entropy.
- **Modular `ui/` package** — `theme.py`, `theme.css`, `components.py`, `tabs/{landing, historical_mood, similar_periods, correlation, intelligence}.py` (Obsidian Quant fidelity, ported from v2.7.0 work).
- **Calibrated Conviction reference constants** `CC_OB_LEVEL_1 / _2 = ±100 / ±80` (for the metric card semantic colouring; the chart pane was removed in this release).
- **Structured console logging system** (`core/logger_config.py`) — phase banners, step lines, item rows, success / warning / failure / checkpoint helpers, boxed summary, per-phase elapsed times.

#### Changed
- **Historical Mood chart layout**: 2 panes (Mood + MSF) → 3 panes (Mood + MSF + WaveTrend). Row heights `[0.50, 0.25, 0.25]`. Vertical spacing 0.06.
- **Calibration architecture**: prototype v1 tuned structural hyperparameters (e.g. `CORR_HALF_LIFE`, `MSF_WINDOW`) per Optuna trial, requiring a full engine re-run per trial (~30-60s × 40 trials = unusable on Streamlit Cloud). v2 calibrates **on the engine output**, reducing per-trial cost by ~1000×.
- **Calibrated Conviction pane removed from chart** — the signal is still produced, persisted, and surfaced as a metric card and on the Intelligence Center dashboard. Removing the chart pane gives the three signal panes (Mood, MSF, WT) more vertical room.
- **OB/OS colour coding on reversed-axis panes** — Mood Score, WaveTrend, and Calibrated Conviction all reverse their y-axes. Reference-band colours now follow *sign*, not visual position: emerald on positive levels, rose on negative.
- **Top OB/OS band alpha** dimmed from 0.55 to 0.30; secondary band alpha from 0.32 to 0.16. Signal lines now dominate the visual hierarchy.
- **Plotly chart wrapper** (`.stPlotlyChart`) — `padding-bottom: 6px` and explicit `box-sizing: content-box` so the wrapper's bottom border-radius doesn't get clipped by the Plotly SVG's flush bottom edge.
- **Sidebar passport split** into pre-analysis (toggle) and post-analysis (status card + import/export/reset) halves via `st.empty()` placeholder. The freshly-saved profile is reflected on the same Run-Analysis click that produced it — no need to switch views.
- **Calibration progress messaging** — phase banner reads *Phase 5/5: Intelligence Calibration · Post-engine ensemble (Nishkarsh pattern)*; per-trial progress bar shows *Trial N/M · Optuna TPE · Best ρ:+X.XXXX*.
- **README.md** restructured around the new architecture: dedicated sections for WaveTrend and Intelligence Mode, full feature matrix and weight bounds tables, granular horizons in Similar Periods.
- **Triangle placement & marker sizes** — MSF Spread triangles moved from ±5 to ±4. Marker size constant `_TRI_SIZE = 9` shared between MSF and WaveTrend so they remain pixel-identical.

#### Removed
- **Calibration Settings expander** in the sidebar (Trials / Folds / Embargo number-inputs). Calibration now runs at factory defaults (40 trials, 5 folds, 5-day embargo) on every Run Analysis. The session-state values remain seeded with the defaults for backwards compatibility with future re-introduction.
- **Top Drivers** section in the Intelligence Center — Feature Analysis already ranks features by importance, so the side-panel was redundant.
- **Separate "Ensemble Weights" + "Parameter Importance"** sections — merged into the single **Feature Analysis** grid (one consolidated card per feature with weight bar + importance bar + direction badge).
- **Calibrated Conviction chart pane** (Row 4 in Intelligence-Mode-ON layout). The signal is still computed and surfaced elsewhere.
- **VISION.md** — superseded by the restructured README. The architecture, pipeline diagrams, and predictor schema are now in one place.

#### Fixed
- **Engine re-running on every UI click** — the v1 prototype's hyperparam-override path bypassed Streamlit's `@st.cache_data` wrapper, so every view switch / timeframe button click triggered a fresh 30-second mood engine run. The new session-state engine output cache resolves this (~150ms cache hit vs ~30s recompute).
- **Sidebar passport stale on first Run Analysis** — passport was rendered before the analysis pipeline ran, so it showed the prior session's profile state. Split into toggle (pre-analysis) + body (post-analysis via `st.empty()` placeholder).
- **WaveTrend divisor blowup during warmup** — `d = ema(|ap − esa|, n1)` underflows during the first ~10 bars, causing `ci` to spike. Floored `d` at 0.5 and masked the first `n1 + n2 = 31` bars (NaN, skipped by Plotly).
- **Historical Mood chart bottom-edge clipping** — Plotly SVG sat flush against `.stPlotlyChart`'s `overflow: hidden` boundary, hiding the rounded bottom border. Added `padding-bottom: 6px` and explicit `box-sizing: content-box`.

---

## [v2.7.0] — 2026-04-15

### Obsidian Quant UI Port

Wholesale UI/UX redesign port from Nishkarsh's "Obsidian Quant" institutional research terminal aesthetic. Engine, math primitives, and data ingestion unchanged.

#### Added
- Modular `ui/` package: `theme.py`, `theme.css` (4 600+ lines), `components.py`, `tabs/{landing, historical_mood, similar_periods, correlation}.py`
- Sanskrit serif masthead (`अर्थगति` overlay on title)
- Section headers with icon badges and animated accent bars
- Analog Period cards (Top Analog Periods view) with eyebrow / symbol / badge / stat-trio / forward-return tile grid / similarity progress-bar footer
- Correlation Cards + Predictor Quality Cards with directional Bullish / Bearish badges
- `core/logger_config.py` — structured terminal logging with phase banners, success / warning / checkpoint helpers, boxed summaries (ported from Nishkarsh)
- Sidebar masthead, view-mode radio, model-configuration expander with predictor staging diff

#### Changed
- Fonts: Space Grotesk (display) + JetBrains Mono / IBM Plex Mono (data)
- Palette: Obsidian deep-navy backgrounds (#050810 → #0A0E17), amber-gold (#D4A853) primary, cyan / emerald / rose accents
- Plotly charts use transparent paper/plot backgrounds, JetBrains Mono ticks, dashed spike crosshairs
- Page favicon: stylised amber chart line on circle

#### Removed
- CRT scanline overlay (legacy retro-broker aesthetic)
- Legacy IBM Plex Sans / IBM Plex Mono token names (some retained as aliases for backwards compatibility)

---

## [v2.6.0] — 2026-04-06

### Google Sheets Infrastructure Simplification

Migrated from Google service account OAuth to the Google Visualization API (`gviz/tq`) with environment variable configuration. No changes to the sentiment engine, math primitives, or UI behavior.

#### Changed
- **Data ingestion endpoint** — switched from `/export?format=csv` with OAuth service account to `/gviz/tq?tqx=out:csv` with no authentication required
- **Configuration model** — replaced `st.secrets` TOML-based secrets with two environment variables: `ARTHAGATI_SHEET_ID` and `ARTHAGATI_SHEET_GID`
- **Timeout resilience** — increased request timeout from 30s to 60s with 3-attempt exponential backoff (2s, 4s, 8s)
- **Deployment simplicity** — no Google Cloud project, no service account JSON, no OAuth scopes needed

#### Removed
- `google-auth` dependency from `requirements.txt` — no longer needed for gviz endpoint
- `_SHEET_SCOPES` constant and OAuth import chain (`google.auth.transport.requests`, `google.oauth2.service_account`)
- Service account credential resolution logic from `_fetch_sheet_csv()`
- `.streamlit/secrets.toml` deployment pattern (replaced by environment variables)

#### Fixed
- Stale progress bar text: "service account auth" → "gviz API"
- VISION.md data ingestion diagram and Q&A section updated to reflect gviz architecture

---

## [v2.5.0] — 2026-04-05

### Production Readiness & Code Cleanup

Production-focused release. Dead code elimination, API surface reduction, and cross-file version synchronization. No behavioral changes to the sentiment engine or UI.

#### Removed
- Dead function `ornstein_uhlenbeck_estimate()` (42 lines) — zero traceable callsites; OU estimation is performed inline via vectorized expanding AR(1) within `calculate_historical_mood()`
- Unused `kalman_gains` return value from `kalman_filter_1d()` — only `filtered_state` and `estimate_variances` are consumed by the smoothing layer
- Stale `ornstein_uhlenbeck_estimate` entry from mathematical primitives documentation table

#### Changed
- `kalman_filter_1d()` signature modernized with PEP 604 type hints: `np.ndarray | pd.Series`, `float | None`, returns `tuple[np.ndarray, np.ndarray]`
- Mathematical primitives count updated: 12 → 11 functions across source code and documentation
- `COMPANY` constant in `arthagati.py` updated to `@thebullishvalue` (branding alignment)
- Version numbers synchronized across all files: `arthagati.py`, `README.md`, `requirements.txt`, `VISION.md` (VISION.md had lagged at v2.2.1 since v2.3.0)

#### Fixed
- Cross-file version consistency: all version identifiers now point to a single source of truth (`VERSION` in `arthagati.py`)

---

## [v2.4.0]

### Adversarial Audit Resolution

Major correctness release. Seven mathematical fixes and nine algorithmic improvements identified through adversarial audit. The sentiment engine now produces mathematically sound scores with no look-ahead bias, correct variance estimation, and stable regime detection.

#### Fixed — Correctness
- **OU Residual Sum of Squares** — Replaced incorrect algebraic expanding RSS formula with per-observation residuals `e²_i = (y_i − a_i − b_i·x_i)²` accumulated via expanding mean; sigma and half-life diagnostics are now correct under time-varying AR(1) coefficients
- **Backward Information Leakage** — Removed `bfill()` from data imputation; only `ffill()` applied, early NaN values remain NaN and are handled by `np.isfinite()` guards in all math primitives
- **DFA Segment Guard** — Increased minimum segment count from 1 to 4 per Peng et al. (1994), preventing degenerate single-segment Hurst estimates
- **MSF Regime Trend Artifact** — Replaced unbounded `cumsum()` with windowed `rolling(MSF_WINDOW).sum()` preventing directional count drift that created false regime signals
- **Rolling Entropy Off-by-One** — Fixed `sliding_window_view` scope and result index alignment
- **Sigmoid Overflow** — Added input clipping (`±500`) before `np.exp()` for extreme z-scores
- **rolling_mean_fast NaN Semantics** — Returns `NaN` instead of `0.0` for all-NaN windows

#### Changed — Algorithm Improvements
- **O(N log N) Adaptive Percentiles** — Replaced O(N²) inner loop with sorted-insert + `np.searchsorted` binary search (Greenwald & Khanna 2001 streaming quantile approach)
- **Kalman Warm-Up Bootstrap** — First 50 observations bootstrapped from first stable window per Harvey (1990), preventing poorly calibrated Kalman gains
- **Freedman-Diaconis Entropy Bins** — Adaptive bin selection via `2·IQR·n^{-1/3}` instead of capped `sqrt(N)`
- **Ledoit-Wolf Covariance Shrinkage** — Mahalanobis distance uses analytical OAS shrinkage (Chen et al. 2010) instead of ad-hoc diagonal regularization
- **Walk-Forward Weight Blending** — Checkpoint weights exponentially blended (α ≈ 0.29, HL = 2 checkpoints) eliminating discontinuous jumps at segment boundaries
- **Confidence Band Soft-Clip** — `tanh(x/100)·100` replaces hard `np.clip(±100)` preserving band width at score extremes
- **Least-Squares Trajectory Detrend** — Replaced endpoint anchoring with least-squares linear detrend (minimizes residual variance on V-shaped and reversal trajectories)
- **Backtest Train/Test Split** — 70/30 chronological split with separate in-sample and out-of-sample Pearson/Spearman correlations

---

## [v2.3.0]

### Walk-Forward Correlations & Bias Corrections

Eliminated look-ahead bias from the correlation engine and applied first-order bias corrections to statistical estimators.

#### Fixed
- **Look-Ahead Bias** — Layers 1–2 restructured to use expanding-window walk-forward correlations at quarterly checkpoints instead of full-sample
- **Percentile Semantics** — Symmetric [−1,+1] adjustments for PE and EY anchors, fixing asymmetric bearish/bullish capacity
- **Hurst Estimator Bias** — Replaced R/S with DFA-1 (Peng et al. 1994, Weron 2002) for robustness on short series
- **OU AR(1) Bias** — Kendall-Marriott-Pope first-order correction applied to expanding AR(1) coefficient
- **Dynamic Y-Axis** — Mood chart now scales to actual data bounds with 8% padding instead of fixed ±100

---

## [v2.2.1]

### UI Rendering & Memory Optimizations

#### Changed
- **WebGL Chart Rendering** — Regime transition markers migrated from individual SVG shapes (`add_vline`) to interleaved WebGL traces (`go.Scattergl`), eliminating DOM bloat on MAX timeframe

#### Fixed
- **Cache Memory Bloat** — Applied `max_entries=5` to all heavy `@st.cache_data` decorators, capping server RAM when users rapidly toggle predictor configurations

---

## [v2.2.0]

### Performance & Vectorization Architecture Rewrite

Execution time reduced by 99%+ through C-level NumPy vectorization of all mathematical primitives.

#### Added
- **C-Level Vectorization Engine** — All explicit Python loops replaced with NumPy `cumsum`, `sliding_window_view`, and array striding
- **O(N) Moving Averages & Variances** — Replaced Pandas `.rolling()`/`.expanding()` with exact NumPy cumulative sums
- **Pure-NumPy Ranking** — Custom vectorized tie-averaging rank algorithm replacing Pandas `.rank()` in weighted Spearman

#### Changed
- **Kalman Filter** — Exponential fading memory factor (Sorenson & Sacks) for non-stationary regime discounting
- **OU Estimation** — O(N²) expanding-window loop converted to single-pass O(N) vectorized algorithm
- **Trajectory Similarity** — 20-day cosine similarity migrated from explicit iteration to matrix striding multiplications
- **Regime Detection** — Fully vectorized Hurst × Entropy quadrant classification

#### Fixed
- **Memory Blowout** — 2D NumPy broadcasting in adaptive percentiles created O(N²) memory (40GB+ allocations); rewritten with O(N) 1D slice lookback reducing engine time from ~120s to <2s

---

## [v2.1.0]

### Diagnostics & Forward Returns

Extended the sentiment engine with forward-looking projections and historical validation.

#### Added
- 90-day OU forward mean-reversion projection on mood chart
- ±1.96σ Kalman confidence bands around smoothed mood score
- Forward return outcomes (30/60/90-day) on similar historical period cards
- Backtest scatter plot: mood score at T vs NIFTY return at T+30
- Data staleness warnings when Google Sheet is more than 3 days old

---

## [v2.0.0]

### Physics-Informed Mathematics

Complete overhaul of the sentiment engine from static correlations to stochastic process modeling.

#### Added
- **Ornstein-Uhlenbeck Normalization** — Mood modeled as mean-reverting diffusion `dx = θ(μ − x)dt + σdW` instead of global z-score
- **Kalman Smoothing** — 1D adaptive state estimation replacing fixed-window EMA
- **Mahalanobis Distance** — Covariance-aware historical period matching replacing Manhattan distance
- **Inverse-Variance MSF Weighting** — Markowitz minimum-variance signal allocation replacing fixed 30/25/25/20 weights
- **Adaptive Percentiles** — Decay-weighted empirical CDF replacing expanding rank percentiles
- **Decay-Spearman Correlations** — Recency-weighted rank correlation replacing full-sample Pearson
- **Shannon Entropy Weighting** — Noisy variable suppression via information-theoretic penalty
- **Predictor Quality Assessment** — Ranked variable scoring by |correlation| × (1 − entropy)
- **Staging → Commit Config** — Apply-button pattern preventing continuous recomputation
- **EY Auto-Derivation** — `1/PE × 100` when Earnings Yield absent from sheet
- **Yield Term Spreads** — `IN10Y − IN02Y` and `US10Y − US02Y` derived as orthogonal predictors

#### Removed
- Fixed Pearson correlations (replaced by decay-Spearman)
- Expanding rank percentiles (replaced by adaptive ECDF)
- Fixed MSF weights (replaced by inverse-variance)
- Manhattan distance similar periods (replaced by Mahalanobis)
- Global z-score normalization (replaced by OU)
- Simple moving average smoothing (replaced by Kalman)

---

## [v1.2.0]

### Initial Release

Baseline sentiment engine with Pearson correlations, expanding percentiles, and fixed-weight MSF oscillator.

---

*© 2026 Arthagati · @thebullishvalue*
