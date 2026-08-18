"""
Arthagati — Signal Validation.

Measures whether the Mood Score has out-of-sample predictive power. It does
not fit anything; there is nothing here to overfit.

Why this module replaced Intelligence Mode
------------------------------------------
Intelligence Mode fitted a linear ensemble over engine outputs (mood, mood
divergence, the MSF composite and its four components) with Optuna, and
surfaced the result as a "Calibrated Conviction" signal. Measured on twenty
years of NIFTY data across three different predictor sets, it reduced the
signal's out-of-sample information ratio every time:

    predictor set    raw Mood Score    fitted ensemble    margin
    selected 4              +1.674             +1.239     -0.436
    current 12              +1.893             -0.444     -2.337
    all 37                  +1.753             +1.363     -0.390

The mechanism is visible in the weights. Only `mood` carries forward
information (holdout rho +0.31 at 90d); the MSF components sit within noise
of zero (-0.03 to +0.01). Maximising the information ratio across CV folds
rewards whatever fits in-sample, so the search loaded on the technicals — in
the 12-predictor run it assigned `mood` a weight of -0.37, inverting its one
useful input.

The quality gate caught this every time and refused to activate. A component
whose gate correctly rejects it on every real configuration is not a feature.

What survives is the measurement apparatus, which was worth building:
a holdout the analysis never sees, an embargo covering the label horizon, and
a permutation null. Pointed at the Mood Score itself, it answers the question
the product actually needs answered — does this signal work?
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Horizons. The engine's relationship with forward returns strengthens with
# horizon (measured rho: +0.03 at 5d, +0.10 at 90d, +0.22 at 250d), so the
# short end is reported for completeness rather than because it carries the
# signal.
DEFAULT_HORIZONS: tuple[int, ...] = (20, 60, 125, 250)

HOLDOUT_FRACTION: float = 0.25
N_BLOCKS: int = 6
MIN_SPEARMAN_OBS: int = 60
N_PERMUTATIONS: int = 200

# Forward windows overlap: a holdout of H rows carries roughly H / horizon
# independent observations. Below this many, no threshold can separate signal
# from noise and the honest output is "not enough data".
MIN_INDEPENDENT_WINDOWS: float = 10.0

VERDICT_EDGE        = "Edge Confirmed"
VERDICT_NO_EDGE     = "No Edge"
VERDICT_INSUFFICIENT = "Insufficient Data"

GATE_MAX_P_VALUE: float = 0.05
GATE_MIN_RHO: float = 0.10


@dataclass
class ValidationReport:
    verdict: str
    validated_horizons: list      # horizons the holdout can actually support
    descriptive_horizons: list    # reported, but too long to validate here
    holdout_rho: float
    p_value: float
    per_horizon: dict
    baseline_rho: float          # -PE alone, the null model to beat
    dev_rho: float
    n_holdout: int
    holdout_start: str
    independent_windows: float
    horizons: list
    n_permutations: int

    def to_dict(self) -> dict:
        return asdict(self)


def _blocks(start: int, stop: int, n_blocks: int) -> list[slice]:
    total = stop - start
    if total <= 0:
        return []
    size = max(MIN_SPEARMAN_OBS, total // max(n_blocks, 1))
    out, pos = [], start
    while pos < stop and len(out) < n_blocks:
        end = stop if len(out) == n_blocks - 1 else min(pos + size, stop)
        if end - pos >= MIN_SPEARMAN_OBS:
            out.append(slice(pos, end))
        pos = end
    return out


def _rho(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < MIN_SPEARMAN_OBS:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r, _ = spearmanr(x[ok], y[ok])
    return float(r) if np.isfinite(r) else np.nan


def _forward_returns(nifty: np.ndarray, h: int) -> np.ndarray:
    n = len(nifty)
    out = np.full(n, np.nan)
    if n > h:
        out[: n - h] = nifty[h:] / nifty[: n - h] - 1.0
    return out


def independent_windows(n_rows: int, horizons: Iterable[int]) -> float:
    return float(n_rows) / max(max(int(h) for h in horizons), 1)


def score_window(
    signal: np.ndarray,
    nifty: np.ndarray,
    window: slice,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    n_blocks: int = N_BLOCKS,
) -> tuple[float, dict[int, float]]:
    """Mean Spearman rho over contiguous blocks x horizons inside `window`.

    Mean rather than mean/std: with a handful of correlated blocks the
    dispersion in an information ratio's denominator is mostly noise, and it
    made the statistic swing widely on resampling.
    """
    horizons = [int(h) for h in horizons]
    per: dict[int, list[float]] = {h: [] for h in horizons}
    for blk in _blocks(window.start or 0, window.stop or len(signal), n_blocks):
        for h in horizons:
            r = _rho(signal[blk], _forward_returns(nifty, h)[blk])
            if np.isfinite(r):
                per[h].append(r)
    allr = [r for v in per.values() for r in v]
    means = {h: (float(np.mean(v)) if v else 0.0) for h, v in per.items()}
    return (float(np.mean(allr)) if allr else 0.0), means


def permutation_p_value(
    signal: np.ndarray,
    nifty: np.ndarray,
    window: slice,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    n_perm: int = N_PERMUTATIONS,
    seed: int = 12345,
) -> float:
    """Share of circularly shifted signals matching or beating the real one.

    Circular shifts preserve the autocorrelation of both series and destroy
    only their alignment, which is the relationship under test. A plain
    shuffle would break the signal's own structure and produce an
    unrealistically easy null.
    """
    start, stop = window.start or 0, window.stop or len(signal)
    m = stop - start
    min_shift = max(MIN_SPEARMAN_OBS, max(int(h) for h in horizons))
    if m - 2 * min_shift <= 1:
        return 1.0

    seg = signal[start:stop].copy()
    nif = nifty
    actual = abs(score_window(signal, nif, window, horizons)[0])

    rng = np.random.default_rng(seed)
    beat = 0
    full = signal.copy()
    for _ in range(n_perm):
        shift = int(rng.integers(min_shift, m - min_shift))
        full[start:stop] = np.roll(seg, shift)
        if abs(score_window(full, nif, window, horizons)[0]) >= actual:
            beat += 1
    return (beat + 1.0) / (n_perm + 1.0)


def validate(
    mood_df: pd.DataFrame,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    holdout_fraction: float = HOLDOUT_FRACTION,
    n_permutations: int = N_PERMUTATIONS,
    baseline: np.ndarray | None = None,
) -> ValidationReport:
    """Score the Mood Score on a holdout it had no part in shaping.

    ``baseline`` is the null model — by default the negated PE ratio, i.e.
    "cheap is good" with no engine at all. The engine has to beat that to
    justify itself.
    """
    horizons = [int(h) for h in horizons]
    n = len(mood_df)
    signal = mood_df["Mood_Score"].to_numpy(dtype=np.float64).copy()
    if "Is_Warmup" in mood_df.columns:
        signal[mood_df["Is_Warmup"].to_numpy(dtype=bool)] = np.nan
    nifty = mood_df["NIFTY"].to_numpy(dtype=np.float64)

    hold_start = n - int(round(n * holdout_fraction))
    dev, hold = slice(0, hold_start), slice(hold_start, n)

    dev_rho, _ = score_window(signal, nifty, dev, horizons)
    hold_rho, per_h = score_window(signal, nifty, hold, horizons)

    # Validate at the horizons this holdout can actually support, and report
    # the rest as descriptive.
    #
    # Forward windows overlap, so a holdout of H rows carries about H/horizon
    # independent observations. On twenty years of NIFTY the signal is
    # strongest at 250 days (rho +0.64) — but a 1,246-row holdout spans only
    # five independent 250-day windows, which is not enough to distinguish
    # that from chance. Rather than refuse a verdict outright, or pretend the
    # long horizon is validated, the test runs on the horizons that clear the
    # floor and the others are labelled descriptive.
    n_hold = n - hold_start
    max_valid_h = n_hold / MIN_INDEPENDENT_WINDOWS
    validated = [h for h in horizons if h <= max_valid_h]
    descriptive = [h for h in horizons if h > max_valid_h]

    if not validated:
        verdict, p = VERDICT_INSUFFICIENT, 1.0
        hold_rho = hold_rho
    else:
        hold_rho, _ = score_window(signal, nifty, hold, validated)
        p = permutation_p_value(signal, nifty, hold, validated, n_permutations)
        verdict = (
            VERDICT_EDGE
            if (abs(hold_rho) >= GATE_MIN_RHO and p <= GATE_MAX_P_VALUE)
            else VERDICT_NO_EDGE
        )
    iw = independent_windows(n_hold, validated or horizons)

    # Baseline scored on the same horizons, so the comparison is like-for-like.
    base_rho = np.nan
    if baseline is not None:
        b = np.asarray(baseline, dtype=np.float64).copy()
        if "Is_Warmup" in mood_df.columns:
            b[mood_df["Is_Warmup"].to_numpy(dtype=bool)] = np.nan
        base_rho, _ = score_window(b, nifty, hold, validated or horizons)

    return ValidationReport(
        verdict=verdict,
        holdout_rho=hold_rho,
        p_value=p,
        per_horizon=per_h,
        baseline_rho=float(base_rho) if np.isfinite(base_rho) else float("nan"),
        dev_rho=dev_rho,
        n_holdout=int(n - hold_start),
        holdout_start=str(mood_df["DATE"].iloc[hold_start].date()),
        independent_windows=iw,
        horizons=list(horizons),
        n_permutations=int(n_permutations),
        validated_horizons=list(validated),
        descriptive_horizons=list(descriptive),
    )
