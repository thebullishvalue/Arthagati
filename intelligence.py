"""
Arthagati Intelligence Mode — post-engine ensemble calibration.

Architecture
------------
The mood engine runs ONCE on factory defaults and emits a feature matrix
``F`` built from its own outputs (mood, mood divergence, the MSF composite
and its four components). Optuna tunes a small weight vector ``w`` so that

    calibrated_conviction = tanh((F @ w) / 3) * 100

carries Spearman information about forward NIFTY returns. Per-trial cost is
one matrix-vector multiply plus a handful of rank correlations, so a full
search finishes in well under a second.

Validation contract
-------------------
This module's previous version could not tell signal from noise. Optuna
maximised ``0.65 * val_IR + 0.35 * train_IR`` and the quality gate then
asked whether ``val_IR > 0`` — testing the objective against itself. On a
dataset whose forward returns were an independent random walk it reported
train IR 1.56, val IR 0.71 and a "Quality OK" badge.

Three changes make the verdict meaningful:

  1. HOLDOUT. The final ``HOLDOUT_FRACTION`` of the series is removed before
     the search begins and is never seen by Optuna, by the CV folds, or by
     the feature standardisation. It is scored once, afterwards. That number
     — not the optimised validation IR — is what the gate and the UI report.

  2. EMBARGO = max(horizon). The gap between a training fold and the
     validation fold that follows it must exceed the longest forward-return
     horizon, or the training labels reach into the validation window. It
     was 5 days against horizons up to 90.

  3. PERMUTATION NULL. The holdout IR is compared against the distribution
     produced by circularly shifting the signal against the same returns.
     Circular shifts preserve the autocorrelation of both series while
     destroying their alignment, which is the right null for overlapping
     forward-return windows. The gate requires the observed IR to sit in the
     upper tail.

Warm-up rows (the engine's ``Is_Warmup`` flag, where correlation weights
were borrowed from a later checkpoint) are excluded from every scoring path.
"""

from __future__ import annotations

import json
import os
import tempfile
import warnings
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import VERSION

# v1 tuned structural hyperparameters; v2 was the ensemble without a holdout;
# v3 adds the holdout, the permutation null and the reduced feature set.
PROFILE_SCHEMA_VERSION = 3

# Deployments with a writable persistent volume can point this elsewhere.
# The default lives beside the code, which on Streamlit Cloud is ephemeral —
# profiles there survive reruns but not container restarts.
PROFILE_DIR = Path(
    os.environ.get("ARTHAGATI_PROFILE_DIR", Path(__file__).resolve().parent / "profiles")
)
ACTIVE_PROFILE_PATH = PROFILE_DIR / "active.json"
MAX_ARCHIVED_PROFILES = 20

DEFAULT_HORIZONS: tuple[int, ...] = (5, 20, 60, 90)
DEFAULT_FOLDS: int = 5
DEFAULT_TRIALS: int = 40
DEFAULT_L2_ALPHA: float = 0.05

# The embargo must cover the longest horizon, otherwise a training row's
# label is drawn from inside the validation window. Kept as a function of
# the horizons rather than a constant so the two cannot drift apart.
def default_embargo(horizons: Iterable[int]) -> int:
    return int(max(horizons))


DEFAULT_EMBARGO_DAYS: int = default_embargo(DEFAULT_HORIZONS)

# Holdout — never seen by the optimiser.
HOLDOUT_FRACTION: float = 0.25
HOLDOUT_MIN_ROWS: int = 250
HOLDOUT_BLOCKS: int = 3
N_PERMUTATIONS: int = 200

MIN_TRAIN_ROWS: int = 252
MIN_SPEARMAN_OBS: int = 60

PROFILE_FRESHNESS_DAYS: int = 14

PRUNER_WARMUP_TRIALS: int = 8
PRUNER_WARMUP_STEPS:  int = 1

# Gate thresholds.
#
# These are deliberately strict. A false "Quality OK" tells someone the
# ensemble has predictive worth when it does not, and the cost of that is
# asymmetric against the cost of withholding a real but marginal signal — so
# the thresholds are set to keep false accepts well under the nominal level
# rather than at it. A p-threshold of 0.10 alone admits noise 10% of the time
# by construction; pairing it with a minimum effect size cuts that sharply.
GATE_MIN_HOLDOUT_IR:    float = 0.25   # effect size, not just a positive sign
GATE_MAX_P_VALUE:       float = 0.05   # must beat the circular-shift null
GATE_OVERFIT_STABILITY: float = 0.30   # holdout / train ratio

# Minimum statistical power before a verdict is issued at all.
#
# Forward-return windows overlap: at a 90-day horizon, consecutive rows share
# 89 of their 90 days, so a holdout of H rows carries roughly H / 90
# INDEPENDENT observations — a 600-row holdout is about six. An information
# ratio computed on six effective observations is so noisy that a
# permutation test against it accepts pure noise a large fraction of the
# time, whatever the nominal p-threshold: measured on random-walk returns,
# a 2,400-row series produced "Quality OK" on 2 of 7 seeds.
#
# Rather than tune the threshold until the symptom disappears, the calibrator
# declines to grade when the holdout cannot support a verdict. The honest
# output there is "not enough data to tell", not a guess. Roughly 10 years of
# daily history is needed before the 90-day horizon can be validated.
GATE_MIN_INDEPENDENT_WINDOWS: float = 10.0

QUALITY_OK           = "Quality OK"
QUALITY_OVERFIT      = "Overfit"
QUALITY_NO_EDGE      = "No Edge"
QUALITY_INSUFFICIENT = "Insufficient Data"

# Only a profile that clears the gate outright is applied to the UI. An
# "Overfit" profile is still written to disk so it can be inspected, but it
# does not drive the Calibrated Conviction card.
ACTIVATABLE_QUALITY = frozenset({QUALITY_OK})


# ──────────────────────────────────────────────────────────────────────────────
# Feature matrix
# ──────────────────────────────────────────────────────────────────────────────
# The v2 matrix carried mood, mood_smooth, mood_diverge, mood_squared and
# mood_sqrt — five near-collinear monotone transforms of one variable
# (pairwise |rho| 0.91 to 0.97, condition number 1.2e17). Linear weights over
# a singular design are not identifiable, so the per-feature weights and the
# Bullish/Bearish badges derived from them were reading noise.
#
# What survives: the level, one genuinely orthogonal derived term (the
# divergence between raw and Kalman-smoothed mood), and the MSF family.

FEATURE_NAMES: tuple[str, ...] = (
    "mood",            # raw Mood_Score (engine output)
    "mood_diverge",    # mood - Kalman(mood): short-vs-long sentiment gap
    "msf_spread",      # MSF composite oscillator
    "msf_momentum",    # NIFTY ROC z-score
    "msf_structure",   # mood trend divergence + acceleration
    "msf_regime",      # adaptive directional count
    "msf_flow",        # breadth participation
)

WEIGHT_BOUNDS: dict[str, tuple[float, float]] = {
    "mood":          (-2.0, 2.0),
    "mood_diverge":  (-1.0, 1.0),
    "msf_spread":    (-2.0, 2.0),
    "msf_momentum":  (-1.0, 1.0),
    "msf_structure": (-1.0, 1.0),
    "msf_regime":    (-1.0, 1.0),
    "msf_flow":      (-1.0, 1.0),
}


def _expanding_standardise(F: np.ndarray, min_periods: int = 60) -> np.ndarray:
    """Causal column standardisation.

    Full-sample mean/std leak the distribution of the whole series — including
    the holdout — into every row. Expanding moments use only observations up
    to and including each row.
    """
    n = len(F)
    counts = np.arange(1, n + 1, dtype=np.float64)[:, None]
    cs = np.cumsum(F, axis=0)
    cs2 = np.cumsum(F ** 2, axis=0)
    mean = cs / counts
    var = (cs2 - (cs ** 2) / counts) / np.maximum(counts - 1.0, 1.0)
    sd = np.sqrt(np.maximum(var, 0.0))
    sd = np.where(sd > 1e-9, sd, 1.0)
    out = (F - mean) / sd
    # Before min_periods the moments are too unstable to be meaningful.
    warm = min(min_periods, n)
    out[:warm, :] = 0.0
    return np.where(np.isfinite(out), out, 0.0)


def build_feature_matrix(mood_df: pd.DataFrame, msf_df: pd.DataFrame) -> np.ndarray:
    """Build the ``F`` matrix from engine output.

    Returns an ``(N, len(FEATURE_NAMES))`` array, causally standardised.
    Missing columns contribute zeros so older engine output still loads.
    """
    n = len(mood_df)

    def _col(df: pd.DataFrame, name: str, default: float = 0.0) -> np.ndarray:
        if name in df.columns:
            arr = df[name].to_numpy(dtype=np.float64)
            return np.where(np.isfinite(arr), arr, default)
        return np.full(n, default, dtype=np.float64)

    mood = _col(mood_df, "Mood_Score")
    mood_smooth = _col(mood_df, "Smoothed_Mood_Score")
    if not np.any(mood_smooth):
        mood_smooth = pd.Series(mood).ewm(span=20, adjust=False).mean().to_numpy(dtype=np.float64)

    F = np.column_stack([
        mood,
        mood - mood_smooth,
        _col(mood_df, "MSF_Spread"),
        _col(msf_df, "momentum"),
        _col(msf_df, "structure"),
        _col(msf_df, "regime"),
        _col(msf_df, "flow"),
    ])
    return _expanding_standardise(F)


def apply_calibration(
    mood_df: pd.DataFrame,
    msf_df: pd.DataFrame,
    weights: dict[str, float],
) -> np.ndarray:
    """Calibrated conviction time-series for a given weight set, in [-100, 100]."""
    F = build_feature_matrix(mood_df, msf_df)
    w = np.array(
        [float(weights.get(name, 0.0)) for name in FEATURE_NAMES],
        dtype=np.float64,
    )
    return np.tanh((F @ w) / 3.0) * 100.0


def valid_mask(mood_df: pd.DataFrame) -> np.ndarray:
    """Rows eligible for scoring — excludes the engine's warm-up region."""
    n = len(mood_df)
    if "Is_Warmup" in mood_df.columns:
        return ~mood_df["Is_Warmup"].to_numpy(dtype=bool)
    return np.ones(n, dtype=bool)


# ──────────────────────────────────────────────────────────────────────────────
# Profile envelope
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CalibrationProfile:
    """Calibrated post-engine ensemble weights + diagnostics."""

    weights: dict[str, float]
    train_ir: float
    val_ir: float               # optimised — reported as a search diagnostic only
    holdout_ir: float           # the honest out-of-sample number
    holdout_p_value: float      # vs the circular-shift null
    stability: float            # holdout / train
    quality_check: str
    horizons: list[int]
    n_folds: int
    n_trials: int
    embargo_days: int
    n_rows_train: int           # unique rows, not the sum over overlapping folds
    n_rows_holdout: int
    n_predictors: int
    data_start: str
    data_end: str
    holdout_start: str
    importance: dict
    timestamp: str
    arthagati_version: str
    schema_version: int = PROFILE_SCHEMA_VERSION
    sensitivity_curves: dict = field(default_factory=dict)

    @property
    def is_default(self) -> bool:
        return not bool(self.weights)

    @property
    def is_activatable(self) -> bool:
        return self.quality_check in ACTIVATABLE_QUALITY and bool(self.weights)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)

    @classmethod
    def from_dict(cls, raw: dict) -> "CalibrationProfile":
        """Build a profile from untrusted JSON.

        Import used to accept whatever the file contained: unknown feature
        names, non-numeric weights, unbounded magnitudes, and a
        ``quality_check`` string that let a hand-edited file bypass the gate
        entirely. Everything is now coerced and bounded.
        """
        if not isinstance(raw, dict):
            raise ValueError("profile must be a JSON object")

        raw_weights = raw.get("weights") or {}
        if not isinstance(raw_weights, dict):
            raise ValueError("'weights' must be an object mapping feature name to number")

        unknown = set(raw_weights) - set(FEATURE_NAMES)
        if unknown:
            raise ValueError(
                f"unknown feature(s): {', '.join(sorted(unknown))}. "
                f"Expected any of: {', '.join(FEATURE_NAMES)}"
            )

        weights: dict[str, float] = {}
        for name, value in raw_weights.items():
            try:
                v = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"weight for '{name}' is not a number: {value!r}")
            if not np.isfinite(v):
                raise ValueError(f"weight for '{name}' is not finite")
            lo, hi = WEIGHT_BOUNDS[name]
            if not (lo <= v <= hi):
                raise ValueError(
                    f"weight for '{name}' is {v:+.3f}, outside the permitted "
                    f"range [{lo:+.1f}, {hi:+.1f}]"
                )
            weights[name] = v

        def _num(key: str, default: float = 0.0) -> float:
            try:
                v = float(raw.get(key, default))
                return v if np.isfinite(v) else default
            except (TypeError, ValueError):
                return default

        quality = str(raw.get("quality_check", "—"))
        if quality not in (QUALITY_OK, QUALITY_OVERFIT, QUALITY_NO_EDGE, QUALITY_INSUFFICIENT):
            quality = "—"

        # Re-derive the verdict from the imported metrics rather than trusting
        # the label in the file.
        holdout_ir = _num("holdout_ir")
        train_ir = _num("train_ir")
        p_value = _num("holdout_p_value", 1.0)
        if quality != "—":
            n_hold = int(_num("n_rows_holdout"))
            hz = raw.get("horizons") or list(DEFAULT_HORIZONS)
            try:
                n_indep = independent_windows(n_hold, [int(h) for h in hz]) if n_hold else None
            except (TypeError, ValueError):
                n_indep = None
            quality, _ = quality_check(train_ir, holdout_ir, p_value, n_indep)

        horizons = raw.get("horizons") or list(DEFAULT_HORIZONS)
        try:
            horizons = [int(h) for h in horizons]
        except (TypeError, ValueError):
            horizons = list(DEFAULT_HORIZONS)

        return cls(
            weights=weights,
            train_ir=train_ir,
            val_ir=_num("val_ir"),
            holdout_ir=holdout_ir,
            holdout_p_value=p_value,
            stability=_num("stability"),
            quality_check=quality,
            horizons=horizons,
            n_folds=int(_num("n_folds", DEFAULT_FOLDS)),
            n_trials=int(_num("n_trials", DEFAULT_TRIALS)),
            embargo_days=int(_num("embargo_days", DEFAULT_EMBARGO_DAYS)),
            n_rows_train=int(_num("n_rows_train")),
            n_rows_holdout=int(_num("n_rows_holdout")),
            n_predictors=int(_num("n_predictors")),
            data_start=str(raw.get("data_start", "")),
            data_end=str(raw.get("data_end", "")),
            holdout_start=str(raw.get("holdout_start", "")),
            importance=raw.get("importance") if isinstance(raw.get("importance"), dict) else {},
            timestamp=str(raw.get("timestamp", "")),
            arthagati_version=str(raw.get("arthagati_version", "")),
            schema_version=PROFILE_SCHEMA_VERSION,
            sensitivity_curves={},
        )


# ──────────────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────────────

def ensure_profile_dir() -> None:
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)


def save_active_profile(profile: CalibrationProfile) -> Path:
    ensure_profile_dir()
    body = profile.to_json()
    fd, tmp = tempfile.mkstemp(prefix="active.", suffix=".tmp", dir=str(PROFILE_DIR))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(body)
        os.replace(tmp, ACTIVE_PROFILE_PATH)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return ACTIVE_PROFILE_PATH


def load_active_profile() -> CalibrationProfile | None:
    if not ACTIVE_PROFILE_PATH.exists():
        return None
    try:
        raw = json.loads(ACTIVE_PROFILE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        warnings.warn(f"Active profile is unreadable and will be ignored: {exc}")
        return None
    if not isinstance(raw, dict) or int(raw.get("schema_version", 0)) < PROFILE_SCHEMA_VERSION:
        return None
    try:
        return CalibrationProfile.from_dict(raw)
    except ValueError as exc:
        warnings.warn(f"Active profile failed validation and will be ignored: {exc}")
        return None


def delete_active_profile() -> bool:
    if ACTIVE_PROFILE_PATH.exists():
        ACTIVE_PROFILE_PATH.unlink()
        return True
    return False


def list_profiles() -> list[Path]:
    if not PROFILE_DIR.exists():
        return []
    return sorted(
        (p for p in PROFILE_DIR.glob("profile_*.json")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def archive_profile(profile: CalibrationProfile, keep: int = MAX_ARCHIVED_PROFILES) -> Path:
    """Write a timestamped snapshot and prune the oldest beyond ``keep``."""
    ensure_profile_dir()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = PROFILE_DIR / f"profile_{ts}.json"
    path.write_text(profile.to_json(), encoding="utf-8")
    for stale in list_profiles()[keep:]:
        try:
            stale.unlink()
        except OSError:
            pass
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Freshness + fingerprints
# ──────────────────────────────────────────────────────────────────────────────

def profile_age_days(profile: CalibrationProfile) -> float:
    try:
        ts = profile.timestamp.replace("Z", "+00:00")
        fit_at = datetime.fromisoformat(ts)
        if fit_at.tzinfo is None:
            fit_at = fit_at.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - fit_at).total_seconds() / 86400.0
    except (ValueError, TypeError):
        return 9e9


def is_profile_fresh(
    profile: CalibrationProfile | None,
    mood_df: pd.DataFrame,
    active_predictors: Iterable[str],
    *,
    max_age_days: float = PROFILE_FRESHNESS_DAYS,
) -> tuple[bool, str]:
    if profile is None:
        return False, "no profile on disk"
    if not profile.is_activatable:
        return False, f"previous calibration was graded '{profile.quality_check}'"

    n_active = len(tuple(active_predictors))
    if profile.n_predictors != n_active:
        return False, f"predictor count changed ({profile.n_predictors} → {n_active})"

    data_end = pd.Timestamp(mood_df["DATE"].max())
    try:
        profile_end = pd.Timestamp(profile.data_end)
        gap_days = (data_end - profile_end).days
    except (ValueError, TypeError):
        return False, "profile data_end is malformed"
    if gap_days > max_age_days:
        return False, f"data extends {gap_days}d beyond profile (>{max_age_days}d threshold)"
    if gap_days < -1:
        return False, f"profile data_end is in the future ({-gap_days}d ahead)"

    age = profile_age_days(profile)
    if age > max_age_days:
        return False, f"profile is {age:.0f}d old (>{max_age_days}d threshold)"

    return True, f"profile fresh · {age:.1f}d old · data within {gap_days}d"


def dataset_fingerprint(raw_df: pd.DataFrame, predictors: Iterable[str]) -> tuple:
    """Hashable fingerprint for the engine-output cache."""
    return (
        int(len(raw_df)),
        str(raw_df["DATE"].iloc[0].date()) if len(raw_df) else "",
        str(raw_df["DATE"].iloc[-1].date()) if len(raw_df) else "",
        tuple(sorted(predictors)),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Folds
# ──────────────────────────────────────────────────────────────────────────────

def _walk_forward_folds(
    n: int, n_folds: int, embargo: int, min_train_size: int,
) -> list[tuple[slice, slice]]:
    """Expanding-window folds with a purge gap of ``embargo`` rows.

    ``embargo`` must be at least the longest forward-return horizon. The last
    training row's label lands at ``train_end - 1 + horizon``; with
    ``val_start = train_end + embargo`` that stays strictly before the
    validation window whenever ``embargo >= horizon``.
    """
    if n_folds < 2 or n < (min_train_size + embargo + n_folds):
        return []
    val_total = n - min_train_size - embargo
    if val_total <= 0:
        return []
    val_size = max(1, val_total // n_folds)
    folds: list[tuple[slice, slice]] = []
    for f in range(n_folds):
        val_start = min_train_size + embargo + f * val_size
        val_end = val_start + val_size if f < n_folds - 1 else n
        if val_end > n or val_end - val_start < MIN_SPEARMAN_OBS:
            continue
        train_end = val_start - embargo
        if train_end < min_train_size:
            continue
        folds.append((slice(0, train_end), slice(val_start, val_end)))
    return folds


def _contiguous_blocks(start: int, stop: int, n_blocks: int) -> list[slice]:
    """Split [start, stop) into ``n_blocks`` contiguous slices."""
    total = stop - start
    if total <= 0 or n_blocks < 1:
        return []
    size = max(MIN_SPEARMAN_OBS, total // n_blocks)
    out: list[slice] = []
    pos = start
    while pos < stop and len(out) < n_blocks:
        end = stop if len(out) == n_blocks - 1 else min(pos + size, stop)
        if end - pos >= MIN_SPEARMAN_OBS:
            out.append(slice(pos, end))
        pos = end
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Scoring kernel
# ──────────────────────────────────────────────────────────────────────────────

def _precompute_forward_returns(
    nifty: np.ndarray, horizons: Iterable[int],
) -> dict[int, np.ndarray]:
    """Forward return from t to t+h, NaN past the end of the series."""
    n = len(nifty)
    out: dict[int, np.ndarray] = {}
    for h in horizons:
        ret = np.full(n, np.nan, dtype=np.float64)
        h = int(h)
        if n > h:
            end_idx = n - h
            with np.errstate(divide="ignore", invalid="ignore"):
                ret[:end_idx] = (nifty[h:] / nifty[:end_idx] - 1.0) * 100.0
            ret = np.where(np.isfinite(ret), ret, np.nan)
        out[h] = ret
    return out


def _spearman_safe(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho, or NaN when there is too little to say.

    Returning NaN rather than 0.0 matters: a 0.0 fallback used to enter the
    IR's mean and standard deviation as though it were an observation,
    shrinking the dispersion and inflating the ratio.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < MIN_SPEARMAN_OBS:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho, _ = spearmanr(x[mask], y[mask])
    return float(rho) if np.isfinite(rho) else np.nan


def _ir(rhos: Iterable[float]) -> float:
    """Information ratio: mean / std across (fold x horizon) correlations.

    Guarded against the degenerate case where near-identical rho values drive
    the denominator toward zero and the ratio toward infinity.
    """
    arr = np.asarray([r for r in rhos if r is not None and np.isfinite(r)], dtype=np.float64)
    if len(arr) < 4:
        return 0.0
    sd = max(float(arr.std(ddof=1)), 1e-3)
    return float(np.clip(arr.mean() / sd, -10.0, 10.0))


def _mean_rho(rhos: Iterable[float]) -> float:
    """Mean rank correlation across (block x horizon).

    This — not the information ratio — is the statistic the permutation test
    compares. IR divides by the dispersion across a dozen highly correlated
    measurements, which adds variance without adding information: on real
    signals it pushed observed p-values from the 0.01 range out past 0.15,
    while noise sometimes landed inside it. The mean is stable, monotone in
    signal strength, and is what the null distribution is built from.
    IR remains the reported effect size.
    """
    arr = np.asarray([r for r in rhos if r is not None and np.isfinite(r)], dtype=np.float64)
    if len(arr) < 4:
        return 0.0
    return float(arr.mean())


def _l2_penalty(weights: np.ndarray, alpha: float) -> float:
    return float(alpha * np.mean(weights * weights))


def independent_windows(n_holdout_rows: int, horizons: Iterable[int]) -> float:
    """Approximate count of non-overlapping forward windows in the holdout."""
    longest = max(int(h) for h in horizons)
    return float(n_holdout_rows) / max(longest, 1)


def quality_check(
    train_ir: float,
    holdout_ir: float,
    p_value: float,
    n_independent: float | None = None,
) -> tuple[str, str]:
    """Grade a calibration against the HOLDOUT, never against the objective.

    Order matters. Statistical power is checked first: if the holdout cannot
    support a verdict, none is issued. Then generalisation — a signal that
    does not carry over is "No Edge" however stable it looks. Then
    significance against the circular-shift null. Stability is last, and only
    separates a weak-but-real signal from a strong-but-fragile one.
    """
    if n_independent is not None and n_independent < GATE_MIN_INDEPENDENT_WINDOWS:
        return QUALITY_INSUFFICIENT, "neutral"
    if not np.isfinite(holdout_ir) or holdout_ir < GATE_MIN_HOLDOUT_IR:
        return QUALITY_NO_EDGE, "danger"
    if not np.isfinite(p_value) or p_value > GATE_MAX_P_VALUE:
        return QUALITY_NO_EDGE, "danger"
    stability = holdout_ir / train_ir if train_ir > 0 else 0.0
    if stability < GATE_OVERFIT_STABILITY:
        return QUALITY_OVERFIT, "warning"
    return QUALITY_OK, "success"


def score_series_ir(
    series: np.ndarray,
    mood_df: pd.DataFrame,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    *,
    window: slice | None = None,
    n_blocks: int = HOLDOUT_BLOCKS,
) -> tuple[float, dict[int, float]]:
    """IR of any 1-D signal over ``window`` (default: the whole series).

    Used by the Intelligence Center to score raw Mood and Calibrated
    Conviction on identical, held-out data — so the comparison can come out
    either way. Previously both were scored on the folds the weights had been
    fitted on, which made the calibrated signal win by construction.
    """
    horizons = [int(h) for h in horizons]
    n = len(mood_df)
    if len(series) != n or n < MIN_TRAIN_ROWS:
        return 0.0, {h: 0.0 for h in horizons}

    win = window or slice(0, n)
    blocks = _contiguous_blocks(win.start or 0, win.stop or n, n_blocks)
    if not blocks:
        return 0.0, {h: 0.0 for h in horizons}

    nifty = mood_df["NIFTY"].to_numpy(dtype=np.float64)
    sig = np.asarray(series, dtype=np.float64).copy()
    sig[~valid_mask(mood_df)] = np.nan
    fwd = _precompute_forward_returns(nifty, horizons)

    per_h: dict[int, list[float]] = {h: [] for h in horizons}
    for blk in blocks:
        for h in horizons:
            per_h[h].append(_spearman_safe(sig[blk], fwd[h][blk]))
    all_rhos = [r for rhos in per_h.values() for r in rhos]
    return _ir(all_rhos), {h: _ir(rhos) for h, rhos in per_h.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Tuner
# ──────────────────────────────────────────────────────────────────────────────

class IntelligenceTuner:
    """Tune the ensemble that maps engine output → calibrated conviction."""

    def __init__(
        self,
        mood_df: pd.DataFrame,
        msf_df: pd.DataFrame,
        n_active_predictors: int,
        horizons: Iterable[int] = DEFAULT_HORIZONS,
        n_folds: int = DEFAULT_FOLDS,
        embargo_days: int | None = None,
        l2_alpha: float = DEFAULT_L2_ALPHA,
        holdout_fraction: float = HOLDOUT_FRACTION,
    ):
        if mood_df is None or mood_df.empty:
            raise ValueError("mood_df is empty — cannot calibrate without engine output.")
        if "NIFTY" not in mood_df.columns:
            raise ValueError("mood_df must contain a NIFTY column.")

        self.mood_df = mood_df
        self.msf_df = msf_df
        self.n_active_preds = int(n_active_predictors)
        self.horizons = tuple(int(h) for h in horizons)
        self.n_folds = int(n_folds)
        self.embargo_days = int(embargo_days if embargo_days is not None
                                else default_embargo(self.horizons))
        if self.embargo_days < max(self.horizons):
            raise ValueError(
                f"embargo ({self.embargo_days}d) is shorter than the longest horizon "
                f"({max(self.horizons)}d) — training labels would overlap validation."
            )
        self.l2_alpha = float(l2_alpha)

        self.n = len(mood_df)
        self.feature_matrix = build_feature_matrix(mood_df, msf_df)
        self.nifty_arr = mood_df["NIFTY"].to_numpy(dtype=np.float64)
        self.fwd_returns = _precompute_forward_returns(self.nifty_arr, self.horizons)
        self.valid = valid_mask(mood_df)

        # Holdout carved off the end. The optimiser never sees these rows.
        holdout_len = max(HOLDOUT_MIN_ROWS, int(round(self.n * holdout_fraction)))
        self.holdout_start = self.n - holdout_len
        if self.holdout_start < MIN_TRAIN_ROWS + self.embargo_days + MIN_SPEARMAN_OBS:
            raise ValueError(
                f"Dataset too small to reserve a holdout "
                f"(n={self.n}, holdout={holdout_len}, min train={MIN_TRAIN_ROWS})."
            )
        self.search_n = self.holdout_start

        self.folds = _walk_forward_folds(
            self.search_n, self.n_folds, self.embargo_days, MIN_TRAIN_ROWS,
        )
        if not self.folds:
            raise ValueError(
                f"Dataset too small for {self.n_folds}-fold CV with a "
                f"{self.embargo_days}d embargo (search rows={self.search_n})."
            )
        self.holdout_blocks = _contiguous_blocks(self.holdout_start, self.n, HOLDOUT_BLOCKS)
        if not self.holdout_blocks:
            raise ValueError("Holdout window is too short to score.")

        self.best_weights: dict = {}
        self.best_train_ir: float = 0.0
        self.best_val_ir: float = 0.0
        self.holdout_ir: float = 0.0
        self.holdout_p_value: float = 1.0
        self.best_stability: float = 0.0
        self.best_quality: str = "—"
        self.importance: dict = {}
        self.study = None

    # ── Scoring ────────────────────────────────────────────────────────
    def _composite(self, w: np.ndarray) -> np.ndarray:
        sig = self.feature_matrix @ w
        sig[~self.valid] = np.nan
        return sig

    def _score_weights(self, w: np.ndarray) -> tuple[float, float]:
        """(train_ir, val_ir) across the search region only."""
        composite = self._composite(w)
        train_rhos: list[float] = []
        val_rhos: list[float] = []
        for train_slc, val_slc in self.folds:
            for h in self.horizons:
                fwd = self.fwd_returns[h]
                train_rhos.append(_spearman_safe(composite[train_slc], fwd[train_slc]))
                val_rhos.append(_spearman_safe(composite[val_slc], fwd[val_slc]))
        return _ir(train_rhos), _ir(val_rhos)

    def _score_holdout(self, w: np.ndarray) -> float:
        composite = self._composite(w)
        rhos = [
            _spearman_safe(composite[blk], self.fwd_returns[h][blk])
            for blk in self.holdout_blocks
            for h in self.horizons
        ]
        return _ir(rhos)

    def _holdout_p_value(self, w: np.ndarray, n_perm: int = N_PERMUTATIONS,
                         seed: int = 12345) -> float:
        """Fraction of circularly shifted signals matching or beating the real one.

        A plain shuffle would destroy the signal's own autocorrelation and
        produce an unrealistically easy null. Circular shifts keep both series
        intact and break only their alignment, which is the relationship being
        tested.
        """
        composite = self._composite(w)
        hold = slice(self.holdout_start, self.n)
        sig = composite[hold]
        m = len(sig)
        if m < 3 * MIN_SPEARMAN_OBS:
            return 1.0

        rel_blocks = [slice(b.start - self.holdout_start, b.stop - self.holdout_start)
                      for b in self.holdout_blocks]
        fwd_h = {h: self.fwd_returns[h][hold] for h in self.horizons}

        def statistic(series: np.ndarray) -> float:
            return _mean_rho([
                _spearman_safe(series[b], fwd_h[h][b])
                for b in rel_blocks for h in self.horizons
            ])

        # Two-sided in effect: the ensemble is free to learn either sign, so
        # the null must be compared on magnitude.
        actual = abs(statistic(sig))

        rng = np.random.default_rng(seed)
        min_shift = max(MIN_SPEARMAN_OBS, max(self.horizons))
        if m - 2 * min_shift <= 1:
            return 1.0

        beat = 0
        for _ in range(n_perm):
            shift = int(rng.integers(min_shift, m - min_shift))
            if abs(statistic(np.roll(sig, shift))) >= actual:
                beat += 1
        # Add-one smoothing: with n_perm draws the p-value cannot be exactly 0.
        return (beat + 1.0) / (n_perm + 1.0)

    # ── Optimisation ───────────────────────────────────────────────────
    def optimize(
        self,
        n_trials: int = DEFAULT_TRIALS,
        seed: int = 42,
        progress_callback: Callable[[int, int, float], None] | None = None,
    ) -> CalibrationProfile:
        try:
            import optuna  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Intelligence Mode requires `optuna` — add it to requirements.txt."
            ) from exc
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        n_trials = int(n_trials)
        n_features = len(FEATURE_NAMES)

        def _suggest(trial) -> np.ndarray:
            w = np.zeros(n_features, dtype=np.float64)
            for i, name in enumerate(FEATURE_NAMES):
                lo, hi = WEIGHT_BOUNDS[name]
                w[i] = trial.suggest_float(name, lo, hi)
            return w

        def _objective(trial) -> float:
            w = _suggest(trial)
            train_ir, val_ir = self._score_weights(w)
            score = 0.65 * val_ir + 0.35 * train_ir
            score -= _l2_penalty(w, self.l2_alpha)
            if progress_callback:
                try:
                    progress_callback(trial.number + 1, n_trials, float(score))
                except Exception:
                    pass
            return float(score)

        self.study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=PRUNER_WARMUP_TRIALS,
                n_warmup_steps=PRUNER_WARMUP_STEPS,
            ),
        )
        self.study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)

        best_params = dict(self.study.best_params)
        w_best = np.array(
            [best_params.get(name, 0.0) for name in FEATURE_NAMES], dtype=np.float64,
        )

        train_ir, val_ir = self._score_weights(w_best)
        holdout_ir = self._score_holdout(w_best)
        p_value = self._holdout_p_value(w_best)

        # Stability is holdout-over-train. The old val-over-train ratio was
        # meaningless here: the objective weights val at 0.65 against train at
        # 0.35, so it actively pushed the ratio above 1.
        stability = (holdout_ir / train_ir) if train_ir > 0 else 0.0
        n_indep = independent_windows(self.n - self.holdout_start, self.horizons)
        q_label, _ = quality_check(train_ir, holdout_ir, p_value, n_indep)

        self.best_weights = best_params
        self.best_train_ir = train_ir
        self.best_val_ir = val_ir
        self.holdout_ir = holdout_ir
        self.holdout_p_value = p_value
        self.best_stability = stability
        self.best_quality = q_label
        self.importance = self._compute_importance()

        return self._make_profile()

    def _compute_importance(self) -> dict:
        """Share of |weight| per feature.

        This used to report Optuna's fANOVA over the objective, which measures
        how sensitive the SEARCH SURFACE is to each parameter — a function of
        the bound widths and the sampler's path, not of the feature's
        contribution to the signal. Presenting that as "explanatory power"
        ranked the feature cards by an unrelated quantity. Contribution share
        of the fitted coefficients is at least the thing it claims to be, and
        is honest now that the design is no longer singular.
        """
        if not self.best_weights:
            return {}
        total = sum(abs(v) for v in self.best_weights.values())
        if total <= 1e-12:
            return {k: 0.0 for k in self.best_weights}
        return {k: abs(v) / total * 100.0 for k, v in self.best_weights.items()}

    def _make_profile(self) -> CalibrationProfile:
        date_col = self.mood_df["DATE"]
        # Unique rows actually used, not the sum over overlapping expanding
        # folds — that used to report 5,140 "train rows" for a 2,200-row series.
        n_train = int(max(tr.stop for tr, _ in self.folds))
        n_holdout = int(self.n - self.holdout_start)

        return CalibrationProfile(
            weights=self.best_weights,
            train_ir=self.best_train_ir,
            val_ir=self.best_val_ir,
            holdout_ir=self.holdout_ir,
            holdout_p_value=self.holdout_p_value,
            stability=self.best_stability,
            quality_check=self.best_quality,
            horizons=list(self.horizons),
            n_folds=len(self.folds),
            n_trials=int(len(self.study.trials)) if self.study else 0,
            embargo_days=self.embargo_days,
            n_rows_train=n_train,
            n_rows_holdout=n_holdout,
            n_predictors=self.n_active_preds,
            data_start=str(date_col.min().date()),
            data_end=str(date_col.max().date()),
            holdout_start=str(date_col.iloc[self.holdout_start].date()),
            importance=self.importance,
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            arthagati_version=VERSION,
        )
