"""
Arthagati Intelligence Mode — self-calibration of mood-engine hyperparameters.

Methodology
-----------
Search space (8 hyperparameters, all integer- or float-bounded):
  - CORR_HALF_LIFE         300 – 700   int (Spearman recency)
  - PCT_HALF_LIFE          150 – 400   int (adaptive percentile recency)
  - KALMAN_HALF_LIFE        60 – 250   int (Kalman fading memory)
  - MSF_WINDOW              10 –  40   int (rolling window for MSF components)
  - MSF_ROC_LEN              5 –  30   int (NIFTY ROC period)
  - MSF_ZSCORE_CLIP        2.0 – 5.0   float (Z-score clip)
  - CORR_MIN_WARMUP        180 – 400   int (first walk-forward checkpoint)
  - CORR_REBALANCE_PERIOD   30 – 126   int (walk-forward rebalance cadence)

Objective: out-of-sample **Information Ratio** of Spearman rank correlation
between Mood Score at t and NIFTY forward return over horizons {30, 60, 90}
trading days. Walk-forward 5-fold expanding-window CV (purged & embargoed
to prevent leakage at fold boundaries).

  IR = mean(Spearman) / std(Spearman) across folds × horizons.

Reported metrics: Train IR, Validation IR, Stability = Val/Train, plus a
quality-check verdict (Quality OK · Overfit · No Edge) used as a hard gate
when saving a profile.

L2 regularisation (alpha = 0.001) is applied to the *normalised* deviation
of each hyperparameter from its factory default — discourages runaway fits
on small N.

Persistence
-----------
Profiles are saved as JSON under ``profiles/`` in the project root, with a
schema-versioned envelope. Atomic writes (tmp + rename) avoid torn files
when Streamlit reruns mid-write. Per-session a single ``active.json`` is
the loaded profile.

Author safety notes
-------------------
- All trials run on a frozen copy of raw_df (deep copy passed in by caller),
  so concurrent Streamlit reruns can't mutate during a long search.
- The hot path uses arthagati._calculate_historical_mood_impl directly to
  avoid filling st.cache_data with thousands of weight-distinct entries.
- A trial that raises is caught, scored as `-100.0`, and logged — never
  silently swallowed.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import warnings
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Local imports are deferred to avoid a hard cycle with arthagati.py at
# module-load time — engine code is fetched inside functions.

PROFILE_SCHEMA_VERSION = 1
PROFILE_DIR = Path(__file__).resolve().parent / "profiles"
ACTIVE_PROFILE_PATH = PROFILE_DIR / "active.json"

DEFAULT_HORIZONS: tuple[int, ...] = (30, 60, 90)
DEFAULT_FOLDS: int = 5
DEFAULT_TRIALS: int = 40
DEFAULT_EMBARGO_DAYS: int = 5     # gap between train end and val start per fold
DEFAULT_TRAIN_FRAC: float = 0.70  # only used for the legacy single-split view
DEFAULT_L2_ALPHA: float = 0.001

# Calibration runs on the FULL dataset for fidelity. Cost is amortised by:
#   1. Cross-session profile caching (profile is reused if (data_end_date,
#      predictor count) match the active profile's fingerprint)
#   2. Session-level engine-output caching (mood_df + msf_df are cached
#      by input fingerprint, so view/timeframe switches are O(1))
# A user calibrates ONCE per genuine input change, not once per Run Analysis.
CALIBRATION_MAX_ROWS: int | None = None   # None == full dataset

# Stale-profile threshold — re-calibrate if the saved profile was fit on
# data older than this. NIFTY's structural regime doesn't change weekly,
# so 14 days is a safe re-calibration cadence.
PROFILE_FRESHNESS_DAYS: int = 14

# Optuna pruner cutoffs (MedianPruner)
PRUNER_WARMUP_TRIALS: int = 8     # finish this many trials before pruning anyone
PRUNER_WARMUP_STEPS:  int = 1     # report intermediate value after this many steps

# ── Quality gate thresholds (used when deciding to ship a profile) ──────
GATE_MIN_VAL_IR: float = 0.0        # val IR must be strictly > 0
GATE_OVERFIT_STABILITY: float = 0.30  # val/train ratio floor — below this == overfit


# ──────────────────────────────────────────────────────────────────────────────
# Search-space definitions (full bounds — fidelity-preserving)
# ──────────────────────────────────────────────────────────────────────────────

SEARCH_SPACE: dict[str, dict] = {
    "CORR_HALF_LIFE":         {"type": "int",   "low": 300, "high": 700},
    "PCT_HALF_LIFE":          {"type": "int",   "low": 150, "high": 400},
    "KALMAN_HALF_LIFE":       {"type": "int",   "low":  60, "high": 250},
    "MSF_WINDOW":             {"type": "int",   "low":  10, "high":  40},
    "MSF_ROC_LEN":            {"type": "int",   "low":   5, "high":  30},
    "MSF_ZSCORE_CLIP":        {"type": "float", "low": 2.0, "high": 5.0},
    "CORR_MIN_WARMUP":        {"type": "int",   "low": 180, "high": 400},
    "CORR_REBALANCE_PERIOD":  {"type": "int",   "low":  30, "high": 126},
}


# ──────────────────────────────────────────────────────────────────────────────
# Profile dataclass + persistence
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CalibrationProfile:
    """A calibrated hyperparameter set with its diagnostics and provenance."""

    weights: dict           # hyperparameter name → value
    train_ir: float
    val_ir: float
    stability: float        # val_ir / train_ir
    quality_check: str      # "Quality OK" | "Overfit" | "No Edge"
    horizons: list[int]
    n_folds: int
    n_trials: int
    embargo_days: int
    n_dates_train: int
    n_dates_val: int
    n_predictors: int
    data_start: str         # ISO date
    data_end: str           # ISO date
    importance: dict        # param → relative importance (0–100)
    timestamp: str          # ISO datetime
    arthagati_version: str
    schema_version: int = PROFILE_SCHEMA_VERSION
    sensitivity_curves: dict = field(default_factory=dict)

    # ── Diagnostics ────────────────────────────────────────────────────
    @property
    def is_default(self) -> bool:
        return not bool(self.weights)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)

    @classmethod
    def from_dict(cls, raw: dict) -> "CalibrationProfile":
        """Tolerant loader — fills missing keys with defaults so older
        profile schemas don't crash on import."""
        defaults: dict = {
            "weights": {},
            "train_ir": 0.0,
            "val_ir": 0.0,
            "stability": 0.0,
            "quality_check": "—",
            "horizons": list(DEFAULT_HORIZONS),
            "n_folds": DEFAULT_FOLDS,
            "n_trials": DEFAULT_TRIALS,
            "embargo_days": DEFAULT_EMBARGO_DAYS,
            "n_dates_train": 0,
            "n_dates_val": 0,
            "n_predictors": 0,
            "data_start": "",
            "data_end": "",
            "importance": {},
            "timestamp": "",
            "arthagati_version": "",
            "schema_version": PROFILE_SCHEMA_VERSION,
            "sensitivity_curves": {},
        }
        merged = {**defaults, **{k: v for k, v in raw.items() if k in defaults}}
        return cls(**merged)


def ensure_profile_dir() -> None:
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)


def save_active_profile(profile: CalibrationProfile) -> Path:
    """Atomic JSON write to PROFILE_DIR/active.json. Returns the path."""
    ensure_profile_dir()
    body = profile.to_json()
    fd, tmp = tempfile.mkstemp(
        prefix="active.", suffix=".tmp", dir=str(PROFILE_DIR)
    )
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
    """Return the currently-active profile, or None if no profile saved."""
    if not ACTIVE_PROFILE_PATH.exists():
        return None
    try:
        raw = json.loads(ACTIVE_PROFILE_PATH.read_text(encoding="utf-8"))
        return CalibrationProfile.from_dict(raw)
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        warnings.warn(f"Failed to load active profile: {exc}")
        return None


def delete_active_profile() -> bool:
    """Remove the active profile from disk. Returns True if it existed."""
    if ACTIVE_PROFILE_PATH.exists():
        ACTIVE_PROFILE_PATH.unlink()
        return True
    return False


def archive_profile(profile: CalibrationProfile) -> Path:
    """Save a timestamped copy alongside the active profile.

    Lets users keep a history of calibrated runs without overwriting.
    """
    ensure_profile_dir()
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = PROFILE_DIR / f"profile_{ts}.json"
    path.write_text(profile.to_json(), encoding="utf-8")
    return path


def list_profiles() -> list[Path]:
    """All archived profile JSONs (excluding the live active.json), newest first."""
    if not PROFILE_DIR.exists():
        return []
    return sorted(
        (p for p in PROFILE_DIR.glob("profile_*.json")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Profile freshness + dataset fingerprints
# ──────────────────────────────────────────────────────────────────────────────

def profile_age_days(profile: CalibrationProfile) -> float:
    """Approximate age of a profile in days (UTC). 9e9 if timestamp is malformed."""
    try:
        ts = profile.timestamp.replace("Z", "")
        fit_at = datetime.fromisoformat(ts)
        return (datetime.utcnow() - fit_at).total_seconds() / 86400.0
    except Exception:
        return 9e9


def is_profile_fresh(
    profile: CalibrationProfile | None,
    raw_df: pd.DataFrame,
    active_predictors: Iterable[str],
    *,
    max_age_days: float = PROFILE_FRESHNESS_DAYS,
) -> tuple[bool, str]:
    """Return (fresh?, reason). Used by the auto-calibrator to decide whether
    to skip calibration this run.

    Fresh ⇔ all of:
      • profile exists and quality_check != "No Edge"
      • profile.data_end is within max_age_days of raw_df's last row
      • profile.n_predictors == len(active_predictors)
      • profile age (timestamp) is within max_age_days
    """
    if profile is None:
        return False, "no profile on disk"
    if profile.quality_check == "No Edge":
        return False, "previous calibration failed quality gate"

    n_active = len(tuple(active_predictors))
    if profile.n_predictors != n_active:
        return False, f"predictor count changed ({profile.n_predictors} → {n_active})"

    data_end = pd.Timestamp(raw_df["DATE"].max())
    try:
        profile_end = pd.Timestamp(profile.data_end)
        gap_days = (data_end - profile_end).days
    except Exception:
        return False, "profile data_end is malformed"
    if gap_days > max_age_days:
        return False, f"data extends {gap_days}d beyond profile (>{max_age_days}d threshold)"
    if gap_days < -1:
        # profile claims newer data than what we have — suspicious; recalibrate
        return False, f"profile data_end is in the future ({-gap_days}d ahead)"

    age = profile_age_days(profile)
    if age > max_age_days:
        return False, f"profile is {age:.0f}d old (>{max_age_days}d threshold)"

    return True, f"profile fresh · {age:.1f}d old · data within {gap_days}d"


def dataset_fingerprint(
    raw_df: pd.DataFrame,
    predictors: Iterable[str],
    hyperparams: dict,
) -> tuple:
    """Hashable fingerprint of the inputs that determine engine output.

    Used as a session-state cache key for ``mood_df`` and ``msf_df`` so
    view-switches and timeframe-button clicks don't rerun the engine.
    """
    return (
        int(len(raw_df)),
        str(raw_df["DATE"].iloc[0].date()) if len(raw_df) else "",
        str(raw_df["DATE"].iloc[-1].date()) if len(raw_df) else "",
        tuple(sorted(predictors)),
        tuple(sorted((k, v) for k, v in hyperparams.items())),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Walk-forward CV: fold generator
# ──────────────────────────────────────────────────────────────────────────────

def _walk_forward_folds(
    n: int, n_folds: int, embargo: int, min_train_size: int,
) -> list[tuple[slice, slice]]:
    """Expanding-window walk-forward CV with a purged embargo gap.

    Returns a list of (train_slice, val_slice) tuples. Each fold's train
    grows; val is a fixed-size window. A gap of ``embargo`` rows sits
    between train end and val start, preventing overlap-leakage from the
    forward-return label space.

    If the dataset is too small for the requested fold count, the result
    may have fewer folds — never errors.
    """
    if n_folds < 2 or n < (min_train_size + embargo + n_folds):
        return []

    # Carve the remaining rows after warmup into roughly equal val windows.
    val_total   = n - min_train_size - embargo
    val_size    = max(1, val_total // n_folds)

    folds: list[tuple[slice, slice]] = []
    for f in range(n_folds):
        val_start = min_train_size + embargo + f * val_size
        val_end   = val_start + val_size if f < n_folds - 1 else n
        if val_end - val_start < 30:
            # Skip degenerate trailing fold — Spearman needs material rows.
            continue
        train_end = val_start - embargo
        if train_end <= 0:
            continue
        folds.append((slice(0, train_end), slice(val_start, val_end)))
    return folds


# ──────────────────────────────────────────────────────────────────────────────
# Scoring: mood vs forward-return Spearman IR across horizons × folds
# ──────────────────────────────────────────────────────────────────────────────

def _spearman_safe(x: np.ndarray, y: np.ndarray) -> float:
    """NaN-safe Spearman. Returns 0.0 on degenerate input."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 30:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho, _ = spearmanr(x[mask], y[mask])
    return float(rho) if np.isfinite(rho) else 0.0


def _fold_score(
    mood_arr: np.ndarray,
    nifty_arr: np.ndarray,
    fold_slice: slice,
    horizons: Iterable[int],
) -> list[float]:
    """Spearman correlations for one fold, one per horizon. NaNs filtered."""
    out: list[float] = []
    mood = mood_arr[fold_slice]
    n_fold = len(mood)
    fold_start = fold_slice.start
    for h in horizons:
        # Forward return over h trading days, anchored at each date in fold.
        # We must look h rows AHEAD into the global nifty array.
        end_global  = min(fold_start + n_fold, len(nifty_arr) - h)
        valid_local = end_global - fold_start
        if valid_local < 30:
            continue
        nifty_start = nifty_arr[fold_start : fold_start + valid_local]
        nifty_end   = nifty_arr[fold_start + h : fold_start + h + valid_local]
        with np.errstate(divide="ignore", invalid="ignore"):
            fwd = (nifty_end / nifty_start - 1.0) * 100.0
        mood_w = mood[:valid_local]
        rho = _spearman_safe(mood_w, fwd)
        out.append(rho)
    return out


def _compute_ir(
    mood_arr: np.ndarray,
    nifty_arr: np.ndarray,
    folds: list[tuple[slice, slice]],
    horizons: Iterable[int],
    split: str,  # "train" or "val"
) -> tuple[float, list[float]]:
    """Information Ratio across (folds × horizons) for either train or val.

    Returns (IR, raw_rho_list).
    """
    rhos: list[float] = []
    for train_slice, val_slice in folds:
        slc = train_slice if split == "train" else val_slice
        rhos.extend(_fold_score(mood_arr, nifty_arr, slc, horizons))
    rhos_arr = np.asarray([r for r in rhos if np.isfinite(r)], dtype=np.float64)
    if len(rhos_arr) < 3:
        return 0.0, rhos
    sd = max(rhos_arr.std(ddof=1), 1e-6)
    return float(rhos_arr.mean() / sd), rhos


# ──────────────────────────────────────────────────────────────────────────────
# L2 regularisation on deviation from factory defaults
# ──────────────────────────────────────────────────────────────────────────────

def _l2_penalty(weights: dict, defaults: dict, alpha: float) -> float:
    """Mean squared *relative* deviation from factory defaults, scaled by alpha."""
    if not weights or alpha <= 0:
        return 0.0
    sq = 0.0
    n = 0
    for k, v in weights.items():
        d = defaults.get(k)
        if d in (None, 0):
            continue
        sq += ((float(v) - float(d)) / float(d)) ** 2
        n += 1
    return alpha * (sq / max(n, 1))


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline runner — evaluates one weight set end-to-end
# ──────────────────────────────────────────────────────────────────────────────

def _run_pipeline(raw_df: pd.DataFrame, dependent_vars, hyperparams: dict) -> np.ndarray:
    """Run the engine with the given hyperparams and return the Mood_Score array.

    Deferred imports avoid circular dependency with arthagati at module load.
    Calibration-quiet mode suppresses the engine's per-call detail log so
    the operator console stays readable across a 20-trial sweep.
    """
    import arthagati  # local import — engine + override CM

    with arthagati.calibration_quiet_mode(), arthagati.hyperparam_overrides(hyperparams):
        mood_df = arthagati._calculate_historical_mood_impl(raw_df, dependent_vars)
    if mood_df is None or mood_df.empty:
        return np.array([])
    return mood_df["Mood_Score"].to_numpy(dtype=np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# Quality gate
# ──────────────────────────────────────────────────────────────────────────────

def quality_check(train_ir: float, val_ir: float) -> tuple[str, str]:
    """Map (train_ir, val_ir) → (label, severity).

    severity is "success" | "warning" | "danger" — drives the UI colour.
    """
    if val_ir <= GATE_MIN_VAL_IR:
        return "No Edge", "danger"
    stability = val_ir / train_ir if train_ir > 0 else 0.0
    if train_ir > 0.05 and stability < GATE_OVERFIT_STABILITY:
        return "Overfit", "warning"
    return "Quality OK", "success"


# ──────────────────────────────────────────────────────────────────────────────
# Public optimizer
# ──────────────────────────────────────────────────────────────────────────────

class IntelligenceTuner:
    """Self-calibration for Arthagati mood-engine hyperparameters.

    Parameters
    ----------
    raw_df:
        The Google-Sheets-loaded market dataframe (with DATE, NIFTY,
        NIFTY50_PE, NIFTY50_EY, predictors).
    dependent_vars:
        Active predictor column names (matches the user's sidebar selection).
    horizons:
        Forward-return horizons (trading days) over which to score Spearman IR.
    n_folds:
        Walk-forward CV folds.
    embargo_days:
        Gap between train end and val start each fold.
    l2_alpha:
        L2 weight on relative deviation from factory defaults.
    """

    def __init__(
        self,
        raw_df: pd.DataFrame,
        dependent_vars: Iterable[str],
        horizons: Iterable[int] = DEFAULT_HORIZONS,
        n_folds: int = DEFAULT_FOLDS,
        embargo_days: int = DEFAULT_EMBARGO_DAYS,
        l2_alpha: float = DEFAULT_L2_ALPHA,
        max_rows: int | None = CALIBRATION_MAX_ROWS,
    ):
        """Build a calibration tuner.

        Parameters
        ----------
        max_rows :
            If set, calibration runs on only the LAST ``max_rows`` rows of
            ``raw_df``. The default (1500 ≈ 6 trading years) keeps trials
            fast while preserving recent regime dynamics. Pass ``None`` to
            calibrate on the full dataset (slow).
        """
        import arthagati  # for default hyperparams

        # Frozen snapshot. Default behaviour (max_rows=None) calibrates on
        # the full dataset for fidelity — engine + profile output caches
        # make the once-per-fortnight wall time acceptable.
        src = raw_df.copy()
        if max_rows is not None and len(src) > max_rows:
            src = src.tail(max_rows).reset_index(drop=True)
        self.raw_df          = src

        self.dependent_vars  = tuple(dependent_vars)
        self.horizons        = tuple(int(h) for h in horizons)
        self.n_folds         = int(n_folds)
        self.embargo_days    = int(embargo_days)
        self.l2_alpha        = float(l2_alpha)
        self.max_rows        = max_rows
        self.defaults        = arthagati.get_default_hyperparams()

        self.nifty_arr       = self.raw_df["NIFTY"].to_numpy(dtype=np.float64)
        self.n               = len(self.raw_df)

        # CV folds are weight-invariant — precompute once.
        min_train = max(self.defaults.get("CORR_MIN_WARMUP", 252), 252)
        self.folds = _walk_forward_folds(
            self.n, self.n_folds, self.embargo_days, min_train,
        )
        if not self.folds:
            raise ValueError(
                f"Dataset too small for {self.n_folds}-fold CV "
                f"(n={self.n}, min_train={min_train})."
            )

        # ── Diagnostics ──
        self.best_weights: dict = {}
        self.best_train_ir: float = 0.0
        self.best_val_ir: float   = 0.0
        self.best_stability: float = 0.0
        self.best_quality: str    = "—"
        self.importance: dict     = {}
        self.study = None

    # ── Trial scoring ──────────────────────────────────────────────────
    def _score(self, hyperparams: dict) -> tuple[float, float, list[float], list[float]]:
        """Evaluate one weight set on all folds. Returns
        (train_ir, val_ir, train_rhos, val_rhos)."""
        try:
            mood_arr = _run_pipeline(self.raw_df, self.dependent_vars, hyperparams)
        except Exception as exc:
            warnings.warn(f"Pipeline failure on hyperparams={hyperparams}: {exc}")
            return -100.0, -100.0, [], []
        if len(mood_arr) != self.n:
            return -100.0, -100.0, [], []
        train_ir, train_rhos = _compute_ir(
            mood_arr, self.nifty_arr, self.folds, self.horizons, "train",
        )
        val_ir, val_rhos = _compute_ir(
            mood_arr, self.nifty_arr, self.folds, self.horizons, "val",
        )
        return train_ir, val_ir, train_rhos, val_rhos

    # ── Optuna search ──────────────────────────────────────────────────
    def optimize(
        self,
        n_trials: int = DEFAULT_TRIALS,
        seed: int = 42,
        progress_callback: Callable[[int, int, float], None] | None = None,
    ) -> CalibrationProfile:
        """Run TPE Bayesian search with median pruning.

        The objective runs the mood engine once per trial, scores it on
        each (fold, horizon) pair, and reports an intermediate value after
        each fold so MedianPruner can short-circuit obviously-bad trials.
        """
        try:
            import optuna  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Intelligence Mode requires `optuna` — add it to requirements.txt."
            ) from exc
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        n_trials = int(n_trials)

        def _suggest(trial) -> dict:
            params: dict = {}
            for name, cfg in SEARCH_SPACE.items():
                if cfg["type"] == "int":
                    params[name] = trial.suggest_int(name, cfg["low"], cfg["high"])
                else:
                    params[name] = trial.suggest_float(name, cfg["low"], cfg["high"])
            return params

        def _objective(trial) -> float:
            hp = _suggest(trial)
            # One engine run per trial — produces the mood-score array.
            try:
                mood_arr = _run_pipeline(self.raw_df, self.dependent_vars, hp)
            except Exception as exc:
                warnings.warn(f"Pipeline failure on hyperparams={hp}: {exc}")
                return -100.0
            if len(mood_arr) != self.n:
                return -100.0

            # Walk folds in order, reporting intermediate val IR each step.
            # MedianPruner uses these reports to abort losing trials early.
            train_rhos: list[float] = []
            val_rhos:   list[float] = []
            for step, (train_slc, val_slc) in enumerate(self.folds):
                train_rhos.extend(_fold_score(mood_arr, self.nifty_arr, train_slc, self.horizons))
                val_rhos.extend(  _fold_score(mood_arr, self.nifty_arr, val_slc,   self.horizons))
                # Intermediate IR (partial — fewer folds so far)
                if val_rhos:
                    v = np.asarray([r for r in val_rhos if np.isfinite(r)], dtype=np.float64)
                    if len(v) >= 3:
                        partial_ir = float(v.mean() / max(v.std(ddof=1), 1e-6))
                        trial.report(partial_ir, step)
                        if trial.should_prune():
                            raise optuna.TrialPruned()

            # Final scores
            def _ir(rhos: list[float]) -> float:
                arr = np.asarray([r for r in rhos if np.isfinite(r)], dtype=np.float64)
                if len(arr) < 3:
                    return 0.0
                return float(arr.mean() / max(arr.std(ddof=1), 1e-6))

            train_ir = _ir(train_rhos)
            val_ir   = _ir(val_rhos)

            score = 0.65 * val_ir + 0.35 * train_ir
            score -= _l2_penalty(hp, self.defaults, self.l2_alpha)
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
        train_ir, val_ir, _, _ = self._score(best_params)
        stability  = (val_ir / train_ir) if train_ir > 0 else 0.0
        q_label, _ = quality_check(train_ir, val_ir)

        self.best_weights   = best_params
        self.best_train_ir  = train_ir
        self.best_val_ir    = val_ir
        self.best_stability = stability
        self.best_quality   = q_label
        self.importance     = self._compute_importance()

        return self._make_profile()

    # ── fANOVA importance with weight-share fallback ───────────────────
    def _compute_importance(self) -> dict:
        if self.study is None or len(self.study.trials) < 3:
            return {}
        try:
            import optuna  # type: ignore
            imp = optuna.importance.get_param_importances(self.study)
            total = sum(imp.values())
            if total <= 0:
                raise ValueError("fANOVA returned zero importance")
            return {k: (v / total) * 100.0 for k, v in imp.items()}
        except Exception:
            # Fallback: relative magnitude of |best - default| / default
            out: dict = {}
            for k, v in self.best_weights.items():
                d = self.defaults.get(k, 1.0) or 1.0
                out[k] = abs((float(v) - float(d)) / float(d)) * 100.0
            total = sum(out.values()) or 1.0
            return {k: (v / total) * 100.0 for k, v in out.items()}

    # ── Build profile envelope ─────────────────────────────────────────
    def _make_profile(self) -> CalibrationProfile:
        import arthagati  # for version string

        date_col = self.raw_df["DATE"]
        n_train  = sum((tr.stop - tr.start) for tr, _ in self.folds)
        n_val    = sum((vl.stop - vl.start) for _, vl in self.folds)

        return CalibrationProfile(
            weights=self.best_weights,
            train_ir=self.best_train_ir,
            val_ir=self.best_val_ir,
            stability=self.best_stability,
            quality_check=self.best_quality,
            horizons=list(self.horizons),
            n_folds=len(self.folds),
            n_trials=int(len(self.study.trials)) if self.study else 0,
            embargo_days=self.embargo_days,
            n_dates_train=int(n_train),
            n_dates_val=int(n_val),
            n_predictors=len(self.dependent_vars),
            data_start=str(date_col.min().date()),
            data_end=str(date_col.max().date()),
            importance=self.importance,
            timestamp=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            arthagati_version=getattr(arthagati, "VERSION", "unknown"),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Convenience runner used by the UI button
# ──────────────────────────────────────────────────────────────────────────────

def run_calibration(
    raw_df: pd.DataFrame,
    dependent_vars: Iterable[str],
    n_trials: int = DEFAULT_TRIALS,
    n_folds: int = DEFAULT_FOLDS,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    embargo_days: int = DEFAULT_EMBARGO_DAYS,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> CalibrationProfile:
    """One-shot calibration: instantiate tuner, run search, return profile.

    Caller is responsible for deciding whether to call ``save_active_profile``.
    """
    tuner = IntelligenceTuner(
        raw_df,
        dependent_vars,
        horizons=horizons,
        n_folds=n_folds,
        embargo_days=embargo_days,
    )
    return tuner.optimize(n_trials=n_trials, progress_callback=progress_callback)
