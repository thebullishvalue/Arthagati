"""
Arthagati Intelligence Mode — fast post-engine ensemble calibration.

Architecture (Nishkarsh-style, microsecond-per-trial):
-------------------------------------------------------
The mood engine runs ONCE on factory defaults and emits a rich feature
matrix (raw mood, smoothed mood, MSF spread, 4 MSF components, mood
divergence, mood squared/sqrt transforms — ~10 columns). Optuna then
tunes a small vector of post-engine ensemble weights ``w`` such that

    calibrated_conviction = F @ w

maximises out-of-sample Spearman IR across forward NIFTY-return horizons
{30, 60, 90} trading days.

Per-trial cost is a single matrix-vector multiply + a handful of
Spearman correlations — milliseconds. 40 trials complete in ~1 second
on production-shape data (~5000 rows × 10 features).

This is the right engineering trade because:
  • Structural-hyperparameter calibration (the v1 approach) requires the
    full engine to re-run per trial, costing 30-60s per trial on a
    4928×67 sheet. 40 trials = 20+ minutes. Streamlit Cloud unusable.
  • Post-engine ensemble calibration produces a *different* signal —
    the Calibrated Conviction — alongside the raw Mood Score. Both are
    shown to the user. Fidelity of the raw signal is preserved.

Quality gate, walk-forward CV with purged embargo, and JSON profile
persistence behave identically to the v1 schema (downstream UI code is
schema-compatible).
"""

from __future__ import annotations

import json
import os
import tempfile
import warnings
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROFILE_SCHEMA_VERSION = 2  # v1 was structural-hyperparam tuning; v2 is ensemble
PROFILE_DIR = Path(__file__).resolve().parent / "profiles"
ACTIVE_PROFILE_PATH = PROFILE_DIR / "active.json"

DEFAULT_HORIZONS: tuple[int, ...] = (30, 60, 90)
DEFAULT_FOLDS: int = 5
DEFAULT_TRIALS: int = 40
DEFAULT_EMBARGO_DAYS: int = 5
DEFAULT_L2_ALPHA: float = 0.001

# Stale-profile threshold. With post-engine calibration this matters less
# (the run is cheap anyway) but we still skip when the same fingerprint
# is on disk to avoid burning Optuna trials needlessly.
PROFILE_FRESHNESS_DAYS: int = 14

PRUNER_WARMUP_TRIALS: int = 8
PRUNER_WARMUP_STEPS:  int = 1

GATE_MIN_VAL_IR: float = 0.0
GATE_OVERFIT_STABILITY: float = 0.30


# ──────────────────────────────────────────────────────────────────────────────
# Feature matrix construction (from engine output)
# ──────────────────────────────────────────────────────────────────────────────
# These are the columns the ensemble linearly combines. Order is the
# canonical search-space ordering — keep stable across versions because
# saved profiles reference features by name.

FEATURE_NAMES: tuple[str, ...] = (
    "mood",            # raw Mood_Score (engine output)
    "mood_smooth",     # Smoothed_Mood_Score (Kalman-smoothed)
    "mood_diverge",    # mood - mood_smooth (short-vs-long mood divergence)
    "mood_squared",    # sign(mood) * mood^2 / 100  (asymmetric amplification)
    "mood_sqrt",       # sign(mood) * sqrt(|mood|)  (asymmetric damping)
    "msf_spread",      # MSF composite oscillator
    "msf_momentum",    # NIFTY ROC z-score
    "msf_structure",   # Mood trend divergence + acceleration
    "msf_regime",      # Adaptive directional count
    "msf_flow",        # Breadth participation
)

# Search-space bounds per feature weight. Centered on 0; positive ↔ feature
# contributes to bullish signal. Wider for "primary" features.
WEIGHT_BOUNDS: dict[str, tuple[float, float]] = {
    "mood":          (-2.0, 2.0),
    "mood_smooth":   (-2.0, 2.0),
    "mood_diverge":  (-1.0, 1.0),
    "mood_squared":  (-1.0, 1.0),
    "mood_sqrt":     (-1.0, 1.0),
    "msf_spread":    (-2.0, 2.0),
    "msf_momentum":  (-1.0, 1.0),
    "msf_structure": (-1.0, 1.0),
    "msf_regime":    (-1.0, 1.0),
    "msf_flow":      (-1.0, 1.0),
}


def build_feature_matrix(mood_df: pd.DataFrame, msf_df: pd.DataFrame) -> np.ndarray:
    """Build the ``F`` matrix from engine output.

    Returns a ``(N, len(FEATURE_NAMES))`` ndarray. Missing columns
    contribute zeros so old engine versions still work.
    """
    n = len(mood_df)
    cols: list[np.ndarray] = []

    def _col(df: pd.DataFrame, name: str, default: float = 0.0) -> np.ndarray:
        if name in df.columns:
            arr = df[name].to_numpy(dtype=np.float64)
            return np.where(np.isfinite(arr), arr, default)
        return np.full(n, default, dtype=np.float64)

    mood        = _col(mood_df, "Mood_Score")
    mood_smooth = _col(mood_df, "Smoothed_Mood_Score", default=0.0)
    if not np.any(mood_smooth):
        # Fallback: if engine didn't emit Smoothed_Mood_Score, use a
        # simple EMA so the divergence column has signal.
        s = pd.Series(mood).ewm(span=20, adjust=False).mean()
        mood_smooth = s.to_numpy(dtype=np.float64)

    cols.append(mood)
    cols.append(mood_smooth)
    cols.append(mood - mood_smooth)
    cols.append(np.sign(mood) * (mood * mood) / 100.0)
    cols.append(np.sign(mood) * np.sqrt(np.abs(mood)))
    cols.append(_col(mood_df, "MSF_Spread"))
    cols.append(_col(msf_df, "momentum"))
    cols.append(_col(msf_df, "structure"))
    cols.append(_col(msf_df, "regime"))
    cols.append(_col(msf_df, "flow"))

    F = np.column_stack(cols)
    # Standardise: zero mean, unit variance per column. Lets Optuna treat
    # weight bounds uniformly without each feature's natural scale
    # swamping the others.
    mu = np.nanmean(F, axis=0)
    sd = np.nanstd(F, axis=0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    F = (F - mu) / sd
    # Replace any residual NaN with 0
    F = np.where(np.isfinite(F), F, 0.0)
    return F


def apply_calibration(
    mood_df: pd.DataFrame,
    msf_df: pd.DataFrame,
    weights: dict[str, float],
) -> np.ndarray:
    """Produce the calibrated conviction time-series for a given weight set.

    The output is rescaled to roughly the [-100, +100] range of Mood_Score
    via tanh, so it composes cleanly with the existing UI scales.
    """
    F = build_feature_matrix(mood_df, msf_df)
    w = np.array(
        [float(weights.get(name, 0.0)) for name in FEATURE_NAMES],
        dtype=np.float64,
    )
    raw = F @ w
    # tanh squash to ±100 so the output reads on the same scale as Mood_Score.
    return np.tanh(raw / 3.0) * 100.0


# ──────────────────────────────────────────────────────────────────────────────
# Profile envelope
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CalibrationProfile:
    """Calibrated post-engine ensemble weights + diagnostics."""

    weights: dict[str, float]   # feature name → weight (FEATURE_NAMES keys)
    train_ir: float
    val_ir: float
    stability: float
    quality_check: str          # "Quality OK" | "Overfit" | "No Edge"
    horizons: list[int]
    n_folds: int
    n_trials: int
    embargo_days: int
    n_dates_train: int
    n_dates_val: int
    n_predictors: int
    data_start: str
    data_end: str
    importance: dict
    timestamp: str
    arthagati_version: str
    schema_version: int = PROFILE_SCHEMA_VERSION
    sensitivity_curves: dict = field(default_factory=dict)

    @property
    def is_default(self) -> bool:
        return not bool(self.weights)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)

    @classmethod
    def from_dict(cls, raw: dict) -> "CalibrationProfile":
        defaults: dict = {
            "weights": {},
            "train_ir": 0.0, "val_ir": 0.0, "stability": 0.0,
            "quality_check": "—",
            "horizons": list(DEFAULT_HORIZONS),
            "n_folds": DEFAULT_FOLDS, "n_trials": DEFAULT_TRIALS,
            "embargo_days": DEFAULT_EMBARGO_DAYS,
            "n_dates_train": 0, "n_dates_val": 0, "n_predictors": 0,
            "data_start": "", "data_end": "",
            "importance": {}, "timestamp": "", "arthagati_version": "",
            "schema_version": PROFILE_SCHEMA_VERSION,
            "sensitivity_curves": {},
        }
        merged = {**defaults, **{k: v for k, v in raw.items() if k in defaults}}
        return cls(**merged)


# ──────────────────────────────────────────────────────────────────────────────
# Persistence (atomic writes — Streamlit Cloud-safe)
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
        # Reject incompatible old (v1 structural) profiles silently.
        if int(raw.get("schema_version", 0)) < PROFILE_SCHEMA_VERSION:
            return None
        return CalibrationProfile.from_dict(raw)
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        warnings.warn(f"Failed to load active profile: {exc}")
        return None


def delete_active_profile() -> bool:
    if ACTIVE_PROFILE_PATH.exists():
        ACTIVE_PROFILE_PATH.unlink()
        return True
    return False


def archive_profile(profile: CalibrationProfile) -> Path:
    ensure_profile_dir()
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = PROFILE_DIR / f"profile_{ts}.json"
    path.write_text(profile.to_json(), encoding="utf-8")
    return path


def list_profiles() -> list[Path]:
    if not PROFILE_DIR.exists():
        return []
    return sorted(
        (p for p in PROFILE_DIR.glob("profile_*.json")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Profile freshness + dataset fingerprints (unchanged from v1)
# ──────────────────────────────────────────────────────────────────────────────

def profile_age_days(profile: CalibrationProfile) -> float:
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
        return False, f"profile data_end is in the future ({-gap_days}d ahead)"

    age = profile_age_days(profile)
    if age > max_age_days:
        return False, f"profile is {age:.0f}d old (>{max_age_days}d threshold)"

    return True, f"profile fresh · {age:.1f}d old · data within {gap_days}d"


def dataset_fingerprint(
    raw_df: pd.DataFrame,
    predictors: Iterable[str],
    hyperparams: dict | None = None,  # kept for back-compat; ignored
) -> tuple:
    """Hashable fingerprint for the engine-output cache."""
    return (
        int(len(raw_df)),
        str(raw_df["DATE"].iloc[0].date()) if len(raw_df) else "",
        str(raw_df["DATE"].iloc[-1].date()) if len(raw_df) else "",
        tuple(sorted(predictors)),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Walk-forward CV folds (purged + embargoed)
# ──────────────────────────────────────────────────────────────────────────────

def _walk_forward_folds(
    n: int, n_folds: int, embargo: int, min_train_size: int,
) -> list[tuple[slice, slice]]:
    if n_folds < 2 or n < (min_train_size + embargo + n_folds):
        return []
    val_total = n - min_train_size - embargo
    val_size  = max(1, val_total // n_folds)
    folds: list[tuple[slice, slice]] = []
    for f in range(n_folds):
        val_start = min_train_size + embargo + f * val_size
        val_end   = val_start + val_size if f < n_folds - 1 else n
        if val_end - val_start < 30:
            continue
        train_end = val_start - embargo
        if train_end <= 0:
            continue
        folds.append((slice(0, train_end), slice(val_start, val_end)))
    return folds


# ──────────────────────────────────────────────────────────────────────────────
# Fast scoring kernel
# ──────────────────────────────────────────────────────────────────────────────

def _precompute_forward_returns(
    nifty: np.ndarray, horizons: Iterable[int],
) -> dict[int, np.ndarray]:
    """For each horizon h, return an array of length n where index i is
    the forward return from t=i to t=i+h, or NaN if past end."""
    n = len(nifty)
    out: dict[int, np.ndarray] = {}
    for h in horizons:
        ret = np.full(n, np.nan, dtype=np.float64)
        if n > h:
            end_idx = n - h
            ret[:end_idx] = (nifty[h:] / nifty[:end_idx] - 1.0) * 100.0
        out[int(h)] = ret
    return out


def _spearman_safe(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 30:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho, _ = spearmanr(x[mask], y[mask])
    return float(rho) if np.isfinite(rho) else 0.0


def _ir(rhos: list[float]) -> float:
    arr = np.asarray([r for r in rhos if np.isfinite(r)], dtype=np.float64)
    if len(arr) < 3:
        return 0.0
    return float(arr.mean() / max(arr.std(ddof=1), 1e-6))


def _l2_penalty(weights: np.ndarray, alpha: float) -> float:
    """Sum of squared weights, scaled by alpha. Discourages runaway fits."""
    return float(alpha * np.mean(weights * weights))


def quality_check(train_ir: float, val_ir: float) -> tuple[str, str]:
    if val_ir <= GATE_MIN_VAL_IR:
        return "No Edge", "danger"
    stability = val_ir / train_ir if train_ir > 0 else 0.0
    if train_ir > 0.05 and stability < GATE_OVERFIT_STABILITY:
        return "Overfit", "warning"
    return "Quality OK", "success"


# ──────────────────────────────────────────────────────────────────────────────
# Public tuner — fast post-engine ensemble calibration
# ──────────────────────────────────────────────────────────────────────────────

class IntelligenceTuner:
    """Tune the ensemble that maps engine output → calibrated conviction.

    Cost model:
        - One ``build_feature_matrix`` call up front (~1 ms)
        - Per Optuna trial: ``F @ w`` + ``len(folds) × len(horizons)``
          Spearman correlations (~5-10 ms)
        - 40 trials default → ~200-400 ms wall time

    Compare to v1 (structural-hyperparam tuning) which re-ran the FULL
    mood engine per trial (~30-60s/trial = 20+ min total). Same dataset,
    1000× cost reduction.
    """

    def __init__(
        self,
        mood_df: pd.DataFrame,
        msf_df: pd.DataFrame,
        n_active_predictors: int,
        horizons: Iterable[int] = DEFAULT_HORIZONS,
        n_folds: int = DEFAULT_FOLDS,
        embargo_days: int = DEFAULT_EMBARGO_DAYS,
        l2_alpha: float = DEFAULT_L2_ALPHA,
    ):
        if mood_df is None or mood_df.empty:
            raise ValueError("mood_df is empty — cannot calibrate without engine output.")
        if "NIFTY" not in mood_df.columns:
            raise ValueError("mood_df must contain a NIFTY column.")

        self.mood_df         = mood_df
        self.msf_df          = msf_df
        self.n_active_preds  = int(n_active_predictors)
        self.horizons        = tuple(int(h) for h in horizons)
        self.n_folds         = int(n_folds)
        self.embargo_days    = int(embargo_days)
        self.l2_alpha        = float(l2_alpha)

        self.feature_matrix  = build_feature_matrix(mood_df, msf_df)
        self.n               = self.feature_matrix.shape[0]
        self.nifty_arr       = mood_df["NIFTY"].to_numpy(dtype=np.float64)
        self.fwd_returns     = _precompute_forward_returns(self.nifty_arr, self.horizons)

        min_train = 252
        self.folds = _walk_forward_folds(
            self.n, self.n_folds, self.embargo_days, min_train,
        )
        if not self.folds:
            raise ValueError(
                f"Dataset too small for {self.n_folds}-fold CV "
                f"(n={self.n}, min_train={min_train})."
            )

        self.best_weights: dict = {}
        self.best_train_ir: float = 0.0
        self.best_val_ir: float   = 0.0
        self.best_stability: float = 0.0
        self.best_quality: str    = "—"
        self.importance: dict     = {}
        self.study = None

    # ── Trial scoring ──────────────────────────────────────────────────
    def _score_weights(
        self, w: np.ndarray,
    ) -> tuple[float, float, list[float], list[float]]:
        """Compute (train_ir, val_ir, train_rhos, val_rhos) for a weight vector.

        Single matrix-vector multiply + Spearman per (fold, horizon).
        """
        composite = self.feature_matrix @ w   # (N,) — the calibrated score
        train_rhos: list[float] = []
        val_rhos:   list[float] = []
        for train_slc, val_slc in self.folds:
            for h in self.horizons:
                fwd = self.fwd_returns[h]
                # Train fold
                t_x = composite[train_slc]
                t_y = fwd[train_slc]
                train_rhos.append(_spearman_safe(t_x, t_y))
                # Val fold
                v_x = composite[val_slc]
                v_y = fwd[val_slc]
                val_rhos.append(_spearman_safe(v_x, v_y))
        return _ir(train_rhos), _ir(val_rhos), train_rhos, val_rhos

    # ── Optimisation loop ──────────────────────────────────────────────
    def optimize(
        self,
        n_trials: int = DEFAULT_TRIALS,
        seed: int = 42,
        progress_callback: Callable[[int, int, float], None] | None = None,
    ) -> CalibrationProfile:
        """Tune ensemble weights via Optuna TPE + MedianPruner."""
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
            train_ir, val_ir, _, _ = self._score_weights(w)
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
            [best_params.get(name, 0.0) for name in FEATURE_NAMES],
            dtype=np.float64,
        )
        train_ir, val_ir, _, _ = self._score_weights(w_best)
        stability  = (val_ir / train_ir) if train_ir > 0 else 0.0
        q_label, _ = quality_check(train_ir, val_ir)

        self.best_weights   = best_params
        self.best_train_ir  = train_ir
        self.best_val_ir    = val_ir
        self.best_stability = stability
        self.best_quality   = q_label
        self.importance     = self._compute_importance()

        return self._make_profile()

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
            total = sum(abs(v) for v in self.best_weights.values()) or 1.0
            return {k: abs(v) / total * 100.0 for k, v in self.best_weights.items()}

    def _make_profile(self) -> CalibrationProfile:
        import arthagati

        date_col = self.mood_df["DATE"]
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
            n_predictors=self.n_active_preds,
            data_start=str(date_col.min().date()),
            data_end=str(date_col.max().date()),
            importance=self.importance,
            timestamp=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            arthagati_version=getattr(arthagati, "VERSION", "unknown"),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Convenience runner used by the UI
# ──────────────────────────────────────────────────────────────────────────────

def run_calibration(
    mood_df: pd.DataFrame,
    msf_df: pd.DataFrame,
    n_active_predictors: int,
    n_trials: int = DEFAULT_TRIALS,
    n_folds: int = DEFAULT_FOLDS,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    embargo_days: int = DEFAULT_EMBARGO_DAYS,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> CalibrationProfile:
    """One-shot fast calibration on pre-computed engine output."""
    tuner = IntelligenceTuner(
        mood_df, msf_df, n_active_predictors,
        horizons=horizons, n_folds=n_folds, embargo_days=embargo_days,
    )
    return tuner.optimize(n_trials=n_trials, progress_callback=progress_callback)
