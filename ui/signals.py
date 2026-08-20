"""
Arthagati — display-state derivation.

Every view needs the same reading of the same numbers: is the mood bullish,
is the MSF overbought, is the WaveTrend stretched, where are the divergences.
Previously each view derived this inline, so the Overview strip, the chart
pane and the summary line could disagree about the same day.

Nothing here computes a signal. It only classifies values the engine has
already produced, so a change to a band moves every view at once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    MOOD_BAND_INNER, MOOD_BAND_OUTER,
    MSF_OB_LEVEL_1, MSF_OB_LEVEL_2, MSF_OS_LEVEL_1, MSF_OS_LEVEL_2,
)


# ── Mood ────────────────────────────────────────────────────────────────────
# Bands are fixed (VISION §6) so "Bullish" means the same thing every day.
# High score = cheap versus recent history = constructive, hence a positive
# tone. This is a valuation reading, not a momentum reading.

#: Regime label → (chart palette key, chip tone). One table, read by every
#: surface that shows a regime, so the colour and the word cannot disagree
#: between views. Only the SEMANTIC name is fixed here; the hex is resolved per
#: render through ui.theme.chart_color so it follows the active appearance.
REGIME_TONE: dict[str, tuple[str, str]] = {
    "Trending":       ("emerald", "success"),
    "Volatile Trend": ("amber",   "warning"),
    "Mean-Reverting": ("cyan",    "info"),
    "Choppy":         ("rose",    "danger"),
    "Unknown":        ("slate",   "neutral"),
}


def mood_state(score: float) -> tuple[str, str]:
    """→ (label, tone)."""
    if score >= MOOD_BAND_OUTER:
        return "VERY BULLISH", "pos"
    if score >= MOOD_BAND_INNER:
        return "BULLISH", "pos"
    if score <= -MOOD_BAND_OUTER:
        return "VERY BEARISH", "neg"
    if score <= -MOOD_BAND_INNER:
        return "BEARISH", "neg"
    return "NEUTRAL", "neutral"


# ── MSF Spread ──────────────────────────────────────────────────────────────
# The oscillator confirms or contradicts the mood reading. High = stretched
# up = a warning, not a positive; the tone is deliberately inverted relative
# to the mood score.

def msf_state(spread: float) -> tuple[str, str]:
    if spread >= MSF_OB_LEVEL_1:
        return "OVERBOUGHT", "neg"
    if spread >= MSF_OB_LEVEL_2:
        return "EXTENDED", "warn"
    if spread <= MSF_OS_LEVEL_1:
        return "OVERSOLD", "pos"
    if spread <= MSF_OS_LEVEL_2:
        return "WEAK", "info"
    return "RANGE", "neutral"


def wt_state(wt1: float, wt2: float, ob_level: float) -> tuple[str, str]:
    """WaveTrend position and slope, as one label."""
    if not np.isfinite(wt1) or not np.isfinite(wt2):
        return "NO DATA", "neutral"
    if wt1 >= ob_level:
        return "OVERBOUGHT", "neg"
    if wt1 <= -ob_level:
        return "OVERSOLD", "pos"
    return ("RISING", "pos") if wt1 > wt2 else ("FALLING", "neg")


def hurst_state(h: float) -> tuple[str, str]:
    if not np.isfinite(h):
        return "—", "neutral"
    if h > 0.55:
        return "TRENDING", "pos"
    if h < 0.45:
        return "REVERTING", "info"
    return "RANDOM", "neutral"


def entropy_state(s: float) -> tuple[str, str]:
    if not np.isfinite(s):
        return "—", "neutral"
    if s > 0.60:
        return "DISORDERED", "neg"
    if s < 0.40:
        return "ORDERED", "pos"
    return "MIXED", "neutral"


# ── Divergence / crossover detection (display markers only) ─────────────────

def msf_divergences(mood: pd.Series, msf: pd.Series, lookback: int = 10):
    """Indices of bullish / bearish mood-vs-MSF divergences.

    A bearish divergence is a new mood low that the oscillator does not
    confirm, and vice versa. Lifted verbatim from the chart view so the
    Overview's signal count and the chart's triangles come from one function.
    """
    rmm_min = mood.rolling(lookback + 1, min_periods=1).min()
    rmm_max = mood.rolling(lookback + 1, min_periods=1).max()
    rms_min = msf.rolling(lookback + 1, min_periods=1).min()
    rms_max = msf.rolling(lookback + 1, min_periods=1).max()
    bear = (mood == rmm_min) & (mood < rmm_min.shift(lookback)) & (rms_min > rms_min.shift(lookback))
    bull = (mood == rmm_max) & (mood > rmm_max.shift(lookback)) & (rms_max < rms_max.shift(lookback))
    valid = np.zeros(len(mood), dtype=bool)
    valid[lookback * 2: len(mood) - 1] = True
    return np.where(bull.to_numpy() & valid)[0], np.where(bear.to_numpy() & valid)[0]


def wt_crossovers(wt1: np.ndarray, wt2: np.ndarray, warmup: int = 32):
    """Indices of bullish / bearish WaveTrend crossovers."""
    p1 = np.concatenate([[np.nan], wt1[:-1]])
    p2 = np.concatenate([[np.nan], wt2[:-1]])
    ok = np.isfinite(wt1) & np.isfinite(wt2) & np.isfinite(p1) & np.isfinite(p2)
    bear = ok & (wt1 > wt2) & (p1 <= p2) & (wt1 < 0)
    bull = ok & (wt2 > wt1) & (p2 <= p1) & (wt1 > 0)
    if len(bull) > warmup:
        bull[:warmup] = False
        bear[:warmup] = False
    return np.where(bull)[0], np.where(bear)[0]


def window(df: pd.DataFrame, timeframes: dict, tf: str, date_col: str = "DATE"):
    """Boolean mask for a named timeframe, selected by DATE not row count.

    Row-count windows stretch by ~1.4x on a trading-day series: "1Y" as
    ``.tail(365)`` spans 510 calendar days.
    """
    last = df[date_col].max()
    if tf == "YTD":
        cutoff = pd.Timestamp(year=last.year, month=1, day=1)
    elif timeframes.get(tf):
        cutoff = last - pd.Timedelta(days=timeframes[tf])
    else:
        return pd.Series(True, index=df.index)
    return df[date_col] >= cutoff
