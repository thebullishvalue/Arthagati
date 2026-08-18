"""
Arthagati — financial display formatting.

One place decides how a number reaches the screen. Before this module the
same quantity appeared as ``{:.2f}``, ``{:+.1f}`` and ``{:,.0f}`` in three
different views, and a missing value rendered as ``0.00``, ``nan`` or an
empty cell depending on which f-string it passed through.

Rules encoded here:

* signed quantities always carry their sign — ``+1.24``, ``-0.86``
* precision follows the quantity, not the caller
* missing is an em dash, never a blank and never a zero
* large counts abbreviate only above a threshold where the digits stop
  being readable, and never inside a column that must be compared
* percentages carry the unit; correlations and z-scores do not
"""

from __future__ import annotations

import math
from datetime import date, datetime

import pandas as pd

DASH = "—"


def _isnull(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def num(v, dp: int = 2, signed: bool = False) -> str:
    """A plain decimal. ``signed`` forces an explicit + on positives."""
    if _isnull(v):
        return DASH
    return f"{float(v):{'+' if signed else ''}.{dp}f}"


def pct(v, dp: int = 1, signed: bool = True) -> str:
    """A percentage that already arrives in percent units (12.4 → +12.4%)."""
    if _isnull(v):
        return DASH
    return f"{float(v):{'+' if signed else ''}.{dp}f}%"


def ratio_pct(v, dp: int = 1) -> str:
    """A 0–1 fraction rendered as a percentage (0.834 → 83.4%)."""
    if _isnull(v):
        return DASH
    return f"{float(v) * 100:.{dp}f}%"


def price(v, dp: int = 0) -> str:
    """An index level or price — grouped, fixed precision, never abbreviated.

    Levels are compared against each other down the column, so 21,732 stays
    21,732; it does not become 21.7K.
    """
    if _isnull(v):
        return DASH
    return f"{float(v):,.{dp}f}"


def compact(v, dp: int = 1) -> str:
    """Abbreviate a magnitude for a KPI slot where the column is one cell wide."""
    if _isnull(v):
        return DASH
    n = float(v)
    sign = "-" if n < 0 else ""
    n = abs(n)
    for cutoff, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if n >= cutoff:
            return f"{sign}{n / cutoff:.{dp}f}{suffix}"
    return f"{sign}{n:,.0f}"


def rho(v, dp: int = 3) -> str:
    """A correlation or Spearman rho — signed, three places, unitless."""
    return num(v, dp, signed=True)


def pvalue(v) -> str:
    """A p-value. Below the resolution of the permutation grid, say so."""
    if _isnull(v):
        return DASH
    p = float(v)
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def days(v) -> str:
    if _isnull(v):
        return DASH
    return f"{float(v):.0f}d"


def when(v, fmt: str = "%d %b %Y") -> str:
    """A date. Unambiguous by construction — day, abbreviated month, year."""
    if _isnull(v):
        return DASH
    if isinstance(v, (datetime, date, pd.Timestamp)):
        return v.strftime(fmt)
    return str(v)


def tone_of(v, invert: bool = False) -> str:
    """Semantic tone for a signed number: ``pos`` / ``neg`` / ``""``.

    ``invert`` is for quantities where up is bad — a spread that reads
    overbought, a drawdown, an error rate.
    """
    if _isnull(v):
        return ""
    n = float(v)
    if n == 0:
        return ""
    up = "neg" if invert else "pos"
    dn = "pos" if invert else "neg"
    return up if n > 0 else dn


def age_label(days_old: int) -> tuple[str, str]:
    """Data freshness → (label, tone). Stale data is labelled, never hidden."""
    if days_old <= 1:
        return "CURRENT", "pos"
    if days_old <= 4:
        return f"T-{days_old}", "info"
    if days_old <= 14:
        return f"STALE {days_old}D", "warn"
    return f"STALE {days_old}D", "neg"
