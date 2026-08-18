"""
Arthagati — the panel primitive.

One surface type with optional slots, replacing the nine card grammars the
app had grown (``position-card`` and its four variants, ``metric-card``,
``system-card``, ``interp-card``, ``info-box``, ``warning-box``,
``profile-stat``, ``progress-card``). Each was locally reasonable; together
they gave the app nine different paddings, borders and header anatomies for
what is conceptually one thing.

Anatomy, all slots optional except the body::

    .fm-panel
      .fm-panel-head    icon · title · subtitle · chip · actions   (44px fixed)
      .fm-panel-body    chart | table | metrics | prose
      .fm-panel-foot    source · last updated

The fixed header height is what aligns panels sitting side by side. Empty,
loading and error are rendered INTO the body rather than replacing the panel,
so a panel with nothing to show keeps its height and its grid position and the
page does not reflow around the gap.
"""

from __future__ import annotations

import html as _h
from contextlib import contextmanager

import streamlit as st

from ui.components import get_icon

# Chip tone → CSS modifier. Kept small on purpose: a chip carries state, and
# there are only so many states worth distinguishing at a glance.
_TONES = {"neutral", "success", "danger", "warning", "info"}


def chip(label: str, tone: str = "neutral") -> str:
    """Status chip markup. Returns a string so it can be embedded in a header."""
    tone = tone if tone in _TONES else "neutral"
    return f'<span class="fm-chip fm-chip--{tone}">{_h.escape(str(label))}</span>'


def panel_head_html(
    title: str,
    subtitle: str = "",
    icon: str = "",
    chip_spec: tuple[str, str] | None = None,
    actions: str = "",
) -> str:
    ic = (
        f'<span class="fm-panel-icon">{get_icon(icon, 16, 1.5)}</span>'
        if icon else ""
    )
    sub = (
        f'<div class="fm-panel-sub">{_h.escape(subtitle)}</div>'
        if subtitle else ""
    )
    cp = chip(*chip_spec) if chip_spec else ""
    return (
        '<div class="fm-panel-head">'
        f'<div class="fm-panel-id">{ic}'
        f'<div><div class="fm-panel-title">{_h.escape(title)}</div>{sub}</div>'
        "</div>"
        f'<div class="fm-panel-act">{cp}{actions}</div>'
        "</div>"
    )


@contextmanager
def panel(
    title: str,
    subtitle: str = "",
    icon: str = "",
    chip_spec: tuple[str, str] | None = None,
    footer: str = "",
    modifier: str = "",
):
    """Open a panel; caller renders the body inside the ``with`` block.

    Streamlit emits each ``st.markdown`` as its own DOM node, so the opening
    and closing markup are separate calls. That works because the wrapper divs
    are unclosed until the exit call — Streamlit's own containers nest inside
    them in document order.
    """
    st.markdown(
        f'<div class="fm-panel {modifier}">'
        + panel_head_html(title, subtitle, icon, chip_spec)
        + '<div class="fm-panel-body">',
        unsafe_allow_html=True,
    )
    try:
        yield
    finally:
        foot = (
            f'<div class="fm-panel-foot">{_h.escape(footer)}</div>'
            if footer else ""
        )
        st.markdown(f"</div>{foot}</div>", unsafe_allow_html=True)


def empty_state(line: str, hint: str = "", icon: str = "search") -> None:
    """Nothing to show — but the panel keeps its shape."""
    st.markdown(
        f'<div class="fm-state">{get_icon(icon, 22, 1.5)}'
        f'<div class="fm-state-line">{_h.escape(line)}</div>'
        + (f'<div class="fm-state-hint">{_h.escape(hint)}</div>' if hint else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def skeleton(kind: str = "chart", rows: int = 5) -> None:
    """Loading placeholder shaped like the content it stands in for."""
    body = (
        '<div class="fm-skel-chart"></div>'
        if kind == "chart"
        else "".join('<div class="fm-skel-line"></div>' for _ in range(rows))
    )
    st.markdown(f'<div class="fm-skeleton">{body}</div>', unsafe_allow_html=True)


# ── Alerts ──────────────────────────────────────────────────────────────────
# Replaces st.warning / st.info / st.error, which had zero CSS rules against
# them and rendered in stock Streamlit chrome at 10 call sites.

_ALERT_ICON = {
    "warning": "alert-triangle",
    "danger":  "alert-triangle",
    "info":    "circle",
    "success": "check-circle",
}


def alert(tone: str, title: str, body: str = "", hint: str = "") -> None:
    """A system alert. Same chrome as every other surface.

    Error copy states the fault and the remedy — no apology, no stack trace.
    """
    tone = tone if tone in _TONES else "info"
    icon = _ALERT_ICON.get(tone, "circle")
    body_html = f'<div class="fm-alert-body">{_h.escape(body)}</div>' if body else ""
    hint_html = f'<div class="fm-alert-hint">{_h.escape(hint)}</div>' if hint else ""
    st.markdown(
        f'<div class="fm-alert fm-alert--{tone}">'
        f'<span class="fm-alert-icon">{get_icon(icon, 16, 1.5)}</span>'
        f'<div class="fm-alert-text">'
        f'<div class="fm-alert-title">{_h.escape(title)}</div>'
        f"{body_html}{hint_html}</div></div>",
        unsafe_allow_html=True,
    )


def kpi_grid(items: list[tuple]) -> None:
    """Equal-height KPI row that reflows continuously.

    ``items``: ``(label, value, sub, tone, icon)``.

    Rendered as one CSS grid rather than ``st.columns``. Streamlit's columns
    emit a fixed flex row that stacks straight from N-up to 1-up with no
    intermediate step, and each column sizes to its own content so cards in a
    row end up unequal heights.
    """
    cards = []
    for label, value, sub, tone, icon in items:
        tone = tone if tone in _TONES else "neutral"
        ic = get_icon(icon, 12, 1.5) if icon else ""
        cards.append(
            f'<div class="fm-kpi fm-kpi--{tone}">'
            f'<div class="fm-kpi-label">{ic}{_h.escape(label)}</div>'
            f'<div class="fm-kpi-value">{_h.escape(str(value))}</div>'
            f'<div class="fm-kpi-sub">{_h.escape(str(sub))}</div>'
            "</div>"
        )
    st.markdown(
        f'<div class="fm-kpi-grid">{"".join(cards)}</div>', unsafe_allow_html=True
    )
