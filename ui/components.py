"""
Arthagati — reusable UI components: panels, metric cards, chips, tables, the tape,
and the conviction chain.
अर्थगति (Arthagati) — "Movement of meaning / market sentiment"

UI — "Graphite" institutional terminal design language (see ui/theme.py).

Every function here emits markup for classes defined in ui/theme.css and
nothing else: no inline colours, no inline type. If a component needs a new
look, the rule belongs in the stylesheet, so the light theme keeps working
as a token swap rather than needing a parallel set of Python branches.
"""

from __future__ import annotations

import datetime as _dt
import html as html_mod
from contextlib import contextmanager as _contextmanager

import pandas as pd
import numpy as np
import streamlit as st
from streamlit.components.v1 import html as _components_html


# ── SVG Icons (inline, no external deps) — with ARIA labels for accessibility

ICONS = {
    "chart":      '<svg aria-label="Chart icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>',
    "cube":       '<svg aria-label="Cube icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/></svg>',
    "target":     '<svg aria-label="Target icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
    "layers":     '<svg aria-label="Layers icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>',
    "bar-chart":  '<svg aria-label="Bar chart icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>',
    "activity":   '<svg aria-label="Activity icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>',
    "crosshair":  '<svg aria-label="Crosshair icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><line x1="22" y1="12" x2="18" y2="12"/><line x1="6" y1="12" x2="2" y2="12"/><line x1="12" y1="6" x2="12" y2="2"/><line x1="12" y1="22" x2="12" y2="18"/></svg>',
    "cpu":        '<svg aria-label="CPU icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>',
    "zap":        '<svg aria-label="Zap icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>',
    "shield":     '<svg aria-label="Shield icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>',
    "grid":       '<svg aria-label="Grid icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>',
    "database":   '<svg aria-label="Database icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>',
    "trending":   '<svg aria-label="Trending icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>',
    "eye":        '<svg aria-label="Eye icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>',
    "play":       '<svg aria-label="Play icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8"/></svg>',
    "chevron-right": '<svg aria-label="Expand icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"/></svg>',
    "sun":        '<svg aria-label="Light mode icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>',
    "moon":       '<svg aria-label="Dark mode icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>',
    "download":   '<svg aria-label="Download icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>',
    "briefcase":  '<svg aria-label="Portfolio icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="2" y="7" width="20" height="14" rx="2" ry="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/></svg>',
    "compass":    '<svg aria-label="Regime icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/></svg>',
    "rocket":     '<svg aria-label="Strong Bull icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z"/><path d="m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-3 11a22.35 22.35 0 0 1-4 2z"/><path d="M9 12H4s.55-3.03 2-4.5c1.62-1.63 5-2.5 5-2.5"/><path d="M12 15v5s3.03-.55 4.5-2c1.63-1.62 2.5-5 2.5-5"/></svg>',
    "trending-up": '<svg aria-label="Bull icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/></svg>',
    "trending-down": '<svg aria-label="Bear icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 17 13.5 8.5 8.5 13.5 2 7"/><polyline points="16 17 22 17 22 11"/></svg>',
    "arrow-up-right": '<svg aria-label="Weak Bull icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><line x1="7" y1="17" x2="17" y2="7"/><polyline points="7 7 17 7 17 17"/></svg>',
    "arrow-down-right": '<svg aria-label="Weak Bear icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><line x1="7" y1="7" x2="17" y2="17"/><polyline points="17 7 17 17 7 17"/></svg>',
    "arrow-up":   '<svg aria-label="Up" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="19" x2="12" y2="5"/><polyline points="5 12 12 5 19 12"/></svg>',
    "arrow-down": '<svg aria-label="Down" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"/><polyline points="19 12 12 19 5 12"/></svg>',
    "move-horizontal": '<svg aria-label="Chop icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="18 8 22 12 18 16"/><polyline points="6 8 2 12 6 16"/><line x1="2" y1="12" x2="22" y2="12"/></svg>',
    "alert-triangle": '<svg aria-label="Crisis icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    "help-circle": '<svg aria-label="Unknown icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    "circle":     '<svg aria-label="Circle" role="img" viewBox="0 0 24 24" fill="currentColor" stroke="none"><circle cx="12" cy="12" r="10"/></svg>',
    "check-circle": '<svg aria-label="Check" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
    "scale":      '<svg aria-label="Weighting icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="M7 21h10"/><path d="M12 3v18"/><path d="M3 7h18"/></svg>',
}


#: The app's single icon drawing style. Every icon is normalised to these by
#: ``get_icon`` regardless of what its own SVG literal declares — the set was
#: pasted in over time and carries three different stroke weights and a mix of
#: butt/round terminals, which is exactly how an icon set stops reading as a
#: set. One weight, round terminals, no fills.
ICON_STROKE = 1.6
_ICON_LINECAP = "round"


def get_icon(name: str, size: int = 18, stroke_width: float | None = None) -> str:
    """Return an SVG icon normalised to the app's one icon style.

    ``stroke_width`` is accepted for call sites that predate ``ICON_STROKE``
    but is deliberately clamped: the two callers that passed 1.8 and 2 were
    drawing the same icons as everything else, one and two notches heavier,
    inside components that sat side by side.
    """
    import re
    base_svg = ICONS.get(name, ICONS["chart"])

    # Strip whatever the literal declared, so the result cannot inherit a
    # per-icon weight or terminal style.
    for attr in ("width", "height", "stroke-width", "stroke-linecap", "stroke-linejoin"):
        base_svg = re.sub(rf'\s+{attr}="[^"]*"', "", base_svg)

    sw = ICON_STROKE if stroke_width is None else min(float(stroke_width), 1.75)
    return base_svg.replace(
        "<svg",
        f'<svg width="{size}" height="{size}" stroke-width="{sw}" '
        f'stroke-linecap="{_ICON_LINECAP}" stroke-linejoin="{_ICON_LINECAP}"',
    )


def render_section_header(
    title: str,
    description: str = "",
    icon: str = "chart",
    accent: str = "",
) -> None:
    """Render a styled section header with icon, title, and optional description.

    Args:
        title: Section title (rendered uppercase).
        description: Optional one-line description below title.
        icon: Key from ICONS dict.
        accent: CSS color class — "", "cyan", "emerald", "violet", "rose".
    """
    svg = get_icon(icon, size=16)
    icon_class = f"icon {accent}" if accent else "icon"
    hdr_class = f"section-hdr {accent}" if accent else "section-hdr"
    # `.desc` is a DIRECT child of the header, not nested inside `.text`. The
    # header is a two-row grid — icon and title on row 1, description on row 2
    # under the title — and a nested description is not a grid item, so it
    # could not be placed and fell back to flowing under the title with its
    # own margin. That is the gap that made the subtitle read as a detached
    # paragraph.
    desc_html = f'<div class="desc">{html_mod.escape(description)}</div>' if description else ""
    st.markdown(
        f'<div class="{hdr_class}">'
        f'<div class="{icon_class}">{svg}</div>'
        f'<div class="text"><h3>{html_mod.escape(title)}</h3></div>'
        f'{desc_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_sub_header(title: str) -> None:
    """A labelled division INSIDE a section — one tier below a section header.

    Mono micro-label, no icon, no rule. Used where a section has two named
    parts (Diagnostics' "Importance Over Time" above its history table), which
    the tab files had been hand-rolling as inline-styled divs at three
    different sizes.
    """
    st.markdown(f'<div class="sub-head">{html_mod.escape(title)}</div>',
                unsafe_allow_html=True)


def render_control_hint(text: str) -> None:
    """Render the canonical terse helper caption beneath a control.

    This is the single source of truth for the "sub-control hint" tier — the
    uppercase micro-caption used beneath a control or a chart. Use it instead of ``st.caption``
    for control helper text so the sidebar/tab fine-print stays one coherent
    visual hierarchy. Keep the text terse and ``·``-separated.
    """
    st.markdown(
        f'<div class="control-hint">{html_mod.escape(text)}</div>',
        unsafe_allow_html=True,
    )


def render_note(text: str) -> None:
    """The one caption tier — a note under a chart, table or control.

    Replaces bare ``st.caption`` everywhere. Streamlit's caption renders in
    its own sans face at its own size with its own margin, so eight of them
    scattered across four tab files read as eight different kinds of aside.
    This is the same object as ``render_control_hint`` (identical styling on
    purpose) named for its other use, so a reader of the tab code does not
    have to know that "control hint" also means "chart footnote".
    """
    st.markdown(
        f'<div class="control-hint">{text}</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════
#  PANEL SYSTEM — one anatomy for every framed thing in the app
# ═══════════════════════════════════════════════════════════════════════
#
# A panel is: header (title / context, meta and chip right) · body · footer.
# Charts, tables and embedded iframes all use it, so a screen mixing them
# reads as one grid instead of as several products sharing a page.
#
# It is a real ``st.container`` rather than an HTML string because the body
# holds WIDGETS — a Plotly figure, a components.v1 iframe — which no amount
# of markdown can wrap. The container carries `key="panel-<id>"`, and
# theme.css styles `[class*="st-key-panel-"]`.

def render_panel_header(
    title: str = "",
    *,
    context: str = "",
    meta: str = "",
    chip: "tuple[str, str] | None" = None,
) -> None:
    """Render a panel header.

    ``title`` — what the panel shows. Omit it when the section header
    directly above already names the panel; a panel header that restates the
    section header is a second title, not a header.
    ``context`` — the panel's own metadata (instrument, window, units).
    ``meta``/``chip`` — right-aligned status: as-of, source, freshness.
    """
    if not (title or context or meta or chip):
        return
    left = ""
    if title:
        left += f'<span class="ph-title">{html_mod.escape(title)}</span>'
    if context:
        left += f'<span class="ph-context">{html_mod.escape(context)}</span>'
    right = ""
    if meta:
        right += f'<span>{html_mod.escape(meta)}</span>'
    if chip:
        right += render_chip(chip[0], chip[1], as_html=True) or ""
    st.markdown(
        f'<div class="panel-hdr"><div class="ph-left">{left}</div>'
        f'<div class="ph-right">{right}</div></div>',
        unsafe_allow_html=True,
    )


@_contextmanager
def panel(
    key: str,
    title: str = "",
    *,
    context: str = "",
    meta: str = "",
    chip: "tuple[str, str] | None" = None,
    footer: str = "",
    window: bool = False,
):
    """Context manager wrapping any content in the shared panel chrome.

    ``with panel("fvo-fairvalue", context="GOLD · 6M"): st.plotly_chart(...)``

    Use it directly for anything that is neither a chart nor a table (an
    embedded widget, a bespoke layout) so that thing still belongs to the
    system rather than sitting on the page unframed.
    """
    with st.container(key=f"panel-{key}"):
        if window:
            # A widget cannot live inside the header's HTML string, so the
            # header and the control are emitted as two siblings of one
            # container, and `.st-key-panelrow-*` turns that container's
            # vertical block into a centred row. Columns were the obvious
            # choice here and the wrong one: `stColumn` computes to zero
            # height, so the header hung 8px off the control's centre line and
            # no amount of override CSS pulled it back. Siblings in a single
            # flex row centre against each other by construction.
            with st.container(key=f"panelrow-{key}"):
                render_panel_header(title, context=context, meta=meta, chip=chip)
                render_window_control(key)
        else:
            render_panel_header(title, context=context, meta=meta, chip=chip)
        yield
        if footer:
            st.markdown(f'<div class="panel-foot">{footer}</div>', unsafe_allow_html=True)


def default_chart_context(units: str = "") -> str:
    """The context line every chart panel gets for free: instrument · window.

    Read from session state rather than threaded through eighteen call sites,
    which is both less plumbing and strictly more correct — a context built
    from the same keys the command bar reads cannot disagree with it.
    """
    parts = [
        str(st.session_state.get("active_instrument", "NIFTY 50") or "").upper(),
        str(st.session_state.get("tf_selected", "") or ""),
    ]
    if units:
        parts.append(units)
    return " · ".join(p for p in parts if p)


def render_chart_panel(
    fig,
    key: str,
    title: str = "",
    *,
    units: str = "",
    context: str | None = None,
    meta: str = "",
    chip: "tuple[str, str] | None" = None,
    footer: str = "",
    window: bool = False,
) -> None:
    """Render a Plotly figure inside the shared panel chrome.

    The ONE way a chart reaches the screen in this app. Every call passes the
    same ``PLOTLY_CONFIG``, which is what removes the stock Plotly toolbar and
    its logo — previously eighteen of the app's charts shipped Plotly's own
    chrome, complete with a link out to plotly.com.

    ``title`` is normally EMPTY. Every chart in Arthagati already sits under a
    ``render_section_header`` that names it and explains how to read it; a
    panel title would be that title again, four pixels lower. What the section
    header cannot say is which instrument and window the plot is actually
    drawn on, so that is what the panel header carries — supplied
    automatically from session state, with ``units`` appended.
    """
    from ui.theme import PLOTLY_CONFIG   # local: avoids a circular import
    ctx = default_chart_context(units) if context is None else context
    with panel(key, title, context=ctx, meta=meta, chip=chip, footer=footer,
               window=window):
        st.plotly_chart(fig, width="stretch", key=f"chart-{key}", config=PLOTLY_CONFIG)


def render_window_control(key: str = "window") -> None:   # noqa: ARG001
    """The chart-window selector, rendered inside a panel header.

    It used to sit in a toolbar strip docked under the command bar — page
    chrome, physically distant from the thing it changes. A control that
    reframes a chart belongs ON that chart, so it now renders in the panel
    header, right-aligned opposite the panel's context line.

    All charts on a page share one window, so exactly one panel per page
    carries the control (``render_chart_panel(..., window=True)``). It writes
    the shared ``tf_selected`` key that the page's series filtering reads.
    """
    from config import TIMEFRAMES
    # 1W is omitted: at ~5 observations a Kalman band and a 90-day OU
    # projection are noise, and the engine's warm-up makes the window
    # meaningless. It stays a valid key in config for any caller that wants it.
    options = [tf for tf in TIMEFRAMES if tf != "1W"]
    if st.session_state.get("tf_selected") not in options:
        st.session_state["tf_selected"] = "1Y"
    st.segmented_control(
        "Window", options, key="tf_selected", label_visibility="collapsed",
        help="Chart window. Applies to every plot on this page; the engines "
             "always fit on the full history.",
    )


def render_loading_skeleton(
    key: str,
    *,
    rows: int = 1,
    height: int = 220,
    title: str = "",
    context: str = "",
) -> None:
    """A panel-shaped placeholder at the size of the thing that is coming.

    Sized to the final content so the layout does not jump when the real
    panel replaces it. ``rows=1, height=N`` is the chart case (one block);
    ``rows=N`` is the table case (a header rule plus N row bars).
    """
    with panel(key, title, context=context):
        if rows <= 1:
            body = f'<div class="skeleton" style="height:{height}px"></div>'
        else:
            body = ('<div class="skeleton sk-head"></div>'
                    + "".join('<div class="skeleton sk-row"></div>' for _ in range(rows)))
        st.markdown(f'<div class="skeleton-body">{body}</div>', unsafe_allow_html=True)


def render_table_panel(
    df,
    key: str,
    title: str = "",
    *,
    units: str = "",
    context: str | None = None,
    meta: str = "",
    chip: "tuple[str, str] | None" = None,
    footer: str = "",
    window: bool = False,
    **table_kwargs,
) -> None:
    """Render a DataFrame inside the shared panel chrome.

    Same header anatomy — and the same ``units``/``context`` contract — as
    ``render_chart_panel``, deliberately, so a table and a chart sitting side
    by side are visibly the same kind of object.

    ``units`` MUST be declared here rather than left to ``**table_kwargs``:
    without it, every ``units=`` at a call site fell through to
    ``render_data_table``, which has no such parameter, and the page died with
    ``render_data_table() got an unexpected keyword argument 'units'``.
    Remaining kwargs still pass through to the table renderer.
    """
    ctx = default_chart_context(units) if context is None else context
    with panel(key, title, context=ctx, meta=meta, chip=chip, footer=footer,
               window=window):
        render_data_table(df, **table_kwargs)


def render_chart_error(key: str, title: str, reason: str) -> None:
    """A failed chart keeps its panel and explains itself inside it.

    A chart that cannot draw must not vanish — a missing panel reads as "there
    was nothing to show", which is a different claim from "this could not be
    computed" and the wrong one to make silently.
    """
    with panel(key, title, chip=("UNAVAILABLE", "warning")):
        st.markdown(
            f'<div class="panel-state">{html_mod.escape(reason)}</div>',
            unsafe_allow_html=True,
        )


def render_chip(label: str, tone: str = "neutral", *, as_html: bool = False) -> str | None:
    """Render (or return) a status chip — one badge system for the whole app.

    ``tone``: ``accent`` / ``success`` / ``danger`` / ``warning`` / ``info`` /
    ``neutral``. Used for analog-tier badges, session/freshness state, and
    any other "state in a word" reading (previously several one-off HTML
    blocks with their own colour logic). Pass ``as_html=True`` to get the raw
    ``<span>`` back for composition inside a larger ``st.markdown`` call
    (e.g. ``render_top_bar``) instead of rendering it standalone.
    """
    html = f'<span class="chip chip-{html_mod.escape(tone)}">{html_mod.escape(label)}</span>'
    if as_html:
        return html
    st.markdown(html, unsafe_allow_html=True)
    return None


def render_metric_card(
    label: str,
    value: str,
    subtext: str = "",
    color_class: str = "neutral",
    tooltip: str = "",
    icon: str = "",
) -> None:
    """Render a terminal-styled metric card with optional tooltip.

    Args:
        label: Card label (rendered uppercase).
        value: Primary metric value.
        subtext: Optional secondary description below value.
        color_class: Semantic color — "neutral", "success", "danger", "warning", "info", "violet".
        tooltip: Optional hover explanation text.
        icon: Optional ICONS key — small icon inlined before the label.
    """
    tooltip_html = ""
    if tooltip:
        tooltip_html = (
            f'<div class="metric-tooltip" data-tooltip="{html_mod.escape(tooltip)}">'
            f'<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">'
            f'<circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/>'
            f'<line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
            f'<span class="metric-tooltip-text">{html_mod.escape(tooltip)}</span>'
            f'</div>'
        )

    sub_metric_html = f'<div class="sub-metric">{html_mod.escape(subtext)}</div>' if subtext else ""
    icon_html = f'<span class="card-icon">{get_icon(icon, size=12)}</span> ' if icon else ""
    st.markdown(
        f'<div class="metric-card {html_mod.escape(color_class)}">'
        f'<span class="label">{icon_html}{html_mod.escape(label)}</span>'
        f"<h2>{html_mod.escape(value)}</h2>"
        f"{sub_metric_html}"
        f"{tooltip_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_kpi_strip(items: list[dict], *, max_cols: int = 5, key: str = "kpi-strip") -> None:
    """Lay out ``render_metric_card`` items in rows of at most ``max_cols``.

    Each item is the keyword set ``render_metric_card`` takes: ``label``,
    ``value``, and optionally ``subtext``/``color_class``/``tooltip``/
    ``icon``. Used for the Overview page's KPI row — caps row width so cards
    never squeeze past legibility (wraps to a second row instead of the
    6-wide layouts elsewhere in the app that get tight on narrow viewports).
    """
    if not items:
        return
    with st.container(key=key):
        for i in range(0, len(items), max_cols):
            row = items[i:i + max_cols]
            cols = st.columns(len(row), gap="small")
            for c, item in zip(cols, row):
                with c:
                    render_metric_card(
                        label=item.get("label", ""),
                        value=item.get("value", ""),
                        subtext=item.get("subtext", ""),
                        color_class=item.get("color_class", "neutral"),
                        tooltip=item.get("tooltip", ""),
                        icon=item.get("icon", ""),
                    )


def render_header(title: str, tagline: str) -> None:
    """Render the cold-start masthead.

    Stacked, not inline: the mark reads at display size on its own line and
    the tagline sits under it as a rule-delimited subtitle. Inline (the
    previous arrangement) the two competed for the same optical line and the
    mark ended up the same size as a section heading — which is what a
    masthead must not be, since it is the only thing on a cold-start screen
    that says what the application is.
    """
    head, tail = (title[:5], title[5:]) if len(title) > 5 else (title, "")
    st.markdown(
        f'<div class="premium-header">'
        f'<div class="title">{html_mod.escape(head)}'
        f'<span class="accent-ink">{html_mod.escape(tail)}</span></div>'
        f'<div class="tagline">{html_mod.escape(tagline)}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_nav_brand(title: str = "ARTHAGATI", tagline: str = "अर्थगति · Market Sentiment") -> None:
    """Render the control rail's brand block.

    The mark is split so the second half carries the accent — a product mark
    that is one flat colour reads as a heading, not as a mark. Left-aligned
    (not centred) because everything below it in the rail is left-aligned,
    and a centred mark over a left-aligned column is the single most common
    tell of a template.
    """
    head, tail = (title[:5], title[5:]) if len(title) > 5 else (title, "")
    st.markdown(
        f'<div class="nav-brand">'
        f'<div class="mark">{html_mod.escape(head)}'
        f'<span class="accent-ink">{html_mod.escape(tail)}</span></div>'
        f'<div class="tagline">{html_mod.escape(tagline)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_top_bar(
    *,
    target: str = "",
    price: float | None = None,
    change_pct: float | None = None,
    status_label: str = "",
    status_tone: str = "neutral",
    meta: str = "",
    meta_items: "list[tuple[str, str]] | None" = None,
    open_strip: bool = False,
) -> None:
    """Render the command bar — the first element on every page.

    Reading order left to right is *identity → value → trust*: which
    instrument, what it is worth, and whether the data behind that is
    current. Nothing renders above this bar; data-quality notices that used
    to sit on top of the page now hang below it in the notice rail, so the
    instrument is always the first thing on screen.

    ``target``/``price``/``change_pct`` describe the active instrument (omit
    price/change when unresolved, e.g. before the target column has loaded).
    ``status_label``/``status_tone`` render as one chip summarising freshness
    at a glance — the full explanation lives in the notice rail below.
    ``meta_items`` are key/value pairs shown right-aligned (as-of date,
    horizon, spine size); ``meta`` is the legacy single-caption form and is
    folded into them. ``open_strip=True`` leaves the bottom corners square
    for a ``toolbar_strip`` to dock flush underneath.
    """
    instrument_html = ""
    if target:
        instrument_html = (
            f'<div class="cb-instrument">'
            f'<span class="eyebrow">Instrument</span>'
            f'<span class="sym">{html_mod.escape(target)}</span>'
            f'</div>'
        )
    quote_html = ""
    if price is not None:
        # `change_pct` is in PERCENT POINTS (-0.42 == -0.42%), the same unit it
        # is printed in. It used to arrive as a fraction and be formatted with
        # "%.2f%%", so every sub-1% session — i.e. most of them — printed as
        # "0.00%". Flat band is a half basis point.
        chg = change_pct if change_pct is not None else 0.0
        chg_cls = "up" if chg > 0.005 else "down" if chg < -0.005 else "flat"
        arrow = "▲" if chg_cls == "up" else "▼" if chg_cls == "down" else "▬"
        # Arrow carries the sign, so the number is unsigned — "▼ 0.42%", not
        # "▼ -0.42%". Direction is stated twice (glyph + colour) and never by
        # colour alone, for red/green deficiency.
        chg_html = (
            f'<span class="chg {chg_cls}">{arrow} {abs(chg):.2f}%</span>'
            if change_pct is not None else ""
        )
        quote_html = (
            f'<div class="cb-quote"><span class="px">{price:,.2f}</span>{chg_html}</div>'
        )

    items = list(meta_items or [])
    if meta:
        items.append(("As of", meta.replace("As of ", "")))
    meta_html = "".join(
        f'<div class="cb-meta"><span class="k">{html_mod.escape(str(k))}</span>'
        f'<span class="v">{html_mod.escape(str(v))}</span></div>'
        for k, v in items if v
    )
    chip_html = render_chip(status_label, status_tone, as_html=True) if status_label else ""
    open_cls = " open" if open_strip else ""
    st.markdown(
        f'<div class="command-bar{open_cls}">'
        f'<div class="cb-left">'
        f'<div class="cb-brand"><span class="mark">ARTHA<span class="accent-ink">GATI</span></span>'
        f'<span class="sub">अर्थगति</span></div>'
        f'{instrument_html}{quote_html}'
        f'</div>'
        f'<div class="cb-right">{meta_html}{chip_html}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_notice_rail(notices: "list[dict] | None") -> None:
    """Render queued data-quality notices as a compact rail under the chrome.

    Each notice is ``{"kind": "warning"|"info", "title": str, "body": str}``.
    These used to render as full-width boxes at the very top of the page —
    three of them (stale source, partial session, carried-forward
    predictors) pushed the instrument itself below the fold, so on exactly
    the days the data most needed scrutiny the interface led with an apology
    instead of a price. One row each, severity on the left rule, title and
    explanation on one line: same information, a third of the vertical cost,
    and now BELOW the thing it qualifies rather than above it.
    """
    if not notices:
        return
    rows = "".join(
        f'<div class="notice {html_mod.escape(n.get("kind", "info"))}">'
        f'<div class="n-title">{html_mod.escape(n.get("title", ""))}</div>'
        f'<div class="n-body">{n.get("body", "")}</div>'
        f'</div>'
        for n in notices
    )
    st.markdown(f'<div class="notice-rail">{rows}</div>', unsafe_allow_html=True)


def render_rail_readout(rows: "list[tuple[str, str, str]]") -> None:
    """Render the rail's session readout — ``(label, value, tone)`` rows.

    ``tone`` is "" / "accent" / "long" / "short" / "caution". Values are
    right-aligned and tabular so a changed number is seen rather than read.
    """
    if not rows:
        return
    body = "".join(
        f'<div class="row"><span class="k">{html_mod.escape(str(k))}</span>'
        f'<span class="v {html_mod.escape(tone)}">{html_mod.escape(str(v))}</span></div>'
        for k, v, tone in rows
    )
    st.markdown(f'<div class="rail-readout">{body}</div>', unsafe_allow_html=True)


#: The tape, in reading order: the index itself, then the valuation anchors
#: it is scored against, then the macro series that drive the score. Ordered
#: by what a reader checks first rather than alphabetically — a tape is
#: scanned peripherally, and a familiar running order is what makes that
#: possible. Anything absent from the loaded sheet is skipped silently.
TICKER_INSTRUMENTS: tuple[str, ...] = (
    "NIFTY", "NIFTY50_PE", "NIFTY50_EY", "AD_RATIO",
    "IN10Y", "IN02Y", "IN_TERM_SPREAD",
    "US10Y", "US02Y", "US_TERM_SPREAD",
    "REPO_RATE", "FED_RATE", "CPI", "USDINR", "DXY", "CRUDE", "GOLD", "VIX",
)


#: Display-name overrides for the tape. Used only where the sheet's column
#: name is not what a desk would call the series — everywhere else the column
#: name IS the right label, because that is what a tape shows.
_TAPE_ALIAS: dict[str, str] = {
    "NIFTY": "NIFTY 50",
    "NIFTY50_PE": "PE",
    "NIFTY50_EY": "EY",
    "AD_RATIO": "A/D",
    "IN_TERM_SPREAD": "IN 10-2",
    "US_TERM_SPREAD": "US 10-2",
    "REPO_RATE": "REPO",
    "FED_RATE": "FED",
}


def _tape_symbol(column: str) -> str:
    """Ticker for a sheet column, as a tape would print it.

    Falls back to a word-boundary-safe abbreviation for anything unmapped,
    because cutting a name mid-word is worse than showing fewer words.
    """
    if column in _TAPE_ALIAS:
        return _TAPE_ALIAS[column]
    out = ""
    for word in column.replace("_", " ").split():
        if len(out) + len(word) + 1 > 12:
            break
        out = f"{out} {word}".strip()
    return (out or column[:12]).upper()


#: Columns whose natural precision is not two decimals. A percentage-point
#: series printed to 2dp is fine; an index level printed to 2dp is noise, and
#: a yield printed to 0dp has lost the reading entirely.
_TAPE_DP: dict[str, int] = {"NIFTY": 0, "AD_RATIO": 3}


def render_ticker(frame, instruments: tuple[str, ...] = TICKER_INSTRUMENTS,
                  seconds_per_item: float = 3.6) -> None:
    """Render the running tape from the already-loaded sheet.

    No additional network call: the frame behind the engine already holds
    every one of these series, so the tape is a view of the data the run is
    using rather than a second, possibly disagreeing, source.

    The track is emitted TWICE and animated to -50%, which is what makes the
    loop seamless — at the moment the first copy leaves the viewport the second
    is exactly where the first began. Duration scales with item count so the
    scroll speed stays constant no matter how many series are listed; a tape
    that accelerates as you add symbols is unreadable.

    Direction is carried by an arrow glyph as well as by colour. Roughly 8% of
    men have red/green colour deficiency, and the sign of a move is the one
    reading here that must never be ambiguous.
    """
    if frame is None or not len(frame):
        return
    cols = [c for c in instruments if c in getattr(frame, "columns", ())]
    if not cols:
        return

    tail = frame[cols].tail(2)
    if len(tail) < 2:
        return
    prev, last = tail.iloc[0], tail.iloc[1]

    items: list[str] = []
    for c in cols:
        try:
            p1, p0 = float(last[c]), float(prev[c])
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(p1) and np.isfinite(p0)) or p0 == 0:
            continue
        chg = (p1 / p0 - 1.0) * 100.0
        cls, arrow = (("up", "\u25b2") if chg > 0.005 else
                      ("down", "\u25bc") if chg < -0.005 else ("flat", "\u2022"))
        dp = _TAPE_DP.get(c, 2)
        px = f"{p1:,.{dp}f}"
        items.append(
            f'<span class="tt-item">'
            f'<span class="tt-sym">{html_mod.escape(_tape_symbol(c))}</span>'
            f'<span class="tt-px">{px}</span>'
            f'<span class="tt-chg {cls}" data-arrow="{arrow}">{abs(chg):.2f}%</span>'
            f'</span><span class="tt-sep">|</span>'
        )
    if not items:
        return

    run = "".join(items)
    duration = max(40.0, len(items) * float(seconds_per_item))
    st.markdown(
        f'<div class="ticker" role="marquee" aria-label="Live series tape">'
        f'<div class="tt-track" style="--tt-duration:{duration:.0f}s">{run}{run}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_empty_state(
    title: str,
    body: str,
    *,
    eyebrow: str = "",
    action_label: str = "",
) -> None:
    """Render a professional, icon-free empty/degraded state.

    One system for every "nothing to show yet" moment in the app — cold
    start (no data loaded), "no weights yet", "no usable precedent", short-
    history guards — rather than each tab hand-rolling its own notice.
    ``body`` may carry simple inline HTML (``<strong>``, line breaks), same
    convention as ``render_interpretation_card``. ``action_label`` is a short
    hint at what to do next (e.g. "Pick a target in the sidebar, then Run
    Analysis"), not a real button — the actual control lives wherever it
    already does (sidebar, etc.); this just points at it.
    """
    eyebrow_html = f'<div class="es-eyebrow">{html_mod.escape(eyebrow)}</div>' if eyebrow else ""
    action_html = f'<div class="es-action">{html_mod.escape(action_label)}</div>' if action_label else ""
    st.markdown(
        f'<div class="empty-state">'
        f'{eyebrow_html}'
        f'<div class="es-title">{html_mod.escape(title)}</div>'
        f'<div class="es-body">{body}</div>'
        f'{action_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_info_box(title: str, content: str, color: str = "cyan") -> None:
    """Render an info box. ``color`` is applied as a modifier class (cyan / amber /
    emerald / rose / violet) so callers can theme it; was previously ignored."""
    st.markdown(
        f'<div class="info-box {html_mod.escape(color)}">'
        f"<h4>{html_mod.escape(title)}</h4>"
        f"<p>{html_mod.escape(content)}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_interpretation_card(
    title: str,
    body: str,
    color: str = "neutral",
) -> None:
    """Render a state-aware interpretation card — terminal readout style.

    Args:
        title: Short state label (e.g. "NEUTRAL", "STRONG OVERSOLD").
        body: One-paragraph explanation (raw HTML allowed — caller is trusted).
        color: Semantic color — "neutral", "success", "danger", "warning", "info".
    """
    st.markdown(
        f'<div class="interp-card {html_mod.escape(color)}">'
        f'<div class="interp-title">{html_mod.escape(title)}</div>'
        f'<div class="interp-body">{body}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# (``render_nishkarsh_signal_card`` lived here. It was a thin wrapper that
# called build_hero_verdict with a signature three rewrites out of date, had no
# callers anywhere in the app, and would have raised TypeError if one had
# appeared.)


# ── Data-table styling tokens ──────────────────────────────────────────
# render_data_table renders into an isolated components.v1.html iframe, which
# does NOT inherit the app's CSS variables — so the theme values it needs are
# mirrored here as literals, in BOTH a dark and light set. Any change to the
# corresponding --token in theme.css/ui.theme.LIGHT_TOKENS has to be made
# here too; there is no way around that while the table lives in an iframe,
# and a stale colour here is the visible symptom. Header rule/hover use the
# primary ACCENT (not amber — amber is caution/warning only in this system).
_TABLE_TOKENS_DARK = {
    "ink_primary":   "#E6EAF1",   # --ink
    "ink_tertiary":  "#8B95A6",   # --ink-tertiary
    "border":        "rgba(255, 255, 255, 0.07)",   # --line
    "border_subtle": "rgba(255, 255, 255, 0.035)",  # --line-faint
    "accent":        "#4C7DF0",   # --accent
    "emerald":       "#2CA36B",   # --long   (positive numeric cells)
    "rose":          "#DD5A5A",   # --short  (negative numeric cells)
    "accent_border": "rgba(76, 125, 240, 0.34)",
    "accent_hover":  "rgba(76, 125, 240, 0.10)",
    "row_odd":       "rgba(255, 255, 255, 0.015)",
    "row_even":      "transparent",
    "surface_a":     "#0F1217",   # --surface-1
    "surface_b":     "#0F1217",
    "header_a":      "#151920",   # --surface-2
    "header_b":      "#151920",
}
_TABLE_TOKENS_LIGHT = {
    "ink_primary":   "#141920",
    "ink_tertiary":  "#5E6979",
    "border":        "rgba(15, 23, 42, 0.10)",
    "border_subtle": "rgba(15, 23, 42, 0.05)",
    "accent":        "#2B5FD9",
    "emerald":       "#0F7A54",
    "rose":          "#C0392F",
    "accent_border": "rgba(43, 95, 217, 0.32)",
    "accent_hover":  "rgba(43, 95, 217, 0.07)",
    "row_odd":       "rgba(15, 23, 42, 0.022)",
    "row_even":      "transparent",
    "surface_a":     "#FFFFFF",
    "surface_b":     "#FFFFFF",
    "header_a":      "#EEF1F5",
    "header_b":      "#EEF1F5",
}


def _table_tokens() -> dict:
    """Active-theme token set for the iframe-isolated data table."""
    return _TABLE_TOKENS_LIGHT if st.session_state.get("theme") == "light" else _TABLE_TOKENS_DARK

#: Webfont the iframe must import for itself, for the same isolation reason.
_TABLE_FONTS = ("https://fonts.googleapis.com/css2?"
                "family=JetBrains+Mono:wght@400;500;600;700&display=swap")


# ═══════════════════════════════════════════════════════════════════════
#  HERO VERDICT — the conviction chain
# ═══════════════════════════════════════════════════════════════════════
#
# The engine does not produce a "signal" that evidence then votes on. It makes
# ONE claim and attaches a series of conditions to it, every one of which can
# independently invalidate it:
#
#   the market is mispriced on valuation   (Mood Score — the claim)
#   ...the oscillator does not contradict  (MSF spread)
#   ...the regime is one reversion works in (Hurst x entropy)
#   ...the last analogous states paid       (precedent base rate)
#   ...this has held out of sample          (holdout rho vs a permutation null)
#   ...and it beats having no engine at all (margin over the -PE baseline)
#
# Every measurement the app makes appears exactly once, and only where it has
# something to say the others do not.
#
# So conviction is a PRODUCT of gates in [0, 1], not a sum of votes. A product
# cannot be rescued by piling on agreement: three enthusiastic confirmations
# and one broken precondition is not "3 - 1 = act smaller", it is "the
# precondition is broken".
#
# And a product has a MINIMUM. Whichever gate is smallest is the binding
# constraint — the single specific reason conviction is not higher, which is
# the most useful sentence a card of this kind can produce and which no amount
# of vote-tallying can express. The card leads with it.
#
# Direction comes from the Mood Score alone. It is the only component making a
# directional claim about the world ("this market is cheap relative to its own
# recent history"); the oscillator, the regime, the precedents and the holdout
# are all statements ABOUT that claim, not rival claims of their own.
#
# The BASELINE gate is Arthagati's own, and it is the honest one. The app's
# validation reports that the negated PE ratio — no engine, no percentiles, no
# correlations — scores nearly as well on the same holdout. A conviction
# number that ignored this would be claiming credit the pipeline has not
# earned, so the margin over that baseline is a gate like any other and it
# routinely binds.


def _gate(value: float, lo: float, hi: float) -> float:
    """Map a raw reading onto [0, 1] with a soft floor and ceiling.

    ``lo`` is where the gate is fully shut, ``hi`` where it is fully open;
    between them it opens linearly. Never returns exactly 0 — a shut gate
    should collapse conviction, not erase the reading and its explanation
    with it.
    """
    if value is None or not np.isfinite(value):
        return 0.5
    if hi == lo:
        return 1.0
    return float(np.clip((value - lo) / (hi - lo), 0.02, 1.0))


#: Minimum DISTINCT analogs before the precedent base rate is treated as
#: probative. Below this, "% positive" is a handful of coin flips.
MIN_PRECEDENT_N = 8

#: Conviction tiers. Products of SIX gates concentrate hard toward zero, so
#: these are not evenly spaced: 0.30 already requires every gate to average
#: ~0.82, and 0.15 requires ~0.73.
_TIERS = (
    (0.30, "high", "HIGH CONVICTION", "position on it"),
    (0.15, "moderate", "MODERATE CONVICTION", "position at reduced size"),
    (0.06, "low", "LOW CONVICTION", "starter size at most"),
    (0.00, "standaside", "STAND ASIDE", "no actionable reading"),
)


def build_hero_verdict(
    *,
    mood: float,
    msf: float,
    regime: str,
    entropy: float,
    hurst: float,
    ou_half_life: float,
    precedent: dict | None,
    validation: dict | None,
    data_age_days: int,
    is_warmup: bool,
    horizon_days: int,
    bands: tuple[float, float, float] = (20.0, 45.0, 5.0),
) -> dict:
    """Build the hero verdict from the conviction chain. Pure data in/out.

    Returns ``{signal, signal_class, direction, conviction, headline, binding,
    limit, gates, risk, trust, action}`` where ``gates`` is the ordered chain
    (each ``{name, value, label, detail}``) and ``binding`` names the smallest.
    Rendering is entirely separate (``render_hero_card``), so these rules stay
    testable without a Streamlit runtime.

    ``bands`` is ``(inner, outer, msf_level)`` from config, threaded in rather
    than imported so the classification here cannot drift from the chart's.
    """
    inner, outer, msf_level = bands
    mood = float(mood)
    msf = float(msf)

    # ── The claim: where valuation sits against its own recent history ──
    # High score = cheap = constructive. This is a valuation reading, not a
    # momentum one: it moves AGAINST recent price action by construction.
    if mood >= inner:
        direction, verb = "bullish", "cheap"
    elif mood <= -inner:
        direction, verb = "bearish", "expensive"
    else:
        direction, verb = "neutral", "fairly valued"

    gates: list[dict] = []

    # ── Gate 1: is it mispriced at all? ────────────────────────────────
    g_val = _gate(abs(mood), inner, outer)
    gates.append({
        "name": "valuation", "value": g_val,
        "label": f"{abs(mood):.0f} pts {verb}" if direction != "neutral" else "within the neutral band",
        "detail": (f"Mood {mood:+.1f} against fixed bands at \u00b1{inner:.0f} and \u00b1{outer:.0f}."
                   if direction != "neutral"
                   else f"Mood {mood:+.1f} is inside \u00b1{inner:.0f} \u2014 no valuation claim to make."),
    })

    # ── Gate 2: does the confirmation oscillator contradict it? ────────
    # MSF is built to be independent of the mood score, so a disagreement is
    # information rather than an error. A bullish valuation reading is NOT
    # confirmed while the oscillator is stretched to the upside.
    aligned = -msf if direction == "bullish" else msf if direction == "bearish" else -abs(msf)
    g_msf = _gate(aligned, -msf_level, msf_level)
    gates.append({
        "name": "confirmation", "value": g_msf,
        "label": ("oscillator confirms" if aligned > 1.0 else
                  "oscillator contradicts" if aligned < -1.0 else "oscillator neutral"),
        "detail": (f"MSF spread {msf:+.2f} on \u00b1{msf_level:.0f} bands"
                   + (f" \u2014 {'confirms' if aligned > 1.0 else 'contradicts' if aligned < -1.0 else 'is neutral on'}"
                      f" the {direction} reading." if direction != "neutral" else ".")),
    })

    # ── Gate 3: is the regime one where a reversion reading works? ─────
    # The score is a mean-reverting valuation measure. A disordered market
    # is one where the percentile it is built from carries less information.
    g_reg = _gate(1.0 - float(entropy), 0.25, 0.62)
    gates.append({
        "name": "regime", "value": g_reg,
        "label": f"{regime.lower()}",
        "detail": (f"Entropy {entropy:.2f}, Hurst {hurst:.2f}"
                   + (f", OU half-life {ou_half_life:.0f}d." if ou_half_life > 0 else ".")),
    })

    # ── Gate 4: did the last analogous states pay? ─────────────────────
    p_n = int((precedent or {}).get("n", 0) or 0)
    p_pos = (precedent or {}).get("positive_pct")
    if precedent and p_n >= MIN_PRECEDENT_N and p_pos is not None:
        p_bull = float(p_pos) / 100.0
        agree = (p_bull if direction == "bullish" else
                 1.0 - p_bull if direction == "bearish" else 0.5)
        g_prec = _gate(agree, 0.35, 0.65)
        gates.append({
            "name": "precedent", "value": g_prec,
            "label": ("precedent agrees" if agree > 0.55 else
                      "precedent disagrees" if agree < 0.45 else "precedent split"),
            "detail": f"{float(p_pos):.0f}% of {p_n} separated analogs rose over +{horizon_days}d.",
        })
    else:
        g_prec = 0.6
        gates.append({
            "name": "precedent", "value": 0.6, "label": "no usable precedent",
            "detail": f"Only {p_n} separated analogs (need {MIN_PRECEDENT_N}) \u2014 "
                      "too few to read as a base rate.",
        })

    # ── Gate 5: has this held out of sample? ───────────────────────────
    v = validation or {}
    rho = v.get("holdout_rho")
    pval = v.get("p_value")
    if rho is None or not np.isfinite(float(rho)):
        # Not "no edge" — no measurement. An unscored run gets a discount,
        # not a verdict.
        g_edge = 0.25
        edge_label = "unvalidated"
        edge_detail = "No holdout score for this predictor set yet \u2014 open Signal Validation."
    else:
        rho = float(rho)
        g_rho = _gate(rho, -0.02, 0.15)
        # Significance is a second, multiplicative condition: a rho that a
        # permutation null reproduces is not an edge however large it looks.
        g_sig = _gate(1.0 - float(pval if pval is not None else 1.0), 0.90, 0.99)
        g_edge = g_rho * g_sig
        edge_label = ("edge holds" if rho > 0.05 and (pval or 1) <= 0.05 else
                      "edge marginal" if rho > 0 else "no measured edge")
        edge_detail = f"Holdout rho {rho:+.3f}"
        if pval is not None:
            edge_detail += f" at p = {float(pval):.3f}"
        if v.get("n_holdout"):
            edge_detail += f" on {int(v['n_holdout']):,} held-out rows."
        else:
            edge_detail += "."
    gates.append({"name": "edge", "value": g_edge,
                  "label": edge_label, "detail": edge_detail})

    # ── Gate 6: is the engine worth more than no engine? ───────────────
    base = v.get("baseline_rho")
    if rho is not None and base is not None and np.isfinite(float(base)) and np.isfinite(float(rho)):
        margin = float(rho) - float(base)
        g_base = _gate(margin, -0.03, 0.05)
        gates.append({
            "name": "baseline", "value": g_base,
            "label": ("engine adds rank information" if margin > 0.02 else
                      "engine matches the anchor" if margin > -0.02 else
                      "anchor alone does better"),
            "detail": f"Negated PE alone scores {float(base):+.3f} on the same window; "
                      f"the five-layer pipeline adds {margin:+.3f}.",
        })
    else:
        g_base = 0.5
        gates.append({
            "name": "baseline", "value": 0.5, "label": "no baseline read",
            "detail": "NIFTY50_PE is absent or flat, so the no-engine baseline "
                      "could not be scored.",
        })

    # ── Conviction: the product, and the constraint that binds it ──────
    conviction = float(np.prod([g["value"] for g in gates]))
    binding = min(gates, key=lambda g: g["value"])

    level, label, prose = "standaside", "STAND ASIDE", "no actionable reading"
    for cut, lvl, lab, pr in _TIERS:
        if conviction >= cut:
            level, label, prose = lvl, lab, pr
            break
    if direction == "neutral":
        level, label, prose = "standaside", "STAND ASIDE", "no directional claim to act on"

    signal = ("CONSTRUCTIVE" if direction == "bullish" else
              "DEFENSIVE" if direction == "bearish" else "NEUTRAL")
    if direction != "neutral" and level == "high":
        signal = f"STRONGLY {signal}"

    headline = (
        f"{signal} \u2014 valuation reads {abs(mood):.0f} points {verb} against its own "
        f"recent history, on a {horizon_days}-day positioning horizon."
        if direction != "neutral"
        else "NEUTRAL \u2014 valuation is inside the band where the score makes no claim."
    )

    # The single most useful sentence the card produces: what holds it back.
    if direction == "neutral":
        limit = f"No directional claim: the reading is within \u00b1{inner:.0f} of neutral."
    elif binding["value"] >= 0.75:
        limit = "Nothing is materially limiting this \u2014 every condition holds."
    else:
        limit = f"Capped by {binding['name']}: {binding['detail']}"

    # ── Standing risk flags, outside the chain ─────────────────────────
    # Data quality is not a condition on the claim; it is a reason to distrust
    # every number on the page, including the gates themselves.
    flags = []
    if data_age_days > 4:
        flags.append(f"Data is {data_age_days} days old \u2014 every reading describes that date, "
                     "not the current market.")
    if is_warmup:
        flags.append("This row is inside the correlation warm-up and carries borrowed statistics.")
    risk = " ".join(flags) or None

    tier = ("solid" if (rho or 0) >= 0.10 else
            "modest" if (rho or 0) >= 0.05 else
            "marginal" if (rho or 0) > 0 else
            "no_edge" if rho is not None else "uncalibrated")
    trust = {
        "tier": tier,
        "chip": {"solid": "SOLID EDGE", "modest": "MODEST EDGE", "marginal": "MARGINAL",
                 "no_edge": "NO EDGE", "uncalibrated": "NO READ"}[tier],
        "oos_ic": rho, "wf_pos": None, "wf_n": None, "prose": edge_detail,
    }

    return {
        "signal": signal,
        "signal_class": ("buy" if direction == "bullish" else
                         "sell" if direction == "bearish" else "hold"),
        "direction": direction,
        "score": float(np.clip(mood / 100.0, -1.0, 1.0)),
        "conviction": conviction,
        "headline": headline,
        "binding": binding["name"] if binding["value"] < 0.75 else None,
        "limit": limit,
        "gates": gates,
        "risk": risk,
        "trust": trust,
        "action": {"level": level, "label": label, "prose": prose,
                   "conviction": conviction},
        "horizon_days": horizon_days,
    }


def render_hero_card(verdict: dict) -> None:
    """Render the verdict: claim, what limits it, then the chain behind both.

    The layout follows the logic rather than decorating it. A reader who stops
    after two lines has the decision (signal + conviction) and the single
    reason it is not stronger; a reader who continues gets every gate with the
    number behind it. The old card put five equal-weight evidence rows above a
    points total, which buried the one line that mattered among four that
    usually did not.
    """
    trust = verdict["trust"]
    # Theme-aware: every colour here is a CSS var, not a literal hex, so the
    # trust chip repaints correctly under the light theme too.
    chip_style = {
        "uncalibrated": ("var(--ink-tertiary)", "var(--surface-2)"),
        "no_edge":      ("var(--short)", "var(--short-fill)"),
        "marginal":     ("var(--caution)", "var(--caution-fill)"),
        "modest":       ("var(--long)", "var(--long-fill)"),
        "solid":        ("var(--long)", "color-mix(in srgb, var(--long) 18%, transparent)"),
    }.get(trust["tier"], ("var(--ink-tertiary)", "var(--surface-2)"))
    ic_text = (f"HOLDOUT \u03c1 {trust['oos_ic']:+.3f}" if trust.get("oos_ic") is not None
               else "no holdout read")

    action = verdict["action"]
    conviction = float(verdict.get("conviction", 0.0))
    tier_color = {"high": "var(--long)", "moderate": "var(--caution)",
                  "low": "var(--ink-secondary)", "standaside": "var(--ink-tertiary)"
                  }.get(action["level"], "var(--ink-tertiary)")

    # ── Gate chain: one row each, bar width = how open the gate is ──────
    binding = verdict.get("binding")
    gate_rows = "".join(
        f'<div class="hero-gate{" binding" if g["name"] == binding else ""}">'
        f'<span class="hero-gate-name">{html_mod.escape(g["name"])}</span>'
        f'<span class="hero-gate-bar"><i style="width:{max(2, round(g["value"] * 100))}%;'
        f'background:{"var(--short)" if g["value"] < 0.35 else "var(--caution)" if g["value"] < 0.7 else "var(--long)"};">'
        f'</i></span>'
        f'<span class="hero-gate-label">{html_mod.escape(g["label"])}</span>'
        f'<span class="hero-gate-detail">{html_mod.escape(g["detail"])}</span>'
        f'</div>'
        for g in verdict["gates"]
    )

    risk_html = (
        f'<div class="hero-risk">{html_mod.escape(verdict["risk"])}</div>'
        if verdict.get("risk") else ""
    )

    st.markdown(
        f"""\
<div class="hero-card {html_mod.escape(verdict["signal_class"])}">
  <div class="hero-top">
    <div class="hero-signal-block">
      <div class="hero-eyebrow">Arthagati &bull; {verdict["horizon_days"]}d positioning horizon</div>
      <div class="hero-signal">{html_mod.escape(verdict["signal"])}</div>
    </div>
    <div class="hero-conviction-block">
      <span class="hero-chip" style="background:{chip_style[1]};color:{chip_style[0]};">\
{html_mod.escape(trust["chip"])} &bull; {ic_text}</span>
      <div class="hero-conviction" style="color:{tier_color};">\
{html_mod.escape(action["label"])} &middot; {conviction:.2f}</div>
      <div class="hero-conviction-sub">{html_mod.escape(action["prose"])}</div>
    </div>
  </div>
  <div class="hero-headline">{html_mod.escape(verdict["headline"])}</div>
  <div class="hero-limit">{html_mod.escape(verdict["limit"])}</div>
  <div class="hero-gates">{gate_rows}</div>
  {risk_html}
  <div class="hero-foot">Conviction is the product of the gates above &mdash; the weakest caps it.</div>
</div>
""",
        unsafe_allow_html=True,
    )


def _fmt_cell(value, precision: int) -> str:
    """Format one cell value for display (NaN → em dash; floats to `precision`).

    Dates render date-only: Arthagati is a DAILY system, so a Timestamp's
    ``00:00:00`` time component is noise — never shown.
    """
    if value is None:
        return "—"
    # Date-only for any datetime-like (pd.Timestamp subclasses datetime.date).
    if isinstance(value, (pd.Timestamp, _dt.date)):
        try:
            if pd.isna(value):
                return "—"
        except (TypeError, ValueError):
            pass
        return value.strftime("%Y-%m-%d")
    if isinstance(value, float):
        if value != value:            # NaN
            return "—"
        return f"{value:,.{precision}f}"
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return f"{value:,}"
    try:
        if pd.isna(value):
            return "—"
    except (TypeError, ValueError):
        pass
    return html_mod.escape(str(value))


# Column-name tokens that must stay UPPER-CASE when a raw column name is
# prettified into a professional header ("MSF_Osc" → "MSF Osc", not "Msf Osc").
# (Deliberately NOT including "OSC" — an oscillator column reads more
# professionally as "Osc" than "OSC", matching the source design.)
_HEADER_ACRONYMS = {
    "RSI", "MA", "MSF", "MMR", "VAP", "IC", "HR", "HMM", "GARCH", "CUSUM",
    "ADF", "KPSS", "DDM", "OU", "PCA", "US", "FX", "ID", "N", "T", "Z", "R2",
    "OHLC", "OHLCV", "ATR", "MACD", "EMA", "SMA",
}


def _prettify_header(name: str) -> str:
    """Turn a raw column/field name into a professional table header.

    ``divergence_type`` → ``Divergence Type``; ``MSF_Osc`` → ``MSF Osc``;
    ``Change_Point`` → ``Change Point``; ``val_ic`` → ``Val IC``. Already-clean
    headers ("Buy Avg Δ", "Period") pass through with only per-word acronym
    casing applied.
    """
    raw = str(name).replace("_", " ").strip()
    if not raw:
        return ""
    out = []
    for word in raw.split():
        up = word.upper()
        if up in _HEADER_ACRONYMS:
            out.append(up)
        elif word.isupper() and len(word) <= 4:   # keep short all-caps as-is
            out.append(word)
        elif any(ch.isdigit() for ch in word) and word.isupper():
            out.append(word)
        else:
            out.append(word[:1].upper() + word[1:])
        # Preserve non-alphanumeric tokens verbatim (Δ, %, etc.)
        if not word[:1].isalnum():
            out[-1] = word
    return " ".join(out)


def render_data_table(
    df: "pd.DataFrame",
    *,
    index_label: str | None = None,
    show_index: bool | None = None,
    max_rows: int | None = None,
    precision: int = 2,
    col_precision: dict[str, int] | None = None,
    sign_color_cols: "set[str] | None" = None,
    label_col: str | None = None,
    col_labels: dict[str, str] | None = None,
    max_height: int = 520,
    row_height: int = 27,
) -> None:
    """Render a DataFrame as the app's one institutional table.

    The only table primitive in Arthagati — there is no bare ``st.dataframe``
    anywhere, because Streamlit's grid brings its own typeface, row height,
    header treatment and hover, none of which can be reached from the app's
    stylesheet. Sticky muted header, hairline row rules, right-aligned tabular
    numerics, a bolder "label" column, and horizontal/vertical scroll under a
    fixed ``max_height`` — safe on both the 10-row divergence table and the
    full dataset viewer.

    Rows are 27px (was 42): the old height came from 0.6rem cell padding at a
    0.75rem font, which is a comfortable READING density, not a scanning one.
    A table a trader scans should fit twice as many rows in the same panel.

    Wrap it in ``render_table_panel`` rather than calling it directly, so the
    table gets the same header anatomy as every chart.

    Parameters
    ----------
    index_label : shown as the first column header when the index is rendered;
        also forces the index to render.
    show_index : override index rendering (default: auto — shown when the index
        is not a plain 0..N RangeIndex, i.e. it carries dates/labels).
    max_rows : cap to the LAST ``max_rows`` rows (tables are newest-relevant).
    precision / col_precision : default and per-column float precision.
    sign_color_cols : numeric columns whose values are tinted emerald/rose by
        sign (the "signal" colouring from Pragyam's per-signal columns).
    label_col : the column to style as the bold Space-Grotesk label (default:
        the index if shown, else the first column).
    col_labels : explicit header overrides ``{raw_name: display}``; any column
        not listed is auto-prettified (``MSF_Osc`` → ``MSF Osc``).
    """
    if df is None or getattr(df, "empty", True):
        st.markdown('<div class="panel-state">No rows to display.</div>',
                    unsafe_allow_html=True)
        return

    view = df.tail(max_rows).copy() if max_rows else df.copy()
    if isinstance(view.columns, pd.MultiIndex):
        view.columns = [" · ".join(str(x) for x in c) for c in view.columns]

    if show_index is None:
        show_index = index_label is not None or not isinstance(view.index, pd.RangeIndex)
    idx_header = (index_label or _prettify_header(view.index.name or "")) if show_index else ""
    col_labels = col_labels or {}

    def _header(c: str) -> str:
        return col_labels.get(c) or _prettify_header(c)

    cols = list(view.columns)
    numeric_cols = {c for c in cols if pd.api.types.is_numeric_dtype(view[c])}
    sign_cols = (sign_color_cols or set()) & numeric_cols
    col_precision = col_precision or {}
    # The label column: explicit, else the index (when shown), else first column.
    if label_col is None:
        label_col = "__index__" if show_index else (cols[0] if cols else None)

    t = _table_tokens()

    def _header_cells() -> str:
        cells = []
        if show_index:
            cells.append(f'<th class="lbl">{html_mod.escape(str(idx_header))}</th>')
        for c in cols:
            cls = "num" if c in numeric_cols and c != label_col else "lbl" if c == label_col else "txt"
            cells.append(f'<th class="{cls}">{html_mod.escape(_header(c))}</th>')
        return "".join(cells)

    def _value_html(c: str, val) -> str:
        p = col_precision.get(c, precision)
        text = _fmt_cell(val, p)
        if c in sign_cols and text != "—":
            try:
                fv = float(val)
                color = (t["emerald"] if fv > 1e-12 else t["rose"] if fv < -1e-12
                         else t["ink_tertiary"])
                return f'<span style="color:{color};font-weight:600;">{text}</span>'
            except (TypeError, ValueError):
                pass
        return text

    body_rows = []
    for idx, row in view.iterrows():
        tds = []
        if show_index:
            tds.append(f'<td class="lbl">{_fmt_cell(idx, precision)}</td>')
        for c in cols:
            cls = "num" if c in numeric_cols and c != label_col else "lbl" if c == label_col else "txt"
            tds.append(f'<td class="{cls}">{_value_html(c, row[c])}</td>')
        body_rows.append(f"<tr>{''.join(tds)}</tr>")

    n_rows = len(view)
    _HEADER_H = 30                      # sticky header row, matches the CSS above
    content_h = _HEADER_H + n_rows * row_height + 4
    iframe_h = min(content_h, max_height)

    # The iframe cannot see the app's stylesheet, so the design tokens it needs
    # are restated here as literals (see _TABLE_TOKENS_*). Values below mirror
    # theme.css exactly: --fs-2xs header, --fs-xs body, s2/s3 cell padding,
    # hairline rules. Nothing here is a one-off number.
    table_html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
    @import url('{_TABLE_FONTS}');
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    /* JetBrains Mono — the app's data face, and the one _TABLE_FONTS actually
       imports. This said 'IBM Plex Mono' while importing JetBrains, so the
       declared family was never loaded and every table in the app fell through
       to the system default (Menlo/Courier). Tables were the one surface
       rendering in a typeface the rest of the UI does not use. */
    body {{ font-family:'JetBrains Mono',ui-monospace,SFMono-Regular,Menlo,monospace;
            background:transparent; color:{t['ink_primary']};
            font-variant-numeric:tabular-nums; font-feature-settings:"tnum" 1,"zero" 1; }}
    /* A hair of top padding so the sticky header cannot sit flush against
       the panel header rendered directly above this iframe — the two read as
       one doubled, overlapping header row without it. */
    .tt-scroll {{ padding-top:2px; max-height:{max_height}px; overflow:auto;
                  scrollbar-width:thin; scrollbar-color:{t['ink_tertiary']} transparent; }}
    .tt-scroll::-webkit-scrollbar {{ width:9px; height:9px; }}
    .tt-scroll::-webkit-scrollbar-track {{ background:transparent; }}
    .tt-scroll::-webkit-scrollbar-thumb {{ background:{t['border']}; border-radius:100px; }}
    .tt-scroll::-webkit-scrollbar-thumb:hover {{ background:{t['ink_tertiary']}; }}
    .tt-scroll::-webkit-scrollbar-corner {{ background:transparent; }}
    table {{ width:100%; border-collapse:collapse; }}
    /* Header: muted, uppercase, hairline rule. It was a 2px accent-coloured
       rule with an accent-coloured first cell — the heaviest horizontal line
       in the app, sitting under its quietest content. A header is a label for
       a column, not a claim about it. */
    thead th {{ position:sticky; top:0; z-index:2;
        background:{t['header_a']};
        color:{t['ink_tertiary']}; font-size:0.625rem; font-weight:600;
        text-transform:uppercase; letter-spacing:0.12em; padding:0.5rem 0.75rem;
        border-bottom:1px solid {t['border']}; text-align:left; white-space:nowrap; }}
    thead th.num {{ text-align:right; }}
    /* Row separation is a hairline OR a tint, never both — the two together
       are what made this read as a spreadsheet export. */
    tbody tr {{ border-bottom:1px solid {t['border_subtle']};
                transition:background 120ms cubic-bezier(0.2,0,0,1); }}
    tbody tr:last-child {{ border-bottom:none; }}
    tbody tr:hover {{ background:{t['accent_hover']}; }}
    tbody td {{ padding:0.4rem 0.75rem; color:{t['ink_primary']}; font-size:0.6875rem;
                line-height:1.5; vertical-align:middle; white-space:nowrap; }}
    tbody td.num {{ text-align:right; }}
    tbody td.lbl {{ font-weight:600; color:{t['ink_primary']}; }}
    tbody td.txt {{ color:{t['ink_tertiary']}; }}
    </style></head><body>
    <div class="tt-scroll"><table>
    <thead><tr>{_header_cells()}</tr></thead>
    <tbody>{''.join(body_rows)}</tbody>
    </table></div></body></html>"""

    _components_html(table_html, height=iframe_h, scrolling=False)


def render_warning_box(title: str, content: str) -> None:
    """Render a themed alert/warning box."""
    st.markdown(
        f"""
        <div class="warning-box">
            <div class="icon"></div>
            <div>
                <div class="title">{html_mod.escape(title)}</div>
                <div class="content">{html_mod.escape(content)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def warmup_note(mood_df) -> str:
    """One sentence naming the newest row as pre-warm-up, or "" when it is not.

    The engine marks rows before ``CORR_MIN_WARMUP`` as ``Is_Warmup``: their
    statistics are borrowed from the first estimated checkpoint rather than
    measured causally, so they are emitted but must not be read as the engine's
    own output. This is the shared accessor, so every surface showing a number
    can say whether it has settled.

    Empty string when the newest row is past warm-up, so callers can treat it
    as falsy and render nothing.
    """
    try:
        if mood_df is None or not len(mood_df):
            return ""
        if "Is_Warmup" not in getattr(mood_df, "columns", ()):
            return ""
        flags = mood_df["Is_Warmup"].to_numpy(dtype=bool)
        if not bool(flags[-1]):
            return ""
        n = int(flags.sum())
        return (f"Latest row is inside the correlation warm-up ({n:,} rows). Its "
                "weights are borrowed from the first estimated checkpoint rather "
                "than measured on data preceding it; later dates are causal.")
    except (IndexError, KeyError, TypeError, ValueError):
        return ""
