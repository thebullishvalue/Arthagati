"""
Arthagati — Shared CSS, chart theming, and colour constants.
अर्थगति (Arthagati) — "Market sentiment / movement of meaning"

UI thesis: "Obsidian Quant" Institutional Research Terminal.
- Display/UI:  Space Grotesk (geometric, authoritative)
- Body/Data:   JetBrains Mono / IBM Plex Mono (tabular precision)
- Palette:     Obsidian (#141A28 -> #0E131F) backgrounds, tuned to WCAG AA
               Amber Gold (#D4A853), Cyan, Emerald, Rose accents
- Surfaces:    Frameless glass panels, thin border strokes
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

# Re-exported from config so there is exactly one place these are defined.
from config import VERSION, PRODUCT_NAME, COMPANY  # noqa: F401

# ── Obsidian Quant colour tokens (mirror :root in theme.css) ────────────────
C_AMBER         = "#D4A853"
C_AMBER_BRIGHT  = "#E8C478"
C_AMBER_DIM     = "rgba(212, 168, 83, 0.6)"
C_AMBER_GLOW    = "rgba(212, 168, 83, 0.25)"
C_CYAN          = "#06B6D4"
C_EMERALD       = "#2DD4A8"
C_EMERALD_BRIGHT = "#6EE7C8"
C_ROSE          = "#E8555A"
C_ROSE_BRIGHT   = "#F07075"
C_VIOLET        = "#9364FE"
C_ORANGE        = "#F59E0B"
C_SLATE_WARM    = "#8B7E6A"

# Semantic shortcuts used throughout the engine code paths
C_PRIMARY = C_AMBER
C_GREEN   = C_EMERALD
C_RED     = C_ROSE
C_MUTED   = "#7D8795"
C_TEXT    = "#F1F5F9"
C_BG_DEEP = "#0E131F"
C_BG_BASE = "#141A28"
C_BG_CARD = "#1B2233"
C_BG_GRID = "rgba(255,255,255,0.07)"

# Path to external CSS file
CSS_PATH = Path(__file__).parent / "theme.css"

# ── Shared Plotly layout configuration ───────────────────────────────────────
PLOTLY_FONT = dict(family="JetBrains Mono, monospace", color="#9BAABF", size=11)
PLOTLY_HOVERLABEL = dict(
    bgcolor="rgba(27, 34, 51, 0.97)",
    font=dict(family="JetBrains Mono, monospace", size=11, color="#F1F5F9"),
    bordercolor="rgba(255,255,255,0.08)",
    align="left",
)
PLOTLY_LEGEND = dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1,
    font=dict(size=10, family="JetBrains Mono, monospace"),
    bgcolor="rgba(0,0,0,0)",
)
PLOTLY_MARGIN = dict(t=20, l=50, r=20, b=40)
PLOTLY_GRID = "rgba(255,255,255,0.07)"
PLOTLY_GRID_ZERO = "rgba(255,255,255,0.14)"

# ── Crosshair ────────────────────────────────────────────────────────────────
# Full crosshair: the x-axis spike draws the vertical line, the y-axis spike
# draws the horizontal one. Both are needed — enabling spikes on one axis only
# gives half a crosshair, which is what this chart had.
#
# `spikesnap="cursor"` makes the lines track the pointer freely rather than
# jumping to the nearest data point, matching how a TradingView crosshair
# behaves. Pair it with `spikedistance=-1` in the layout so the spikes stay
# visible anywhere in the plot area rather than only near a trace.
#
# Both spikes render under `hovermode="x unified"`, so the unified tooltip is
# kept (verified against plotly.js 5.24: the y-axis spike is emitted as a
# horizontal <line> in unified, closest and x hover modes alike).
PLOTLY_SPIKE_COLOR = "rgba(155,170,191,0.38)"

PLOTLY_SPIKE_X: dict = dict(
    showspikes=True,
    spikemode="across",     # spans every stacked pane, not just the hovered one
    spikesnap="cursor",
    spikethickness=0.5,
    spikedash="dash",
    spikecolor=PLOTLY_SPIKE_COLOR,
)

PLOTLY_SPIKE_Y: dict = dict(
    showspikes=True,
    spikemode="across",     # spans the width of the pane being hovered
    spikesnap="cursor",
    spikethickness=0.5,
    spikedash="dash",
    spikecolor=PLOTLY_SPIKE_COLOR,
)

# Shared base layout — paper/plot backgrounds are transparent so the page's
# glass containers show through.
PLOTLY_BASE: dict = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=PLOTLY_FONT,
    hoverlabel=PLOTLY_HOVERLABEL,
)


def chart_layout(
    height: int = 360,
    show_legend: bool = True,
    margin: dict | None = None,
    responsive: bool = False,
) -> dict:
    """Return a base Plotly layout dict for the Obsidian Quant theme."""
    base = dict(
        height=height,
        showlegend=show_legend,
        legend=PLOTLY_LEGEND if show_legend else None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=PLOTLY_FONT,
        hovermode="x unified",
        hoverlabel=PLOTLY_HOVERLABEL,
        margin=margin or PLOTLY_MARGIN,
        # -1 keeps the crosshair alive anywhere in the plot area rather than
        # only when the pointer is near a trace.
        spikedistance=-1,
        hoverdistance=-1,
    )
    if responsive:
        base["autosize"] = True
    return base


def style_axes(fig, y_title: str = "", x_title: str = "", y_range=None, row=None, col=None) -> None:
    """Apply consistent axis styling to a Plotly figure."""
    kw = {}
    if row is not None:
        kw["row"] = row
    if col is not None:
        kw["col"] = col

    fig.update_xaxes(
        showgrid=True,
        gridcolor=PLOTLY_GRID,
        gridwidth=0.5,
        zeroline=False,
        linecolor="rgba(255,255,255,0.04)",
        title_text=x_title,
        tickfont=dict(size=10, family="JetBrains Mono, monospace", color="#9BAABF"),
        **PLOTLY_SPIKE_X,
        **kw,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=PLOTLY_GRID,
        gridwidth=0.5,
        zeroline=True,
        zerolinecolor=PLOTLY_GRID_ZERO,
        zerolinewidth=0.5,
        linecolor="rgba(255,255,255,0.04)",
        title_text=y_title,
        range=y_range,
        tickfont=dict(size=10, family="JetBrains Mono, monospace", color="#9BAABF"),
        hoverformat=".2f",
        **PLOTLY_SPIKE_Y,
        **kw,
    )


def inject_css() -> None:
    """Inject the Obsidian Quant Terminal CSS into the Streamlit app."""
    if CSS_PATH.exists():
        css = CSS_PATH.read_text()
    else:
        css = "/* theme.css not found */"
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def progress_bar(slot, pct: int, label: str, sub: str = "") -> None:
    """Render a themed progress card into an ``st.empty()`` slot."""
    is_complete = pct >= 100
    bar_color = C_EMERALD if is_complete else C_AMBER if pct > 50 else C_CYAN
    dot_class = "pulse-dot complete" if is_complete else "pulse-dot"
    sub_html = f'<div class="progress-sub">{sub}</div>' if sub else ""
    slot.markdown(
        f"""
    <div class="progress-card">
        <div class="progress-label">
            <span class="{dot_class}"></span>{label}
        </div>
        {sub_html}
        <div class="progress-track">
            <div class="progress-fill" style="width:{pct}%;background:{bar_color};box-shadow:0 0 10px {bar_color};"></div>
        </div>
        <div class="progress-pct">{pct}%</div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def apply_chart_theme(fig) -> None:
    """Apply the Obsidian Quant Terminal theme to a Plotly figure (mutates in place)."""
    fig.update_layout(**chart_layout())
    style_axes(fig)
