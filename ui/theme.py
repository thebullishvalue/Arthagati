"""
Arthagati UI Theming — Extracted from Pragyam & Nishkarsh
"""

from __future__ import annotations

from pathlib import Path
import streamlit as st

CSS_PATH = Path(__file__).parent / "theme.css"

# ── Shared Plotly layout config ─────────────────────────────────────────────
# Adapted from Nishkarsh Institutional Research Terminal

PLOTLY_FONT = dict(family="JetBrains Mono, monospace", color="#94A3B8", size=10)
PLOTLY_HOVERLABEL = dict(
    bgcolor="rgba(10, 14, 23, 0.95)",
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
PLOTLY_GRID = "rgba(255,255,255,0.035)"
PLOTLY_GRID_ZERO = "rgba(255,255,255,0.06)"

def chart_layout(
    height: int = 360,
    show_legend: bool = True,
    margin: dict | None = None,
    responsive: bool = False,
) -> dict:
    """Return a base Plotly layout dict."""
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
        spikedistance=-1,
    )
    if responsive:
        base["autosize"] = True
    return base


def style_axes(fig, y_title: str = "", x_title: str = "", y_range=None, row=None, col=None) -> None:
    """Apply consistent axis styling to a Plotly figure."""
    kw = {}
    if row is not None: kw["row"] = row
    if col is not None: kw["col"] = col

    fig.update_xaxes(
        showgrid=True,
        gridcolor=PLOTLY_GRID,
        gridwidth=0.5,
        zeroline=False,
        linecolor="rgba(255,255,255,0.04)",
        title_text=x_title,
        tickfont=dict(size=9, family="JetBrains Mono, monospace", color="#64748B"),
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikethickness=0.5,
        spikedash="dash",
        spikecolor="rgba(148,163,184,0.18)",
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
        tickfont=dict(size=9, family="JetBrains Mono, monospace", color="#64748B"),
        hoverformat=".2f",
        **kw,
    )

def inject_css() -> None:
    """Inject the Pragyam-based CSS into the Streamlit app."""
    if CSS_PATH.exists():
        css = CSS_PATH.read_text()
    else:
        css = "/* theme.css not found */"

    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def apply_chart_theme(fig) -> None:
    """Apply the Nishkarsh theme to a Plotly figure."""
    fig.update_layout(**chart_layout())
    style_axes(fig)
