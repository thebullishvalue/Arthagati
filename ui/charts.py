"""
Arthagati — one charting system.

The app previously had three Plotly figures with three unrelated grammars.
They shared ``PLOTLY_BASE`` and diverged on everything else::

                     hovermode      height  margin(l,t)  crosshair
    Historical Mood  "x unified"    880     60, 60       yes
    Backtest scatter "closest"      420     60, 20       yes
    Signal Validation (unset)       300     50, 30       no

A registered Plotly template moves font, margin, grid, hover, legend, colorway
and spike behaviour out of the individual figures. Charts then declare data and
structure — subplot layout, axis ranges, reversal — and nothing about styling.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

from ui.theme import (
    C_AMBER, C_CYAN, C_EMERALD, C_ROSE, C_VIOLET,
    PLOTLY_GRID, PLOTLY_GRID_ZERO, PLOTLY_SPIKE_X, PLOTLY_SPIKE_Y,
)

TEMPLATE_NAME = "arthagati"

# One height scale. 880 / 420 / 300 were three arbitrary numbers; these are
# named sizes that the responsive helper below scales per breakpoint.
CHART_H = {"sm": 260, "md": 380, "lg": 560, "stack3": 840}

# One margin. The 56px left gutter puts every plot's axis on the same vertical
# line as the panel's text edge, which is what makes stacked charts read as a
# single instrument rather than three separate ones.
CHART_MARGIN = dict(l=56, r=16, t=8, b=36)

FONT_STACK = "JetBrains Mono, IBM Plex Mono, ui-monospace, monospace"

# Categorical series take this in order. Semantic colour (emerald/rose) is
# reserved for values whose sign carries meaning and is applied explicitly.
COLORWAY = [C_AMBER, C_CYAN, C_EMERALD, C_ROSE, C_VIOLET]

PLOT_CONFIG = {
    "displaylogo": False,
    "displayModeBar": False,
    "responsive": True,          # resize with the container, not a fixed px width
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}

_registered = False


def register_theme() -> None:
    """Register and activate the ``arthagati`` Plotly template. Idempotent."""
    global _registered
    if _registered:
        return

    axis_common = dict(
        linecolor="rgba(255,255,255,0.08)",
        tickfont=dict(size=10, family=FONT_STACK, color="#9BAABF"),
        title=dict(font=dict(size=11, family=FONT_STACK, color="#7D8795")),
    )

    pio.templates[TEMPLATE_NAME] = go.layout.Template(
        layout=dict(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family=FONT_STACK, size=11, color="#9BAABF"),
            colorway=COLORWAY,
            margin=CHART_MARGIN,
            # One hover grammar across every chart in the product.
            hovermode="x unified",
            hoverlabel=dict(
                bgcolor="rgba(27,34,51,0.97)",
                bordercolor="rgba(255,255,255,0.13)",
                font=dict(family=FONT_STACK, size=11.5, color="#F1F5F9"),
                align="left",
            ),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                xanchor="right", x=1, bgcolor="rgba(0,0,0,0)",
                font=dict(size=10.5, family=FONT_STACK),
            ),
            # Full crosshair everywhere; -1 keeps it alive away from traces.
            spikedistance=-1,
            hoverdistance=-1,
            xaxis=dict(showgrid=False, zeroline=False, **axis_common, **PLOTLY_SPIKE_X),
            yaxis=dict(
                showgrid=True, gridcolor=PLOTLY_GRID, gridwidth=0.5,
                zeroline=True, zerolinecolor=PLOTLY_GRID_ZERO, zerolinewidth=0.5,
                **axis_common, **PLOTLY_SPIKE_Y,
            ),
        )
    )
    pio.templates.default = TEMPLATE_NAME
    _registered = True


def chart_height(size: str = "md", viewport: str | None = None) -> int:
    """Height for a named size.

    Streamlit cannot read the viewport server-side, so the responsive step for
    chart height is carried by CSS ``max-height`` on the panel body (see the
    layout layer). This returns the desktop figure; the container clamps it on
    smaller screens rather than the figure guessing.
    """
    return CHART_H.get(size, CHART_H["md"])


def axis_muted(**overrides) -> dict:
    """Axis dict for structural overrides — ranges, reversal, titles.

    Everything cosmetic already comes from the template, so callers pass only
    what is genuinely per-chart.
    """
    base = dict(showgrid=True, gridcolor=PLOTLY_GRID, gridwidth=0.5)
    base.update(overrides)
    return base
