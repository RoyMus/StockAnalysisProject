"""Plotly figure builders for the Streamlit UI.

Every chart shares one dark visual system (see ``_base_layout``) so the app
reads as a single coherent product rather than a grab-bag of default Plotly
themes. Functions here are pure: given data (and occasionally pre-computed
indicator frames), they return a ``go.Figure`` and never touch Streamlit or
the network.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from stockanalysis.backtest import BacktestResult
from stockanalysis.forecasting import ForecastResult
from stockanalysis.indicators import bollinger, macd, rsi, sma
from stockanalysis.types import PillarScore

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------

BG = "#1a1a19"
GRID = "#2c2c2a"
AXIS_LINE = "#383835"
INK_MUTED = "#c3c2b7"
INK_PRIMARY = "#ffffff"

BLUE = "#3987e5"
AQUA = "#199e70"
YELLOW = "#c98500"
VIOLET = "#9085e9"
RED = "#e66767"
GRAY = "#898781"

GOOD = "#0ca30c"
WARNING = "#fab219"
SERIOUS = "#ec835a"
CRITICAL = "#d03b3b"

BAND_STRONG_SELL = "#d03b3b"
BAND_SELL = "#ec835a"
BAND_HOLD = "#898781"
BAND_BUY = "#0ca30c"
BAND_STRONG_BUY = "#006300"

QUINTILE_BLUES = ["#86b6ef", "#5f9de8", "#3987e5", "#2668bb", "#184f95"]


def _base_layout(**overrides: object) -> go.Layout:
    """Shared dark-theme layout, as a starting point for every figure."""
    layout = go.Layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=INK_MUTED, size=13),
        legend=dict(font=dict(color=INK_MUTED), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=50, r=30, t=50, b=40),
        hoverlabel=dict(bgcolor="#262624", font=dict(color=INK_PRIMARY)),
    )
    layout.update(**overrides)
    return layout


def _style_axes(fig: go.Figure, **kwargs: object) -> None:
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS_LINE, zerolinecolor=AXIS_LINE, **kwargs)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS_LINE, zerolinecolor=AXIS_LINE, **kwargs)


# ---------------------------------------------------------------------------
# Hero row
# ---------------------------------------------------------------------------


def gauge_chart(composite: float, label: str) -> go.Figure:
    """Composite score gauge (0-100) with colored rating bands and a needle."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=composite,
            number=dict(suffix="", font=dict(color=INK_PRIMARY, size=40)),
            title=dict(text=f"Composite — {label}", font=dict(color=INK_PRIMARY, size=16)),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor=INK_MUTED, tickfont=dict(color=INK_MUTED)),
                bar=dict(color=BLUE, thickness=0.28),
                bgcolor=BG,
                borderwidth=0,
                steps=[
                    dict(range=[0, 20], color=BAND_STRONG_SELL),
                    dict(range=[20, 40], color=BAND_SELL),
                    dict(range=[40, 60], color=BAND_HOLD),
                    dict(range=[60, 80], color=BAND_BUY),
                    dict(range=[80, 100], color=BAND_STRONG_BUY),
                ],
            ),
        )
    )
    fig.update_layout(_base_layout(height=280))
    return fig


def radar_chart(pillars: list[PillarScore]) -> go.Figure:
    """Single-trace radar of pillar scores (missing pillars plotted at 0)."""
    # Break long pillar names at the ampersand so they fit inside the
    # polar chart's margins instead of clipping at the plot edge.
    names = [p.name.replace(" & ", " &<br>") for p in pillars]
    values = [p.score if p.score is not None else 0.0 for p in pillars]
    # Close the polygon.
    theta = [*names, names[0]]
    r = [*values, values[0]]

    fig = go.Figure(
        go.Scatterpolar(
            r=r,
            theta=theta,
            fill="toself",
            line=dict(color=BLUE, width=2),
            fillcolor="rgba(57,135,229,0.25)",
            hovertemplate="%{theta}: %{r:.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        _base_layout(height=280, showlegend=False),
        polar=dict(
            bgcolor=BG,
            radialaxis=dict(
                range=[0, 100], gridcolor=GRID, linecolor=AXIS_LINE, color=INK_MUTED
            ),
            angularaxis=dict(gridcolor=GRID, linecolor=AXIS_LINE, color=INK_MUTED),
        ),
        margin=dict(l=80, r=80, t=30, b=30),
    )
    return fig


# ---------------------------------------------------------------------------
# Technical chart
# ---------------------------------------------------------------------------


def technical_chart(history: pd.DataFrame) -> go.Figure:
    """3-row shared-x subplot: candles+SMA+Bollinger, RSI, MACD."""
    close = history["Close"]
    sma50 = sma(close, 50)
    sma200 = sma(close, 200)
    boll = bollinger(close)
    rsi14 = rsi(close, 14)
    macd_df = macd(close)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2],
        vertical_spacing=0.03,
    )

    fig.add_trace(
        go.Candlestick(
            x=history.index,
            open=history["Open"],
            high=history["High"],
            low=history["Low"],
            close=close,
            name="Price",
            increasing_line_color=GOOD,
            decreasing_line_color=CRITICAL,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=boll["upper"],
            name="Bollinger upper",
            line=dict(color="rgba(57,135,229,0.5)", width=1),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=boll["lower"],
            name="Bollinger band",
            line=dict(color="rgba(57,135,229,0.5)", width=1),
            fill="tonexty",
            fillcolor="rgba(57,135,229,0.12)",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=history.index, y=sma50, name="SMA50", line=dict(color=YELLOW, width=2)
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=history.index, y=sma200, name="SMA200", line=dict(color=VIOLET, width=2)
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=history.index, y=rsi14, name="RSI(14)", line=dict(color=BLUE, width=2)
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=70, line=dict(color=GRAY, width=1, dash="dash"), row=2, col=1)
    fig.add_hline(y=30, line=dict(color=GRAY, width=1, dash="dash"), row=2, col=1)

    hist_colors = [GOOD if v >= 0 else CRITICAL for v in macd_df["histogram"].fillna(0)]
    fig.add_trace(
        go.Bar(
            x=history.index,
            y=macd_df["histogram"],
            name="MACD histogram",
            marker_color=hist_colors,
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=history.index, y=macd_df["macd"], name="MACD", line=dict(color=BLUE, width=2)
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=macd_df["signal"],
            name="Signal",
            line=dict(color=YELLOW, width=2),
        ),
        row=3,
        col=1,
    )

    fig.update_layout(
        _base_layout(height=750, hovermode="x unified"),
        xaxis_rangeslider_visible=False,
        xaxis3=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all", label="all"),
                ],
                bgcolor=BG,
                activecolor=BLUE,
                font=dict(color=INK_MUTED),
            )
        ),
    )
    fig.update_yaxes(title_text="Price ($)", tickprefix="$", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    _style_axes(fig)
    return fig


# ---------------------------------------------------------------------------
# Fundamentals
# ---------------------------------------------------------------------------


def revenue_bar_chart(values: list[float], title: str, newest_first: bool = True) -> go.Figure:
    """Bar chart of a revenue series (annual or quarterly), oldest to newest, left to right."""
    ordered = list(reversed(values)) if newest_first else list(values)
    labels = [f"P-{len(ordered) - 1 - i}" for i in range(len(ordered))]
    if labels:
        labels[-1] = "Latest"

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=ordered,
            marker_color=BLUE,
            hovertemplate="%{x}: $%{y:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        _base_layout(
            height=300,
            title=dict(text=title, font=dict(color=INK_PRIMARY, size=16)),
            showlegend=False,
        )
    )
    fig.update_yaxes(title_text="USD", tickprefix="$")
    _style_axes(fig)
    return fig


# ---------------------------------------------------------------------------
# Analysts
# ---------------------------------------------------------------------------


def analyst_rating_bar(
    strong_buy: int, buy: int, hold: int, sell: int, strong_sell: int
) -> go.Figure:
    """Single stacked horizontal bar of analyst rating counts."""
    categories = ["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"]
    counts = [strong_buy, buy, hold, sell, strong_sell]
    colors = [BAND_STRONG_BUY, BAND_BUY, BAND_HOLD, BAND_SELL, BAND_STRONG_SELL]

    fig = go.Figure()
    for cat, count, color in zip(categories, counts, colors, strict=False):
        fig.add_trace(
            go.Bar(
                y=["Ratings"],
                x=[count],
                name=cat,
                orientation="h",
                marker_color=color,
                text=[str(count) if count else ""],
                textposition="inside",
                textfont=dict(color=INK_PRIMARY),
                hovertemplate=f"{cat}: %{{x}}<extra></extra>",
            )
        )
    fig.update_layout(
        _base_layout(height=180, barmode="stack", legend=dict(orientation="h", y=-0.3)),
    )
    fig.update_xaxes(title_text="Analysts")
    fig.update_yaxes(showticklabels=False)
    _style_axes(fig)
    return fig


def price_target_range_chart(
    current_price: float, low: float | None, mean: float | None, high: float | None
) -> go.Figure:
    """Horizontal dumbbell showing current price against low/mean/high targets."""
    fig = go.Figure()

    if low is not None and high is not None:
        fig.add_trace(
            go.Scatter(
                x=[low, high],
                y=["Target range", "Target range"],
                mode="lines",
                line=dict(color=GRAY, width=4),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    markers_x, markers_labels, markers_colors = [], [], []
    if low is not None:
        markers_x.append(low)
        markers_labels.append("Low")
        markers_colors.append(GRAY)
    if mean is not None:
        markers_x.append(mean)
        markers_labels.append("Mean")
        markers_colors.append(YELLOW)
    if high is not None:
        markers_x.append(high)
        markers_labels.append("High")
        markers_colors.append(GRAY)

    fig.add_trace(
        go.Scatter(
            x=markers_x,
            y=["Target range"] * len(markers_x),
            mode="markers+text",
            marker=dict(size=14, color=markers_colors, line=dict(color=BG, width=1)),
            text=markers_labels,
            textposition="top center",
            textfont=dict(color=INK_MUTED),
            showlegend=False,
            hovertemplate="%{text}: $%{x:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[current_price],
            y=["Target range"],
            mode="markers+text",
            marker=dict(size=18, color=BLUE, symbol="diamond", line=dict(color=BG, width=1)),
            text=["Current"],
            textposition="bottom center",
            textfont=dict(color=INK_PRIMARY),
            name="Current price",
            hovertemplate="Current: $%{x:.2f}<extra></extra>",
        )
    )

    fig.update_layout(_base_layout(height=220, showlegend=False))
    fig.update_xaxes(title_text="Price ($)", tickprefix="$")
    fig.update_yaxes(showticklabels=False)
    _style_axes(fig)
    return fig


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------


def forecast_chart(
    history: pd.DataFrame, forecast: ForecastResult, lookback_bars: int = 120
) -> go.Figure:
    """Recent history plus forecast mean (dashed) and CI band."""
    recent = history.tail(lookback_bars)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recent.index,
            y=recent["Close"],
            name="History",
            line=dict(color=BLUE, width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(forecast.dates) + list(forecast.dates[::-1]),
            y=list(forecast.upper) + list(forecast.lower[::-1]),
            fill="toself",
            fillcolor="rgba(25,158,112,0.18)",
            line=dict(color="rgba(0,0,0,0)"),
            name="80% CI",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast.dates,
            y=forecast.mean,
            name="Forecast mean",
            line=dict(color=AQUA, width=2, dash="dash"),
        )
    )

    fig.update_layout(_base_layout(height=420, hovermode="x unified"))
    fig.update_yaxes(title_text="Price ($)", tickprefix="$")
    _style_axes(fig)
    return fig


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------


def equity_curve_chart(result: BacktestResult) -> go.Figure:
    """Strategy vs buy-and-hold equity curves."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=result.strategy_equity.index,
            y=result.strategy_equity,
            name="Strategy",
            line=dict(color=BLUE, width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=result.buyhold_equity.index,
            y=result.buyhold_equity,
            name="Buy & hold",
            line=dict(color=GRAY, width=2),
        )
    )
    fig.update_layout(_base_layout(height=380, hovermode="x unified"))
    fig.update_yaxes(title_text="Growth of $1")
    _style_axes(fig)
    return fig


def quintile_return_chart(quintile_returns: pd.Series) -> go.Figure:
    """Mean forward return by score quintile, ascending-blue sequential fill."""
    n = len(quintile_returns)
    colors = QUINTILE_BLUES[:n] if n <= len(QUINTILE_BLUES) else QUINTILE_BLUES * (
        n // len(QUINTILE_BLUES) + 1
    )
    fig = go.Figure(
        go.Bar(
            x=list(quintile_returns.index),
            y=(quintile_returns * 100),
            marker_color=colors[:n],
            hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(_base_layout(height=320, showlegend=False))
    fig.update_yaxes(title_text="Mean forward return (%)")
    _style_axes(fig)
    return fig
