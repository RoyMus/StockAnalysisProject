"""Trend & mean-reversion pillar.

Detects whether a ticker is currently trending, ranging, or in-between
(via ADX) and blends a trend-following read with a mean-reversion read
accordingly. Both reads are always computed and both sets of sub-signals
are always emitted (with weights scaled by the regime blend) so the UI can
show the full breakdown rather than just the winning branch.

Works on any prefix slice of ``history`` (no lookahead) so it can be
replayed point-in-time by a backtester.
"""

from __future__ import annotations

import pandas as pd

from stockanalysis.indicators import adx, bollinger, ma_slope, rsi, zscore
from stockanalysis.types import PillarScore, Signal, linear_scale, weighted_signal_score

_PILLAR_NAME = "Trend & Mean-Reversion"


def _rescale(value: float, x0: float, x1: float, y0: float, y1: float) -> float:
    """Map ``value`` from the [x0, x1] domain onto the [y0, y1] range."""
    return y0 + (y1 - y0) * linear_scale(value, x0, x1) / 100.0


def score_trend_reversion(history: pd.DataFrame) -> PillarScore:
    """Blend trend-following and mean-reversion reads based on ADX regime."""
    if len(history) < 60:
        return PillarScore(name=_PILLAR_NAME, score=None, note="Insufficient price history")

    close, high, low = history["Close"], history["High"], history["Low"]

    adx_last = adx(high, low, close).iloc[-1]
    slope = ma_slope(close, window=50, lookback=10)
    rsi_last = rsi(close).iloc[-1]
    boll = bollinger(close)
    percent_b = boll["percent_b"].iloc[-1]
    z_last = zscore(close, window=50).iloc[-1]

    if any(pd.isna(v) for v in (adx_last, rsi_last, percent_b, z_last)) or slope is None:
        return PillarScore(name=_PILLAR_NAME, score=None, note="Insufficient price history")

    adx_last, rsi_last, percent_b, z_last = (
        float(adx_last),
        float(rsi_last),
        float(percent_b),
        float(z_last),
    )

    if adx_last > 25:
        regime = "trending"
        w_trend, w_revert = 0.8, 0.2
    elif adx_last < 20:
        regime = "ranging"
        w_trend, w_revert = 0.2, 0.8
    else:
        regime = "transitional"
        w_trend, w_revert = 0.5, 0.5

    # Trend-following branch.
    slope_score = _rescale(slope, -0.003, 0.003, 10, 90)
    adx_conviction_score = linear_scale(adx_last, 15, 40)
    rsi_continuation_score = linear_scale(rsi_last, 35, 70)

    # Mean-reversion branch.
    percent_b_score = _rescale(percent_b, 0, 1, 85, 15)
    zscore_score = _rescale(z_last, -2, 2, 85, 15)
    if rsi_last < 30:
        rsi_extreme_score = 80.0
    elif rsi_last <= 70:
        rsi_extreme_score = _rescale(rsi_last, 30, 70, 65, 35)
    else:
        rsi_extreme_score = 20.0

    signals = [
        Signal(
            name="SMA50 slope (trend)",
            value=f"SMA50 slope {slope * 100:+.3f}%/bar",
            score=slope_score,
            weight=4.0 * w_trend,
        ),
        Signal(
            name="ADX conviction (trend)",
            value=f"ADX {adx_last:.1f}",
            score=adx_conviction_score,
            weight=3.0 * w_trend,
        ),
        Signal(
            name="RSI continuation (trend)",
            value=f"RSI {rsi_last:.1f}",
            score=rsi_continuation_score,
            weight=3.0 * w_trend,
        ),
        Signal(
            name="Bollinger %B (mean-reversion)",
            value=f"%B {percent_b:.2f}",
            score=percent_b_score,
            weight=4.0 * w_revert,
        ),
        Signal(
            name="Z-score (mean-reversion)",
            value=f"Z-score {z_last:+.2f} vs 50-bar mean",
            score=zscore_score,
            weight=3.0 * w_revert,
        ),
        Signal(
            name="RSI extremes (mean-reversion)",
            value=f"RSI {rsi_last:.1f}",
            score=rsi_extreme_score,
            weight=3.0 * w_revert,
        ),
    ]

    note = f"Regime: {regime} (ADX {adx_last:.1f})"
    return PillarScore(
        name=_PILLAR_NAME, score=weighted_signal_score(signals), signals=signals, note=note
    )
