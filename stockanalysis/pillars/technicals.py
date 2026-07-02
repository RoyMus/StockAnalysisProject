"""Technicals pillar: trend alignment, momentum, volume and price positioning.

Every signal here is computed from ``history`` alone and only ever looks at
data up to the last row it is given — the function must behave identically
whether it is called on the full price history or an arbitrary prefix of it,
since a backtester replays it point-in-time.
"""

from __future__ import annotations

import pandas as pd

from stockanalysis.indicators import macd, rsi, sma, volume_trend
from stockanalysis.types import PillarScore, Signal, linear_scale, weighted_signal_score


def _rescale(value: float, x0: float, x1: float, y0: float, y1: float) -> float:
    """Map ``value`` from the [x0, x1] domain onto the [y0, y1] range.

    Built on top of ``linear_scale`` (which always targets 0-100) so callers
    can express sub-ranges like "50 -> 85" without a second scaling helper.
    """
    return y0 + (y1 - y0) * linear_scale(value, x0, x1) / 100.0


def _trend_alignment(history: pd.DataFrame) -> Signal | None:
    bars = len(history)
    if bars < 50:
        return None

    close = history["Close"]
    price = float(close.iloc[-1])
    sma50 = float(sma(close, 50).iloc[-1])

    if bars < 200:
        score = 70.0 if price > sma50 else 30.0
        state = "above" if price > sma50 else "below"
        return Signal(
            name="Trend alignment",
            value=f"Price {price:.2f} {state} SMA50 {sma50:.2f} (<200 bars, SMA200 unavailable)",
            score=score,
            weight=3.0,
        )

    sma200 = float(sma(close, 200).iloc[-1])
    price_above_50 = price > sma50
    fifty_above_200 = sma50 > sma200
    if price_above_50 and fifty_above_200:
        score, desc = 90.0, "bullish alignment (price > SMA50 > SMA200)"
    elif price_above_50 and not fifty_above_200:
        score, desc = 65.0, "price reclaiming SMA50 while SMA50 still below SMA200"
    elif not price_above_50 and fifty_above_200:
        score, desc = 40.0, "price slipping below SMA50 in an uptrend"
    else:
        score, desc = 10.0, "bearish alignment (price < SMA50 < SMA200)"

    return Signal(
        name="Trend alignment",
        value=f"Price {price:.2f}, SMA50 {sma50:.2f}, SMA200 {sma200:.2f} ({desc})",
        score=score,
        weight=3.0,
    )


def _macd_signal(history: pd.DataFrame) -> Signal | None:
    if len(history) < 35:
        return None

    macd_df = macd(history["Close"])
    macd_line = float(macd_df["macd"].iloc[-1])
    signal_line = float(macd_df["signal"].iloc[-1])
    last_hist = float(macd_df["histogram"].iloc[-1])
    prev_hist = float(macd_df["histogram"].iloc[-2])
    if pd.isna(macd_line) or pd.isna(signal_line) or pd.isna(prev_hist):
        return None

    bullish = macd_line > signal_line
    rising = last_hist > prev_hist
    if bullish and rising:
        score = 80.0
    elif bullish and not rising:
        score = 62.0
    elif not bullish and rising:
        score = 38.0
    else:
        score = 20.0

    direction = "above" if bullish else "below"
    momentum = "rising" if rising else "falling"
    return Signal(
        name="MACD",
        value=f"MACD {macd_line:.3f} {direction} signal {signal_line:.3f}, histogram {momentum}",
        score=score,
        weight=2.5,
    )


def _rsi_momentum(history: pd.DataFrame) -> Signal | None:
    if len(history) < 15:
        return None

    value = rsi(history["Close"]).iloc[-1]
    if pd.isna(value):
        return None
    r = float(value)

    if r < 30:
        score = 35.0
    elif r < 50:
        score = _rescale(r, 30, 50, 35, 50)
    elif r < 70:
        score = _rescale(r, 50, 70, 50, 85)
    elif r <= 85:
        score = _rescale(r, 70, 85, 85, 60)
    else:
        score = 50.0

    return Signal(name="RSI momentum", value=f"RSI {r:.1f}", score=score, weight=2.0)


def _volume_trend_signal(history: pd.DataFrame) -> Signal | None:
    if len(history) < 60:
        return None

    ratio = volume_trend(history["Volume"])
    if ratio is None:
        return None

    close = history["Close"]
    prior = float(close.iloc[-21])
    price_change_20 = (float(close.iloc[-1]) - prior) / prior if prior else 0.0

    raw = linear_scale(ratio, 0.7, 1.5)
    score = 100.0 - raw if price_change_20 < 0 else raw

    direction = "declining" if price_change_20 < 0 else "rising"
    return Signal(
        name="Volume trend",
        value=(
            f"20d/60d volume ratio {ratio:.2f}x, price {direction} "
            f"{price_change_20 * 100:+.1f}% over 20d"
        ),
        score=score,
        weight=1.5,
    )


def _fifty_two_week_position(history: pd.DataFrame) -> Signal | None:
    bars = len(history)
    if bars < 120:
        return None

    window = history.tail(min(bars, 252))
    low = float(window["Low"].min())
    high = float(window["High"].max())
    price = float(history["Close"].iloc[-1])

    pos = (price - low) / (high - low) if high > low else 0.5
    score = _rescale(pos, 0, 1, 25, 85)

    return Signal(
        name="52-week position",
        value=f"{pos * 100:.0f}% of {low:.2f}-{high:.2f} range (price {price:.2f})",
        score=score,
        weight=1.0,
    )


def score_technicals(history: pd.DataFrame) -> PillarScore:
    """Score price-action technicals from an OHLCV frame.

    Works on any prefix slice of ``history`` (no lookahead) so it can be
    replayed point-in-time by a backtester.
    """
    signals = [
        s
        for s in (
            _trend_alignment(history),
            _macd_signal(history),
            _rsi_momentum(history),
            _volume_trend_signal(history),
            _fifty_two_week_position(history),
        )
        if s is not None
    ]

    if not signals:
        return PillarScore(name="Technicals", score=None, note="Insufficient price history")

    return PillarScore(name="Technicals", score=weighted_signal_score(signals), signals=signals)
