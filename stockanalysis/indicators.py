"""Technical indicators, hand-rolled with pandas.

Implemented from the standard formulas (Wilder's RSI/ADX, classic
MACD/Bollinger) rather than pulled from a TA library, so every number in
the score breakdown can be traced to a formula in this file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window).mean()


def ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing."""
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.fillna(100.0).where(avg_loss.notna(), np.nan)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Returns columns: macd, signal, histogram."""
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "histogram": macd_line - signal_line}
    )


def bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Returns columns: middle, upper, lower, percent_b, width."""
    middle = sma(close, window)
    std = close.rolling(window).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    band_range = (upper - lower).replace(0.0, np.nan)
    return pd.DataFrame(
        {
            "middle": middle,
            "upper": upper,
            "lower": lower,
            "percent_b": (close - lower) / band_range,
            "width": band_range / middle,
        }
    )


def zscore(close: pd.Series, window: int = 50) -> pd.Series:
    """How many rolling standard deviations price sits from its mean."""
    mean = close.rolling(window).mean()
    std = close.rolling(window).std().replace(0.0, np.nan)
    return (close - mean) / std


def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average Directional Index (Wilder). Values > 25 indicate a trending market."""
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    prev_close = close.shift()
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    alpha = 1 / window
    atr = true_range.ewm(alpha=alpha, adjust=False).mean().replace(0.0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr
    di_sum = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    return dx.ewm(alpha=alpha, adjust=False).mean()


def ma_slope(close: pd.Series, window: int = 50, lookback: int = 10) -> float | None:
    """Annualized-ish slope of the SMA over the last `lookback` bars,
    normalized by price so it's comparable across tickers (fraction/bar)."""
    ma = sma(close, window).dropna()
    if len(ma) < lookback:
        return None
    recent = ma.iloc[-lookback:]
    slope = np.polyfit(np.arange(lookback), recent.to_numpy(), 1)[0]
    last_price = float(close.iloc[-1])
    return float(slope / last_price) if last_price else None


def volume_trend(volume: pd.Series, short: int = 20, long: int = 60) -> float | None:
    """Ratio of recent average volume to longer-run average volume."""
    if len(volume.dropna()) < long:
        return None
    short_avg = float(volume.rolling(short).mean().iloc[-1])
    long_avg = float(volume.rolling(long).mean().iloc[-1])
    return short_avg / long_avg if long_avg else None
