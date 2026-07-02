"""Close-only signal variants, validated against real historical data.

These are pure functions of a Close price series — the subset of the
technical toolkit that works when High/Low/Volume are unavailable, and the
variants exercised by the dev/holdout study documented in
``docs/SIGNAL_FINDINGS.md``.

Headline finding from that study (20 real S&P 500 names, 2005-2022 daily,
21-day horizon, pre-registered dev/holdout split): ``reversion_score``
carried genuine predictive information (dev pooled IC +0.048 p=0.045;
holdout confirmation +0.059 p=0.004), ``momentum_score`` anti-predicted
(dev −0.045), and the balanced blend cancelled to zero. See the findings
doc for scope and caveats before treating this as gospel — large caps
only, close-only data, and one era of market history.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockanalysis.indicators import bollinger, macd, rsi, sma, zscore
from stockanalysis.types import clamp, linear_scale


def momentum_score(close: pd.Series) -> float | None:
    """Trend-following read from Close alone: SMA alignment, MACD, RSI.

    Mirrors the momentum-leaning sub-signals of the technicals pillar,
    minus the volume and 52-week components that need more than Close.
    """
    if len(close) < 210:
        return None
    price = float(close.iloc[-1])
    sma50 = float(sma(close, 50).iloc[-1])
    sma200 = float(sma(close, 200).iloc[-1])
    if any(np.isnan(v) for v in (sma50, sma200)):
        return None
    if price > sma50 > sma200:
        trend = 90.0
    elif price > sma50:
        trend = 65.0
    elif sma50 > sma200:
        trend = 40.0
    else:
        trend = 10.0

    m = macd(close)
    macd_now, sig_now = float(m["macd"].iloc[-1]), float(m["signal"].iloc[-1])
    hist_now, hist_prev = float(m["histogram"].iloc[-1]), float(m["histogram"].iloc[-2])
    if macd_now > sig_now:
        macd_score = 80.0 if hist_now > hist_prev else 62.0
    else:
        macd_score = 38.0 if hist_now > hist_prev else 20.0

    r = float(rsi(close).iloc[-1])
    if r < 30:
        rsi_score = 35.0
    elif r <= 50:
        rsi_score = 35 + 0.15 * linear_scale(r, 30, 50)
    elif r <= 70:
        rsi_score = 50 + 0.35 * linear_scale(r, 50, 70)
    elif r <= 85:
        rsi_score = 85 - 0.25 * linear_scale(r, 70, 85)
    else:
        rsi_score = 50.0

    return clamp((3 * trend + 2.5 * macd_score + 2 * rsi_score) / 7.5)


def reversion_score(close: pd.Series) -> float | None:
    """Mean-reversion read from Close alone: Bollinger %B, z-score, RSI extremes.

    The variant that survived holdout confirmation in the signal study —
    stretched-low readings score high (expected bounce), stretched-high
    readings score low.
    """
    if len(close) < 60:
        return None
    pb = float(bollinger(close)["percent_b"].iloc[-1])
    z = float(zscore(close, 50).iloc[-1])
    r = float(rsi(close).iloc[-1])
    if any(np.isnan(v) for v in (pb, z, r)):
        return None

    pb_score = 85 - 70 * clamp(pb * 100) / 100  # %B 0 -> 85, 1 -> 15
    z_score_val = 85 - 70 * linear_scale(z, -2, 2) / 100  # z -2 -> 85, +2 -> 15
    if r < 30:
        rsi_score = 80.0
    elif r <= 70:
        rsi_score = 65 - 30 * linear_scale(r, 30, 70) / 100
    else:
        rsi_score = 20.0

    return clamp((4 * pb_score + 3 * z_score_val + 3 * rsi_score) / 10)


def balanced_score(close: pd.Series) -> float | None:
    """50/50 momentum/reversion blend — kept as the null-hypothesis baseline.

    The study measured this blend at exactly zero IC (the two halves
    cancel); it exists so future runs can re-check that finding, not
    because it is recommended.
    """
    parts = [v for v in (momentum_score(close), reversion_score(close)) if v is not None]
    return sum(parts) / len(parts) if parts else None
