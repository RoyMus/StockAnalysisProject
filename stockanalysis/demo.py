"""Deterministic synthetic data for the "DEMO" ticker.

This machine (and, more importantly, Streamlit Cloud's free tier under
heavy traffic) can't always reach Yahoo Finance. Entering ticker ``DEMO``
(case-insensitive) in the app bypasses ``stockanalysis.data`` entirely and
builds a hand-crafted, seeded bundle instead, so the full UI — charts,
scoring, forecast, backtest — can be exercised offline and produces the
exact same numbers on every run.

None of this touches yfinance or the network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockanalysis.data import AnalystData, CompanyInfo, FinancialHistory

DEMO_TICKER = "DEMO"

_SEED = 20260702  # arbitrary, fixed — determinism is the point


def _regime_segment(
    rng: np.random.Generator,
    start_price: float,
    n_bars: int,
    drift: float,
    vol: float,
) -> np.ndarray:
    """Geometric random walk segment starting at ``start_price``."""
    daily_returns = rng.normal(drift, vol, n_bars)
    return start_price * np.exp(np.cumsum(daily_returns))


def _ranging_segment(
    rng: np.random.Generator, start_price: float, n_bars: int, amplitude: float
) -> np.ndarray:
    """Sine-wave oscillation around ``start_price`` — a non-trending regime."""
    t = np.linspace(0, 4 * np.pi, n_bars)
    sine_component = amplitude * np.sin(t)
    noise = rng.normal(0, amplitude * 0.1, n_bars)
    return start_price + sine_component + noise


def _build_history(rng: np.random.Generator) -> pd.DataFrame:
    """~750-bar OHLCV frame with distinct trend/range/drawdown/recovery regimes.

    Segment lengths and regimes are chosen so the ADX-based regime detector
    in ``stockanalysis.pillars.regime`` sees genuine trending and ranging
    stretches, rather than one long homogeneous random walk.
    """
    segments: list[np.ndarray] = []
    last_price = 100.0

    uptrend_1 = _regime_segment(rng, last_price, 200, drift=0.0015, vol=0.010)
    segments.append(uptrend_1)
    last_price = float(uptrend_1[-1])

    ranging = _ranging_segment(rng, last_price, 150, amplitude=last_price * 0.05)
    segments.append(ranging)
    last_price = float(ranging[-1])

    drawdown = _regime_segment(rng, last_price, 200, drift=-0.0008, vol=0.014)
    segments.append(drawdown)
    last_price = float(drawdown[-1])

    recovery = _regime_segment(rng, last_price, 200, drift=0.0012, vol=0.011)
    segments.append(recovery)

    close = np.concatenate(segments)
    n_bars = len(close)

    open_price = close * (1 + rng.normal(0, 0.005, n_bars))
    high_price = np.maximum(open_price, close) * (1 + np.abs(rng.normal(0, 0.008, n_bars)))
    low_price = np.minimum(open_price, close) * (1 - np.abs(rng.normal(0, 0.008, n_bars)))

    base_vol = 1_500_000 * (1 + rng.normal(0, 0.2, n_bars))
    volume = np.maximum(base_vol, 100_000)

    dates = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=n_bars)

    return pd.DataFrame(
        {
            "Open": open_price,
            "High": high_price,
            "Low": low_price,
            "Close": close,
            "Volume": volume,
        },
        index=dates,
    )


def _build_company_info(history: pd.DataFrame) -> CompanyInfo:
    current_price = float(history["Close"].iloc[-1])
    window = history.tail(min(len(history), 252))
    return CompanyInfo(
        ticker=DEMO_TICKER,
        long_name="Demo Corp (synthetic data)",
        sector="Technology",
        industry="Software - Infrastructure",
        quote_type="EQUITY",
        current_price=current_price,
        market_cap=3.1e11,
        trailing_pe=24.5,
        forward_pe=19.8,
        peg_ratio=1.6,
        price_to_book=5.2,
        ev_to_ebitda=16.1,
        free_cashflow=8e9,
        revenue_growth=0.14,
        earnings_growth=0.18,
        profit_margin=0.21,
        fifty_two_week_low=float(window["Low"].min()),
        fifty_two_week_high=float(window["High"].max()),
    )


def _build_analyst_data(current_price: float) -> AnalystData:
    strong_buy, buy, hold, sell, strong_sell = 12, 18, 6, 1, 0
    trend = pd.DataFrame(
        {
            "strongBuy": [strong_buy, 11, 9, 8],
            "buy": [buy, 17, 18, 17],
            "hold": [hold, 7, 8, 9],
            "sell": [sell, 2, 2, 3],
            "strongSell": [strong_sell, 0, 0, 1],
        },
        index=["0m", "-1m", "-2m", "-3m"],
    )
    return AnalystData(
        strong_buy=strong_buy,
        buy=buy,
        hold=hold,
        sell=sell,
        strong_sell=strong_sell,
        num_analysts=37,
        target_mean=current_price * 1.15,
        target_median=current_price * 1.14,
        target_high=current_price * 1.35,
        target_low=current_price * 0.90,
        trend=trend,
    )


def _build_financial_history() -> FinancialHistory:
    # Newest first. ~12%/yr revenue CAGR with modest margin expansion, which
    # makes earnings grow a bit faster than revenue (mirrors the info-level
    # revenue_growth=0.14 / earnings_growth=0.18 split).
    newest_revenue = 6.0e10
    annual_revenue = [newest_revenue / (1.12**i) for i in range(4)]
    annual_margins = [0.21, 0.20, 0.185, 0.17]
    annual_net_income = [
        rev * margin for rev, margin in zip(annual_revenue, annual_margins, strict=True)
    ]

    # 8 quarters, mostly sequential growth with one seasonal dip.
    quarterly_revenue = [
        15.6e9,
        15.9e9,
        14.8e9,  # seasonal dip
        14.5e9,
        14.1e9,
        13.6e9,
        13.2e9,
        12.7e9,
    ]
    quarterly_margins = [0.215, 0.212, 0.205, 0.208, 0.20, 0.198, 0.192, 0.188]
    quarterly_net_income = [
        rev * margin for rev, margin in zip(quarterly_revenue, quarterly_margins, strict=True)
    ]

    return FinancialHistory(
        annual_revenue=annual_revenue,
        annual_net_income=annual_net_income,
        quarterly_revenue=quarterly_revenue,
        quarterly_net_income=quarterly_net_income,
    )


def get_demo_bundle() -> tuple[pd.DataFrame, CompanyInfo, AnalystData, FinancialHistory]:
    """Build the full synthetic data bundle for the ``DEMO`` ticker.

    Seeded, so every field — prices, ratios, analyst counts, financials —
    is identical across runs. Returns ``(history, info, analysts,
    financials)``, mirroring the four fetchers in ``stockanalysis.data``.
    """
    rng = np.random.default_rng(_SEED)
    history = _build_history(rng)
    info = _build_company_info(history)
    analysts = _build_analyst_data(info.current_price or 100.0)
    financials = _build_financial_history()
    return history, info, analysts, financials


def is_demo_ticker(ticker: str) -> bool:
    """True if ``ticker`` (any case/whitespace) should route to demo data."""
    return ticker.strip().upper() == DEMO_TICKER
