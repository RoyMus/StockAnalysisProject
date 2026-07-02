"""Pytest fixtures for testing the stock analysis library."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def no_network_calls(monkeypatch):
    """Prevent any network calls in tests by monkeypatching yfinance.Ticker."""
    def raise_on_network(*args, **kwargs):
        raise AssertionError("Network call in tests! Use fixtures instead.")

    import yfinance
    monkeypatch.setattr(yfinance, "Ticker", raise_on_network)


@pytest.fixture
def synthetic_history():
    """300-bar OHLCV DataFrame with geometric random walk and mild upward drift.

    Seeded for reproducibility. Columns: Open, High, Low, Close, Volume.
    Index: pd.bdate_range (business dates).
    """
    rng = np.random.default_rng(42)
    n_bars = 300

    # Geometric random walk with mild upward drift
    daily_returns = rng.normal(0.0003, 0.015, n_bars)  # ~0.03% drift, 1.5% vol
    close = 100.0 * np.exp(np.cumsum(daily_returns))

    # Generate OHLC from close
    open_price = close * (1 + rng.normal(0, 0.005, n_bars))
    high_price = np.maximum(open_price, close) * (1 + np.abs(rng.normal(0, 0.008, n_bars)))
    low_price = np.minimum(open_price, close) * (1 - np.abs(rng.normal(0, 0.008, n_bars)))

    # Volume with some autocorrelation
    base_vol = 1_000_000 * (1 + rng.normal(0, 0.2, n_bars))
    volume = np.maximum(base_vol, 100_000)

    dates = pd.bdate_range(end=pd.Timestamp.now(), periods=n_bars)

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


@pytest.fixture
def short_history():
    """30-bar OHLCV DataFrame for testing minimum bar requirements."""
    rng = np.random.default_rng(123)
    n_bars = 30

    daily_returns = rng.normal(0.0002, 0.012, n_bars)
    close = 100.0 * np.exp(np.cumsum(daily_returns))

    open_price = close * (1 + rng.normal(0, 0.005, n_bars))
    high_price = np.maximum(open_price, close) * (1 + np.abs(rng.normal(0, 0.008, n_bars)))
    low_price = np.minimum(open_price, close) * (1 - np.abs(rng.normal(0, 0.008, n_bars)))
    volume = 1_000_000 * (1 + rng.normal(0, 0.15, n_bars))

    dates = pd.bdate_range(end=pd.Timestamp.now(), periods=n_bars)

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


@pytest.fixture
def trending_history():
    """400-bar OHLCV DataFrame with strong steady uptrend (daily drift 0.3%, low noise)."""
    rng = np.random.default_rng(456)
    n_bars = 400

    # Strong upward drift with low noise
    daily_returns = rng.normal(0.003, 0.005, n_bars)  # 0.3% drift, 0.5% vol
    close = 100.0 * np.exp(np.cumsum(daily_returns))

    open_price = close * (1 + rng.normal(0, 0.002, n_bars))
    high_price = np.maximum(open_price, close) * (1 + np.abs(rng.normal(0, 0.003, n_bars)))
    low_price = np.minimum(open_price, close) * (1 - np.abs(rng.normal(0, 0.003, n_bars)))
    volume = 1_000_000 * (1 + rng.normal(0, 0.1, n_bars))

    dates = pd.bdate_range(end=pd.Timestamp.now(), periods=n_bars)

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


@pytest.fixture
def ranging_history():
    """400-bar OHLCV DataFrame with sine-wave oscillation around flat mean plus small noise."""
    rng = np.random.default_rng(789)
    n_bars = 400

    # Sine-wave oscillation + small noise (ranging behavior)
    t = np.linspace(0, 8 * np.pi, n_bars)  # 4 full cycles
    sine_component = 5.0 * np.sin(t)  # Oscillates ±5 around mean
    noise = rng.normal(0, 0.5, n_bars)  # Small random noise
    close = 100.0 + sine_component + noise

    open_price = close * (1 + rng.normal(0, 0.002, n_bars))
    high_price = np.maximum(open_price, close) * (1 + np.abs(rng.normal(0, 0.003, n_bars)))
    low_price = np.minimum(open_price, close) * (1 - np.abs(rng.normal(0, 0.003, n_bars)))
    volume = 1_000_000 * (1 + rng.normal(0, 0.1, n_bars))

    dates = pd.bdate_range(end=pd.Timestamp.now(), periods=n_bars)

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
