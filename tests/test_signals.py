"""Tests for the close-only signal variants."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockanalysis.signals import balanced_score, momentum_score, reversion_score


@pytest.fixture
def uptrend_close() -> pd.Series:
    # Realistic noise level matters: an implausibly smooth uptrend pegs RSI
    # above 85, which momentum_score deliberately tapers back to neutral.
    rng = np.random.default_rng(31)
    n = 400
    return pd.Series(
        100 * np.exp(np.cumsum(rng.normal(0.002, 0.012, n))),
        index=pd.bdate_range("2023-01-02", periods=n),
    )


@pytest.fixture
def crash_close() -> pd.Series:
    """Steady series that just fell hard — stretched low, reversion setup."""
    rng = np.random.default_rng(32)
    n = 300
    flat = 100 + np.cumsum(rng.normal(0, 0.2, n - 10))
    crash = np.linspace(flat[-1], flat[-1] * 0.80, 10)
    return pd.Series(
        np.concatenate([flat, crash]), index=pd.bdate_range("2023-01-02", periods=n)
    )


def test_scores_bounded(uptrend_close, crash_close):
    for series in (uptrend_close, crash_close):
        for fn in (momentum_score, reversion_score, balanced_score):
            score = fn(series)
            assert score is not None
            assert 0 <= score <= 100


def test_short_series_returns_none():
    short = pd.Series([100.0] * 30, index=pd.bdate_range("2024-01-01", periods=30))
    assert momentum_score(short) is None
    assert reversion_score(short) is None
    assert balanced_score(short) is None  # both parts unavailable


def test_momentum_high_in_uptrend_low_after_crash(uptrend_close, crash_close):
    assert momentum_score(uptrend_close) > 60
    assert momentum_score(crash_close) < 40


def test_reversion_high_after_crash_low_when_stretched_up(uptrend_close, crash_close):
    assert reversion_score(crash_close) > 60
    assert reversion_score(uptrend_close) < 50


def test_balanced_is_mean_of_parts(uptrend_close):
    m = momentum_score(uptrend_close)
    r = reversion_score(uptrend_close)
    assert balanced_score(uptrend_close) == pytest.approx((m + r) / 2)
