"""Tests for the statistical validation utilities.

The load-bearing property is *calibration*: on data constructed to contain
no signal, the tests must not report significance beyond their nominal
false-positive rate; on data constructed with real signal, they must find
it. Both directions are exercised with seeded synthetic series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockanalysis.validation import (
    block_bootstrap_sharpe,
    monte_carlo_random_entry,
    permutation_test_hit_rate,
    permutation_test_ic,
)


@pytest.fixture
def index_400() -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-01", periods=400)


@pytest.fixture
def forward_returns(index_400: pd.DatetimeIndex) -> pd.Series:
    rng = np.random.default_rng(1)
    return pd.Series(rng.normal(0.002, 0.02, 400), index=index_400)


def test_ic_detects_informative_signal(index_400, forward_returns):
    rng = np.random.default_rng(2)
    scores = (50 + 800 * forward_returns + rng.normal(0, 8, 400)).clip(0, 100)
    result = permutation_test_ic(scores, forward_returns, n_permutations=300, seed=3)
    assert result is not None
    assert result.observed > 0.5
    assert result.p_value < 0.05
    assert result.significant


def test_ic_rejects_noise_signal(index_400, forward_returns):
    rng = np.random.default_rng(4)
    scores = pd.Series(rng.uniform(0, 100, 400), index=index_400)
    result = permutation_test_ic(scores, forward_returns, n_permutations=300, seed=3)
    assert result is not None
    assert abs(result.observed) < 0.15
    assert result.p_value > 0.05


def test_ic_rotation_null_calibrated_on_overlapping_windows():
    """Autocorrelated (overlapping-window) returns must not inflate significance."""
    rng = np.random.default_rng(5)
    n = 300
    daily = rng.normal(0, 0.01, n + 21)
    # 21-day overlapping forward returns sampled every bar: heavy overlap.
    fwd = pd.Series(
        [np.prod(1 + daily[i : i + 21]) - 1 for i in range(n)],
        index=pd.bdate_range("2023-01-02", periods=n),
    )
    false_positives = 0
    trials = 20
    for seed in range(trials):
        noise_scores = pd.Series(
            np.random.default_rng(100 + seed).uniform(0, 100, n), index=fwd.index
        )
        result = permutation_test_ic(noise_scores, fwd, n_permutations=200, seed=seed)
        assert result is not None
        false_positives += result.p_value < 0.05
    # Nominal rate is 5%; allow generous headroom for 20 trials.
    assert false_positives <= 3


def test_ic_too_few_observations_returns_none(index_400):
    scores = pd.Series([50.0] * 5, index=index_400[:5])
    rets = pd.Series([0.01] * 5, index=index_400[:5])
    assert permutation_test_ic(scores, rets) is None


def test_ic_unknown_method_raises(index_400, forward_returns):
    scores = pd.Series(np.linspace(0, 100, 400), index=index_400)
    with pytest.raises(ValueError, match="Unknown method"):
        permutation_test_ic(scores, forward_returns, method="bogus")


def test_hit_rate_detects_informative_signal(index_400, forward_returns):
    scores = pd.Series(np.where(forward_returns > 0, 80.0, 20.0), index=index_400)
    result = permutation_test_hit_rate(scores, forward_returns, n_permutations=300, seed=6)
    assert result is not None
    assert result.observed == 1.0
    assert result.p_value < 0.05


def test_hit_rate_too_few_bullish_returns_none(index_400, forward_returns):
    scores = pd.Series(30.0, index=index_400)  # never crosses the bullish threshold
    assert permutation_test_hit_rate(scores, forward_returns) is None


def test_monte_carlo_clairvoyant_beats_null(index_400):
    rng = np.random.default_rng(7)
    daily = pd.Series(rng.normal(0.0005, 0.015, 400), index=index_400)
    clairvoyant = (daily > 0).astype(float)
    result = monte_carlo_random_entry(daily, clairvoyant, n_trials=300, seed=8)
    assert result is not None
    assert result.percentile > 99.0


def test_monte_carlo_random_position_is_unremarkable(index_400):
    rng = np.random.default_rng(9)
    daily = pd.Series(rng.normal(0.0005, 0.015, 400), index=index_400)
    percentiles = []
    for seed in range(10):
        pos = pd.Series(
            (np.random.default_rng(200 + seed).uniform(size=400) > 0.5).astype(float),
            index=index_400,
        )
        result = monte_carlo_random_entry(daily, pos, n_trials=300, seed=8)
        assert result is not None
        percentiles.append(result.percentile)
    assert 20 < float(np.mean(percentiles)) < 80


def test_monte_carlo_all_flat_returns_none(index_400):
    daily = pd.Series(0.01, index=index_400)
    pos = pd.Series(0.0, index=index_400)
    assert monte_carlo_random_entry(daily, pos) is None


def test_block_bootstrap_brackets_observed(index_400):
    rng = np.random.default_rng(10)
    daily = pd.Series(rng.normal(0.001, 0.012, 400), index=index_400)
    result = block_bootstrap_sharpe(daily, n_trials=300, seed=11)
    assert result is not None
    assert result.p5 < result.p50 < result.p95
    assert result.p5 < result.observed < result.p95


def test_block_bootstrap_short_series_returns_none():
    daily = pd.Series([0.01] * 20, index=pd.bdate_range("2024-01-01", periods=20))
    assert block_bootstrap_sharpe(daily) is None
