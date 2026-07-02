"""Tests for the point-in-time backtester.

Uses constant / clairvoyant signal functions to pin down the strategy
mechanics exactly (a constant signal has known positions, so equity and
costs are computable by hand), plus structural checks on the default
combined signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockanalysis.backtest import backtest_signal, summary_table


@pytest.fixture(scope="module")
def history_600() -> pd.DataFrame:
    rng = np.random.default_rng(21)
    n = 600
    close = 100 * np.exp(np.cumsum(rng.normal(0.0006, 0.015, n)))
    return pd.DataFrame(
        {
            "Open": close * (1 + rng.normal(0, 0.003, n)),
            "High": close * (1 + np.abs(rng.normal(0, 0.008, n))),
            "Low": close * (1 - np.abs(rng.normal(0, 0.008, n))),
            "Close": close,
            "Volume": rng.integers(int(1e6), int(5e6), n).astype(float),
        },
        index=pd.bdate_range("2023-01-02", periods=n),
    )


def test_too_short_history_returns_none(history_600):
    assert backtest_signal(history_600.iloc[:200]) is None


def test_default_signal_structure(history_600):
    result = backtest_signal(history_600)
    assert result is not None
    assert result.n_signals > 0
    assert ((result.scores >= 0) & (result.scores <= 100)).all()
    assert len(result.scores) == len(result.forward_returns)
    assert set(result.position.unique()) <= {0.0, 1.0}
    assert result.strategy_equity.iloc[0] == pytest.approx(1.0, abs=0.01)
    assert result.buyhold_equity.iloc[0] == pytest.approx(1.0, abs=0.01)
    assert len(result.strategy_equity) == len(result.buyhold_equity)


def test_always_flat_signal_stays_at_one(history_600):
    """A signal pinned at 0 never enters, so equity stays exactly 1.0."""
    result = backtest_signal(history_600, signal_fn=lambda df: 0.0)
    assert result is not None
    assert (result.position == 0.0).all()
    assert (result.strategy_equity == 1.0).all()
    assert result.strategy_return == 0.0
    assert result.strategy_max_drawdown == 0.0


def test_always_long_signal_tracks_buyhold_minus_entry_cost(history_600):
    """A signal pinned at 100 is long throughout; equity differs from
    buy-and-hold only by the single entry cost and the one-bar entry lag."""
    result = backtest_signal(history_600, signal_fn=lambda df: 100.0, cost_bps=0.0)
    assert result is not None
    assert result.position.iloc[5:].eq(1.0).all()
    # With zero costs and full-time exposure (after the first bar), total
    # return matches buy-and-hold computed from the same window.
    assert result.strategy_return == pytest.approx(result.buyhold_return, rel=1e-9)


def test_costs_reduce_returns(history_600):
    free = backtest_signal(history_600, cost_bps=0.0)
    costly = backtest_signal(history_600, cost_bps=50.0)
    assert free is not None and costly is not None
    # Same positions, so returns can only go down with costs (strictly, if
    # any trade happened).
    assert costly.strategy_return <= free.strategy_return
    if free.position.diff().abs().sum() > 0:
        assert costly.strategy_return < free.strategy_return


def test_signal_fn_receives_prefix_only(history_600):
    """The backtester must never show the signal future bars."""
    seen_lengths: list[int] = []

    def probe(df: pd.DataFrame) -> float:
        seen_lengths.append(len(df))
        return 50.0

    backtest_signal(history_600, horizon_days=21, step=10, warmup=220, signal_fn=probe)
    assert seen_lengths
    # Slices grow by `step` and never reach past len - horizon.
    assert seen_lengths == sorted(seen_lengths)
    assert max(seen_lengths) <= len(history_600) - 21
    assert min(seen_lengths) == 221  # warmup bar + 1 (prefix is iloc[:i+1])


def test_summary_table_renders(history_600):
    result = backtest_signal(history_600)
    table = summary_table(result)
    assert list(table.columns) == ["Metric", "Value"]
    assert len(table) >= 10
