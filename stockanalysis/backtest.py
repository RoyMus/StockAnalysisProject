"""Point-in-time backtest for the price-based composite signal.

Only two of the five scoring pillars can be honestly backtested:
``score_technicals`` and ``score_trend_reversion``. Both are pure functions
of an OHLCV prefix — call them on ``history.iloc[:i+1]`` for any ``i`` and
they only ever look at data up to bar ``i``, so replaying them at each
historical bar is a faithful simulation of "what would this signal have
said on that day."

The other three pillars (Value, Growth, Analysts) cannot be backtested this
way. ``get_company_info`` / ``get_analyst_data`` / ``get_financial_history``
return yfinance's *current* snapshot only — there is no historical P/E,
analyst rating, or revenue-growth series to replay. Scoring a historical
bar with today's fundamentals would leak the future into the past (e.g.
crediting 2019-era price action with a P/E ratio computed from 2026
earnings), which is a textbook look-ahead bias. So this module deliberately
backtests a *price-only* composite (mean of the two backtestable pillars)
rather than the full five-pillar ``score_stock`` composite. Treat the
results here as a read on "does the price/technical half of the signal
have any predictive power," not a validation of the whole product.

Methodology, in short:
  1. Walk the price history bar by bar (every ``step`` bars, after a
     ``warmup`` period), scoring only the prefix available at that bar.
  2. Pair each score with its realized forward return ``horizon_days``
     later.
  3. Summarize predictiveness via hit rates, quintile spreads, and a
     Spearman-style rank information coefficient (implemented with plain
     pandas ``.rank().corr()`` so this module has no scipy dependency).
  4. Simulate a simple long/flat, no-shorting strategy that enters when the
     score crosses ``entry_threshold`` and exits when it drops below
     ``exit_threshold`` (hysteresis, to avoid whipsawing around a single
     level), and compare its equity curve to buy-and-hold.

This is research tooling, not a production trading system: no transaction
costs, slippage, taxes, or position sizing are modeled.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from stockanalysis.pillars.regime import score_trend_reversion
from stockanalysis.pillars.technicals import score_technicals

# Fixed thresholds for the hit-rate / quintile diagnostics. These are
# independent of the ``entry_threshold`` / ``exit_threshold`` parameters
# used by the strategy simulation, which are tunable.
_BULLISH_HIT_THRESHOLD = 60.0
_BEARISH_HIT_THRESHOLD = 40.0

_TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class BacktestResult:
    """Everything needed to judge (and show) how predictive the signal was."""

    ticker: str
    n_signals: int
    horizon_days: int
    scores: pd.Series
    forward_returns: pd.Series
    hit_rate_bullish: float | None
    hit_rate_bearish: float | None
    n_bullish: int
    n_bearish: int
    quintile_returns: pd.Series
    ic_spearman: float | None
    strategy_equity: pd.Series
    buyhold_equity: pd.Series
    position: pd.Series  # daily 0/1 exposure of the simulated strategy
    strategy_return: float
    buyhold_return: float
    strategy_sharpe: float | None
    buyhold_sharpe: float | None
    strategy_max_drawdown: float
    buyhold_max_drawdown: float


def _combined_score(history: pd.DataFrame) -> float | None:
    """Mean of the two backtestable pillar scores at this prefix, if any."""
    technicals = score_technicals(history).score
    trend_reversion = score_trend_reversion(history).score
    values = [v for v in (technicals, trend_reversion) if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _build_signal_series(
    history: pd.DataFrame,
    horizon_days: int,
    step: int,
    warmup: int,
    signal_fn: Callable[[pd.DataFrame], float | None],
) -> tuple[pd.Series, pd.Series]:
    """Score every ``step``-th bar from ``warmup`` on, paired with its forward return.

    Each iteration slices ``history.iloc[:i + 1]`` — a prefix only, so the
    score is exactly what it would have been if computed live on that day.
    NOTE on cost: ``score_technicals``/``score_trend_reversion`` each
    recompute their indicators over the full prefix, i.e. O(i) work per
    call, so this loop is O(n^2 / step) overall. That's negligible for a
    few hundred/thousand daily bars (the intended use case) but would need
    an incremental-indicator rewrite to scale to intraday/tick data.
    """
    close = history["Close"]
    scores: dict[pd.Timestamp, float] = {}
    forward_returns: dict[pd.Timestamp, float] = {}

    for i in range(warmup, len(history) - horizon_days, step):
        combined = signal_fn(history.iloc[: i + 1])
        if combined is None:
            continue
        date = history.index[i]
        scores[date] = combined
        entry_price = float(close.iloc[i])
        exit_price = float(close.iloc[i + horizon_days])
        forward_returns[date] = (exit_price / entry_price) - 1.0 if entry_price else 0.0

    return pd.Series(scores, dtype=float), pd.Series(forward_returns, dtype=float)


def _hit_rate(scores: pd.Series, forward_returns: pd.Series, threshold: float, bullish: bool):
    mask = scores >= threshold if bullish else scores <= threshold
    subset = forward_returns[mask]
    n = int(len(subset))
    if n == 0:
        return None, 0
    rate = float((subset > 0).mean()) if bullish else float((subset < 0).mean())
    return rate, n


def _quintile_labels(n_bins: int) -> list[str]:
    if n_bins <= 0:
        return []
    if n_bins == 1:
        return ["Q1 (weakest)"]
    labels = [f"Q{i}" for i in range(1, n_bins + 1)]
    labels[0] = "Q1 (weakest)"
    labels[-1] = f"Q{n_bins} (strongest)"
    return labels


def _quintile_returns(scores: pd.Series, forward_returns: pd.Series) -> pd.Series:
    """Mean forward return by score quintile (fewer bins if scores are degenerate)."""
    if len(scores) < 2 or scores.nunique() < 2:
        return pd.Series(dtype=float)
    try:
        bins = pd.qcut(scores, 5, duplicates="drop")
    except ValueError:
        return pd.Series(dtype=float)

    n_bins = len(bins.cat.categories)
    if n_bins == 0:
        return pd.Series(dtype=float)
    labels = _quintile_labels(n_bins)
    bins = bins.cat.rename_categories(labels)
    return forward_returns.groupby(bins, observed=True).mean().reindex(labels)


def _information_coefficient(scores: pd.Series, forward_returns: pd.Series) -> float | None:
    """Spearman rank correlation via pandas ``.rank().corr()`` — no scipy needed."""
    if len(scores) < 2:
        return None
    ic = scores.rank().corr(forward_returns.rank())
    return None if pd.isna(ic) else float(ic)


def _annualized_sharpe(daily_returns: pd.Series) -> float | None:
    std = daily_returns.std()
    if not std:
        return None
    return float(daily_returns.mean() / std * (_TRADING_DAYS_PER_YEAR**0.5))


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def _simulate_strategy(
    history: pd.DataFrame,
    scores: pd.Series,
    warmup: int,
    entry_threshold: float,
    exit_threshold: float,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Simulate the score-gated long/flat strategy on daily closes.

    The score only updates every ``step`` bars (see ``_build_signal_series``);
    it is forward-filled onto the daily index here, which is the realistic
    behavior of "use the last signal you computed until a new one arrives."
    Position enters at ``entry_threshold`` and exits at ``exit_threshold``
    (hysteresis) rather than flipping on every crossing of one level.
    """
    window = history.iloc[warmup:]
    close = window["Close"]
    daily_returns = close.pct_change().fillna(0.0)

    score_ff = scores.reindex(window.index, method="ffill")
    # Neutral (flat-inducing) fallback for any bars before the first
    # successfully computed score.
    score_ff = score_ff.fillna(50.0)

    positions: list[float] = []
    in_position = False
    for value in score_ff:
        if in_position:
            if value < exit_threshold:
                in_position = False
        elif value >= entry_threshold:
            in_position = True
        positions.append(1.0 if in_position else 0.0)
    position = pd.Series(positions, index=window.index)

    strategy_daily_returns = position.shift(1).fillna(0.0) * daily_returns
    buyhold_daily_returns = daily_returns

    strategy_equity = (1.0 + strategy_daily_returns).cumprod()
    buyhold_equity = (1.0 + buyhold_daily_returns).cumprod()

    return (
        strategy_equity,
        buyhold_equity,
        strategy_daily_returns,
        buyhold_daily_returns,
        position,
    )


def backtest_signal(
    history: pd.DataFrame,
    ticker: str = "",
    horizon_days: int = 21,
    step: int = 5,
    warmup: int = 220,
    entry_threshold: float = 60.0,
    exit_threshold: float = 45.0,
    signal_fn: Callable[[pd.DataFrame], float | None] | None = None,
) -> BacktestResult | None:
    """Backtest the price-only composite (technicals + trend/mean-reversion).

    ``signal_fn`` overrides the default combined score with any pure
    function of an OHLCV prefix returning a 0-100 score (or None) — this is
    what ablation studies plug alternative signal variants into.

    Returns ``None`` if ``history`` is too short to produce a meaningful
    sample (``len(history) < warmup + horizon_days + 30``).

    See the module docstring for why only these two pillars are used and
    what the resulting numbers do (and don't) tell you.
    """
    if len(history) < warmup + horizon_days + 30:
        return None

    scores, forward_returns = _build_signal_series(
        history, horizon_days, step, warmup, signal_fn or _combined_score
    )

    hit_rate_bullish, n_bullish = _hit_rate(
        scores, forward_returns, _BULLISH_HIT_THRESHOLD, bullish=True
    )
    hit_rate_bearish, n_bearish = _hit_rate(
        scores, forward_returns, _BEARISH_HIT_THRESHOLD, bullish=False
    )
    quintile_returns = _quintile_returns(scores, forward_returns)
    ic_spearman = _information_coefficient(scores, forward_returns)

    strategy_equity, buyhold_equity, strategy_daily, buyhold_daily, position = (
        _simulate_strategy(history, scores, warmup, entry_threshold, exit_threshold)
    )

    strategy_return = float(strategy_equity.iloc[-1] - 1.0) if len(strategy_equity) else 0.0
    buyhold_return = float(buyhold_equity.iloc[-1] - 1.0) if len(buyhold_equity) else 0.0

    return BacktestResult(
        ticker=ticker,
        n_signals=int(len(scores)),
        horizon_days=horizon_days,
        scores=scores,
        forward_returns=forward_returns,
        hit_rate_bullish=hit_rate_bullish,
        hit_rate_bearish=hit_rate_bearish,
        n_bullish=n_bullish,
        n_bearish=n_bearish,
        quintile_returns=quintile_returns,
        ic_spearman=ic_spearman,
        strategy_equity=strategy_equity,
        buyhold_equity=buyhold_equity,
        position=position,
        strategy_return=strategy_return,
        buyhold_return=buyhold_return,
        strategy_sharpe=_annualized_sharpe(strategy_daily),
        buyhold_sharpe=_annualized_sharpe(buyhold_daily),
        strategy_max_drawdown=_max_drawdown(strategy_equity),
        buyhold_max_drawdown=_max_drawdown(buyhold_equity),
    )


def _fmt_hit_rate(rate: float | None, n: int) -> str:
    if rate is None:
        return "n/a"
    return f"{rate * 100:.1f}% (n={n})"


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:+.1f}%"


def _fmt_ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def summary_table(result: BacktestResult) -> pd.DataFrame:
    """Human-readable ("Metric", "Value") summary of a ``BacktestResult``."""
    rows = [
        ("Ticker", result.ticker or "n/a"),
        ("Signals evaluated", str(result.n_signals)),
        ("Forward-return horizon (trading days)", str(result.horizon_days)),
        (
            f"Bullish hit rate (score >= {_BULLISH_HIT_THRESHOLD:.0f})",
            _fmt_hit_rate(result.hit_rate_bullish, result.n_bullish),
        ),
        (
            f"Bearish hit rate (score <= {_BEARISH_HIT_THRESHOLD:.0f})",
            _fmt_hit_rate(result.hit_rate_bearish, result.n_bearish),
        ),
        ("Information coefficient (rank correlation)", _fmt_ratio(result.ic_spearman)),
        ("Strategy total return", _fmt_pct(result.strategy_return)),
        ("Buy & hold total return", _fmt_pct(result.buyhold_return)),
        ("Strategy Sharpe (annualized)", _fmt_ratio(result.strategy_sharpe)),
        ("Buy & hold Sharpe (annualized)", _fmt_ratio(result.buyhold_sharpe)),
        ("Strategy max drawdown", _fmt_pct(result.strategy_max_drawdown)),
        ("Buy & hold max drawdown", _fmt_pct(result.buyhold_max_drawdown)),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])
