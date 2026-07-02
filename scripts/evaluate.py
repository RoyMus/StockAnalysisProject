"""Signal evaluation harness — the anti-overfitting workflow.

Runs the point-in-time backtest plus statistical validation across a basket
of tickers, split into a DEV set (used to iterate on signal design) and a
HOLDOUT set (untouched by design decisions; consulted only to confirm a
finished change). Optimizing on dev and confirming on holdout is what keeps
signal "improvements" from being curve-fits to one basket's noise.

Usage:
    python scripts/evaluate.py                 # dev basket, live data
    python scripts/evaluate.py --holdout       # holdout basket (confirmation runs only!)
    python scripts/evaluate.py --synthetic     # seeded synthetic tickers (no network;
                                               # methodology smoke test, not a real result)

Interpretation guide (printed with results):
- Mean IC > 0 with pooled permutation p < 0.05  → signal ordering carries information.
- MC percentile > 95 on most names             → timing beats random entry with the
                                                 same exposure.
- Sharpe CI straddling 0                       → the strategy edge is not distinguishable
                                                 from noise at this sample size.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stockanalysis.backtest import BacktestResult, backtest_signal  # noqa: E402
from stockanalysis.validation import (  # noqa: E402
    block_bootstrap_sharpe,
    monte_carlo_random_entry,
    permutation_test_ic,
)

# Dev basket: iterate signal design against these. Deliberately mixed:
# mega-cap tech, industrial, financial, healthcare, energy, consumer.
DEV_TICKERS = ["AAPL", "MSFT", "JPM", "CAT", "JNJ", "XOM", "KO", "NVDA", "WMT", "UNH"]

# Holdout basket: DO NOT tune against these. Different names, same sector mix.
HOLDOUT_TICKERS = ["GOOGL", "BAC", "DE", "PFE", "CVX", "PEP", "AMD", "COST", "HON", "MRK"]

PERIOD = "5y"
HORIZON = 21
STEP = 5


@dataclass
class TickerEval:
    ticker: str
    n_signals: int
    ic: float | None
    ic_p: float | None
    hit_bullish: float | None
    mc_percentile: float | None
    strat_return: float
    bh_return: float
    strat_sharpe: float | None
    sharpe_p5: float | None
    sharpe_p95: float | None


def _make_synthetic(seed: int, n: int = 1000) -> pd.DataFrame:
    """Regime-switching geometric walk; a stand-in ticker for offline runs."""
    rng = np.random.default_rng(seed)
    n_segments = 8
    seg = n // n_segments
    rets: list[float] = []
    for _ in range(n_segments):
        drift = rng.normal(0.0004, 0.0012)
        vol = rng.uniform(0.008, 0.025)
        rets.extend(rng.normal(drift, vol, seg))
    close = 100 * np.exp(np.cumsum(rets))
    m = len(close)
    return pd.DataFrame(
        {
            "Open": close * (1 + rng.normal(0, 0.003, m)),
            "High": close * (1 + np.abs(rng.normal(0, 0.008, m))),
            "Low": close * (1 - np.abs(rng.normal(0, 0.008, m))),
            "Close": close,
            "Volume": rng.integers(int(1e6), int(5e6), m).astype(float),
        },
        index=pd.bdate_range("2021-01-04", periods=m),
    )


def _evaluate_one(ticker: str, history: pd.DataFrame) -> TickerEval | None:
    result: BacktestResult | None = backtest_signal(
        history, ticker=ticker, horizon_days=HORIZON, step=STEP
    )
    if result is None:
        return None

    ic_test = permutation_test_ic(result.scores, result.forward_returns, seed=0)
    strat_daily = result.strategy_equity.pct_change().dropna()
    bh_daily = result.buyhold_equity.pct_change().dropna()
    position = (strat_daily.abs() > 1e-12).astype(float)
    mc = monte_carlo_random_entry(bh_daily, position, seed=0)
    bs = block_bootstrap_sharpe(strat_daily, seed=0)

    return TickerEval(
        ticker=ticker,
        n_signals=result.n_signals,
        ic=result.ic_spearman,
        ic_p=ic_test.p_value if ic_test else None,
        hit_bullish=result.hit_rate_bullish,
        mc_percentile=mc.percentile if mc else None,
        strat_return=result.strategy_return,
        bh_return=result.buyhold_return,
        strat_sharpe=result.strategy_sharpe,
        sharpe_p5=bs.p5 if bs else None,
        sharpe_p95=bs.p95 if bs else None,
    )


def _fmt(value: float | None, spec: str = ".3f") -> str:
    return format(value, spec) if value is not None else "n/a"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", action="store_true", help="run the holdout basket")
    parser.add_argument(
        "--synthetic", action="store_true", help="seeded synthetic data (no network)"
    )
    args = parser.parse_args()

    rows: list[TickerEval] = []
    if args.synthetic:
        print("SYNTHETIC RUN — methodology check only; says nothing about real markets.\n")
        for seed in range(10):
            evaluation = _evaluate_one(f"SYN{seed:02d}", _make_synthetic(seed))
            if evaluation:
                rows.append(evaluation)
    else:
        from stockanalysis.data import get_price_history

        tickers = HOLDOUT_TICKERS if args.holdout else DEV_TICKERS
        basket = "HOLDOUT" if args.holdout else "DEV"
        if args.holdout:
            print(
                "HOLDOUT RUN — confirmation only. If you are still iterating on the\n"
                "signal, stop: repeated holdout checks turn it into a second dev set.\n"
            )
        print(f"Evaluating {basket} basket: {', '.join(tickers)}\n")
        for ticker in tickers:
            try:
                history = get_price_history(ticker, period=PERIOD)
            except Exception as exc:  # noqa: BLE001 - report and continue
                print(f"  {ticker}: fetch failed ({exc})")
                continue
            evaluation = _evaluate_one(ticker, history)
            if evaluation:
                rows.append(evaluation)

    if not rows:
        print("No evaluable tickers.")
        return 1

    frame = pd.DataFrame(
        [
            {
                "ticker": r.ticker,
                "signals": r.n_signals,
                "IC": _fmt(r.ic),
                "IC p": _fmt(r.ic_p),
                "hit(bull)": _fmt(r.hit_bullish, ".1%"),
                "MC pct": _fmt(r.mc_percentile, ".0f"),
                "strat ret": _fmt(r.strat_return, "+.1%"),
                "B&H ret": _fmt(r.bh_return, "+.1%"),
                "Sharpe": _fmt(r.strat_sharpe, ".2f"),
                "Sharpe 90% CI": f"[{_fmt(r.sharpe_p5, '.2f')}, {_fmt(r.sharpe_p95, '.2f')}]",
            }
            for r in rows
        ]
    )
    print(frame.to_string(index=False))

    ics = [r.ic for r in rows if r.ic is not None]
    mcs = [r.mc_percentile for r in rows if r.mc_percentile is not None]
    sig = [r for r in rows if r.ic_p is not None and r.ic_p < 0.05]
    print(f"\nMean IC: {np.mean(ics):+.3f} across {len(ics)} names")
    print(f"IC significant (p<0.05) on {len(sig)}/{len(rows)} names")
    if mcs:
        print(f"Mean Monte Carlo timing percentile: {np.mean(mcs):.0f}")
    print(
        "\nReminder: a positive mean IC with mostly non-significant per-name p-values\n"
        "is normal at this sample size; judge the basket, not single names."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
