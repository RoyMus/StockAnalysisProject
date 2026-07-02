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
from stockanalysis.pillars import score_technicals, score_trend_reversion  # noqa: E402
from stockanalysis.signals import momentum_score, reversion_score  # noqa: E402
from stockanalysis.validation import (  # noqa: E402
    block_bootstrap_sharpe,
    monte_carlo_random_entry,
    permutation_test_ic,
    pooled_ic_test,
)

# Signal variants for ablation runs. Each is a pure function of an OHLCV
# prefix returning a 0-100 score (or None). Keep this list short and
# hypothesis-driven: every variant evaluated is another chance to fit noise,
# so only add one when there's a reason to believe it could matter.
# The close-only variants were validated on real bundled data — see
# docs/SIGNAL_FINDINGS.md for the study (reversion carried the signal).
SIGNAL_VARIANTS: dict[str, object] = {
    "combined (baseline)": None,  # backtest_signal's default
    "technicals only": lambda df: score_technicals(df).score,
    "trend/mean-rev only": lambda df: score_trend_reversion(df).score,
    "close-only reversion": lambda df: reversion_score(df["Close"]),
    "close-only momentum": lambda df: momentum_score(df["Close"]),
}

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


def _evaluate_one(
    ticker: str, history: pd.DataFrame, signal_fn: object = None
) -> tuple[TickerEval, tuple[pd.Series, pd.Series]] | None:
    result: BacktestResult | None = backtest_signal(
        history,
        ticker=ticker,
        horizon_days=HORIZON,
        step=STEP,
        signal_fn=signal_fn,  # type: ignore[arg-type]
    )
    if result is None:
        return None

    ic_test = permutation_test_ic(result.scores, result.forward_returns, seed=0)
    strat_daily = result.strategy_equity.pct_change().dropna()
    bh_daily = result.buyhold_equity.pct_change().dropna()
    effective_position = result.position.shift(1).fillna(0.0)
    mc = monte_carlo_random_entry(bh_daily, effective_position, seed=0)
    bs = block_bootstrap_sharpe(strat_daily, seed=0)

    evaluation = TickerEval(
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
    return evaluation, (result.scores, result.forward_returns)


def _fmt(value: float | None, spec: str = ".3f") -> str:
    return format(value, spec) if value is not None else "n/a"


def _load_basket(args: argparse.Namespace) -> list[tuple[str, pd.DataFrame]]:
    """Fetch (or synthesize, or read from disk) the histories once."""
    if args.synthetic:
        return [(f"SYN{seed:02d}", _make_synthetic(seed)) for seed in range(10)]

    tickers = HOLDOUT_TICKERS if args.holdout else DEV_TICKERS

    if args.data_dir:
        # Offline mode: read <TICKER>.csv files (Date index + OHLCV columns,
        # the exact format `history.to_csv()` produces). Lets the evaluation
        # run on real exported data in environments without market-data
        # network access.
        data_dir = Path(args.data_dir)
        loaded_csv: list[tuple[str, pd.DataFrame]] = []
        for ticker in tickers:
            path = data_dir / f"{ticker}.csv"
            if not path.exists():
                print(f"  {ticker}: no file at {path}, skipped")
                continue
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            missing = {"Open", "High", "Low", "Close", "Volume"} - set(df.columns)
            if missing:
                print(f"  {ticker}: missing columns {sorted(missing)}, skipped")
                continue
            loaded_csv.append((ticker, df))
        return loaded_csv

    from stockanalysis.data import get_price_history

    loaded: list[tuple[str, pd.DataFrame]] = []
    for ticker in tickers:
        try:
            loaded.append((ticker, get_price_history(ticker, period=PERIOD)))
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  {ticker}: fetch failed ({exc})")
    return loaded


def _run_ablation(basket: list[tuple[str, pd.DataFrame]]) -> None:
    """Compare signal variants on the same histories, basket-level view only.

    Per-name p-values are deliberately not shown here: with several variants
    and many names, cherry-picking the significant cells is exactly the
    multiple-comparisons trap this harness exists to avoid. Compare variants
    on basket-level aggregates, pick at most one change, then confirm on the
    holdout basket.
    """
    summary_rows = []
    for variant_name, signal_fn in SIGNAL_VARIANTS.items():
        outcomes = [
            out for t, h in basket if (out := _evaluate_one(t, h, signal_fn)) is not None
        ]
        evals = [e for e, _pair in outcomes]
        pairs = [pair for _e, pair in outcomes]
        ics = [e.ic for e in evals if e.ic is not None]
        mcs = [e.mc_percentile for e in evals if e.mc_percentile is not None]
        sharpes = [e.strat_sharpe for e in evals if e.strat_sharpe is not None]
        pooled = pooled_ic_test(pairs, seed=0)
        summary_rows.append(
            {
                "variant": variant_name,
                "names": len(evals),
                "mean IC": f"{np.mean(ics):+.3f}" if ics else "n/a",
                "pooled IC p": f"{pooled.p_value:.3f}" if pooled else "n/a",
                "mean MC pct": f"{np.mean(mcs):.0f}" if mcs else "n/a",
                "mean Sharpe": f"{np.mean(sharpes):.2f}" if sharpes else "n/a",
            }
        )
    print(pd.DataFrame(summary_rows).to_string(index=False))
    n_variants = len(SIGNAL_VARIANTS)
    print(
        f"\nMultiple-comparisons note: {n_variants} variants were tested, so judge\n"
        f"the best variant against a Bonferroni-adjusted threshold of\n"
        f"p < {0.05 / n_variants:.3f} (raw 0.05 / {n_variants}) — picking the best of "
        f"several tries and\nusing the unadjusted threshold is itself a form of overfitting."
    )
    print(
        "\nDecision rule: prefer the variant with the best pooled IC p-value AND\n"
        "mean Sharpe on the dev basket; confirm the single chosen variant on the\n"
        "holdout basket before adopting it."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", action="store_true", help="run the holdout basket")
    parser.add_argument(
        "--synthetic", action="store_true", help="seeded synthetic data (no network)"
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="compare signal variants (basket-level aggregates only)",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="read <TICKER>.csv OHLCV files from this directory instead of fetching",
    )
    args = parser.parse_args()

    if args.ablation:
        if args.synthetic:
            print("SYNTHETIC ABLATION — methodology check only.\n")
        basket = _load_basket(args)
        if not basket:
            print("No data available.")
            return 1
        _run_ablation(basket)
        return 0

    if args.synthetic:
        print("SYNTHETIC RUN — methodology check only; says nothing about real markets.\n")
    elif args.holdout:
        print(
            "HOLDOUT RUN — confirmation only. If you are still iterating on the\n"
            "signal, stop: repeated holdout checks turn it into a second dev set.\n"
        )
    else:
        print(f"Evaluating DEV basket: {', '.join(DEV_TICKERS)}\n")

    rows: list[TickerEval] = []
    pairs: list[tuple[pd.Series, pd.Series]] = []
    for ticker, history in _load_basket(args):
        outcome = _evaluate_one(ticker, history)
        if outcome:
            rows.append(outcome[0])
            pairs.append(outcome[1])

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
    print(f"\nMean IC: {np.mean(ics):+.3f} across {len(ics)} names")
    pooled = pooled_ic_test(pairs, seed=0)
    if pooled:
        verdict = "SIGNIFICANT" if pooled.significant else "not significant"
        print(
            f"Pooled basket test: mean IC {pooled.observed:+.3f}, "
            f"p={pooled.p_value:.3f} ({verdict} at 5%)"
        )
    if mcs:
        print(f"Mean Monte Carlo timing percentile: {np.mean(mcs):.0f}")
    print(
        "\nReminder: per-name p-values have little power at this sample size —\n"
        "the pooled basket test is the load-bearing number here."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
