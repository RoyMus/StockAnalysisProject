"""Methodology page — documents the scoring, backtest, and validation methods.

Pure documentation: no data fetching happens here, so this page never fails
regardless of network access or ticker.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from stockanalysis.scoring import DEFAULT_WEIGHTS, RATING_BANDS

st.set_page_config(page_title="Methodology — Stock Analyzer", layout="wide", page_icon="📖")

st.title("Methodology")
st.caption(
    "How the composite score, backtest, and forecast are built — and where "
    "each one should (and shouldn't) be trusted."
)

# ---------------------------------------------------------------------------
# Pillars
# ---------------------------------------------------------------------------

st.header("The five pillars")
st.markdown(
    "Every ticker is scored on five independent pillars, each 0-100 (50 = "
    "neutral). A pillar can be `None` when it has no usable data for a "
    "given ticker (e.g. Value on an ETF) — the composite renormalizes "
    "around missing pillars rather than penalizing the gap."
)

st.subheader("Value")
st.caption("Classic valuation multiples, piecewise-linear banded (lower multiple = higher score).")
st.dataframe(
    pd.DataFrame(
        [
            ("Trailing P/E", 1.0, "8→90, 15→75, 25→55, 40→35, 60→15"),
            ("PEG ratio", 1.0, "0.5→90, 1→80, 2→50, 3→30, 4→15"),
            ("Price/Book", 1.0, "0.8→85, 1.5→70, 3→55, 6→35, 10→15"),
            ("EV/EBITDA", 1.0, "6→90, 8→75, 15→50, 25→30, 35→15"),
            ("FCF yield", 1.0, "0%→30, 2%→45, 5%→70, 8%→85, 12%→95"),
            ("Forward vs trailing P/E", 0.5, "70 if forward < trailing, else 35"),
        ],
        columns=["Signal", "Weight", "Band"],
    ),
    use_container_width=True,
    hide_index=True,
)

st.subheader("Growth")
st.caption("YoY growth rates plus multi-period trend confirmation from statement history.")
st.dataframe(
    pd.DataFrame(
        [
            ("Revenue growth YoY", 3.0, "linear scale -10% → +30%"),
            ("Earnings growth YoY", 3.0, "linear scale -15% → +40%"),
            ("Revenue CAGR (3+ yrs)", 2.0, "linear scale -5% → +25%"),
            ("Growth consistency", 2.0, "fraction of quarters growing sequentially, 20% → 100%"),
        ],
        columns=["Signal", "Weight", "Band"],
    ),
    use_container_width=True,
    hide_index=True,
)

st.subheader("Technicals")
st.caption("Trend alignment, momentum, volume, and price positioning — pure price-action.")
st.dataframe(
    pd.DataFrame(
        [
            ("Trend alignment", 3.0, "price vs SMA50 vs SMA200 ordering"),
            ("MACD", 2.5, "line vs signal, histogram direction"),
            ("RSI momentum", 2.0, "RSI(14) mapped through a rescaled curve"),
            ("Volume trend", 1.5, "20d/60d volume ratio, direction-aware"),
            ("52-week position", 1.0, "% of trailing 52-week range"),
        ],
        columns=["Signal", "Weight", "Band"],
    ),
    use_container_width=True,
    hide_index=True,
)

st.subheader("Trend & Mean-Reversion")
st.caption(
    "Regime-aware: detects trending vs ranging markets via ADX and blends a "
    "trend-following read with a mean-reversion read accordingly. See "
    "'Regime detection' below."
)
st.dataframe(
    pd.DataFrame(
        [
            ("SMA50 slope (trend)", "4.0 × w_trend", "normalized 50-bar SMA slope"),
            ("ADX conviction (trend)", "3.0 × w_trend", "linear scale ADX 15 → 40"),
            ("RSI continuation (trend)", "3.0 × w_trend", "linear scale RSI 35 → 70"),
            ("Bollinger %B (mean-reversion)", "4.0 × w_revert", "inverse: low %B scores high"),
            ("Z-score (mean-reversion)", "3.0 × w_revert", "inverse: -2σ → 85, +2σ → 15"),
            ("RSI extremes (mean-reversion)", "3.0 × w_revert", "oversold/overbought fade"),
        ],
        columns=["Signal", "Base weight", "Band"],
    ),
    use_container_width=True,
    hide_index=True,
)

st.subheader("Analysts")
st.caption(
    "Sell-side consensus and price-target upside — the most lagging pillar, "
    "shrunk toward neutral (50) when coverage is thin (< 5 analysts)."
)
st.dataframe(
    pd.DataFrame(
        [
            ("Analyst consensus", 5.0, "weighted mean of strongBuy..strongSell counts"),
            ("Price target upside", 5.0, "linear scale -20% → +30% upside to mean target"),
        ],
        columns=["Signal", "Weight", "Band"],
    ),
    use_container_width=True,
    hide_index=True,
)

# ---------------------------------------------------------------------------
# Composite weights
# ---------------------------------------------------------------------------

st.header("Composite weights")
st.markdown(
    "The five pillar scores are combined into one 0-100 composite via a "
    "weighted mean. Default weights:"
)
st.dataframe(
    pd.DataFrame(
        [(name, f"{w:.0%}") for name, w in DEFAULT_WEIGHTS.items()],
        columns=["Pillar", "Default weight"],
    ),
    use_container_width=True,
    hide_index=True,
)
st.markdown(
    """
**Rationale:**

- **Value (25%)** and **Technicals (20%)** get the highest weight because
  they capture durable, price-anchored information: what you pay relative
  to fundamentals, and where price sits relative to its own history. Both
  are hard to game and slow to go stale.
- **Trend & Mean-Reversion (20%)** rounds out the price-positioning view
  with a regime-aware read (following momentum vs. fading an extreme),
  complementary to raw Technicals rather than redundant with it.
- **Growth (20%)** matters for the durability of the value thesis but is
  more volatile quarter to quarter than the multiples themselves.
- **Analysts (15%)** gets the lowest weight: sell-side ratings and price
  targets are the most lagging and herd-prone signal of the five — they
  tend to catch up to price moves rather than lead them.

When a pillar has no usable data, its weight is dropped and the rest are
renormalized to sum to 1 — a stock missing analyst coverage is judged
entirely on the other four pillars rather than penalized for the gap.
The sidebar lets you override these weights; they're renormalized the
same way before scoring.
"""
)

st.header("Rating bands")
st.dataframe(
    pd.DataFrame(
        [(f"{low:g}–{high:g}", label) for low, high, label in RATING_BANDS],
        columns=["Composite score", "Label"],
    ),
    use_container_width=True,
    hide_index=True,
)

# ---------------------------------------------------------------------------
# Regime detection
# ---------------------------------------------------------------------------

st.header("Regime detection")
st.markdown(
    """
The Trend & Mean-Reversion pillar uses **ADX(14)** (Average Directional
Index, Wilder's formula) to classify the current regime:

- **ADX > 25 → trending.** Trend-following signals get 80% of the blend
  weight, mean-reversion signals 20%.
- **ADX < 20 → ranging.** Mean-reversion signals get 80% of the blend
  weight, trend-following signals 20%.
- **20 ≤ ADX ≤ 25 → transitional.** An even 50/50 blend.

Both branches are always computed and both sets of sub-signals are always
shown in the Score Breakdown tab (weighted by the regime blend), rather
than only showing the winning branch — so you can see what the "losing"
read would have said too.
"""
)

# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

st.header("Backtest methodology")
st.markdown(
    """
Only two of the five pillars — **Technicals** and **Trend & Mean-Reversion**
— can be honestly backtested. Both are pure functions of an OHLCV prefix:
call them on `history.iloc[:i+1]` for any `i` and they only ever look at
data up to bar `i`, so replaying them at each historical bar is a faithful
simulation of "what would this signal have said on that day."

The other three pillars (Value, Growth, Analysts) **cannot** be backtested
this way, because `yfinance` only exposes each metric's *current* snapshot
— there's no historical P/E, analyst rating, or revenue-growth series to
replay. Scoring a historical bar with today's fundamentals would leak the
future into the past (crediting 2019-era price action with a P/E computed
from 2026 earnings, for instance). So the backtest deliberately scores a
**price-only composite** — the mean of the two backtestable pillars — not
the full five-pillar score. Treat backtest results as a read on "does the
price/technical half of the signal have any predictive power," not a
validation of the whole product.

**Procedure:**

1. Walk the price history bar by bar (every `step` bars, after a `warmup`
   period), scoring only the prefix available at that bar.
2. Pair each score with its realized forward return `horizon_days` later.
3. Summarize predictiveness via hit rates, quintile spreads, and a
   Spearman-style rank information coefficient.
4. Simulate a simple long/flat, no-shorting strategy that enters when the
   score crosses an entry threshold and exits below an exit threshold
   (hysteresis, to avoid whipsawing), and compare its equity curve to
   buy-and-hold.

**Honest limitations:** this is research tooling, not a production trading
system. **A flat 10 bps cost per position change is charged; slippage,
taxes, and position sizing are not modeled.** Past predictiveness is no
guarantee of future predictiveness,
and the walk-forward loop re-scores overlapping windows of the same price
series, so successive signals are not independent observations.
"""
)

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

st.header("Statistical validation")
st.markdown(
    """
A backtest that looks good can still be luck. Four checks quantify that:

- **Permutation test (information coefficient).** Shuffles the pairing
  between signal scores and forward returns thousands of times to build a
  null distribution of the rank correlation under "the signal has no
  information," then reports where the observed IC sits (p-value).
- **Permutation test (hit rate).** Same idea applied to the bullish hit
  rate: under the null that forward returns are exchangeable, does
  conditioning on a high score really beat the unconditional positive
  rate?
- **Monte Carlo random-entry test.** Circularly shifts the strategy's
  actual position sequence by a random offset in each trial — this
  preserves both the fraction of time in market and the run-length
  structure of positions, isolating *timing skill* from *raw exposure*.
  The observed strategy's total return is compared against this null.
- **Block bootstrap (Sharpe ratio).** Resamples daily strategy returns in
  contiguous blocks (preserving short-range autocorrelation) to build a
  confidence interval around the annualized Sharpe ratio, instead of
  trusting a single point estimate.

A p-value below 0.05 is the conventional (not sacred) threshold for
"probably not noise."
"""
)

# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------

st.header("Forecast (ARIMA)")
st.markdown(
    """
The Forecast tab fits an ARIMA model to log-transformed closing prices:

1. Log-transform Close for variance stabilization.
2. Choose the differencing order `d` via the Augmented Dickey–Fuller test
   (stationary series → `d=0`, first difference stationary → `d=1`,
   otherwise `d=2`).
3. Grid-search `p, q ∈ {0,1,2}` for the given `d` and pick the order with
   the lowest AIC.
4. Generate a point forecast and an 80% confidence interval, transformed
   back into price space.

**This is a statistical baseline for educational purposes, not investment
advice.** ARIMA extrapolates historical trend and volatility patterns; it
cannot anticipate news, earnings surprises, or macro events, and its
confidence intervals only reflect the model's own uncertainty about the
historical series, not real-world risk.
"""
)

# ---------------------------------------------------------------------------
# Data source & disclaimer
# ---------------------------------------------------------------------------

st.header("Data source")
st.markdown(
    """
All live data comes from **Yahoo Finance via the `yfinance` package**,
which scrapes Yahoo's undocumented, unofficial endpoints rather than a
supported API. Fields can be missing, stale, or occasionally wrong, and
Yahoo can change or rate-limit these endpoints without notice — the app
degrades gracefully (skipping a signal or a whole pillar) rather than
crashing when a field is unavailable, but it cannot guarantee accuracy.
Entering ticker `DEMO` bypasses this entirely with deterministic synthetic
data, useful for exploring the UI without depending on network access.
"""
)

st.header("Disclaimer")
st.warning(
    "This tool is for educational and research purposes only. Nothing in "
    "this app — scores, charts, forecasts, or backtests — is financial "
    "advice or a recommendation to buy or sell any security. Past "
    "performance and backtested results do not guarantee future results. "
    "Consult a licensed financial advisor before making investment "
    "decisions."
)

st.page_link("main.py", label="Back to analyzer", icon="📊")
