# Stock Analyzer 📊

**A multi-factor stock analysis engine that scores stocks 0–100 across five
pillars — technicals, trend & mean-reversion, analyst consensus, value, and
growth — and validates its own signal with a point-in-time backtest and
statistical significance tests.**

Built with Python, pandas, and Streamlit; market data from Yahoo Finance via
`yfinance` (no API key needed).

---

## What it does

Enter a ticker and get:

- **Composite score & rating** (0–100 → Strong Sell … Strong Buy) with a
  transparent, configurable weighting of five analysis pillars
- **Full score breakdown** — every sub-signal shows the raw reading it was
  derived from (e.g. "RSI 63.4", "P/E 24.1", "+18% upside to mean target"),
  so no number is a black box
- **Technical chart** — candlesticks with SMA50/200, Bollinger Bands, RSI,
  and MACD
- **Fundamentals view** — valuation ratios and growth metrics with their
  individual scores
- **Analyst consensus** — aggregated buy/hold/sell distribution and price-target
  range vs. the current price
- **ARIMA price forecast** — a clearly-labeled statistical baseline with 80%
  confidence intervals (ADF-driven differencing, AIC order selection)
- **Backtest with significance testing** — replays the price-based signal
  bar-by-bar through history and reports whether its predictive power is
  statistically distinguishable from luck

Type `DEMO` as the ticker to explore the app on deterministic synthetic data
without touching the network.

## The five pillars

| Pillar | Weight | What it measures |
|---|---|---|
| Value | 25% | P/E, PEG, P/B, EV/EBITDA, FCF yield against classic valuation bands |
| Growth | 20% | YoY revenue/earnings growth, multi-year revenue CAGR, growth consistency |
| Technicals | 20% | Trend alignment (price/SMA50/SMA200), MACD, RSI momentum, volume trend, 52-week position |
| Trend & Mean-Reversion | 20% | ADX regime detection; blends trend-following vs. reversion signals (Bollinger %B, z-score, RSI extremes) according to the active regime |
| Analysts | 15% | Rating consensus and price-target upside, shrunk toward neutral for thin coverage |

Weights are configurable in the UI and renormalize automatically when a pillar
has no data (an ETF has no P/E; a fresh IPO has no SMA200) — a stock is judged
on the evidence that exists rather than penalized for gaps. The rationale for
the default weights is documented in the in-app **Methodology** page.

## Honest backtesting

Most hobby projects skip the question "does this signal actually work?" This
one answers it carefully:

- **Point-in-time replay** — the price-based pillars are pure functions of an
  OHLCV prefix, so the backtest scores each historical bar using only data
  available on that day. No lookahead.
- **Scope honesty** — fundamental and analyst pillars are *excluded* from the
  backtest because yfinance only provides current snapshots; scoring the past
  with today's P/E would be textbook lookahead bias. The backtest evaluates
  the price-based half of the composite, and says so.
- **Significance, not vibes** — results ship with a permutation test on the
  information coefficient (using a circular-rotation null that stays calibrated
  when forward-return windows overlap), a random-entry Monte Carlo that
  isolates timing skill from market exposure, and a block-bootstrap confidence
  interval on the Sharpe ratio.
- **Dev/holdout discipline** — `scripts/evaluate.py` evaluates the signal
  across a development basket of tickers, with a separate holdout basket
  reserved for confirming finished changes, so signal tweaks aren't curve-fit
  to one basket's noise.
- **A real, confirmed finding** — a pre-registered study on 20 real S&P 500
  names (2005–2022) found the mean-reversion signal carries genuine 21-day
  predictive information (holdout pooled IC +0.059, p = 0.004) while the
  momentum signal anti-predicts. Full methodology, results, and caveats in
  [`docs/SIGNAL_FINDINGS.md`](docs/SIGNAL_FINDINGS.md).

## Architecture

```
main.py                     Streamlit app (entry point)
pages/1_Methodology.py      Scoring methodology, in the app
stockanalysis/
  data.py                   Typed yfinance wrapper (the only network module)
  indicators.py             Hand-rolled RSI, MACD, Bollinger, ADX, …
  pillars/                  One module per scoring pillar, all pure functions
  scoring.py                Composite engine: weights, renormalization, rating bands
  forecasting.py            ARIMA forecast (statsmodels)
  backtest.py               Point-in-time signal replay + strategy simulation
  validation.py             Permutation tests, Monte Carlo, block bootstrap
  charts.py                 Plotly figure builders
  demo.py                   Deterministic synthetic data for offline demo
scripts/evaluate.py         Dev/holdout signal evaluation harness
tests/                      62 unit tests, no network access required
```

Indicators are implemented from their standard formulas with pandas rather
than imported from a TA library — every number in the score breakdown traces
to a formula in `indicators.py`.

## Getting started

```bash
git clone https://github.com/RoyMus/StockAnalysisProject.git
cd StockAnalysisProject
python3 -m venv venv && source venv/bin/activate   # .\venv\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run main.py
```

Run the tests and linter:

```bash
pip install -r requirements-dev.txt
pytest
ruff check .
```

Evaluate the signal across a ticker basket:

```bash
python scripts/evaluate.py               # development basket
python scripts/evaluate.py --holdout     # holdout basket (confirmation only)
python scripts/evaluate.py --synthetic   # offline methodology check
```

## Limitations

- yfinance is an unofficial API; field availability varies by ticker and can
  change without notice. The data layer degrades gracefully but can't conjure
  missing fundamentals.
- Valuation bands are classic absolute heuristics, not sector-relative — a
  software company and a bank are held to the same P/E scale.
- The backtest charges a flat 10 bps per position change but models no
  slippage or taxes.
- ARIMA extrapolates historical patterns; it cannot know about earnings
  surprises, news, or regime changes.

## Disclaimer

This tool is for educational and informational purposes only and does not
constitute financial advice. Consult a licensed financial advisor before
making investment decisions.

## License

MIT
