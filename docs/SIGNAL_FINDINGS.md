# Signal study: what actually predicts, on real data

**TL;DR — on 20 real S&P 500 large caps (2005–2022, daily), the
mean-reversion signal carried genuine 21-day predictive information and
survived an independent holdout confirmation (pooled IC +0.059, p = 0.004).
The momentum signal anti-predicted, and blending the two cancelled to
exactly zero.**

## Why this study exists

A backtest that was tuned until it looked good proves nothing. This study
was run under pre-registered rules — data split, evaluation window,
variants, and decision thresholds all fixed before any result was seen —
so the outcome is evidence, not curve-fitting.

## Data

Real historical prices bundled inside published Python packages (no
network market-data access was available in the build environment):

- **skfolio** S&P 500 dataset: 20 real tickers, daily adjusted closes
  1990–2022. Close-only, so the study exercises the close-only signal
  subset (SMA alignment, MACD, RSI, Bollinger %B, z-score).
- **backtesting.py** GOOG dataset: full OHLCV, 2004–2013, used as a
  single-name check of the full signal including ADX and volume.

## Pre-registered design

| Choice | Value | Why |
|---|---|---|
| Dev basket | AAPL BAC CVX HD JPM LLY MSFT PFE RRC WMT | even-indexed alphabetically |
| Holdout basket | AMD BBY GE JNJ KO MRK PEP PG UNH XOM | odd-indexed; touched exactly once |
| Window | 2005-01-01 → 2022-12-28 | fixed before running |
| Horizon / step / warmup | 21d / 5 / 220 | backtester defaults |
| Costs | 10 bps per position change | backtester default |
| Variants | balanced, momentum-only, reversion-only | 3 hypotheses, no more |
| Dev threshold | Bonferroni p < 0.0167 (0.05 / 3) | best-of-3 must beat corrected bar |
| Confirmation | single holdout run of the one chosen variant | no second chances |

Signals were replayed point-in-time (prefix slices only — verified by a
unit test that probes the backtester never shows the signal future bars),
and significance uses rotation-null permutation tests pooled across names
with a shared offset (see `stockanalysis/validation.py` for why naive
shuffling is miscalibrated on overlapping windows).

## Results

### Dev basket (10 names)

| Variant | Pooled IC | p | Mean MC percentile | Mean Sharpe |
|---|---|---|---|---|
| Balanced blend | +0.000 | 0.983 | 48 | 0.28 |
| Momentum only | −0.045 | 0.040 | 40 | 0.36 |
| **Reversion only** | **+0.048** | **0.045** | **70** | **0.38** |

No variant cleared the corrected dev threshold outright, but reversion-only
won every decision metric and became the single pre-registered candidate
for confirmation.

### Holdout confirmation (10 different names, one run)

| Metric | Value |
|---|---|
| Pooled IC | **+0.059** |
| Permutation p | **0.004** |
| Names with positive IC | 8 / 10 |
| Mean Monte Carlo timing percentile | 65 |
| Mean strategy Sharpe | 0.33 |

Same direction as dev, independently significant, and clears even the
corrected threshold. Two independent baskets agreeing is the strongest
statement this dataset can make.

### GOOG full-OHLCV check (single name, 2004–2013)

All variants insignificant (full signal IC +0.001, p = 0.99; reversion IC
−0.040, p = 0.60). Two honest lessons: single-name tests have almost no
statistical power, and 2004–2013 hyper-growth GOOG is exactly the regime
where betting on reversion fails. Reversion's edge is a *basket-level,
large-cap* finding, not a per-stock guarantee.

## What changed in the code because of this

1. The three studied variants now live in `stockanalysis/signals.py`
   (they were scratch code during the study) so the finding is
   reproducible and the validated `reversion_score` is available as a
   building block.
2. `scripts/evaluate.py` includes them as ablation variants for future
   re-runs on fresh data.

## What deliberately did NOT change

- **Pillar formulas and weights.** The study measured three specific
  close-only composites, unconditionally. It did not isolate which
  momentum sub-signal hurts, nor test the ADX-regime-gated blend the
  product actually uses, nor condition on regime. Re-tuning those from
  this data would be fitting parameters to a study that wasn't designed
  to answer them — the exact failure mode this project's validation
  tooling exists to prevent. The right follow-up is a new pre-registered
  study on fresh data (different names and/or a post-2022 window) that
  isolates those questions.
- **The five-pillar product composite.** Value/growth/analyst pillars are
  not backtestable with snapshot-only data and were out of scope here.

## Scope and caveats

- Close-only data; the volume and ADX components were only smoke-checked
  on one name.
- Large-cap S&P constituents only — the reversal effect documented here is
  known to be strongest in exactly this universe; do not extrapolate to
  small caps or crypto.
- One era of market history (2005–2022). Regimes change.
- 10 bps costs are optimistic for retail execution and generous for
  institutional.
- Survivorship: the 20 bundled tickers are names that existed through
  2022, which flatters buy-and-hold baselines more than signal ICs, but
  is still a bias worth naming.
