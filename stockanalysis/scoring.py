"""Composite scoring: blends the five pillars into a single 0-100 score.

This module owns the only network-touching entry point (``score_stock``);
``composite_from_pillars`` is pure and takes already-computed pillar scores,
which is what makes it independently testable and what a backtester can
call directly without re-fetching data for every historical bar.
"""

from __future__ import annotations

from stockanalysis.data import (
    get_analyst_data,
    get_company_info,
    get_financial_history,
    get_price_history,
)
from stockanalysis.pillars import (
    score_analysts,
    score_growth,
    score_technicals,
    score_trend_reversion,
    score_value,
)
from stockanalysis.types import PillarScore, StockScore

# Default pillar weights for the composite score.
#
# Rationale:
#   - Value (0.25) and Technicals (0.20) are weighted highest because they
#     capture durable, price-anchored information: what you pay relative to
#     fundamentals, and where price sits relative to its own history. Both
#     are hard to game and slow to go stale.
#   - Trend & Mean-Reversion (0.20) rounds out the price-positioning view
#     with a regime-aware read (are we following momentum or fading an
#     extreme), which is complementary to raw Technicals rather than
#     redundant with it.
#   - Growth (0.20) matters for durability of the value thesis but is more
#     volatile quarter to quarter than the multiples themselves.
#   - Analysts (0.15) gets the lowest weight: sell-side ratings and price
#     targets are the most lagging and herd-prone signal of the five — they
#     tend to catch up to price moves rather than lead them.
DEFAULT_WEIGHTS: dict[str, float] = {
    "Value": 0.25,
    "Growth": 0.20,
    "Technicals": 0.20,
    "Trend & Mean-Reversion": 0.20,
    "Analysts": 0.15,
}

# (low, high, label) bands, low inclusive / high exclusive except the final
# band which also includes 100.
RATING_BANDS: list[tuple[float, float, str]] = [
    (0, 20, "Strong Sell"),
    (20, 40, "Sell"),
    (40, 60, "Hold"),
    (60, 80, "Buy"),
    (80, 100, "Strong Buy"),
]


def label_for(score: float) -> str:
    """Map a 0-100 composite score onto its rating label."""
    last = len(RATING_BANDS) - 1
    for i, (_low, high, label) in enumerate(RATING_BANDS):
        if score < high or i == last:
            return label
    return RATING_BANDS[-1][2]  # unreachable given the loop above


def composite_from_pillars(
    pillars: list[PillarScore], weights: dict[str, float] | None = None
) -> tuple[float, dict[str, float]]:
    """Weighted-average the pillar scores, renormalizing around missing ones.

    Pure and side-effect free: pillars with ``score is None`` are dropped
    and the remaining weights are renormalized to sum to 1, so a stock
    missing (say) analyst coverage is judged entirely on the other four
    pillars rather than being penalized for the gap. Returns the composite
    score and the effective (post-renormalization) weights actually used.

    Raises ValueError if every pillar is None (nothing to score).
    """
    weights = weights if weights is not None else DEFAULT_WEIGHTS
    usable = [p for p in pillars if p.score is not None]
    if not usable:
        raise ValueError("All pillars are None; cannot compute a composite score")

    raw_weights = {p.name: weights.get(p.name, 0.0) for p in usable}
    total = sum(raw_weights.values())
    if total <= 0:
        # Configured weights don't cover any usable pillar; fall back to
        # equal weighting rather than raising, since there IS usable data.
        effective_weights = {name: 1.0 / len(usable) for name in raw_weights}
    else:
        effective_weights = {name: w / total for name, w in raw_weights.items()}

    composite = sum(p.score * effective_weights[p.name] for p in usable)  # type: ignore[operator]
    return composite, effective_weights


def score_stock(
    ticker: str, weights: dict[str, float] | None = None, period: str = "2y"
) -> StockScore:
    """Fetch data for ``ticker`` and compute its composite multi-factor score.

    Lets ``TickerNotFoundError`` from ``get_price_history`` propagate — an
    unknown ticker has no history to score at all.
    """
    history = get_price_history(ticker, period=period)
    info = get_company_info(ticker)
    analysts = get_analyst_data(ticker)
    financials = get_financial_history(ticker)

    pillars = [
        score_value(info),
        score_growth(info, financials),
        score_technicals(history),
        score_trend_reversion(history),
        score_analysts(analysts, info.current_price),
    ]

    composite, effective_weights = composite_from_pillars(pillars, weights)
    return StockScore(
        ticker=info.ticker or ticker.upper(),
        composite=composite,
        label=label_for(composite),
        pillars=pillars,
        weights_used=effective_weights,
    )
