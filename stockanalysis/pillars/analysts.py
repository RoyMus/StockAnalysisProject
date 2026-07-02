"""Analysts pillar: consensus ratings and price-target upside.

The most lagging of the five pillars (analysts tend to herd and revise
after the market has already moved), so it carries the composite's lowest
default weight and is further shrunk toward neutral when coverage is thin.
"""

from __future__ import annotations

from stockanalysis.data import AnalystData
from stockanalysis.types import PillarScore, Signal, linear_scale, weighted_signal_score

_PILLAR_NAME = "Analysts"


def _consensus_signal(analysts: AnalystData) -> Signal | None:
    total = analysts.total_ratings
    if total <= 0:
        return None

    score = (
        analysts.strong_buy * 100
        + analysts.buy * 75
        + analysts.hold * 50
        + analysts.sell * 25
        + analysts.strong_sell * 0
    ) / total
    value = (
        f"{analysts.strong_buy} strong buy / {analysts.buy} buy / {analysts.hold} hold / "
        f"{analysts.sell} sell / {analysts.strong_sell} strong sell ({total} ratings)"
    )
    return Signal(name="Analyst consensus", value=value, score=score, weight=5.0)


def _price_target_signal(analysts: AnalystData, current_price: float | None) -> Signal | None:
    if analysts.target_mean is None or current_price is None or current_price <= 0:
        return None

    upside = (analysts.target_mean - current_price) / current_price
    score = linear_scale(upside, -0.20, 0.30)

    dispersion_note = ""
    have_range = analysts.target_high is not None and analysts.target_low is not None
    if have_range and analysts.target_mean:
        spread = (analysts.target_high - analysts.target_low) / analysts.target_mean
        if spread > 0.5:
            score = 50 + 0.7 * (score - 50)
            dispersion_note = f", wide target dispersion ({spread:.0%}) haircut applied"

    value = (
        f"{upside * 100:+.1f}% upside to mean target ${analysts.target_mean:.2f}{dispersion_note}"
    )
    return Signal(name="Price target upside", value=value, score=score, weight=5.0)


def score_analysts(analysts: AnalystData, current_price: float | None) -> PillarScore:
    """Score consensus ratings and price-target upside, shrunk for thin coverage."""
    signals = [
        s
        for s in (_consensus_signal(analysts), _price_target_signal(analysts, current_price))
        if s is not None
    ]

    if not signals:
        return PillarScore(name=_PILLAR_NAME, score=None, note="No analyst coverage")

    raw = weighted_signal_score(signals)
    assert raw is not None  # signals is non-empty

    n = analysts.num_analysts or analysts.total_ratings
    shrink = min(1.0, n / 5) if n else 0.0
    final = 50 + shrink * (raw - 50)

    note = f"Thin coverage ({n} analyst(s)): shrunk toward neutral" if n < 5 else ""
    return PillarScore(name=_PILLAR_NAME, score=final, signals=signals, note=note)
