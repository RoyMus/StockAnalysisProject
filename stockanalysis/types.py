"""Shared result types for the analysis pillars and composite score."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Signal:
    """A single named sub-signal inside a pillar.

    ``score`` is normalized to 0-100 (50 = neutral). ``value`` is the
    human-readable raw reading the score was derived from, so the UI can
    show *why* the pillar scored the way it did.
    """

    name: str
    value: str
    score: float
    weight: float = 1.0


@dataclass
class PillarScore:
    """Score for one analysis pillar (0-100), with full signal breakdown.

    ``score`` is None when the pillar has no usable data (e.g. value
    analysis on an ETF); the composite engine re-weights around it.
    """

    name: str
    score: float | None
    signals: list[Signal] = field(default_factory=list)
    note: str = ""


@dataclass
class StockScore:
    """Composite result for a ticker."""

    ticker: str
    composite: float
    label: str
    pillars: list[PillarScore]
    weights_used: dict[str, float]


def weighted_signal_score(signals: list[Signal]) -> float | None:
    """Combine sub-signals into a pillar score via weighted mean."""
    if not signals:
        return None
    total_weight = sum(s.weight for s in signals)
    if total_weight == 0:
        return None
    return sum(s.score * s.weight for s in signals) / total_weight


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def linear_scale(value: float, worst: float, best: float) -> float:
    """Map ``value`` linearly onto 0-100 between ``worst`` and ``best``.

    Works for both ascending (best > worst) and descending metrics
    (best < worst, e.g. P/E where lower is better).
    """
    if best == worst:
        return 50.0
    return clamp((value - worst) / (best - worst) * 100.0)
