"""Growth pillar: YoY growth rates plus multi-period trend confirmation."""

from __future__ import annotations

from stockanalysis.data import CompanyInfo, FinancialHistory
from stockanalysis.types import PillarScore, Signal, linear_scale, weighted_signal_score

_PILLAR_NAME = "Growth"


def _revenue_growth_signal(info: CompanyInfo) -> Signal | None:
    if info.revenue_growth is None:
        return None
    g = info.revenue_growth
    return Signal(
        name="Revenue growth YoY",
        value=f"Revenue growth {g * 100:+.1f}% YoY",
        score=linear_scale(g, -0.10, 0.30),
        weight=3.0,
    )


def _earnings_growth_signal(info: CompanyInfo) -> Signal | None:
    if info.earnings_growth is None:
        return None
    g = info.earnings_growth
    return Signal(
        name="Earnings growth YoY",
        value=f"Earnings growth {g * 100:+.1f}% YoY",
        score=linear_scale(g, -0.15, 0.40),
        weight=3.0,
    )


def _revenue_cagr_signal(financials: FinancialHistory) -> Signal | None:
    values = financials.annual_revenue  # newest first
    if len(values) < 3 or any(v <= 0 for v in values):
        return None
    n = len(values)
    newest, oldest = values[0], values[-1]
    cagr = (newest / oldest) ** (1 / (n - 1)) - 1
    return Signal(
        name="Revenue CAGR",
        value=f"Revenue CAGR {cagr * 100:+.1f}% over {n} yrs",
        score=linear_scale(cagr, -0.05, 0.25),
        weight=2.0,
    )


def _growth_consistency_signal(financials: FinancialHistory) -> Signal | None:
    values = financials.quarterly_revenue  # newest first
    if len(values) < 5:
        return None
    pairs = len(values) - 1
    grew = sum(1 for newer, older in zip(values, values[1:]) if newer > older)
    fraction = grew / pairs
    return Signal(
        name="Growth consistency",
        value=f"{grew}/{pairs} quarters grew sequentially ({fraction * 100:.0f}%)",
        score=linear_scale(fraction, 0.2, 1.0),
        weight=2.0,
    )


def score_growth(info: CompanyInfo, financials: FinancialHistory) -> PillarScore:
    """Score revenue/earnings growth and its consistency over time."""
    if not info.is_equity:
        return PillarScore(
            name=_PILLAR_NAME, score=None, note="Growth analysis not applicable to ETFs/funds"
        )

    signals = [
        s
        for s in (
            _revenue_growth_signal(info),
            _earnings_growth_signal(info),
            _revenue_cagr_signal(financials),
            _growth_consistency_signal(financials),
        )
        if s is not None
    ]

    if not signals:
        return PillarScore(name=_PILLAR_NAME, score=None, note="No growth data available")

    return PillarScore(name=_PILLAR_NAME, score=weighted_signal_score(signals), signals=signals)
