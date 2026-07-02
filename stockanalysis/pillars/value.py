"""Value pillar: classic valuation multiples scored via piecewise-linear bands.

Each ratio is scored independently and skipped (not zeroed) when its inputs
are missing or non-sensical, so a stock with a thin data set is judged on
what's actually known about it rather than being penalized for gaps.
"""

from __future__ import annotations

from stockanalysis.data import CompanyInfo
from stockanalysis.types import PillarScore, Signal, weighted_signal_score

_PILLAR_NAME = "Value"

# (threshold, score) pairs, ascending threshold, interpolated linearly between
# points and clamped at the ends. Lower multiples score higher throughout.
_PE_BANDS: list[tuple[float, float]] = [(8, 90), (15, 75), (25, 55), (40, 35), (60, 15)]
_PEG_BANDS: list[tuple[float, float]] = [(0.5, 90), (1, 80), (2, 50), (3, 30), (4, 15)]
_PB_BANDS: list[tuple[float, float]] = [(0.8, 85), (1.5, 70), (3, 55), (6, 35), (10, 15)]
_EV_EBITDA_BANDS: list[tuple[float, float]] = [(6, 90), (8, 75), (15, 50), (25, 30), (35, 15)]
_FCF_YIELD_BANDS: list[tuple[float, float]] = [
    (0.0, 30),
    (0.02, 45),
    (0.05, 70),
    (0.08, 85),
    (0.12, 95),
]


def band_score(value: float, bands: list[tuple[float, float]]) -> float:
    """Piecewise-linear interpolation over ``bands`` = [(threshold, score), ...].

    ``bands`` must be sorted ascending by threshold. Values outside the
    range are clamped to the nearest endpoint's score.
    """
    if value <= bands[0][0]:
        return bands[0][1]
    if value >= bands[-1][0]:
        return bands[-1][1]
    for (x0, y0), (x1, y1) in zip(bands, bands[1:]):
        if x0 <= value <= x1:
            t = (value - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return bands[-1][1]  # unreachable given the bounds checks above


def _ratio_signal(
    name: str, value: float | None, bands: list[tuple[float, float]], label: str
) -> Signal | None:
    if value is None or value <= 0:
        return None
    score = band_score(value, bands)
    return Signal(name=name, value=f"{label} {value:.2f}", score=score, weight=1.0)


def _fcf_yield_signal(info: CompanyInfo) -> Signal | None:
    if info.free_cashflow is None or info.market_cap is None or info.market_cap <= 0:
        return None
    yield_ = info.free_cashflow / info.market_cap
    if yield_ < 0:
        score = 20.0
        value = f"FCF yield {yield_ * 100:.1f}% (negative — cash burn)"
    else:
        score = band_score(yield_, _FCF_YIELD_BANDS)
        value = f"FCF yield {yield_ * 100:.1f}%"
    return Signal(name="FCF yield", value=value, score=score, weight=1.0)


def _forward_pe_signal(info: CompanyInfo) -> Signal | None:
    if info.forward_pe is None or info.trailing_pe is None:
        return None
    if info.forward_pe <= 0 or info.trailing_pe <= 0:
        return None
    if info.forward_pe < info.trailing_pe:
        score, note = 70.0, "earnings expected to grow"
    elif info.forward_pe > info.trailing_pe:
        score, note = 35.0, "earnings expected to contract"
    else:
        score, note = 50.0, "flat"
    value = f"Forward P/E {info.forward_pe:.1f} vs trailing {info.trailing_pe:.1f} ({note})"
    return Signal(name="Forward vs trailing P/E", value=value, score=score, weight=0.5)


def score_value(info: CompanyInfo) -> PillarScore:
    """Score classic valuation multiples for an equity."""
    if not info.is_equity:
        return PillarScore(
            name=_PILLAR_NAME, score=None, note="Valuation not applicable to ETFs/funds"
        )

    signals = [
        s
        for s in (
            _ratio_signal("Trailing P/E", info.trailing_pe, _PE_BANDS, "P/E"),
            _ratio_signal("PEG ratio", info.peg_ratio, _PEG_BANDS, "PEG"),
            _ratio_signal("Price/Book", info.price_to_book, _PB_BANDS, "P/B"),
            _ratio_signal("EV/EBITDA", info.ev_to_ebitda, _EV_EBITDA_BANDS, "EV/EBITDA"),
            _fcf_yield_signal(info),
            _forward_pe_signal(info),
        )
        if s is not None
    ]

    if not signals:
        return PillarScore(name=_PILLAR_NAME, score=None, note="No valuation data available")

    return PillarScore(name=_PILLAR_NAME, score=weighted_signal_score(signals), signals=signals)
