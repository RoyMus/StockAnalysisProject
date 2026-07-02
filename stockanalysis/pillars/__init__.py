"""The five analysis pillars that feed the composite stock score."""

from __future__ import annotations

from stockanalysis.pillars.analysts import score_analysts
from stockanalysis.pillars.growth import score_growth
from stockanalysis.pillars.regime import score_trend_reversion
from stockanalysis.pillars.technicals import score_technicals
from stockanalysis.pillars.value import score_value

__all__ = [
    "score_technicals",
    "score_trend_reversion",
    "score_analysts",
    "score_value",
    "score_growth",
]
