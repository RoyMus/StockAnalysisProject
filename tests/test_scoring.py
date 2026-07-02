"""Tests for composite scoring."""

from __future__ import annotations

import pytest

from stockanalysis.scoring import composite_from_pillars, label_for
from stockanalysis.types import PillarScore


class TestCompositeFromPillars:
    """Test composite score calculation from pillars."""

    def test_known_scores_exact_result(self):
        """Known pillar scores should produce exact expected weighted mean."""
        pillars = [
            PillarScore(name="Value", score=80.0),
            PillarScore(name="Growth", score=60.0),
            PillarScore(name="Technicals", score=70.0),
            PillarScore(name="Trend & Mean-Reversion", score=50.0),
            PillarScore(name="Analysts", score=40.0),
        ]
        weights = {
            "Value": 0.25,
            "Growth": 0.20,
            "Technicals": 0.20,
            "Trend & Mean-Reversion": 0.20,
            "Analysts": 0.15,
        }

        expected = (
            80.0 * 0.25 + 60.0 * 0.20 + 70.0 * 0.20 + 50.0 * 0.20 + 40.0 * 0.15
        )
        composite, used_weights = composite_from_pillars(pillars, weights)
        assert abs(composite - expected) < 1e-10

    def test_one_none_renormalizes(self):
        """One None pillar should renormalize weights to sum to 1."""
        pillars = [
            PillarScore(name="Value", score=80.0),
            PillarScore(name="Growth", score=60.0),
            PillarScore(name="Technicals", score=None),  # Missing
            PillarScore(name="Trend & Mean-Reversion", score=50.0),
            PillarScore(name="Analysts", score=40.0),
        ]
        weights = {
            "Value": 0.25,
            "Growth": 0.20,
            "Technicals": 0.20,
            "Trend & Mean-Reversion": 0.20,
            "Analysts": 0.15,
        }

        composite, used_weights = composite_from_pillars(pillars, weights)
        # Check that used weights sum to 1
        total_weight = sum(used_weights.values())
        assert abs(total_weight - 1.0) < 1e-10

        # Technicals should not be in used_weights
        assert "Technicals" not in used_weights

    def test_all_none_raises_error(self):
        """All-None pillars should raise ValueError."""
        pillars = [
            PillarScore(name="Value", score=None),
            PillarScore(name="Growth", score=None),
            PillarScore(name="Technicals", score=None),
            PillarScore(name="Trend & Mean-Reversion", score=None),
            PillarScore(name="Analysts", score=None),
        ]

        with pytest.raises(ValueError, match="All pillars are None"):
            composite_from_pillars(pillars)


class TestLabelFor:
    """Test rating label mapping."""

    def test_label_boundaries(self):
        """Test boundaries: 19.99→Strong Sell, 20→Sell, 59.9→Hold, 60→Buy, 80→Strong Buy."""
        assert label_for(19.99) == "Strong Sell"
        assert label_for(20.0) == "Sell"
        assert label_for(59.9) == "Hold"
        assert label_for(60.0) == "Buy"
        assert label_for(80.0) == "Strong Buy"
        assert label_for(100.0) == "Strong Buy"

    def test_label_in_middle_ranges(self):
        """Test mid-range scores."""
        assert label_for(0.0) == "Strong Sell"
        assert label_for(50.0) == "Hold"
        assert label_for(70.0) == "Buy"
        assert label_for(90.0) == "Strong Buy"
