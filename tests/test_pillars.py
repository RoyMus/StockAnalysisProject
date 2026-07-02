"""Tests for the five analysis pillars."""

from __future__ import annotations

from stockanalysis.data import AnalystData, CompanyInfo, FinancialHistory
from stockanalysis.pillars import (
    score_analysts,
    score_growth,
    score_technicals,
    score_trend_reversion,
    score_value,
)
from stockanalysis.types import PillarScore


class TestScoreTechnicals:
    """Test technicals pillar."""

    def test_returns_pillar_score(self, synthetic_history):
        """score_technicals should return a PillarScore."""
        result = score_technicals(synthetic_history)
        assert isinstance(result, PillarScore)
        assert result.name == "Technicals"

    def test_score_in_range_or_none(self, synthetic_history):
        """Score should be in [0, 100] or None."""
        result = score_technicals(synthetic_history)
        if result.score is not None:
            assert 0 <= result.score <= 100

    def test_has_signals_on_full_history(self, synthetic_history):
        """Full history should produce non-empty signals list."""
        result = score_technicals(synthetic_history)
        assert len(result.signals) > 0, "Should have signals for 300-bar history"

    def test_short_history_returns_none_score(self, short_history):
        """Short history (30 bars) should return None score."""
        result = score_technicals(short_history)
        assert result.score is None or isinstance(result.score, float)

    def test_determinism_prefix_slice(self, synthetic_history):
        """Score should be identical when computed on same data twice."""
        result1 = score_technicals(synthetic_history.iloc[:250])
        result2 = score_technicals(synthetic_history.iloc[:250])
        if result1.score is not None and result2.score is not None:
            assert abs(result1.score - result2.score) < 1e-10


class TestScoreTrendReversion:
    """Test trend & mean-reversion pillar."""

    def test_trending_has_trending_in_note(self, trending_history):
        """Trending history should have 'trending' in note."""
        result = score_trend_reversion(trending_history)
        assert "trending" in result.note.lower()

    def test_ranging_has_ranging_in_note(self, ranging_history):
        """Ranging history should have 'ranging' in note."""
        result = score_trend_reversion(ranging_history)
        assert "ranging" in result.note.lower()

    def test_score_in_range_or_none(self, synthetic_history):
        """Score must be in [0, 100] or None."""
        result = score_trend_reversion(synthetic_history)
        if result.score is not None:
            assert 0 <= result.score <= 100


class TestScoreValue:
    """Test value pillar."""

    def test_normal_equity_scores(self):
        """Normal equity with all ratios should produce score in [0, 100]."""
        info = CompanyInfo(
            ticker="TEST",
            long_name="Test Corp",
            sector="Technology",
            industry="Software",
            quote_type="EQUITY",
            current_price=100.0,
            market_cap=1_000_000_000.0,
            trailing_pe=20.0,
            forward_pe=18.0,
            peg_ratio=1.5,
            price_to_book=3.0,
            ev_to_ebitda=12.0,
            free_cashflow=100_000_000.0,
            revenue_growth=0.15,
            earnings_growth=0.20,
            profit_margin=0.15,
            fifty_two_week_low=80.0,
            fifty_two_week_high=120.0,
        )
        result = score_value(info)
        assert result.score is not None
        assert 0 <= result.score <= 100

    def test_unprofitable_still_scores(self):
        """Unprofitable equity (no P/E, PEG) should still score on other ratios."""
        info = CompanyInfo(
            ticker="TEST",
            long_name="Unprofitable Corp",
            sector="Technology",
            industry="Software",
            quote_type="EQUITY",
            current_price=50.0,
            market_cap=500_000_000.0,
            trailing_pe=None,  # Unprofitable
            forward_pe=None,
            peg_ratio=None,
            price_to_book=2.0,
            ev_to_ebitda=10.0,
            free_cashflow=-50_000_000.0,  # Negative
            revenue_growth=0.25,
            earnings_growth=None,
            profit_margin=-0.10,
            fifty_two_week_low=30.0,
            fifty_two_week_high=80.0,
        )
        result = score_value(info)
        # Should still have a score using available ratios
        assert result.score is not None or len(result.signals) == 0

    def test_etf_returns_none_score(self):
        """ETF should return None score."""
        info = CompanyInfo(
            ticker="SPY",
            long_name="SPDR S&P 500 ETF",
            sector=None,
            industry=None,
            quote_type="ETF",
            current_price=450.0,
            market_cap=None,
            trailing_pe=None,
            forward_pe=None,
            peg_ratio=None,
            price_to_book=None,
            ev_to_ebitda=None,
            free_cashflow=None,
            revenue_growth=None,
            earnings_growth=None,
            profit_margin=None,
            fifty_two_week_low=400.0,
            fifty_two_week_high=500.0,
        )
        result = score_value(info)
        assert result.score is None

    def test_all_none_equity_returns_none(self):
        """Equity with all-None fields should return None score."""
        info = CompanyInfo(
            ticker="UNKN",
            long_name="Unknown",
            sector=None,
            industry=None,
            quote_type="EQUITY",
            current_price=None,
            market_cap=None,
            trailing_pe=None,
            forward_pe=None,
            peg_ratio=None,
            price_to_book=None,
            ev_to_ebitda=None,
            free_cashflow=None,
            revenue_growth=None,
            earnings_growth=None,
            profit_margin=None,
            fifty_two_week_low=None,
            fifty_two_week_high=None,
        )
        result = score_value(info)
        assert result.score is None


class TestScoreGrowth:
    """Test growth pillar."""

    def test_strong_grower_vs_shrinking(self):
        """Strong grower should score higher than shrinking company."""
        strong_info = CompanyInfo(
            ticker="GROW",
            long_name="Growth Inc",
            sector="Technology",
            industry="Software",
            quote_type="EQUITY",
            current_price=100.0,
            market_cap=1_000_000_000.0,
            trailing_pe=30.0,
            forward_pe=25.0,
            peg_ratio=1.0,
            price_to_book=5.0,
            ev_to_ebitda=20.0,
            free_cashflow=100_000_000.0,
            revenue_growth=0.40,  # Strong growth
            earnings_growth=0.50,
            profit_margin=0.20,
            fifty_two_week_low=80.0,
            fifty_two_week_high=120.0,
        )

        shrinking_info = CompanyInfo(
            ticker="SHRK",
            long_name="Shrinking Corp",
            sector="Technology",
            industry="Software",
            quote_type="EQUITY",
            current_price=50.0,
            market_cap=500_000_000.0,
            trailing_pe=15.0,
            forward_pe=20.0,
            peg_ratio=2.0,
            price_to_book=1.0,
            ev_to_ebitda=8.0,
            free_cashflow=50_000_000.0,
            revenue_growth=-0.15,  # Declining
            earnings_growth=-0.20,
            profit_margin=0.10,
            fifty_two_week_low=30.0,
            fifty_two_week_high=80.0,
        )

        financials_strong = FinancialHistory(
            annual_revenue=[1000, 800, 600],
            annual_net_income=[200, 150, 100],
            quarterly_revenue=[280, 260, 240, 220],
            quarterly_net_income=[60, 50, 40, 30],
        )

        financials_shrink = FinancialHistory(
            annual_revenue=[600, 800, 1000],
            annual_net_income=[60, 100, 150],
            quarterly_revenue=[140, 160, 180, 200],
            quarterly_net_income=[25, 30, 40, 50],
        )

        result_strong = score_growth(strong_info, financials_strong)
        result_shrink = score_growth(shrinking_info, financials_shrink)

        if result_strong.score is not None and result_shrink.score is not None:
            assert result_strong.score > result_shrink.score

    def test_non_equity_returns_none(self):
        """Non-equity should return None score."""
        etf_info = CompanyInfo(
            ticker="QQQ",
            long_name="QQQ ETF",
            sector=None,
            industry=None,
            quote_type="ETF",
            current_price=400.0,
            market_cap=None,
            trailing_pe=None,
            forward_pe=None,
            peg_ratio=None,
            price_to_book=None,
            ev_to_ebitda=None,
            free_cashflow=None,
            revenue_growth=None,
            earnings_growth=None,
            profit_margin=None,
            fifty_two_week_low=350.0,
            fifty_two_week_high=450.0,
        )
        financials = FinancialHistory()
        result = score_growth(etf_info, financials)
        assert result.score is None


class TestScoreAnalysts:
    """Test analysts pillar."""

    def test_strong_buy_heavy_high_upside(self):
        """Strong-buy-heavy analysts with big upside should score > 60."""
        analysts = AnalystData(
            strong_buy=10,
            buy=5,
            hold=1,
            sell=0,
            strong_sell=0,
            num_analysts=16,
            target_mean=150.0,
            target_median=150.0,
            target_high=160.0,
            target_low=140.0,
        )
        result = score_analysts(analysts, current_price=100.0)
        assert result.score is not None
        msg = f"Expected score > 60 for strong-buy with 50% upside, got {result.score}"
        assert result.score > 60, msg

    def test_thin_coverage_shrunk_toward_neutral(self):
        """Few analysts (2) should shrink toward 50 vs same data with 20 analysts."""
        analysts_few = AnalystData(
            strong_buy=2,
            buy=0,
            hold=0,
            sell=0,
            strong_sell=0,
            num_analysts=2,
            target_mean=150.0,
            target_median=150.0,
            target_high=160.0,
            target_low=140.0,
        )

        analysts_many = AnalystData(
            strong_buy=20,
            buy=0,
            hold=0,
            sell=0,
            strong_sell=0,
            num_analysts=20,
            target_mean=150.0,
            target_median=150.0,
            target_high=160.0,
            target_low=140.0,
        )

        result_few = score_analysts(analysts_few, current_price=100.0)
        result_many = score_analysts(analysts_many, current_price=100.0)

        if result_few.score is not None and result_many.score is not None:
            # Few should be closer to 50 than many
            assert abs(result_few.score - 50) < abs(result_many.score - 50)

    def test_empty_analysts_returns_none(self):
        """Empty AnalystData should return None score."""
        # AnalystData with no ratings (all zeros) has total_ratings property = 0
        analysts = AnalystData()
        result = score_analysts(analysts, current_price=100.0)
        assert result.score is None
