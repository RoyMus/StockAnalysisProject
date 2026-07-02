"""Tests for price forecasting."""

from __future__ import annotations

from stockanalysis.forecasting import forecast_prices


class TestForecastPrices:
    """Test ARIMA price forecasting."""

    def test_forecast_returns_result_on_full_history(self, synthetic_history):
        """Forecast on 300-bar seeded synthetic frame should not be None."""
        result = forecast_prices(synthetic_history, horizon_days=30)
        assert result is not None

    def test_forecast_length_matches_horizon(self, synthetic_history):
        """Forecast length should match horizon."""
        horizon = 30
        result = forecast_prices(synthetic_history, horizon_days=horizon)
        if result is not None:
            assert len(result.mean) == horizon

    def test_forecast_bounds_sensible(self, synthetic_history):
        """Lower <= mean <= upper for forecast bounds."""
        result = forecast_prices(synthetic_history, horizon_days=30)
        if result is not None:
            assert (result.lower <= result.mean).all()
            assert (result.mean <= result.upper).all()

    def test_forecast_ci_widens(self, synthetic_history):
        """Confidence interval width should increase over horizon."""
        result = forecast_prices(synthetic_history, horizon_days=30)
        if result is not None:
            width_first = result.upper.iloc[0] - result.lower.iloc[0]
            width_last = result.upper.iloc[-1] - result.lower.iloc[-1]
            # CI should widen into the future
            msg = f"CI width {width_last} should be > first {width_first}"
            assert width_last > width_first, msg

    def test_forecast_dates_after_history(self, synthetic_history):
        """Forecast dates should start after history end."""
        result = forecast_prices(synthetic_history, horizon_days=30)
        if result is not None:
            history_last = synthetic_history.index[-1]
            forecast_first = result.dates[0]
            assert forecast_first > history_last

    def test_insufficient_bars_returns_none(self, short_history):
        """Fewer than 100 bars should return None."""
        result = forecast_prices(short_history, horizon_days=30)
        assert result is None

    def test_missing_close_column_returns_none(self, synthetic_history):
        """DataFrame missing Close column should return None."""
        df_no_close = synthetic_history.drop(columns=["Close"])
        result = forecast_prices(df_no_close, horizon_days=30)
        assert result is None
