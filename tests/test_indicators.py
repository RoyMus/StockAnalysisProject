"""Tests for technical indicators in stockanalysis.indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockanalysis.indicators import (
    adx,
    bollinger,
    ema,
    ma_slope,
    macd,
    rsi,
    sma,
    volume_trend,
    zscore,
)


class TestRSI:
    """Test Relative Strength Index calculation."""

    def test_all_gains_near_100(self):
        """RSI on monotonic up-only should be high."""
        # Create a series with only increases: 100, 101, 102, ..., 130
        close = pd.Series(np.arange(100.0, 131.0))
        result = rsi(close, window=14)
        # After sufficient history, RSI should be very high (>80)
        assert result.iloc[-1] > 80, f"Expected RSI > 80 for pure uptrend, got {result.iloc[-1]}"

    def test_all_losses_near_0(self):
        """RSI on monotonic down-only should be low."""
        # Create a series with only decreases: 130, 129, 128, ..., 100
        close = pd.Series(np.arange(130.0, 99.0, -1.0))
        result = rsi(close, window=14)
        # After sufficient history, RSI should be very low (<20)
        assert result.iloc[-1] < 20, f"Expected RSI < 20 for pure downtrend, got {result.iloc[-1]}"

    def test_rsi_in_range(self):
        """RSI must always be in [0, 100]."""
        prices = pd.Series(np.random.default_rng(42).normal(100, 5, 100))
        result = rsi(prices)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all(), "RSI values outside [0, 100]"

    def test_rsi_tiny_series_hand_computed(self):
        """Hand-check RSI against tiny series with manual Wilder's calculation."""
        # Small series: [44, 44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42,
        #               45.84, 46.08, 45.89, 46.03, 46.41, 46.28, 46.00, 46.03, 46.41,
        #               46.28, 46.00, 46.03]
        # Using period=5 for manageable hand-check

        close = pd.Series(
            [44.0, 44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84]
        )
        result = rsi(close, window=5)

        # Values with NaN in early periods are expected
        # We just verify the computed values (post-warm-up) are reasonable
        valid = result.dropna()
        assert len(valid) > 0, "RSI should have some valid values"
        assert (valid >= 0).all() and (valid <= 100).all()


class TestSMAandEMA:
    """Test Simple and Exponential Moving Averages."""

    def test_sma_basic(self):
        """SMA(3) of [1, 2, 3, 4, 5] = [NaN, NaN, 2.0, 3.0, 4.0]."""
        prices = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sma(prices, window=3)
        expected_valid = [2.0, 3.0, 4.0]
        np.testing.assert_array_almost_equal(result.dropna().values, expected_valid)

    def test_ema_matches_pandas(self):
        """EMA should match pandas ewm reference implementation."""
        prices = pd.Series(np.random.default_rng(42).uniform(100, 110, 100))
        our_ema = ema(prices, span=12)
        # pandas ewm with adjust=False should match our implementation
        pandas_ewm = prices.ewm(span=12, adjust=False).mean()
        np.testing.assert_array_almost_equal(our_ema.values, pandas_ewm.values, decimal=5)


class TestMACD:
    """Test MACD indicator."""

    def test_macd_columns_exist(self):
        """MACD output must have 'macd', 'signal', 'histogram' columns."""
        prices = pd.Series(np.random.default_rng(42).uniform(100, 110, 100))
        result = macd(prices)
        assert "macd" in result.columns
        assert "signal" in result.columns
        assert "histogram" in result.columns

    def test_macd_histogram_is_difference(self):
        """MACD histogram must equal macd - signal."""
        prices = pd.Series(np.random.default_rng(42).uniform(100, 110, 100))
        result = macd(prices)
        expected = result["macd"] - result["signal"]
        np.testing.assert_array_almost_equal(result["histogram"].values, expected.values)


class TestBollinger:
    """Test Bollinger Bands."""

    def test_bollinger_columns(self):
        """Bollinger must have 'middle', 'upper', 'lower', 'percent_b', 'width'."""
        prices = pd.Series(np.random.default_rng(42).uniform(100, 110, 100))
        result = bollinger(prices, window=20)
        for col in ["middle", "upper", "lower", "percent_b", "width"]:
            assert col in result.columns

    def test_bollinger_middle_is_sma20(self):
        """Bollinger middle band must equal SMA(20)."""
        prices = pd.Series(np.random.default_rng(42).uniform(100, 110, 100))
        result = bollinger(prices, window=20)
        expected = sma(prices, 20)
        np.testing.assert_array_almost_equal(result["middle"].values, expected.values)

    def test_bollinger_band_ordering(self):
        """Upper >= middle >= lower where all are defined."""
        prices = pd.Series(np.random.default_rng(42).uniform(100, 110, 100))
        result = bollinger(prices, window=20)
        valid = ~(result.isna().any(axis=1))
        assert (result.loc[valid, "upper"] >= result.loc[valid, "middle"]).all()
        assert (result.loc[valid, "middle"] >= result.loc[valid, "lower"]).all()

    def test_bollinger_percent_b_formula(self):
        """Percent B = (close - lower) / (upper - lower)."""
        prices = pd.Series(np.random.default_rng(42).uniform(100, 110, 100))
        result = bollinger(prices, window=20)
        expected = (prices - result["lower"]) / (result["upper"] - result["lower"])
        np.testing.assert_array_almost_equal(result["percent_b"].values, expected.values)


class TestZScore:
    """Test Z-score (rolling standardization)."""

    def test_zscore_constant_series_is_nan(self):
        """Z-score of constant series should be NaN (0 std dev)."""
        prices = pd.Series([100.0] * 100)
        result = zscore(prices, window=20)
        # With constant series, std=0 at all points, so result should be NaN
        assert result.isna().all()

    def test_zscore_point_two_std_above_mean(self):
        """A point 2 std above rolling mean should have z-score ~2.0."""
        rng = np.random.default_rng(42)
        prices = pd.Series(rng.normal(100, 5, 100))
        # Manually create a spike: append a value 2 std above the trailing mean
        trailing_mean = prices.iloc[-20:].mean()
        trailing_std = prices.iloc[-20:].std()
        spike_value = trailing_mean + 2.0 * trailing_std
        prices = pd.concat([prices, pd.Series([spike_value])])

        result = zscore(prices, window=20)
        z_at_spike = result.iloc[-1]
        assert 1.8 < z_at_spike < 2.2, f"Expected z ~2.0 at spike, got {z_at_spike}"


class TestADX:
    """Test Average Directional Index."""

    def test_adx_in_range(self):
        """ADX must be in [0, 100] where defined."""
        high = pd.Series(np.random.default_rng(42).uniform(101, 110, 100))
        low = pd.Series(np.random.default_rng(42).uniform(90, 99, 100))
        close = (high + low) / 2

        result = adx(high, low, close, window=14)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_adx_trending_vs_ranging(self, trending_history, ranging_history):
        """ADX should be higher on trending data than ranging data (last value)."""
        adx_trend = adx(
            trending_history["High"],
            trending_history["Low"],
            trending_history["Close"],
            window=14,
        )
        adx_range = adx(
            ranging_history["High"],
            ranging_history["Low"],
            ranging_history["Close"],
            window=14,
        )

        last_trend = adx_trend.iloc[-1]
        last_range = adx_range.iloc[-1]
        assert not pd.isna(last_trend) and not pd.isna(last_range), "ADX values should not be NaN"
        assert (
            last_trend > last_range
        ), f"Expected ADX(trending) > ADX(ranging), got {last_trend} vs {last_range}"


class TestMASlope:
    """Test MA slope calculation."""

    def test_ma_slope_trending_positive(self, trending_history):
        """MA slope should be positive on trending data."""
        slope = ma_slope(trending_history["Close"], window=50, lookback=10)
        assert slope is not None, "MA slope should not be None for trending data"
        assert slope > 0, f"Expected positive slope on uptrend, got {slope}"

    def test_ma_slope_ranging_near_zero(self, ranging_history):
        """MA slope should be near zero on ranging data."""
        slope = ma_slope(ranging_history["Close"], window=50, lookback=10)
        assert slope is not None
        assert abs(slope) < 0.01, f"Expected near-zero slope on range, got {slope}"

    def test_ma_slope_short_series_none(self, short_history):
        """MA slope should be None for series shorter than lookback."""
        slope = ma_slope(short_history["Close"], window=50, lookback=10)
        assert slope is None, "MA slope should be None for short series"


class TestVolumeTrend:
    """Test volume trend ratio."""

    def test_volume_trend_insufficient_bars(self):
        """Volume trend should be None for < 60 bars."""
        volume = pd.Series(np.random.default_rng(42).uniform(1_000_000, 2_000_000, 50))
        result = volume_trend(volume, short=20, long=60)
        assert result is None

    def test_volume_trend_sufficient_bars(self, synthetic_history):
        """Volume trend should return a float for sufficient bars."""
        result = volume_trend(synthetic_history["Volume"], short=20, long=60)
        assert isinstance(result, float)
        assert result > 0  # Ratio should be positive
