"""Statistical price forecast using ARIMA from statsmodels.

This module provides a baseline forecast for educational and research purposes
only. The ARIMA model is a statistical technique that captures historical
price trends and volatility patterns, but is NOT investment advice. Stock
prices are influenced by many factors (news, earnings, macro events) that
historical data alone cannot predict. Do not use this forecast as a basis
for trading or investment decisions.

Methodology:
- Log-transforms Close prices for variance stabilization
- Chooses differencing order (d) via Augmented Dickey–Fuller test
- Grid-searches ARIMA(p, d, q) orders over p, q in {0,1,2}
- Selects model with lowest AIC
- Generates point forecast and 80% confidence intervals
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller


@dataclass(frozen=True)
class ForecastResult:
    """Results of a statistical price forecast.

    Attributes:
        dates: Future business days (pd.DatetimeIndex).
        mean: Point forecast in price space (pd.Series).
        lower: 80% confidence interval lower bound in price space (pd.Series).
        upper: 80% confidence interval upper bound in price space (pd.Series).
        order: ARIMA order tuple (p, d, q).
        aic: Akaike Information Criterion of the fitted model.
    """

    dates: pd.DatetimeIndex
    mean: pd.Series
    lower: pd.Series
    upper: pd.Series
    order: tuple[int, int, int]
    aic: float


def forecast_prices(history: pd.DataFrame, horizon_days: int = 30) -> ForecastResult | None:
    """Forecast future stock prices using ARIMA.

    Args:
        history: DataFrame with DatetimeIndex and 'Close' column (daily prices).
        horizon_days: Number of business days to forecast (default 30).

    Returns:
        ForecastResult with dates, mean, and 80% confidence intervals, or None
        if the model cannot be fit or history is too short.

    Caveats:
        This is a statistical baseline, not investment advice. The ARIMA model
        fits historical price patterns but cannot capture unexpected events or
        regime changes. Always consult other analysis and professional advisors.
    """
    # Guard: minimum 100 rows of history
    if history is None or len(history) < 100:
        return None

    # Use recent 500 rows for speed and regime relevance
    recent = history.tail(500).copy()
    if recent is None or len(recent) < 100:
        return None

    # Ensure we have a Close column
    if "Close" not in recent.columns:
        return None

    close = recent["Close"].astype(float)

    # Log-transform for variance stabilization
    log_close = np.log(close)

    # Step 1: Choose differencing order (d) via ADF test
    d = _choose_differencing_order(log_close)

    # Step 2: Grid search for best p, q
    best_order, best_aic, best_model = _grid_search_arima(log_close, d)

    if best_model is None:
        return None

    # Step 3: Generate forecast
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecast = best_model.get_forecast(steps=horizon_days)
        conf_int = forecast.conf_int(alpha=0.2)  # 80% CI

    # Transform back to price space
    mean_log = forecast.predicted_mean.values
    lower_log = conf_int.iloc[:, 0].values
    upper_log = conf_int.iloc[:, 1].values

    mean_price = np.exp(mean_log)
    lower_price = np.exp(lower_log)
    upper_price = np.exp(upper_log)

    # Generate future business days starting from the next business day
    last_date = close.index[-1]
    future_dates = pd.bdate_range(start=last_date, periods=horizon_days + 1)[1:]

    # Build result Series with aligned indices
    result_mean = pd.Series(mean_price, index=future_dates)
    result_lower = pd.Series(lower_price, index=future_dates)
    result_upper = pd.Series(upper_price, index=future_dates)

    return ForecastResult(
        dates=future_dates,
        mean=result_mean,
        lower=result_lower,
        upper=result_upper,
        order=best_order,
        aic=best_aic,
    )


def _choose_differencing_order(log_series: pd.Series) -> int:
    """Determine differencing order via Augmented Dickey–Fuller test.

    Returns:
        d = 0 if log_series is stationary (p < 0.05)
        d = 1 if first difference is stationary
        d = 2 otherwise
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Test log series
        clean_series = log_series.dropna()
        if len(clean_series) < 10:
            return 1  # Not enough data for ADF test, use default d=1

        result = adfuller(clean_series, autolag="AIC")
        pvalue = result[1]

        if pvalue < 0.05:
            return 0

        # Test first difference
        diff1 = log_series.diff().dropna()
        if len(diff1) < 10:
            return 1

        result = adfuller(diff1, autolag="AIC")
        pvalue = result[1]

        if pvalue < 0.05:
            return 1

        return 2


def _grid_search_arima(
    log_series: pd.Series, d: int
) -> tuple[tuple[int, int, int], float, ARIMA | None]:
    """Grid search over p, q in {0,1,2} for given differencing order.

    Returns:
        (best_order, best_aic, best_model) or (None, float('inf'), None) if all fits fail.
    """
    best_order = None
    best_aic = float("inf")
    best_model = None

    for p in range(3):
        for q in range(3):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = ARIMA(log_series, order=(p, d, q))
                    fitted = model.fit()

                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_order = (p, d, q)
                    best_model = fitted

            except Exception:
                # Some orders fail to converge or have other issues
                continue

    return best_order, best_aic, best_model
