"""Typed data layer — the only module that talks to yfinance.

Every fetcher returns a frozen dataclass with normalized, defensively
extracted fields, so the analysis pillars never touch raw yfinance
dicts/DataFrames and never need their own try/except plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import yfinance as yf


class TickerNotFoundError(Exception):
    """Raised when a ticker returns no price data at all."""


@dataclass(frozen=True)
class CompanyInfo:
    ticker: str
    long_name: str | None
    sector: str | None
    industry: str | None
    quote_type: str | None  # EQUITY / ETF / MUTUALFUND / INDEX ...
    current_price: float | None
    market_cap: float | None
    trailing_pe: float | None
    forward_pe: float | None
    peg_ratio: float | None
    price_to_book: float | None
    ev_to_ebitda: float | None
    free_cashflow: float | None
    revenue_growth: float | None  # YoY, decimal
    earnings_growth: float | None  # YoY, decimal
    profit_margin: float | None
    fifty_two_week_low: float | None
    fifty_two_week_high: float | None

    @property
    def is_equity(self) -> bool:
        return (self.quote_type or "").upper() == "EQUITY"


@dataclass(frozen=True)
class AnalystData:
    strong_buy: int = 0
    buy: int = 0
    hold: int = 0
    sell: int = 0
    strong_sell: int = 0
    num_analysts: int | None = None
    target_mean: float | None = None
    target_median: float | None = None
    target_high: float | None = None
    target_low: float | None = None
    trend: pd.DataFrame | None = None  # raw recommendations table for charting

    @property
    def total_ratings(self) -> int:
        return self.strong_buy + self.buy + self.hold + self.sell + self.strong_sell


@dataclass(frozen=True)
class FinancialHistory:
    """Annual and quarterly revenue / net income series, newest first."""

    annual_revenue: list[float] = field(default_factory=list)
    annual_net_income: list[float] = field(default_factory=list)
    quarterly_revenue: list[float] = field(default_factory=list)
    quarterly_net_income: list[float] = field(default_factory=list)


def _num(value: Any) -> float | None:
    """Coerce a yfinance field to float, treating junk as missing."""
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None  # drop NaN


def _first_num(info: dict, *keys: str) -> float | None:
    for key in keys:
        value = _num(info.get(key))
        if value is not None:
            return value
    return None


def get_price_history(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Daily OHLCV history (auto-adjusted). Raises TickerNotFoundError if empty."""
    df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
    if df is None or df.empty:
        raise TickerNotFoundError(f"No price data found for '{ticker}'")
    return df.dropna(subset=["Close"])


def get_company_info(ticker: str) -> CompanyInfo:
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        info = {}
    return CompanyInfo(
        ticker=ticker.upper(),
        long_name=info.get("longName") or info.get("shortName"),
        sector=info.get("sector"),
        industry=info.get("industry"),
        quote_type=info.get("quoteType"),
        current_price=_first_num(info, "currentPrice", "regularMarketPrice", "previousClose"),
        market_cap=_num(info.get("marketCap")),
        trailing_pe=_num(info.get("trailingPE")),
        forward_pe=_num(info.get("forwardPE")),
        peg_ratio=_first_num(info, "trailingPegRatio", "pegRatio"),
        price_to_book=_num(info.get("priceToBook")),
        ev_to_ebitda=_num(info.get("enterpriseToEbitda")),
        free_cashflow=_num(info.get("freeCashflow")),
        revenue_growth=_num(info.get("revenueGrowth")),
        earnings_growth=_num(info.get("earningsGrowth")),
        profit_margin=_num(info.get("profitMargins")),
        fifty_two_week_low=_num(info.get("fiftyTwoWeekLow")),
        fifty_two_week_high=_num(info.get("fiftyTwoWeekHigh")),
    )


def get_analyst_data(ticker: str) -> AnalystData:
    tk = yf.Ticker(ticker)

    counts = {"strongBuy": 0, "buy": 0, "hold": 0, "sell": 0, "strongSell": 0}
    trend = None
    try:
        rec = tk.recommendations
        if rec is not None and not rec.empty and "strongBuy" in rec.columns:
            trend = rec
            current = rec.iloc[0]  # '0m' row = current month
            for key in counts:
                counts[key] = int(current.get(key, 0) or 0)
    except Exception:
        pass

    targets: dict[str, Any] = {}
    try:
        targets = tk.analyst_price_targets or {}
    except Exception:
        pass
    if not targets:
        try:
            info = tk.info or {}
            targets = {
                "mean": info.get("targetMeanPrice"),
                "median": info.get("targetMedianPrice"),
                "high": info.get("targetHighPrice"),
                "low": info.get("targetLowPrice"),
            }
        except Exception:
            targets = {}

    num_analysts = None
    try:
        num_analysts_raw = _num((tk.info or {}).get("numberOfAnalystOpinions"))
        num_analysts = int(num_analysts_raw) if num_analysts_raw else None
    except Exception:
        pass

    return AnalystData(
        strong_buy=counts["strongBuy"],
        buy=counts["buy"],
        hold=counts["hold"],
        sell=counts["sell"],
        strong_sell=counts["strongSell"],
        num_analysts=num_analysts,
        target_mean=_num(targets.get("mean")),
        target_median=_num(targets.get("median")),
        target_high=_num(targets.get("high")),
        target_low=_num(targets.get("low")),
        trend=trend,
    )


def _row_values(df: pd.DataFrame | None, *row_names: str) -> list[float]:
    """Extract a statement row as a list of floats, newest first."""
    if df is None or df.empty:
        return []
    for name in row_names:
        if name in df.index:
            values = [_num(v) for v in df.loc[name].tolist()]
            return [v for v in values if v is not None]
    return []


def get_financial_history(ticker: str) -> FinancialHistory:
    tk = yf.Ticker(ticker)
    annual = quarterly = None
    try:
        annual = tk.income_stmt
    except Exception:
        pass
    try:
        quarterly = tk.quarterly_income_stmt
    except Exception:
        pass
    return FinancialHistory(
        annual_revenue=_row_values(annual, "Total Revenue", "Operating Revenue"),
        annual_net_income=_row_values(annual, "Net Income", "Net Income Common Stockholders"),
        quarterly_revenue=_row_values(quarterly, "Total Revenue", "Operating Revenue"),
        quarterly_net_income=_row_values(quarterly, "Net Income", "Net Income Common Stockholders"),
    )
