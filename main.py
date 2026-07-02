"""Streamlit UI for the multi-factor stock analysis engine.

Entry point for Streamlit Cloud. Layout and flow live here; all Plotly
figure construction is delegated to ``stockanalysis.charts`` so this module
stays readable. Enter ticker ``DEMO`` to explore the whole app offline with
deterministic synthetic data (see ``stockanalysis.demo``).
"""

from __future__ import annotations

import pandas as pd
import requests
import streamlit as st

from stockanalysis import charts
from stockanalysis.backtest import BacktestResult, backtest_signal, summary_table
from stockanalysis.data import (
    AnalystData,
    CompanyInfo,
    FinancialHistory,
    TickerNotFoundError,
    get_analyst_data,
    get_company_info,
    get_financial_history,
    get_price_history,
)
from stockanalysis.demo import get_demo_bundle, is_demo_ticker
from stockanalysis.forecasting import forecast_prices
from stockanalysis.pillars import (
    score_analysts,
    score_growth,
    score_technicals,
    score_trend_reversion,
    score_value,
)
from stockanalysis.scoring import DEFAULT_WEIGHTS, composite_from_pillars, label_for
from stockanalysis.types import PillarScore
from stockanalysis.validation import (
    block_bootstrap_sharpe,
    monte_carlo_random_entry,
    permutation_test_hit_rate,
    permutation_test_ic,
)

st.set_page_config(page_title="Stock Analyzer", layout="wide", page_icon="📊")


# ---------------------------------------------------------------------------
# Cached data access
#
# The package fetchers in stockanalysis.data are deliberately cache-agnostic
# (no Streamlit dependency), so caching is layered on here.
# ---------------------------------------------------------------------------


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_price_history(ticker: str, period: str) -> pd.DataFrame:
    return get_price_history(ticker, period=period)


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_company_info(ticker: str) -> CompanyInfo:
    return get_company_info(ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_analyst_data(ticker: str) -> AnalystData:
    return get_analyst_data(ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_financial_history(ticker: str) -> FinancialHistory:
    return get_financial_history(ticker)


def _fetch_bundle(
    ticker: str, period: str
) -> tuple[pd.DataFrame, CompanyInfo, AnalystData, FinancialHistory]:
    """Fetch (history, info, analysts, financials), routing DEMO to synthetic data."""
    if is_demo_ticker(ticker):
        return get_demo_bundle()
    history = _cached_price_history(ticker, period)
    info = _cached_company_info(ticker)
    analysts = _cached_analyst_data(ticker)
    financials = _cached_financial_history(ticker)
    return history, info, analysts, financials


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_backtest_bundle(ticker: str, period: str, _history: pd.DataFrame) -> dict:
    """Backtest + all four validation checks, cached together by ticker+period.

    ``_history`` is prefixed with an underscore so Streamlit uses it for the
    call but doesn't hash the (potentially large) frame itself — the cache
    key is the ticker/period pair, which is what the caller actually varies.
    """
    result = backtest_signal(_history, ticker=ticker)
    if result is None:
        return {"result": None}

    daily_returns = result.buyhold_equity.pct_change().fillna(0.0)
    strategy_daily_returns = result.strategy_equity.pct_change().fillna(0.0)
    # monte_carlo_random_entry evaluates timing of the *effective* exposure,
    # which lags the position decision by one bar in the simulation.
    effective_position = result.position.shift(1).fillna(0.0)

    return {
        "result": result,
        "perm_ic": permutation_test_ic(result.scores, result.forward_returns),
        "perm_hit": permutation_test_hit_rate(result.scores, result.forward_returns),
        "monte_carlo": monte_carlo_random_entry(daily_returns, effective_position),
        "bootstrap": block_bootstrap_sharpe(strategy_daily_returns),
    }


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _sidebar() -> dict:
    with st.sidebar:
        st.header("Stock Analyzer")
        ticker = st.text_input("Ticker", value="DEMO").strip()
        period = st.selectbox("History period", ["1y", "2y", "5y"], index=1)
        horizon = st.slider("Forecast horizon (trading days)", 5, 60, 30)

        with st.expander("Pillar weights"):
            st.caption(
                "Sliders are renormalized to sum to 1 before scoring. If a "
                "pillar has no usable data for this ticker, its weight is "
                "further redistributed across the remaining pillars."
            )
            raw_weights = {
                name: st.slider(name, 0.0, 1.0, float(default), 0.05)
                for name, default in DEFAULT_WEIGHTS.items()
            }
            total = sum(raw_weights.values())
            weights = (
                {name: w / total for name, w in raw_weights.items()}
                if total > 0
                else dict(DEFAULT_WEIGHTS)
            )

        st.caption("Data source: Yahoo Finance via yfinance")

    return {"ticker": ticker, "period": period, "horizon": horizon, "weights": weights}


# ---------------------------------------------------------------------------
# Header / hero
# ---------------------------------------------------------------------------


def _render_header(info: CompanyInfo, history: pd.DataFrame) -> None:
    name = info.long_name or info.ticker
    st.title(f"{name} ({info.ticker})")

    meta = " · ".join(x for x in (info.sector, info.industry) if x)
    if meta:
        st.caption(meta)

    price = info.current_price
    if price is not None and len(history) >= 2:
        prev_close = float(history["Close"].iloc[-2])
        change = price - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0.0
        st.metric("Current price", f"${price:,.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
    elif price is not None:
        st.metric("Current price", f"${price:,.2f}")
    else:
        st.metric("Current price", "n/a")


def _render_hero(composite: float, label: str, pillars: list[PillarScore]) -> None:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(charts.gauge_chart(composite, label), use_container_width=True, theme=None)
    with col2:
        st.plotly_chart(charts.radar_chart(pillars), use_container_width=True, theme=None)
    with col3:
        for pillar in pillars:
            value = f"{pillar.score:.0f}" if pillar.score is not None else "n/a"
            st.metric(pillar.name, value)

        available = [p for p in pillars if p.score is not None]
        missing_notes = [f"{p.name} not applicable" for p in pillars if p.score is None]
        summary = f"{len(available)}/{len(pillars)} pillars available"
        if missing_notes:
            summary += " — " + "; ".join(missing_notes)
        st.caption(summary)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------


def _render_score_breakdown(
    pillars: list[PillarScore], effective_weights: dict[str, float]
) -> None:
    for pillar in pillars:
        if pillar.score is not None:
            title = f"{pillar.name} — {pillar.score:.0f}"
        else:
            title = f"{pillar.name} — n/a"
        with st.expander(title, expanded=pillar.score is not None):
            if pillar.signals:
                rows = [
                    {
                        "Signal": s.name,
                        "Reading": s.value,
                        "Score (0-100)": round(s.score, 1),
                        "Weight": s.weight,
                    }
                    for s in pillar.signals
                ]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            if pillar.note:
                st.caption(pillar.note)
            if not pillar.signals and not pillar.note:
                st.caption("No data available for this pillar.")

    weight_str = ", ".join(f"{name} {w:.0%}" for name, w in effective_weights.items())
    st.caption(f"Effective composite weights used: {weight_str}")


def _render_technical_chart(history: pd.DataFrame) -> None:
    if len(history) < 20:
        st.info("Not enough price history to draw the technical chart.")
        return
    st.plotly_chart(charts.technical_chart(history), use_container_width=True, theme=None)


def _render_fundamentals(
    pillars: list[PillarScore], financials: FinancialHistory
) -> None:
    value_pillar = next(p for p in pillars if p.name == "Value")
    growth_pillar = next(p for p in pillars if p.name == "Growth")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Value ratios")
        if value_pillar.signals:
            rows = [
                {"Metric": s.name, "Reading": s.value, "Score": round(s.score, 1)}
                for s in value_pillar.signals
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info(value_pillar.note or "No valuation data available.")

    with col2:
        st.subheader("Growth")
        if growth_pillar.signals:
            rows = [
                {"Metric": s.name, "Reading": s.value, "Score": round(s.score, 1)}
                for s in growth_pillar.signals
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info(growth_pillar.note or "No growth data available.")

    if financials.annual_revenue:
        st.plotly_chart(
            charts.revenue_bar_chart(financials.annual_revenue, "Annual revenue"),
            use_container_width=True,
            theme=None,
        )
    if financials.quarterly_revenue:
        st.plotly_chart(
            charts.revenue_bar_chart(financials.quarterly_revenue, "Quarterly revenue"),
            use_container_width=True,
            theme=None,
        )
    if not financials.annual_revenue and not financials.quarterly_revenue:
        st.caption("No financial statement history available for this ticker.")


def _render_analysts(analysts: AnalystData, info: CompanyInfo) -> None:
    if analysts.total_ratings > 0:
        st.plotly_chart(
            charts.analyst_rating_bar(
                analysts.strong_buy,
                analysts.buy,
                analysts.hold,
                analysts.sell,
                analysts.strong_sell,
            ),
            use_container_width=True,
            theme=None,
        )
    else:
        st.info("No analyst rating data available.")

    if info.current_price is not None and analysts.target_mean is not None:
        st.plotly_chart(
            charts.price_target_range_chart(
                info.current_price, analysts.target_low, analysts.target_mean, analysts.target_high
            ),
            use_container_width=True,
            theme=None,
        )
    else:
        st.info("No price-target data available.")

    st.caption(f"Coverage: {analysts.num_analysts or analysts.total_ratings or 0} analyst(s)")


def _render_forecast(history: pd.DataFrame, horizon: int) -> None:
    with st.spinner("Fitting ARIMA model..."):
        forecast = forecast_prices(history, horizon_days=horizon)

    if forecast is None:
        st.info("Not enough price history (need at least 100 bars) to fit a forecast.")
        return

    st.plotly_chart(charts.forecast_chart(history, forecast), use_container_width=True, theme=None)
    st.caption(f"ARIMA order {forecast.order}, AIC {forecast.aic:.1f}")
    st.warning(
        "This is a statistical baseline fit to historical price patterns only — "
        "not investment advice. It cannot anticipate news, earnings, or macro events."
    )


def _render_validation_metrics(bundle: dict) -> None:
    col1, col2, col3, col4 = st.columns(4)

    perm_ic = bundle.get("perm_ic")
    with col1:
        if perm_ic is not None:
            st.metric("IC permutation p-value", f"{perm_ic.p_value:.3f}")
            st.caption("p < 0.05 suggests the signal carries real information.")
        else:
            st.metric("IC permutation p-value", "n/a")

    perm_hit = bundle.get("perm_hit")
    with col2:
        if perm_hit is not None:
            st.metric("Hit-rate permutation p-value", f"{perm_hit.p_value:.3f}")
            st.caption("p < 0.05 suggests bullish calls beat random chance.")
        else:
            st.metric("Hit-rate permutation p-value", "n/a")

    monte_carlo = bundle.get("monte_carlo")
    with col3:
        if monte_carlo is not None:
            st.metric("MC timing percentile", f"{monte_carlo.percentile:.0f}th")
            st.caption("Share of random-timing trials beaten by the real strategy.")
        else:
            st.metric("MC timing percentile", "n/a")

    bootstrap = bundle.get("bootstrap")
    with col4:
        if bootstrap is not None:
            st.metric("Sharpe 90% CI", f"[{bootstrap.p5:.2f}, {bootstrap.p95:.2f}]")
            st.caption(f"Block-bootstrap CI around observed Sharpe {bootstrap.observed:.2f}.")
        else:
            st.metric("Sharpe 90% CI", "n/a")


def _render_backtest(history: pd.DataFrame, ticker: str, period: str) -> None:
    st.info(
        "Only the two price-based pillars (Technicals, Trend & Mean-Reversion) are "
        "backtested here. Value, Growth, and Analysts rely on yfinance's *current* "
        "snapshot only — there's no historical P/E, rating, or revenue series to "
        "replay without leaking the future into the past. Treat these results as a "
        "read on the price/technical half of the signal, not a validation of the "
        "full five-pillar composite. No transaction costs, slippage, or position "
        "sizing are modeled."
    )

    cache_key = (ticker.upper(), period)
    if st.button("Run backtest", type="primary"):
        with st.spinner("Running backtest and validation checks — this can take 30-60s..."):
            bundle = _cached_backtest_bundle(ticker.upper(), period, history)
        st.session_state["backtest_bundle"] = (cache_key, bundle)

    stored = st.session_state.get("backtest_bundle")
    if stored is None or stored[0] != cache_key:
        st.caption('Click "Run backtest" to compute (compute-heavy, roughly 30-60s).')
        return

    bundle = stored[1]
    result: BacktestResult | None = bundle["result"]
    if result is None:
        st.info("Not enough price history for a meaningful backtest (need a longer series).")
        return

    st.dataframe(summary_table(result), use_container_width=True, hide_index=True)
    st.plotly_chart(charts.equity_curve_chart(result), use_container_width=True, theme=None)

    if not result.quintile_returns.empty:
        st.plotly_chart(
            charts.quintile_return_chart(result.quintile_returns),
            use_container_width=True,
            theme=None,
        )

    st.subheader("Statistical significance")
    _render_validation_metrics(bundle)


def _render_footer() -> None:
    st.divider()
    st.caption(
        "Educational tool only — not financial advice. Scores, forecasts, and "
        "backtests are statistical artifacts of historical data and carry no "
        "guarantee of future performance."
    )
    st.page_link("pages/1_Methodology.py", label="Methodology", icon="📖")


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------


def main() -> None:
    controls = _sidebar()
    ticker = controls["ticker"]

    if not ticker:
        st.info("Enter a ticker in the sidebar to begin (try `DEMO`).")
        return

    demo_mode = is_demo_ticker(ticker)

    try:
        with st.spinner(f"Fetching data for {ticker.upper()}..."):
            history, info, analysts, financials = _fetch_bundle(ticker, controls["period"])
    except TickerNotFoundError:
        st.error(
            f"Could not find ticker '{ticker}'. Double-check the symbol, or enter "
            "`DEMO` to explore the app with synthetic data."
        )
        return
    except (requests.exceptions.RequestException, OSError) as exc:
        st.error(
            f"Network error while fetching '{ticker}': {exc}. This environment may "
            "not have access to Yahoo Finance — enter `DEMO` to explore the app "
            "with synthetic data instead."
        )
        return
    except Exception as exc:  # yfinance raises a variety of undocumented errors
        st.error(
            f"Unexpected error fetching '{ticker}': {exc}. Enter `DEMO` to explore "
            "the app with synthetic data instead."
        )
        return

    if demo_mode:
        st.info("Synthetic demonstration data — not a real security", icon="ℹ️")

    _render_header(info, history)

    pillars = [
        score_value(info),
        score_growth(info, financials),
        score_technicals(history),
        score_trend_reversion(history),
        score_analysts(analysts, info.current_price),
    ]
    try:
        composite, effective_weights = composite_from_pillars(pillars, controls["weights"])
    except ValueError:
        st.error("No pillar has usable data for this ticker — cannot compute a composite score.")
        return
    label = label_for(composite)

    _render_hero(composite, label, pillars)

    tabs = st.tabs(
        ["Score Breakdown", "Technical Chart", "Fundamentals", "Analysts", "Forecast", "Backtest"]
    )
    with tabs[0]:
        _render_score_breakdown(pillars, effective_weights)
    with tabs[1]:
        _render_technical_chart(history)
    with tabs[2]:
        _render_fundamentals(pillars, financials)
    with tabs[3]:
        _render_analysts(analysts, info)
    with tabs[4]:
        _render_forecast(history, controls["horizon"])
    with tabs[5]:
        _render_backtest(history, ticker, controls["period"])

    _render_footer()


if __name__ == "__main__":
    main()
