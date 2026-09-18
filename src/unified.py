import pandas as pd

from src.account_risk import build_account_risk_results
from src.benchmark import normalize_benchmark_ticker, run_benchmark_analysis
from src.data import (
    DEFAULT_RISK_WINDOW_LABEL,
    DEFAULT_MAX_PRICE_STALENESS_BUSINESS_DAYS,
    MAXIMUM_HISTORY_DOWNLOAD_START,
    download_price_views,
    prepare_price_data,
    resolve_risk_window
)
from src.engine import run_portfolio_analysis
from src.investor import run_investor_performance_analysis
from src.transactions import (
    build_current_portfolio,
    normalize_transaction_ledger
)


SECURITY_RISK_SCOPE = "invested_securities_only"
ACCOUNT_RISK_SCOPE = "total_account_current_allocation"


def _normalize_date(value, input_name):
    """Return a normalized timestamp or raise a caller-friendly error."""
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{input_name} must be a valid date.") from error

    if pd.isna(timestamp):
        raise ValueError(f"{input_name} must be a valid date.")

    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)

    return timestamp.normalize()


def _resolve_price_views(
    market_tickers,
    start_date,
    prices,
    valuation_prices,
    return_prices
):
    """Resolve explicit channels or the legacy shared prepared-price input."""
    explicit_view_supplied = (
        valuation_prices is not None
        or return_prices is not None
    )
    if prices is not None and explicit_view_supplied:
        raise ValueError(
            "Use either legacy prices= or both valuation_prices= and "
            "return_prices=, not a mixture."
        )

    if prices is not None:
        return {
            "valuation_prices": prices,
            "return_prices": prices,
            "price_source": "injected",
            "price_input_mode": "shared_prepared_prices_compatibility",
            "valuation_price_basis": "shared_prepared_price_view",
            "return_price_basis": "shared_prepared_price_view"
        }

    if (valuation_prices is None) != (return_prices is None):
        raise ValueError(
            "valuation_prices and return_prices must be supplied together."
        )

    if valuation_prices is None:
        price_views = download_price_views(
            market_tickers,
            start_date.strftime("%Y-%m-%d")
        )
        price_source = "yahoo_finance"
        price_input_mode = "explicit_price_channels"
    else:
        price_views = {
            "valuation_prices": valuation_prices,
            "return_prices": return_prices
        }
        price_source = "injected"
        price_input_mode = "explicit_price_channels"

    return {
        **price_views,
        "price_source": price_source,
        "price_input_mode": price_input_mode,
        "valuation_price_basis": "raw_close",
        "return_price_basis": "adjusted_close"
    }


def run_unified_portfolio_analysis(
    transactions,
    cash_flows=None,
    stress_scenarios=None,
    prices=None,
    risk_start_date=None,
    risk_window_label=DEFAULT_RISK_WINDOW_LABEL,
    valuation_prices=None,
    return_prices=None,
    max_price_staleness_business_days=(
        DEFAULT_MAX_PRICE_STALENESS_BUSINESS_DAYS
    ),
    risk_free_rate_annual=0.0,
    benchmark_ticker="SPY"
):
    """Run investor accounting, benchmark analytics and current risk.

    Top-level results have one explicit responsibility:

    - ``investor``: actual ledger reconstruction and TWR performance;
    - ``risk``: the backward-compatible Invested Securities risk result;
    - ``account_risk``: today's total-account allocation with zero-return cash;
    - ``benchmark``: actual TWR versus adjusted-Close benchmark performance;
    - ``diversification``: compatibility alias of ``risk["diversification"]``;
    - ``current_portfolio``: open ledger positions in engine input form; and
    - ``metadata``: price-channel, scope, date and observation provenance.

    The live path downloads once and separates raw Close valuation prices from
    adjusted Close return prices. Supplying both explicit price views avoids
    live downloads and makes the workflow deterministic. Legacy ``prices`` is
    retained as a shared-view compatibility input. Current raw valuation prices
    are checked against the configurable business-day freshness threshold.
    The benchmark joins the single market-data request and uses adjusted Close
    for exact-date, total-return-style comparison with investor daily TWR.
    """
    ledger = normalize_transaction_ledger(transactions, cash_flows)
    benchmark_ticker = normalize_benchmark_ticker(benchmark_ticker)
    normalized_explicit_risk_start = (
        _normalize_date(risk_start_date, "risk_start_date")
        if risk_start_date is not None
        else None
    )

    transaction_tickers = (
        ledger.loc[ledger["type"].isin(["BUY", "SELL"]), "ticker"]
        .drop_duplicates()
        .tolist()
    )
    market_tickers = list(dict.fromkeys([
        *transaction_tickers,
        benchmark_ticker
    ]))
    investor_start_date = ledger["date"].min().normalize()
    if investor_start_date.tzinfo is not None:
        investor_start_date = investor_start_date.tz_localize(None)

    current_portfolio = build_current_portfolio(ledger)
    if current_portfolio.empty:
        raise ValueError(
            "Current-risk analysis requires at least one open security "
            "position; all transaction-ledger positions are closed. "
            "Cash-only accounts are not analyzed by the invested-securities "
            "risk engine."
        )

    current_risk_tickers = current_portfolio["ticker"].tolist()
    if normalized_explicit_risk_start is None:
        # The Yahoo endpoint is not known before download. Requesting from a
        # deliberately early fixed date supports every preset, including
        # Maximum available, without consulting the computer clock.
        shared_price_start = min(
            pd.Timestamp(MAXIMUM_HISTORY_DOWNLOAD_START),
            investor_start_date
        )
    else:
        shared_price_start = min(
            normalized_explicit_risk_start,
            investor_start_date
        )
    price_views = _resolve_price_views(
        market_tickers=market_tickers,
        start_date=shared_price_start,
        prices=prices,
        valuation_prices=valuation_prices,
        return_prices=return_prices
    )
    shared_valuation_prices = prepare_price_data(
        price_views["valuation_prices"],
        required_assets=market_tickers
    )
    shared_return_prices = prepare_price_data(
        price_views["return_prices"],
        required_assets=market_tickers
    )

    relevant_return_prices = shared_return_prices[
        current_risk_tickers
    ].dropna(how="all")
    if relevant_return_prices.empty:
        raise ValueError(
            "No relevant return-price observations are available for the "
            "currently held securities."
        )
    risk_reference_date = relevant_return_prices.index[-1]
    earliest_available_return_price_date = relevant_return_prices.index[0]
    if normalized_explicit_risk_start is None:
        requested_risk_start = resolve_risk_window(
            risk_window_label=risk_window_label,
            reference_date=risk_reference_date,
            earliest_available_date=earliest_available_return_price_date
        )
        resolved_risk_window_label = risk_window_label
    else:
        requested_risk_start = normalized_explicit_risk_start
        resolved_risk_window_label = "Explicit start date"

    investor_valuation_prices = shared_valuation_prices.loc[
        shared_valuation_prices.index >= investor_start_date,
        transaction_tickers
    ].copy()
    if investor_valuation_prices.empty:
        raise ValueError(
            "Transaction and cash-flow dates must not be later than the "
            "last supplied valuation-price date "
            f"({shared_valuation_prices.index[-1].date()})."
        )

    investor_results = run_investor_performance_analysis(
        transactions=ledger,
        valuation_prices=investor_valuation_prices,
        max_price_staleness_business_days=(
            max_price_staleness_business_days
        ),
        risk_free_rate_annual=risk_free_rate_annual
    )
    benchmark_results = run_benchmark_analysis(
        daily_twr=investor_results["daily_twr"],
        benchmark_prices=shared_return_prices[benchmark_ticker],
        risk_free_rate_annual=risk_free_rate_annual,
        benchmark_ticker=benchmark_ticker
    )

    risk_valuation_prices = shared_valuation_prices.loc[
        shared_valuation_prices.index >= requested_risk_start,
        current_risk_tickers
    ].copy()
    risk_return_prices = shared_return_prices.loc[
        (shared_return_prices.index >= requested_risk_start)
        & (shared_return_prices.index <= risk_reference_date),
        current_risk_tickers
    ].copy()
    if risk_valuation_prices.empty or risk_return_prices.empty:
        raise ValueError(
            "Insufficient price history for current-risk analysis beginning "
            f"at {requested_risk_start.date()}: both valuation and return "
            "price observations are required."
        )

    risk_results = run_portfolio_analysis(
        portfolio=current_portfolio,
        stress_scenarios=stress_scenarios,
        start_date=requested_risk_start.strftime("%Y-%m-%d"),
        valuation_prices=risk_valuation_prices,
        return_prices=risk_return_prices,
        max_price_staleness_business_days=(
            max_price_staleness_business_days
        )
    )
    account_risk_results = build_account_risk_results(
        security_risk_results=risk_results,
        security_market_value=investor_results["current_security_value"],
        cash_balance=investor_results["current_cash_balance"],
        total_account_value=investor_results["current_portfolio_value"]
    )
    risk_sample_start = risk_results["portfolio_returns"].index[0]
    risk_sample_end = risk_results["portfolio_returns"].index[-1]
    risk_observations = len(risk_results["portfolio_returns"])

    return {
        "investor": investor_results,
        "risk": risk_results,
        "account_risk": account_risk_results,
        # Retained as the documented V3.8 convenience API. It is the same
        # object as risk["diversification"], not a second calculation.
        "diversification": risk_results["diversification"],
        "benchmark": benchmark_results,
        "current_portfolio": current_portfolio,
        "metadata": {
            "risk_scope": SECURITY_RISK_SCOPE,
            "risk_value_basis": "current_security_market_value",
            "default_risk_view": "account",
            "available_risk_views": ["account", "invested_securities"],
            "risk_scopes": {
                "account_risk": ACCOUNT_RISK_SCOPE,
                "risk": SECURITY_RISK_SCOPE
            },
            "invested_ratio": account_risk_results["invested_ratio"],
            "cash_weight": account_risk_results["cash_weight"],
            "cash_return_assumption": 0.0,
            "var_backtest_scale_invariant": (
                account_risk_results["backtest_scale_invariant"]
            ),
            "price_source": price_views["price_source"],
            "price_input_mode": price_views["price_input_mode"],
            "valuation_price_basis": (
                price_views["valuation_price_basis"]
            ),
            "return_price_basis": price_views["return_price_basis"],
            "investor_income_scope": "ledger_income_and_price_return",
            "valuation_reference_date": (
                risk_results["valuation_reference_date"]
            ),
            "valuation_price_dates": risk_results["valuation_price_dates"],
            "valuation_price_age_business_days": (
                risk_results["valuation_price_age_business_days"]
            ),
            "max_price_staleness_business_days": (
                risk_results["max_price_staleness_business_days"]
            ),
            "transaction_tickers": transaction_tickers,
            "market_data_tickers": market_tickers,
            "benchmark_ticker": benchmark_ticker,
            "benchmark_price_basis": "adjusted_close_total_return_style",
            "current_risk_tickers": current_risk_tickers,
            "investor_history_start_date": investor_start_date,
            "risk_window_label": resolved_risk_window_label,
            "risk_requested_start_date": requested_risk_start,
            "risk_reference_date": risk_reference_date,
            "earliest_available_return_price_date": (
                earliest_available_return_price_date
            ),
            "risk_start_date": risk_sample_start,
            "risk_end_date": risk_sample_end,
            "risk_observations": risk_observations,
            "constant_weight_assumption": True,
            "constant_weight_methodology": (
                "Historical risk uses today's portfolio weights held "
                "constant through the selected risk window (equivalent to "
                "daily rebalancing)."
            ),
            "raw_price_observations": risk_results[
                "raw_price_observations"
            ],
            "asset_return_observations_before_alignment": risk_results[
                "asset_return_observations_before_alignment"
            ],
            "aligned_portfolio_return_observations": risk_results[
                "aligned_portfolio_return_observations"
            ],
            "observations_dropped_during_alignment": risk_results[
                "observations_dropped_during_alignment"
            ],
            "usable_return_observations_by_asset": risk_results[
                "usable_return_observations_by_asset"
            ],
            "shared_price_start_date": shared_price_start
        }
    }
