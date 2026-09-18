from src.data import (
    DEFAULT_MAX_PRICE_STALENESS_BUSINESS_DAYS,
    download_price_views,
    validate_valuation_price_freshness
)
from src.performance import (
    calculate_investor_performance_metrics,
    calculate_time_weighted_return,
    calculate_twr_drawdown
)
from src.transactions import (
    normalize_transaction_ledger,
    reconstruct_accounting_history,
    reconstruct_holdings,
)


def run_investor_performance_analysis(
    transactions,
    external_cash_flows=None,
    prices=None,
    valuation_prices=None,
    max_price_staleness_business_days=(
        DEFAULT_MAX_PRICE_STALENESS_BUSINESS_DAYS
    ),
    risk_free_rate_annual=0.0
):
    """Reconstruct price-return investor history and calculate TWR.

    Raw Close valuation prices are downloaded only for assets in the validated
    transaction ledger. The earliest transaction or external cash-flow date
    becomes the download start date. Supplying ``valuation_prices`` skips the
    download. Legacy ``prices`` remains a compatibility alias for prepared raw
    valuation prices and cannot be combined with ``valuation_prices``.

    Ledger dividends, interest and expenses are internal performance events;
    only deposits and withdrawals are external TWR flows. Current open
    positions must pass the shared raw-price freshness policy.
    """
    ledger = normalize_transaction_ledger(
        transactions,
        external_cash_flows
    )

    tickers = (
        ledger.loc[ledger["type"].isin(["BUY", "SELL"]), "ticker"]
        .drop_duplicates()
        .tolist()
    )
    earliest_relevant_date = ledger["date"].min().normalize()
    start_date = earliest_relevant_date.strftime("%Y-%m-%d")

    if prices is not None and valuation_prices is not None:
        raise ValueError(
            "Use either legacy prices= or valuation_prices=, not both."
        )

    if valuation_prices is None:
        if prices is not None:
            valuation_prices = prices
        else:
            valuation_prices = download_price_views(
                tickers,
                start_date
            )["valuation_prices"]

    holdings = reconstruct_holdings(ledger)
    current_holdings = (
        holdings[holdings["quantity"] > 0]
        .copy()
        .reset_index(drop=True)
    )
    valuation_prices, valuation_freshness = (
        validate_valuation_price_freshness(
            valuation_prices,
            required_assets=current_holdings["ticker"].tolist(),
            max_price_staleness_business_days=(
                max_price_staleness_business_days
            )
        )
    )

    accounting_history = reconstruct_accounting_history(
        ledger,
        prices=valuation_prices
    )
    twr_results = calculate_time_weighted_return(
        accounting_history["daily_portfolio_value"],
        accounting_history["external_cash_flows_by_day"]
    )
    drawdown_results = calculate_twr_drawdown(twr_results["daily_twr"])
    performance_metrics = calculate_investor_performance_metrics(
        daily_twr=twr_results["daily_twr"],
        investor_max_drawdown=drawdown_results["investor_max_drawdown"],
        risk_free_rate_annual=risk_free_rate_annual
    )

    return {
        "prices": valuation_prices,
        "valuation_prices": valuation_prices,
        **valuation_freshness,
        "holdings": holdings,
        "current_holdings": current_holdings,
        **accounting_history,
        **twr_results,
        **drawdown_results,
        **performance_metrics,
        "current_portfolio_value": float(
            accounting_history["daily_portfolio_value"].iloc[-1]
        ),
        "current_security_value": float(
            accounting_history["daily_security_value"].iloc[-1]
        ),
        "current_cash_balance": float(
            accounting_history["daily_cash"].iloc[-1]
        ),
        "analysis_start_date": earliest_relevant_date
    }
