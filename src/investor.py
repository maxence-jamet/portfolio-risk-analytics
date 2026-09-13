from src.data import download_prices, validate_price_data
from src.performance import calculate_time_weighted_return
from src.transactions import (
    reconstruct_accounting_history,
    reconstruct_holdings,
    validate_external_cash_flows,
    validate_transactions
)


def run_investor_performance_analysis(
    transactions,
    external_cash_flows,
    prices=None
):
    """Reconstruct actual investor history and calculate TWR.

    Prices are downloaded only for assets in the validated transaction ledger.
    The earliest transaction or external cash-flow date becomes the download
    start date. Supplying prepared prices skips the download for deterministic
    tests and other reusable workflows.
    """
    validated_transactions = validate_transactions(transactions)
    validated_cash_flows = validate_external_cash_flows(
        external_cash_flows
    )

    tickers = (
        validated_transactions["ticker"]
        .drop_duplicates()
        .tolist()
    )
    earliest_relevant_date = min(
        validated_transactions["date"].min(),
        validated_cash_flows["date"].min()
    ).normalize()
    start_date = earliest_relevant_date.strftime("%Y-%m-%d")

    if prices is None:
        prices = download_prices(tickers, start_date)

    prices = validate_price_data(
        prices,
        required_assets=tickers
    )

    accounting_history = reconstruct_accounting_history(
        validated_transactions,
        validated_cash_flows,
        prices
    )
    holdings = reconstruct_holdings(validated_transactions)
    current_holdings = (
        holdings[holdings["quantity"] > 0]
        .copy()
        .reset_index(drop=True)
    )
    twr_results = calculate_time_weighted_return(
        accounting_history["daily_portfolio_value"],
        accounting_history["external_cash_flows_by_day"]
    )

    return {
        "prices": prices,
        "holdings": holdings,
        "current_holdings": current_holdings,
        **accounting_history,
        **twr_results,
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
