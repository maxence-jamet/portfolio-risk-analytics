import numpy as np
import pandas as pd

from src.data import validate_price_data


REQUIRED_TRANSACTION_COLUMNS = {
    "date",
    "ticker",
    "side",
    "quantity",
    "price"
}
SUPPORTED_TRANSACTION_SIDES = {"BUY", "SELL"}
REQUIRED_CASH_FLOW_COLUMNS = {"date", "type", "amount"}
SUPPORTED_CASH_FLOW_TYPES = {"DEPOSIT", "WITHDRAWAL"}


def validate_transactions(transactions):
    """Validate and normalize a defensive copy of a transaction ledger."""
    if not isinstance(transactions, pd.DataFrame):
        raise ValueError("Transactions must be provided as a pandas DataFrame.")

    if transactions.empty:
        raise ValueError("Transactions must contain at least one row.")

    missing_columns = (
        REQUIRED_TRANSACTION_COLUMNS
        - set(transactions.columns)
    )
    if missing_columns:
        raise ValueError(
            f"Missing transaction columns: {sorted(missing_columns)}"
        )

    transactions = transactions.copy()

    transactions["date"] = pd.to_datetime(
        transactions["date"],
        errors="coerce"
    )
    if transactions["date"].isna().any():
        raise ValueError("Transaction dates must contain only valid dates.")

    missing_tickers = transactions["ticker"].isna()
    transactions["ticker"] = (
        transactions["ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
    )
    if missing_tickers.any() or (transactions["ticker"] == "").any():
        raise ValueError("Transaction tickers must not be blank.")

    transactions["side"] = (
        transactions["side"]
        .astype(str)
        .str.strip()
        .str.upper()
    )
    unsupported_sides = sorted(
        set(transactions["side"])
        - SUPPORTED_TRANSACTION_SIDES
    )
    if unsupported_sides:
        raise ValueError(
            "Unsupported transaction side values: "
            f"{unsupported_sides}. Supported values are BUY and SELL."
        )

    try:
        transactions["quantity"] = pd.to_numeric(
            transactions["quantity"],
            errors="raise"
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Transaction quantities must contain only numeric values."
        ) from error

    try:
        transactions["price"] = pd.to_numeric(
            transactions["price"],
            errors="raise"
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Transaction prices must contain only numeric values."
        ) from error

    if not np.isfinite(transactions["quantity"]).all():
        raise ValueError(
            "Transaction quantities must contain only finite numeric values."
        )

    if not np.isfinite(transactions["price"]).all():
        raise ValueError(
            "Transaction prices must contain only finite numeric values."
        )

    if (transactions["quantity"] <= 0).any():
        raise ValueError("Transaction quantities must be strictly positive.")

    if (transactions["price"] <= 0).any():
        raise ValueError("Transaction prices must be strictly positive.")

    return transactions


def reconstruct_holdings(transactions):
    """Reconstruct current holdings using average-cost accounting."""
    transactions = validate_transactions(transactions)
    transactions = transactions.sort_values(
        "date",
        kind="stable"
    )

    holdings = {}

    for transaction in transactions.itertuples(index=False):
        ticker = transaction.ticker
        quantity = float(transaction.quantity)
        price = float(transaction.price)

        if ticker not in holdings:
            holdings[ticker] = {
                "ticker": ticker,
                "quantity": 0.0,
                "average_cost": 0.0,
                "total_cost_basis": 0.0,
                "realized_pnl": 0.0
            }

        position = holdings[ticker]

        if transaction.side == "BUY":
            new_quantity = position["quantity"] + quantity
            new_cost_basis = (
                position["total_cost_basis"]
                + quantity * price
            )

            position["quantity"] = new_quantity
            position["total_cost_basis"] = new_cost_basis
            position["average_cost"] = new_cost_basis / new_quantity
            continue

        if quantity > position["quantity"]:
            raise ValueError(
                f"Cannot SELL {quantity:g} units of {ticker} on "
                f"{transaction.date.date()}: only "
                f"{position['quantity']:g} units are available. "
                "Short selling is not supported."
            )

        position["realized_pnl"] += (
            quantity
            * (price - position["average_cost"])
        )
        remaining_quantity = position["quantity"] - quantity

        if np.isclose(remaining_quantity, 0.0):
            position["quantity"] = 0.0
            position["average_cost"] = 0.0
            position["total_cost_basis"] = 0.0
        else:
            position["quantity"] = remaining_quantity
            position["total_cost_basis"] = (
                remaining_quantity
                * position["average_cost"]
            )

    return pd.DataFrame(holdings.values())


def build_current_portfolio(transactions):
    """Convert open transaction-derived holdings to the engine input schema."""
    holdings = reconstruct_holdings(transactions)
    open_holdings = holdings[holdings["quantity"] > 0].copy()

    current_portfolio = (
        open_holdings
        .rename(columns={"average_cost": "purchase_price"})
        [["ticker", "quantity", "purchase_price"]]
        .reset_index(drop=True)
    )

    return current_portfolio


def validate_external_cash_flows(cash_flows):
    """Validate and normalize a defensive copy of external investor flows."""
    if not isinstance(cash_flows, pd.DataFrame):
        raise ValueError(
            "External cash flows must be provided as a pandas DataFrame."
        )

    if cash_flows.empty:
        raise ValueError("External cash flows must contain at least one row.")

    missing_columns = (
        REQUIRED_CASH_FLOW_COLUMNS
        - set(cash_flows.columns)
    )
    if missing_columns:
        raise ValueError(
            f"Missing external cash-flow columns: {sorted(missing_columns)}"
        )

    cash_flows = cash_flows.copy()

    cash_flows["date"] = pd.to_datetime(
        cash_flows["date"],
        errors="coerce"
    )
    if cash_flows["date"].isna().any():
        raise ValueError("External cash-flow dates must be valid dates.")
    cash_flows["date"] = cash_flows["date"].dt.normalize()

    cash_flows["type"] = (
        cash_flows["type"]
        .astype(str)
        .str.strip()
        .str.upper()
    )
    unsupported_types = sorted(
        set(cash_flows["type"])
        - SUPPORTED_CASH_FLOW_TYPES
    )
    if unsupported_types:
        raise ValueError(
            "Unsupported external cash-flow type values: "
            f"{unsupported_types}. Supported values are DEPOSIT and "
            "WITHDRAWAL."
        )

    try:
        cash_flows["amount"] = pd.to_numeric(
            cash_flows["amount"],
            errors="raise"
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "External cash-flow amounts must contain only numeric values."
        ) from error

    if not np.isfinite(cash_flows["amount"]).all():
        raise ValueError(
            "External cash-flow amounts must contain only finite values."
        )

    if (cash_flows["amount"] <= 0).any():
        raise ValueError(
            "External cash-flow amounts must be strictly positive."
        )

    return cash_flows


def _prepare_accounting_prices(prices, required_tickers):
    """Validate prices and return a chronological copy with daily dates."""
    prices = validate_price_data(
        prices,
        required_assets=required_tickers
    )

    price_dates = pd.to_datetime(prices.index, errors="coerce")
    if price_dates.isna().any():
        raise ValueError("The price index must contain only valid dates.")

    normalized_dates = price_dates.normalize()
    if price_dates.equals(normalized_dates):
        prices.index = price_dates
    else:
        prices.index = normalized_dates
    if prices.index.duplicated().any():
        raise ValueError("The price index must not contain duplicate dates.")

    return prices.sort_index(kind="stable")


def _reject_events_after_price_history(event_dates, last_price_date):
    if event_dates.max() > last_price_date:
        raise ValueError(
            "Transaction and cash-flow dates must not be later than the "
            f"last supplied price date ({last_price_date.date()})."
        )


def reconstruct_daily_holdings(transactions, prices):
    """Return end-of-day quantities on each supplied market-price date.

    An event affects holdings on the first supplied price date on or after its
    transaction date. Stable sorting preserves input order for same-day trades.
    """
    transactions = validate_transactions(transactions)
    transactions["date"] = transactions["date"].dt.normalize()
    transactions = transactions.sort_values("date", kind="stable")

    tickers = transactions["ticker"].drop_duplicates().tolist()
    prices = _prepare_accounting_prices(prices, tickers)
    _reject_events_after_price_history(
        transactions["date"],
        prices.index[-1]
    )

    daily_holdings = pd.DataFrame(
        0.0,
        index=prices.index,
        columns=tickers
    )
    current_quantities = {
        ticker: 0.0
        for ticker in tickers
    }
    ordered_transactions = list(
        transactions.itertuples(index=False)
    )
    transaction_number = 0

    for market_date in prices.index:
        while (
            transaction_number < len(ordered_transactions)
            and ordered_transactions[transaction_number].date <= market_date
        ):
            transaction = ordered_transactions[transaction_number]
            quantity = float(transaction.quantity)

            if transaction.side == "BUY":
                current_quantities[transaction.ticker] += quantity
            else:
                available_quantity = current_quantities[transaction.ticker]
                if quantity > available_quantity:
                    raise ValueError(
                        f"Cannot SELL {quantity:g} units of "
                        f"{transaction.ticker} on {transaction.date.date()}: "
                        f"only {available_quantity:g} units are available. "
                        "Short selling is not supported."
                    )
                current_quantities[transaction.ticker] -= quantity

                if np.isclose(
                    current_quantities[transaction.ticker],
                    0.0
                ):
                    current_quantities[transaction.ticker] = 0.0

            transaction_number += 1

        daily_holdings.loc[market_date] = [
            current_quantities[ticker]
            for ticker in tickers
        ]

    return daily_holdings


def reconstruct_daily_cash(transactions, cash_flows, prices):
    """Return daily cash and signed external flows on supplied price dates.

    External flows are processed before BUY/SELL trades on the same date.
    Original input order is preserved within each separate ledger.
    """
    transactions = validate_transactions(transactions)
    transactions["date"] = transactions["date"].dt.normalize()
    cash_flows = validate_external_cash_flows(cash_flows)

    # Validate the complete trade sequence independently of the cash account.
    reconstruct_holdings(transactions)

    tickers = transactions["ticker"].drop_duplicates().tolist()
    prices = _prepare_accounting_prices(prices, tickers)
    _reject_events_after_price_history(
        pd.concat([transactions["date"], cash_flows["date"]]),
        prices.index[-1]
    )

    events = []

    for input_order, cash_flow in enumerate(
        cash_flows.itertuples(index=False)
    ):
        signed_amount = float(cash_flow.amount)
        if cash_flow.type == "WITHDRAWAL":
            signed_amount = -signed_amount

        events.append({
            "date": cash_flow.date,
            "source_priority": 0,
            "input_order": input_order,
            "event_type": cash_flow.type,
            "ticker": None,
            "cash_change": signed_amount
        })

    for input_order, transaction in enumerate(
        transactions.itertuples(index=False)
    ):
        trade_value = float(transaction.quantity * transaction.price)
        cash_change = trade_value
        if transaction.side == "BUY":
            cash_change = -trade_value

        events.append({
            "date": transaction.date,
            "source_priority": 1,
            "input_order": input_order,
            "event_type": transaction.side,
            "ticker": transaction.ticker,
            "cash_change": cash_change
        })

    events.sort(key=lambda event: (
        event["date"],
        event["source_priority"],
        event["input_order"]
    ))

    daily_cash = pd.Series(
        0.0,
        index=prices.index,
        name="cash_balance"
    )
    external_cash_flows_by_day = pd.Series(
        0.0,
        index=prices.index,
        name="external_cash_flow"
    )
    cash_balance = 0.0
    event_number = 0

    for market_date in prices.index:
        while (
            event_number < len(events)
            and events[event_number]["date"] <= market_date
        ):
            event = events[event_number]
            required_cash = -event["cash_change"]

            if required_cash > cash_balance:
                if event["event_type"] == "BUY":
                    raise ValueError(
                        f"Insufficient cash for BUY of {event['ticker']} on "
                        f"{event['date'].date()}: requires "
                        f"{required_cash:.2f}, but only "
                        f"{cash_balance:.2f} is available. "
                        "Margin and borrowing are not supported."
                    )

                raise ValueError(
                    f"Insufficient cash for WITHDRAWAL on "
                    f"{event['date'].date()}: requires "
                    f"{required_cash:.2f}, but only "
                    f"{cash_balance:.2f} is available."
                )

            cash_balance += event["cash_change"]
            if np.isclose(cash_balance, 0.0):
                cash_balance = 0.0

            if event["source_priority"] == 0:
                external_cash_flows_by_day.loc[market_date] += (
                    event["cash_change"]
                )

            event_number += 1

        daily_cash.loc[market_date] = cash_balance

    return daily_cash, external_cash_flows_by_day


def reconstruct_accounting_history(transactions, cash_flows, prices):
    """Build daily holdings, cash and market value without performance metrics.

    Valuation uses each date's supplied market price. Missing prices may use
    only an earlier supplied observation through forward-filling, never a
    future price.
    """
    validated_transactions = validate_transactions(transactions)
    tickers = validated_transactions["ticker"].drop_duplicates().tolist()
    prices = _prepare_accounting_prices(prices, tickers)

    daily_holdings = reconstruct_daily_holdings(
        validated_transactions,
        prices
    )
    daily_cash, external_cash_flows_by_day = reconstruct_daily_cash(
        validated_transactions,
        cash_flows,
        prices
    )

    valuation_prices = prices.reindex(
        columns=daily_holdings.columns
    ).ffill()
    missing_held_prices = (
        daily_holdings.gt(0)
        & valuation_prices.isna()
    )
    if missing_held_prices.any().any():
        missing_date, missing_ticker = next(
            (date, ticker)
            for date, row in missing_held_prices.iterrows()
            for ticker, is_missing in row.items()
            if is_missing
        )
        raise ValueError(
            f"No current or earlier price is available for {missing_ticker} "
            f"on {missing_date.date()}; portfolio value cannot be calculated."
        )

    security_values_by_asset = daily_holdings.mul(valuation_prices)
    security_values_by_asset = security_values_by_asset.mask(
        daily_holdings.eq(0),
        0.0
    )
    daily_security_value = security_values_by_asset.sum(axis=1)
    daily_security_value.name = "security_market_value"

    daily_portfolio_value = daily_security_value + daily_cash
    daily_portfolio_value.name = "portfolio_value"

    return {
        "daily_holdings": daily_holdings,
        "daily_cash": daily_cash,
        "daily_security_value": daily_security_value,
        "daily_portfolio_value": daily_portfolio_value,
        "external_cash_flows_by_day": external_cash_flows_by_day
    }
