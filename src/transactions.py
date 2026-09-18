import numpy as np
import pandas as pd

from src.data import prepare_price_data


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
NORMALIZED_LEDGER_COLUMNS = [
    "date",
    "type",
    "ticker",
    "quantity",
    "price",
    "amount",
    "fees",
    "taxes"
]
SUPPORTED_LEDGER_TYPES = {
    "BUY",
    "SELL",
    "DIVIDEND",
    "DEPOSIT",
    "WITHDRAWAL",
    "FEE",
    "TAX",
    "INTEREST"
}
TRADE_TYPES = {"BUY", "SELL"}
INCOME_TYPES = {"DIVIDEND", "INTEREST"}
AMOUNT_TYPES = {
    "DIVIDEND",
    "DEPOSIT",
    "WITHDRAWAL",
    "FEE",
    "TAX",
    "INTEREST"
}


def validate_transaction_ledger(ledger):
    """Validate and return a canonical, defensive ledger copy.

    The returned columns are always ordered as ``NORMALIZED_LEDGER_COLUMNS``.
    Fields that do not apply to a row remain missing, while fees and taxes are
    always numeric and default to zero.
    """
    if not isinstance(ledger, pd.DataFrame):
        raise ValueError("Transaction ledger must be a pandas DataFrame.")
    if ledger.empty:
        raise ValueError("Transaction ledger must contain at least one row.")

    missing_columns = {"date", "type"} - set(ledger.columns)
    if missing_columns:
        raise ValueError(
            f"Missing transaction ledger columns: {sorted(missing_columns)}"
        )

    ledger = ledger.copy(deep=True)
    for column in NORMALIZED_LEDGER_COLUMNS:
        if column not in ledger.columns:
            ledger[column] = 0.0 if column in {"fees", "taxes"} else np.nan
    ledger = ledger[NORMALIZED_LEDGER_COLUMNS]

    ledger["date"] = pd.to_datetime(ledger["date"], errors="coerce")
    if ledger["date"].isna().any():
        raise ValueError("Transaction ledger dates must contain valid dates.")
    ledger["date"] = ledger["date"].dt.normalize()

    ledger["type"] = ledger["type"].astype(str).str.strip().str.upper()
    unsupported_types = sorted(set(ledger["type"]) - SUPPORTED_LEDGER_TYPES)
    if unsupported_types:
        raise ValueError(
            "Unsupported transaction type values: "
            f"{unsupported_types}."
        )

    for column in ["quantity", "price", "amount", "fees", "taxes"]:
        try:
            ledger[column] = pd.to_numeric(ledger[column], errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Transaction ledger {column} values must be numeric when "
                "present."
            ) from error
        present = ledger[column].notna()
        if not np.isfinite(ledger.loc[present, column]).all():
            raise ValueError(
                f"Transaction ledger {column} values must be finite when "
                "present."
            )

    ledger[["fees", "taxes"]] = ledger[["fees", "taxes"]].fillna(0.0)
    if (ledger[["fees", "taxes"]] < 0).any().any():
        raise ValueError("Transaction ledger fees and taxes cannot be negative.")

    cost_supported_rows = ledger["type"].isin(TRADE_TYPES | INCOME_TYPES)
    unsupported_costs = (
        ~cost_supported_rows
        & (ledger["fees"].ne(0.0) | ledger["taxes"].ne(0.0))
    )
    if unsupported_costs.any():
        raise ValueError(
            "fees and taxes may be attached only to BUY, SELL, DIVIDEND or "
            "INTEREST rows."
        )

    trade_rows = ledger["type"].isin(TRADE_TYPES)
    missing_trade_values = ledger.loc[
        trade_rows, ["ticker", "quantity", "price"]
    ].isna().any(axis=1)
    if missing_trade_values.any():
        raise ValueError("BUY and SELL require ticker, quantity and price.")

    dividend_rows = ledger["type"].eq("DIVIDEND")
    if ledger.loc[dividend_rows, ["ticker", "amount"]].isna().any(axis=1).any():
        raise ValueError("DIVIDEND requires ticker and amount.")

    ticker_rows = trade_rows | dividend_rows
    missing_tickers = ledger.loc[ticker_rows, "ticker"].isna()
    normalized_tickers = (
        ledger.loc[ticker_rows, "ticker"].astype(str).str.strip().str.upper()
    )
    if missing_tickers.any() or normalized_tickers.eq("").any():
        raise ValueError("Required transaction tickers must not be blank.")
    if ticker_rows.any():
        ledger["ticker"] = ledger["ticker"].astype("object")
        ledger.loc[ticker_rows, "ticker"] = normalized_tickers

    if (ledger.loc[trade_rows, "quantity"] <= 0).any():
        raise ValueError("BUY and SELL quantities must be strictly positive.")
    if (ledger.loc[trade_rows, "price"] <= 0).any():
        raise ValueError("BUY and SELL prices must be strictly positive.")

    amount_rows = ledger["type"].isin(AMOUNT_TYPES)
    if ledger.loc[amount_rows, "amount"].isna().any():
        raise ValueError(
            "DIVIDEND, DEPOSIT, WITHDRAWAL, FEE, TAX and INTEREST require "
            "amount."
        )
    if (ledger.loc[amount_rows, "amount"] <= 0).any():
        raise ValueError("Transaction ledger amounts must be strictly positive.")

    return ledger


def _combine_legacy_ledgers(transactions, cash_flows):
    """Convert the former two-input API to the canonical ledger."""
    trades = validate_transactions(transactions).rename(
        columns={"side": "type"}
    )
    trades["amount"] = np.nan
    trades["fees"] = 0.0
    trades["taxes"] = 0.0

    flows = validate_external_cash_flows(cash_flows)
    flows["ticker"] = np.nan
    flows["quantity"] = np.nan
    flows["price"] = np.nan
    flows["fees"] = 0.0
    flows["taxes"] = 0.0

    # Preserve the established same-day convention: external funding is
    # available before trades from the separate legacy ledger.
    combined = pd.concat([flows, trades], ignore_index=True, sort=False)
    return validate_transaction_ledger(combined)


def _resolve_ledger(transactions, cash_flows=None):
    if not isinstance(transactions, pd.DataFrame):
        raise ValueError("Transactions must be provided as a pandas DataFrame.")
    if "type" in transactions.columns and "side" not in transactions.columns:
        if cash_flows is not None:
            raise ValueError(
                "Do not supply separate cash flows with a normalized ledger."
            )
        return validate_transaction_ledger(transactions)
    if cash_flows is None:
        raise ValueError(
            "Legacy BUY/SELL transactions require separate external cash "
            "flows."
        )
    return _combine_legacy_ledgers(transactions, cash_flows)


def normalize_transaction_ledger(transactions, cash_flows=None):
    """Return one canonical ledger from normalized or legacy inputs."""
    return _resolve_ledger(transactions, cash_flows)


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
    if "type" in transactions.columns and "side" not in transactions.columns:
        transactions = validate_transaction_ledger(transactions)
        transactions = transactions[
            transactions["type"].isin(TRADE_TYPES)
        ].rename(columns={"type": "side"})
    else:
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

    return pd.DataFrame(
        holdings.values(),
        columns=[
            "ticker",
            "quantity",
            "average_cost",
            "total_cost_basis",
            "realized_pnl"
        ]
    )


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
    if "type" in transactions.columns and "side" not in transactions.columns:
        transactions = validate_transaction_ledger(transactions)
        transactions = transactions[
            transactions["type"].isin(TRADE_TYPES)
        ].rename(columns={"type": "side"})
    else:
        transactions = validate_transactions(transactions)
    transactions["date"] = transactions["date"].dt.normalize()
    transactions = transactions.sort_values("date", kind="stable")

    tickers = transactions["ticker"].drop_duplicates().tolist()
    prices = prepare_price_data(prices, required_assets=tickers)
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


def reconstruct_daily_cash(transactions, cash_flows=None, prices=None):
    """Return daily cash and signed external flows on supplied price dates.

    A normalized ledger preserves stable input order. Under the legacy
    two-ledger API, external flows are processed before same-day trades.
    """
    if prices is None:
        raise ValueError("prices must be supplied.")
    ledger = _resolve_ledger(transactions, cash_flows)

    # Validate the complete trade sequence independently of the cash account.
    reconstruct_holdings(ledger)

    tickers = (
        ledger.loc[ledger["type"].isin(TRADE_TYPES), "ticker"]
        .drop_duplicates()
        .tolist()
    )
    prices = prepare_price_data(prices, required_assets=tickers)
    _reject_events_after_price_history(
        ledger["date"],
        prices.index[-1]
    )

    events = []
    for input_order, transaction in enumerate(ledger.itertuples(index=False)):
        event_type = transaction.type
        fees = float(transaction.fees)
        taxes = float(transaction.taxes)

        if event_type == "BUY":
            cash_change = -(
                float(transaction.quantity * transaction.price)
                + fees
                + taxes
            )
        elif event_type == "SELL":
            cash_change = (
                float(transaction.quantity * transaction.price)
                - fees
                - taxes
            )
        elif event_type in INCOME_TYPES:
            cash_change = float(transaction.amount) - fees - taxes
        elif event_type == "DEPOSIT":
            cash_change = float(transaction.amount)
        else:
            cash_change = -float(transaction.amount)

        events.append({
            "date": transaction.date,
            "source_priority": (
                0 if event_type in SUPPORTED_CASH_FLOW_TYPES else 1
            ),
            "input_order": input_order,
            "event_type": event_type,
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
                    f"Insufficient cash for {event['event_type']} on "
                    f"{event['date'].date()}: requires "
                    f"{required_cash:.2f}, but only "
                    f"{cash_balance:.2f} is available."
                )

            cash_balance += event["cash_change"]
            if np.isclose(cash_balance, 0.0):
                cash_balance = 0.0

            if event["event_type"] in SUPPORTED_CASH_FLOW_TYPES:
                external_cash_flows_by_day.loc[market_date] += (
                    event["cash_change"]
                )

            event_number += 1

        daily_cash.loc[market_date] = cash_balance

    return daily_cash, external_cash_flows_by_day


def reconstruct_accounting_history(transactions, cash_flows=None, prices=None):
    """Build daily holdings, cash and market value without performance metrics.

    Valuation uses each date's supplied market price. Missing prices may use
    only an earlier supplied observation through forward-filling, never a
    future price.
    """
    if prices is None:
        raise ValueError("prices must be supplied.")
    ledger = _resolve_ledger(transactions, cash_flows)
    tickers = (
        ledger.loc[ledger["type"].isin(TRADE_TYPES), "ticker"]
        .drop_duplicates()
        .tolist()
    )
    prices = prepare_price_data(prices, required_assets=tickers)

    daily_holdings = reconstruct_daily_holdings(
        ledger,
        prices
    )
    daily_cash, external_cash_flows_by_day = reconstruct_daily_cash(
        ledger,
        prices=prices
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

    holdings = reconstruct_holdings(ledger)
    latest_prices = valuation_prices.iloc[-1]
    unrealized_market_pnl = float(sum(
        position.quantity
        * (latest_prices[position.ticker] - position.average_cost)
        for position in holdings.itertuples(index=False)
        if position.quantity > 0
    ))
    realized_market_pnl = float(holdings["realized_pnl"].sum())
    dividend_income = float(
        ledger.loc[ledger["type"].eq("DIVIDEND"), "amount"].sum()
    )
    interest_income = float(
        ledger.loc[ledger["type"].eq("INTEREST"), "amount"].sum()
    )
    fees_paid = float(
        ledger["fees"].sum()
        + ledger.loc[ledger["type"].eq("FEE"), "amount"].sum()
    )
    taxes_paid = float(
        ledger["taxes"].sum()
        + ledger.loc[ledger["type"].eq("TAX"), "amount"].sum()
    )
    net_external_contributions = float(
        ledger.loc[ledger["type"].eq("DEPOSIT"), "amount"].sum()
        - ledger.loc[ledger["type"].eq("WITHDRAWAL"), "amount"].sum()
    )
    economic_total_pnl = (
        realized_market_pnl
        + unrealized_market_pnl
        + dividend_income
        + interest_income
        - fees_paid
        - taxes_paid
    )

    return {
        "transaction_ledger": ledger,
        "daily_holdings": daily_holdings,
        "daily_cash": daily_cash,
        "daily_security_value": daily_security_value,
        "daily_portfolio_value": daily_portfolio_value,
        "external_cash_flows_by_day": external_cash_flows_by_day,
        "realized_market_pnl": realized_market_pnl,
        "unrealized_market_pnl": unrealized_market_pnl,
        "dividend_income": dividend_income,
        "interest_income": interest_income,
        "fees_paid": fees_paid,
        "taxes_paid": taxes_paid,
        "net_external_contributions": net_external_contributions,
        "economic_total_pnl": economic_total_pnl
    }
