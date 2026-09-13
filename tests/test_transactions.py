import numpy as np
import pandas as pd
import pytest

from src.transactions import (
    build_current_portfolio,
    reconstruct_accounting_history,
    reconstruct_daily_cash,
    reconstruct_daily_holdings,
    reconstruct_holdings,
    validate_external_cash_flows,
    validate_transactions
)


def make_transactions(rows):
    return pd.DataFrame(
        rows,
        columns=["date", "ticker", "side", "quantity", "price"]
    )


def test_one_simple_buy():
    transactions = make_transactions([
        ["2024-01-10", "AAA", "BUY", 10, 100]
    ])

    holdings = reconstruct_holdings(transactions)
    position = holdings.iloc[0]

    assert position["ticker"] == "AAA"
    assert np.isclose(position["quantity"], 10.0)
    assert np.isclose(position["average_cost"], 100.0)
    assert np.isclose(position["total_cost_basis"], 1000.0)
    assert np.isclose(position["realized_pnl"], 0.0)


def test_multiple_buys_use_weighted_average_cost():
    transactions = make_transactions([
        ["2024-01-10", "AAA", "BUY", 10, 100],
        ["2024-06-15", "AAA", "BUY", 5, 130]
    ])

    position = reconstruct_holdings(transactions).iloc[0]

    assert np.isclose(position["quantity"], 15.0)
    assert np.isclose(position["average_cost"], 110.0)
    assert np.isclose(position["total_cost_basis"], 1650.0)


def test_partial_sell_keeps_average_cost_and_calculates_realized_pnl():
    transactions = make_transactions([
        ["2024-01-10", "AAA", "BUY", 10, 100],
        ["2024-06-15", "AAA", "BUY", 5, 130],
        ["2025-02-03", "AAA", "SELL", 3, 160]
    ])

    position = reconstruct_holdings(transactions).iloc[0]

    assert np.isclose(position["quantity"], 12.0)
    assert np.isclose(position["average_cost"], 110.0)
    assert np.isclose(position["total_cost_basis"], 1320.0)
    assert np.isclose(position["realized_pnl"], 150.0)


def test_full_position_closure_zeros_cost_fields():
    transactions = make_transactions([
        ["2024-01-10", "AAA", "BUY", 4, 100],
        ["2024-02-10", "AAA", "SELL", 4, 125]
    ])

    position = reconstruct_holdings(transactions).iloc[0]

    assert np.isclose(position["quantity"], 0.0)
    assert np.isclose(position["average_cost"], 0.0)
    assert np.isclose(position["total_cost_basis"], 0.0)
    assert np.isclose(position["realized_pnl"], 100.0)


def test_sell_greater_than_available_quantity_is_rejected():
    transactions = make_transactions([
        ["2024-01-10", "AAA", "BUY", 4, 100],
        ["2024-02-10", "AAA", "SELL", 5, 125]
    ])

    with pytest.raises(ValueError, match="only 4 units are available"):
        reconstruct_holdings(transactions)


def test_transactions_are_processed_chronologically_with_stable_ties():
    transactions = make_transactions([
        ["2024-02-10", "AAA", "BUY", 5, 130],
        ["2024-01-10", "AAA", "BUY", 10, 100],
        ["2024-02-10", "AAA", "SELL", 12, 160]
    ])

    position = reconstruct_holdings(transactions).iloc[0]

    assert np.isclose(position["quantity"], 3.0)
    assert np.isclose(position["average_cost"], 110.0)
    assert np.isclose(position["realized_pnl"], 600.0)


@pytest.mark.parametrize(
    ("column", "invalid_value", "message"),
    [
        ("quantity", "many", "quantities.*numeric"),
        ("price", "expensive", "prices.*numeric"),
        ("quantity", np.inf, "quantities.*finite"),
        ("price", -np.inf, "prices.*finite"),
        ("quantity", 0, "quantities must be strictly positive"),
        ("price", 0, "prices must be strictly positive")
    ]
)
def test_invalid_quantity_or_price_is_rejected(
    column,
    invalid_value,
    message
):
    row = {
        "date": "2024-01-10",
        "ticker": "AAA",
        "side": "BUY",
        "quantity": 10,
        "price": 100
    }
    row[column] = invalid_value
    transactions = pd.DataFrame([row])

    with pytest.raises(ValueError, match=message):
        validate_transactions(transactions)


def test_invalid_side_is_rejected():
    transactions = make_transactions([
        ["2024-01-10", "AAA", "HOLD", 10, 100]
    ])

    with pytest.raises(ValueError, match="Unsupported transaction side"):
        validate_transactions(transactions)


def test_invalid_date_is_rejected():
    transactions = make_transactions([
        ["not-a-date", "AAA", "BUY", 10, 100]
    ])

    with pytest.raises(ValueError, match="valid dates"):
        validate_transactions(transactions)


@pytest.mark.parametrize(
    ("transactions", "message"),
    [
        ("not a DataFrame", "pandas DataFrame"),
        (
            pd.DataFrame(
                columns=["date", "ticker", "side", "quantity", "price"]
            ),
            "at least one row"
        ),
        (
            pd.DataFrame({
                "date": ["2024-01-10"],
                "ticker": ["AAA"],
                "side": ["BUY"],
                "quantity": [10]
            }),
            "Missing transaction columns"
        )
    ]
)
def test_invalid_transaction_container_is_rejected(transactions, message):
    with pytest.raises(ValueError, match=message):
        validate_transactions(transactions)


def test_blank_ticker_is_rejected():
    transactions = make_transactions([
        ["2024-01-10", "   ", "BUY", 10, 100]
    ])

    with pytest.raises(ValueError, match="tickers must not be blank"):
        validate_transactions(transactions)


def test_validation_does_not_mutate_caller_dataframe():
    transactions = make_transactions([
        ["2024-01-10", " aaa ", " buy ", "10", "100"]
    ])
    original_transactions = transactions.copy(deep=True)

    validated = validate_transactions(transactions)

    pd.testing.assert_frame_equal(transactions, original_transactions)
    assert validated.loc[0, "ticker"] == "AAA"
    assert validated.loc[0, "side"] == "BUY"
    assert pd.api.types.is_datetime64_any_dtype(validated["date"])
    assert validated.loc[0, "quantity"] == 10
    assert validated.loc[0, "price"] == 100


def test_build_current_portfolio_excludes_closed_positions():
    transactions = make_transactions([
        ["2024-01-10", "AAA", "BUY", 10, 100],
        ["2024-02-10", "AAA", "SELL", 10, 120],
        ["2024-03-10", "BBB", "BUY", 5, 80]
    ])

    portfolio = build_current_portfolio(transactions)
    expected = pd.DataFrame({
        "ticker": ["BBB"],
        "quantity": [5.0],
        "purchase_price": [80.0]
    })

    pd.testing.assert_frame_equal(portfolio, expected)


def make_accounting_case():
    dates = pd.bdate_range("2024-01-02", periods=4)
    prices = pd.DataFrame(
        {
            "AAA": [10.0, 12.0, 11.0, 13.0],
            "BBB": [20.0, 21.0, 22.0, 23.0]
        },
        index=dates
    )
    transactions = make_transactions([
        [dates[1], "AAA", "BUY", 5, 10],
        [dates[2], "BBB", "BUY", 2, 20],
        [dates[3], "AAA", "SELL", 2, 12]
    ])
    cash_flows = pd.DataFrame({
        "date": [dates[0], dates[3]],
        "type": ["DEPOSIT", "WITHDRAWAL"],
        "amount": [100, 10]
    })

    return dates, prices, transactions, cash_flows


def test_external_cash_flow_validation_normalizes_without_mutating():
    cash_flows = pd.DataFrame({
        "date": ["2024-01-02"],
        "type": [" deposit "],
        "amount": ["100"]
    })
    original_cash_flows = cash_flows.copy(deep=True)

    validated = validate_external_cash_flows(cash_flows)

    pd.testing.assert_frame_equal(cash_flows, original_cash_flows)
    assert validated.loc[0, "type"] == "DEPOSIT"
    assert validated.loc[0, "amount"] == 100
    assert pd.api.types.is_datetime64_any_dtype(validated["date"])


@pytest.mark.parametrize(
    ("cash_flows", "message"),
    [
        ("not a DataFrame", "pandas DataFrame"),
        (
            pd.DataFrame(columns=["date", "type", "amount"]),
            "at least one row"
        ),
        (
            pd.DataFrame({
                "date": ["2024-01-02"],
                "type": ["DEPOSIT"]
            }),
            "Missing external cash-flow columns"
        ),
        (
            pd.DataFrame({
                "date": ["not-a-date"],
                "type": ["DEPOSIT"],
                "amount": [100]
            }),
            "dates must be valid"
        ),
        (
            pd.DataFrame({
                "date": ["2024-01-02"],
                "type": ["TRANSFER"],
                "amount": [100]
            }),
            "Unsupported external cash-flow type"
        ),
        (
            pd.DataFrame({
                "date": ["2024-01-02"],
                "type": ["DEPOSIT"],
                "amount": ["many"]
            }),
            "amounts.*numeric"
        ),
        (
            pd.DataFrame({
                "date": ["2024-01-02"],
                "type": ["DEPOSIT"],
                "amount": [np.inf]
            }),
            "amounts.*finite"
        ),
        (
            pd.DataFrame({
                "date": ["2024-01-02"],
                "type": ["DEPOSIT"],
                "amount": [0]
            }),
            "amounts must be strictly positive"
        )
    ]
)
def test_invalid_external_cash_flows_are_rejected(cash_flows, message):
    with pytest.raises(ValueError, match=message):
        validate_external_cash_flows(cash_flows)


def test_daily_holdings_before_and_after_transactions():
    dates, prices, transactions, _ = make_accounting_case()

    daily_holdings = reconstruct_daily_holdings(transactions, prices)

    expected = pd.DataFrame(
        {
            "AAA": [0.0, 5.0, 5.0, 3.0],
            "BBB": [0.0, 0.0, 2.0, 2.0]
        },
        index=dates
    )
    pd.testing.assert_frame_equal(daily_holdings, expected)


def test_daily_cash_tracks_internal_trades_and_external_flows():
    dates, prices, transactions, cash_flows = make_accounting_case()

    daily_cash, external_flows = reconstruct_daily_cash(
        transactions,
        cash_flows,
        prices
    )

    expected_cash = pd.Series(
        [100.0, 50.0, 10.0, 24.0],
        index=dates,
        name="cash_balance"
    )
    expected_external_flows = pd.Series(
        [100.0, 0.0, 0.0, -10.0],
        index=dates,
        name="external_cash_flow"
    )
    pd.testing.assert_series_equal(daily_cash, expected_cash)
    pd.testing.assert_series_equal(external_flows, expected_external_flows)


def test_daily_portfolio_value_known_case():
    dates, prices, transactions, cash_flows = make_accounting_case()

    history = reconstruct_accounting_history(
        transactions,
        cash_flows,
        prices
    )

    expected_security_value = pd.Series(
        [0.0, 60.0, 99.0, 85.0],
        index=dates,
        name="security_market_value"
    )
    expected_portfolio_value = pd.Series(
        [100.0, 110.0, 109.0, 109.0],
        index=dates,
        name="portfolio_value"
    )
    pd.testing.assert_series_equal(
        history["daily_security_value"],
        expected_security_value
    )
    pd.testing.assert_series_equal(
        history["daily_portfolio_value"],
        expected_portfolio_value
    )


def test_same_day_deposit_is_processed_before_buy():
    date = pd.Timestamp("2024-01-02")
    prices = pd.DataFrame({"AAA": [10.0]}, index=[date])
    transactions = make_transactions([
        [date, "AAA", "BUY", 10, 10]
    ])
    cash_flows = pd.DataFrame({
        "date": [date],
        "type": ["DEPOSIT"],
        "amount": [100]
    })

    daily_cash, _ = reconstruct_daily_cash(
        transactions,
        cash_flows,
        prices
    )

    assert np.isclose(daily_cash.iloc[0], 0.0)


def test_buy_with_insufficient_cash_is_rejected():
    date = pd.Timestamp("2024-01-02")
    prices = pd.DataFrame({"AAA": [10.0]}, index=[date])
    transactions = make_transactions([
        [date, "AAA", "BUY", 6, 10]
    ])
    cash_flows = pd.DataFrame({
        "date": [date],
        "type": ["DEPOSIT"],
        "amount": [50]
    })

    with pytest.raises(ValueError, match="Insufficient cash for BUY"):
        reconstruct_daily_cash(transactions, cash_flows, prices)


def test_withdrawal_with_insufficient_cash_is_rejected():
    dates = pd.bdate_range("2024-01-02", periods=2)
    prices = pd.DataFrame({"AAA": [10.0, 10.0]}, index=dates)
    transactions = make_transactions([
        [dates[1], "AAA", "BUY", 1, 10]
    ])
    cash_flows = pd.DataFrame({
        "date": dates,
        "type": ["DEPOSIT", "WITHDRAWAL"],
        "amount": [50, 60]
    })

    with pytest.raises(ValueError, match="Insufficient cash for WITHDRAWAL"):
        reconstruct_daily_cash(transactions, cash_flows, prices)


def test_full_closure_is_reflected_in_daily_holdings():
    dates = pd.bdate_range("2024-01-02", periods=3)
    prices = pd.DataFrame({"AAA": [10.0, 11.0, 12.0]}, index=dates)
    transactions = make_transactions([
        [dates[0], "AAA", "BUY", 3, 10],
        [dates[2], "AAA", "SELL", 3, 12]
    ])

    daily_holdings = reconstruct_daily_holdings(transactions, prices)

    assert daily_holdings["AAA"].tolist() == [3.0, 3.0, 0.0]


def test_non_market_day_event_applies_on_next_price_date():
    dates = pd.to_datetime(["2024-01-05", "2024-01-08"])
    prices = pd.DataFrame({"AAA": [10.0, 11.0]}, index=dates)
    transactions = make_transactions([
        ["2024-01-06", "AAA", "BUY", 1, 10]
    ])

    daily_holdings = reconstruct_daily_holdings(transactions, prices)

    assert daily_holdings["AAA"].tolist() == [0.0, 1.0]


def test_valuation_uses_earlier_price_but_not_future_price():
    dates = pd.bdate_range("2024-01-02", periods=2)
    transactions = make_transactions([
        [dates[0], "AAA", "BUY", 1, 10]
    ])
    cash_flows = pd.DataFrame({
        "date": [dates[0]],
        "type": ["DEPOSIT"],
        "amount": [10]
    })

    prices_with_past_value = pd.DataFrame(
        {"AAA": [10.0, np.nan]},
        index=dates
    )
    history = reconstruct_accounting_history(
        transactions,
        cash_flows,
        prices_with_past_value
    )
    assert history["daily_security_value"].tolist() == [10.0, 10.0]

    prices_with_only_future_value = pd.DataFrame(
        {"AAA": [np.nan, 10.0]},
        index=dates
    )
    with pytest.raises(ValueError, match="No current or earlier price"):
        reconstruct_accounting_history(
            transactions,
            cash_flows,
            prices_with_only_future_value
        )


def test_accounting_history_does_not_mutate_caller_dataframes():
    _, prices, transactions, cash_flows = make_accounting_case()
    original_prices = prices.copy(deep=True)
    original_transactions = transactions.copy(deep=True)
    original_cash_flows = cash_flows.copy(deep=True)

    reconstruct_accounting_history(transactions, cash_flows, prices)

    pd.testing.assert_frame_equal(prices, original_prices)
    pd.testing.assert_frame_equal(transactions, original_transactions)
    pd.testing.assert_frame_equal(cash_flows, original_cash_flows)
