from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.investor import run_investor_performance_analysis


def make_investor_case():
    dates = pd.bdate_range("2024-01-02", periods=3)
    prices = pd.DataFrame({"AAA": [10.0, 11.0, 12.0]}, index=dates)
    transactions = pd.DataFrame({
        "date": [dates[0], dates[1]],
        "ticker": ["AAA", "AAA"],
        "side": ["BUY", "SELL"],
        "quantity": [10, 5],
        "price": [10, 11]
    })
    cash_flows = pd.DataFrame({
        "date": [dates[0], dates[2]],
        "type": ["DEPOSIT", "WITHDRAWAL"],
        "amount": [100, 55]
    })
    return dates, prices, transactions, cash_flows


def test_investor_analysis_uses_injected_prices_and_returns_results(
    monkeypatch
):
    dates, prices, transactions, cash_flows = make_investor_case()
    original_prices = prices.copy(deep=True)
    original_transactions = transactions.copy(deep=True)
    original_cash_flows = cash_flows.copy(deep=True)

    def fail_if_download_is_called(*args, **kwargs):
        raise AssertionError(
            "Injected prices should skip the market-data download."
        )

    monkeypatch.setattr(
        "src.investor.download_price_views",
        fail_if_download_is_called
    )

    results = run_investor_performance_analysis(
        transactions,
        cash_flows,
        prices=prices
    )

    assert np.isclose(results["current_portfolio_value"], 60.0)
    assert np.isclose(results["current_security_value"], 60.0)
    assert np.isclose(results["current_cash_balance"], 0.0)
    assert np.isclose(results["total_twr"], 0.20)
    assert results["daily_portfolio_value"].tolist() == [100.0, 110.0, 60.0]

    expected_current_holdings = pd.DataFrame({
        "ticker": ["AAA"],
        "quantity": [5.0],
        "average_cost": [10.0],
        "total_cost_basis": [50.0],
        "realized_pnl": [5.0]
    })
    pd.testing.assert_frame_equal(
        results["current_holdings"],
        expected_current_holdings
    )
    assert results["daily_twr"].index.equals(dates)
    assert np.isclose(results["cumulative_twr"].iloc[-1], 0.20)
    assert np.isclose(results["investor_wealth_index"].iloc[-1], 1.20)
    assert np.isclose(results["investor_max_drawdown"], 0.0)
    assert results["investor_drawdown"].name == "investor_drawdown"
    assert results["performance_observations"] == 3
    assert np.isclose(
        results["annualized_twr"],
        1.20 ** (252 / 3) - 1.0
    )
    assert np.isfinite(results["sharpe_ratio"])
    assert np.isnan(results["sortino_ratio"])
    assert np.isnan(results["calmar_ratio"])
    assert results["risk_free_rate_annual"] == 0.0

    pd.testing.assert_frame_equal(prices, original_prices)
    pd.testing.assert_frame_equal(transactions, original_transactions)
    pd.testing.assert_frame_equal(cash_flows, original_cash_flows)


def test_default_examples_run_with_prepared_spy_and_aapl_prices(
    monkeypatch
):
    project_dir = Path(__file__).resolve().parents[1]
    transactions = pd.read_csv(
        project_dir / "inputs" / "transactions_example.csv"
    )
    cash_flows = pd.read_csv(
        project_dir / "inputs" / "cash_flows_example.csv"
    )
    dates = pd.bdate_range("2024-01-05", "2025-04-01")
    downloaded = {}

    def fake_download_price_views(tickers, start_date):
        downloaded["tickers"] = tickers
        downloaded["start_date"] = start_date
        valuation_prices = pd.DataFrame(
            {
                "SPY": np.full(len(dates), 500.0),
                "AAPL": np.full(len(dates), 200.0)
            },
            index=dates
        )
        return {
            "valuation_prices": valuation_prices,
            "return_prices": valuation_prices * 0.90
        }

    monkeypatch.setattr(
        "src.investor.download_price_views",
        fake_download_price_views
    )

    results = run_investor_performance_analysis(
        transactions,
        cash_flows
    )

    assert downloaded == {
        "tickers": ["SPY", "AAPL"],
        "start_date": "2024-01-05"
    }
    assert results["current_holdings"]["ticker"].tolist() == ["SPY", "AAPL"]
    assert results["current_holdings"]["quantity"].tolist() == [2.0, 7.0]
    assert np.isclose(results["daily_cash"].min(), 1125.0)
    assert np.isclose(results["current_cash_balance"], 1245.0)
    assert np.isclose(results["current_security_value"], 2400.0)
    assert np.isclose(results["current_portfolio_value"], 3645.0)


def test_investor_freshness_check_does_not_backfill_a_future_price():
    dates = pd.bdate_range("2024-01-02", periods=2)
    valuation_prices = pd.DataFrame(
        {"AAA": [np.nan, 10.0]},
        index=dates
    )
    transactions = pd.DataFrame({
        "date": [dates[0]],
        "ticker": ["AAA"],
        "side": ["BUY"],
        "quantity": [1],
        "price": [10.0]
    })
    cash_flows = pd.DataFrame({
        "date": [dates[0]],
        "type": ["DEPOSIT"],
        "amount": [10.0]
    })

    with pytest.raises(ValueError, match="No current or earlier price"):
        run_investor_performance_analysis(
            transactions=transactions,
            external_cash_flows=cash_flows,
            valuation_prices=valuation_prices
        )


@pytest.mark.parametrize(
    ("event_type", "amount", "expected_twr", "summary_field"),
    [
        ("DIVIDEND", 10.0, 0.10, "dividend_income"),
        ("INTEREST", 10.0, 0.10, "interest_income"),
        ("FEE", 10.0, -0.10, "fees_paid"),
        ("TAX", 10.0, -0.10, "taxes_paid")
    ]
)
def test_internal_income_and_expenses_change_performance_not_external_flow(
    event_type,
    amount,
    expected_twr,
    summary_field
):
    dates = pd.bdate_range("2024-01-02", periods=2)
    prices = pd.DataFrame({"AAA": [10.0, 10.0]}, index=dates)
    event_ticker = "AAA" if event_type == "DIVIDEND" else np.nan
    ledger = pd.DataFrame({
        "date": [dates[0], dates[0], dates[1]],
        "type": ["DEPOSIT", "BUY", event_type],
        "ticker": [np.nan, "AAA", event_ticker],
        "quantity": [np.nan, 9.0, np.nan],
        "price": [np.nan, 10.0, np.nan],
        "amount": [100.0, np.nan, amount]
    })

    results = run_investor_performance_analysis(
        transactions=ledger,
        valuation_prices=prices
    )

    assert results["external_cash_flows_by_day"].tolist() == [100.0, 0.0]
    assert np.isclose(results["daily_twr"].iloc[1], expected_twr)
    assert np.isclose(results[summary_field], amount)


def test_economic_pnl_reconciles_income_and_costs_exactly_once():
    dates = pd.bdate_range("2024-01-02", periods=4)
    prices = pd.DataFrame({"AAA": [10.0] * 4}, index=dates)
    ledger = pd.DataFrame({
        "date": [
            dates[0], dates[0], dates[1], dates[2], dates[2], dates[3]
        ],
        "type": ["DEPOSIT", "BUY", "DIVIDEND", "INTEREST", "FEE", "TAX"],
        "ticker": [np.nan, "AAA", "AAA", np.nan, np.nan, np.nan],
        "quantity": [np.nan, 9.0, np.nan, np.nan, np.nan, np.nan],
        "price": [np.nan, 10.0, np.nan, np.nan, np.nan, np.nan],
        "amount": [120.0, np.nan, 10.0, 5.0, 2.0, 3.0],
        "fees": [0.0, 2.0, 1.0, 1.0, 0.0, 0.0],
        "taxes": [0.0, 1.0, 2.0, 1.0, 0.0, 0.0]
    })

    results = run_investor_performance_analysis(
        transactions=ledger,
        valuation_prices=prices
    )

    assert np.isclose(results["dividend_income"], 10.0)
    assert np.isclose(results["interest_income"], 5.0)
    assert np.isclose(results["fees_paid"], 6.0)
    assert np.isclose(results["taxes_paid"], 7.0)
    assert np.isclose(results["net_external_contributions"], 120.0)
    assert np.isclose(results["economic_total_pnl"], 2.0)
    assert np.isclose(
        results["economic_total_pnl"],
        results["current_portfolio_value"]
        - results["net_external_contributions"]
    )


def test_deposit_withdrawal_are_neutral_and_trades_are_internal_twr_events():
    dates = pd.bdate_range("2024-01-02", periods=3)
    prices = pd.DataFrame({"AAA": [10.0, 10.0, 10.0]}, index=dates)
    ledger = pd.DataFrame({
        "date": [dates[0], dates[0], dates[1], dates[1], dates[2]],
        "type": ["DEPOSIT", "BUY", "SELL", "WITHDRAWAL", "BUY"],
        "ticker": [np.nan, "AAA", "AAA", np.nan, "AAA"],
        "quantity": [np.nan, 5.0, 2.0, np.nan, 1.0],
        "price": [np.nan, 10.0, 10.0, np.nan, 10.0],
        "amount": [100.0, np.nan, np.nan, 20.0, np.nan]
    })

    results = run_investor_performance_analysis(
        transactions=ledger,
        valuation_prices=prices
    )

    assert results["external_cash_flows_by_day"].tolist() == [100.0, -20.0, 0.0]
    assert np.allclose(results["daily_twr"], [0.0, 0.0, 0.0])
