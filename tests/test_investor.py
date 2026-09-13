from pathlib import Path

import numpy as np
import pandas as pd

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
        "src.investor.download_prices",
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

    def fake_download_prices(tickers, start_date):
        downloaded["tickers"] = tickers
        downloaded["start_date"] = start_date
        return pd.DataFrame(
            {
                "SPY": np.full(len(dates), 500.0),
                "AAPL": np.full(len(dates), 200.0)
            },
            index=dates
        )

    monkeypatch.setattr(
        "src.investor.download_prices",
        fake_download_prices
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
