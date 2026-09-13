import numpy as np
import pandas as pd
import pytest

from src.performance import calculate_time_weighted_return
from src.transactions import reconstruct_accounting_history


def make_series(values, dates, name):
    return pd.Series(values, index=pd.DatetimeIndex(dates), name=name)


def calculate(values, flows, dates):
    portfolio_value = make_series(values, dates, "portfolio_value")
    external_flows = make_series(flows, dates, "external_cash_flow")
    return calculate_time_weighted_return(portfolio_value, external_flows)


def test_initial_deposit_with_no_performance_has_zero_return():
    result = calculate([100.0], [100.0], ["2024-01-02"])

    assert np.isclose(result["daily_twr"].iloc[0], 0.0)
    assert np.isclose(result["total_twr"], 0.0)


def test_deposit_does_not_create_artificial_return():
    result = calculate(
        [100.0, 200.0],
        [100.0, 100.0],
        ["2024-01-02", "2024-01-03"]
    )

    assert np.isclose(result["daily_twr"].iloc[1], 0.0)


def test_real_investment_gain_after_funding():
    result = calculate(
        [200.0, 220.0],
        [200.0, 0.0],
        ["2024-01-02", "2024-01-03"]
    )

    assert np.isclose(result["daily_twr"].iloc[1], 0.10)


def test_withdrawal_does_not_create_artificial_return():
    result = calculate(
        [200.0, 150.0],
        [200.0, -50.0],
        ["2024-01-02", "2024-01-03"]
    )

    assert np.isclose(result["daily_twr"].iloc[1], 0.0)


def test_daily_returns_are_linked_geometrically():
    dates = pd.date_range("2024-01-02", periods=3)
    result = calculate([110.0, 110.0, 121.0], [100.0, 0.0, 0.0], dates)

    expected_daily = make_series([0.10, 0.0, 0.10], dates, "daily_twr")
    expected_cumulative = make_series(
        [0.10, 0.10, 0.21],
        dates,
        "cumulative_twr"
    )
    pd.testing.assert_series_equal(result["daily_twr"], expected_daily)
    pd.testing.assert_series_equal(
        result["cumulative_twr"],
        expected_cumulative
    )
    assert np.isclose(result["total_twr"], 0.21)


def test_zero_capital_period_is_nan_and_does_not_change_cumulative_twr():
    dates = pd.date_range("2024-01-02", periods=3)
    result = calculate([0.0, 100.0, 110.0], [0.0, 100.0, 0.0], dates)

    assert np.isnan(result["daily_twr"].iloc[0])
    assert np.isclose(result["cumulative_twr"].iloc[0], 0.0)
    assert np.isclose(result["cumulative_twr"].iloc[1], 0.0)
    assert np.isclose(result["total_twr"], 0.10)


def test_zero_capital_with_nonzero_portfolio_value_is_rejected():
    with pytest.raises(ValueError, match="value must be zero.*capital base"):
        calculate([10.0], [0.0], ["2024-01-02"])


def test_negative_capital_base_is_rejected():
    with pytest.raises(ValueError, match="capital base cannot be negative"):
        calculate(
            [100.0, 10.0],
            [100.0, -110.0],
            ["2024-01-02", "2024-01-03"]
        )


def test_index_mismatch_is_rejected():
    values = make_series([100.0], ["2024-01-02"], "portfolio_value")
    flows = make_series([100.0], ["2024-01-03"], "external_cash_flow")

    with pytest.raises(ValueError, match="indexes must align exactly"):
        calculate_time_weighted_return(values, flows)


def test_duplicate_dates_are_rejected():
    dates = ["2024-01-02", "2024-01-02"]

    with pytest.raises(ValueError, match="duplicate dates"):
        calculate([100.0, 100.0], [100.0, 0.0], dates)


def test_non_chronological_index_is_rejected():
    dates = ["2024-01-03", "2024-01-02"]

    with pytest.raises(ValueError, match="chronological order"):
        calculate([100.0, 100.0], [100.0, 0.0], dates)


@pytest.mark.parametrize(
    ("values", "flows", "message"),
    [
        ([100.0], pd.Series([100.0]), "portfolio_value.*pandas Series"),
        (pd.Series([100.0]), [100.0], "cash_flows.*pandas Series"),
        (
            pd.Series(["100"]),
            pd.Series([100.0]),
            "portfolio_value.*numeric"
        ),
        (
            pd.Series([100.0]),
            pd.Series([np.inf]),
            "cash_flows.*finite"
        )
    ]
)
def test_invalid_input_values_are_rejected(values, flows, message):
    with pytest.raises(ValueError, match=message):
        calculate_time_weighted_return(values, flows)


def test_caller_series_are_not_mutated():
    dates = pd.date_range("2024-01-02", periods=2)
    values = make_series([100, 110], dates, "portfolio_value")
    flows = make_series([100, 0], dates, "external_cash_flow")
    original_values = values.copy(deep=True)
    original_flows = flows.copy(deep=True)

    calculate_time_weighted_return(values, flows)

    pd.testing.assert_series_equal(values, original_values)
    pd.testing.assert_series_equal(flows, original_flows)


def test_twr_integrates_with_reconstructed_accounting_history():
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

    history = reconstruct_accounting_history(
        transactions,
        cash_flows,
        prices
    )
    result = calculate_time_weighted_return(
        history["daily_portfolio_value"],
        history["external_cash_flows_by_day"]
    )

    # Day 1: 100 / (0 + 100) - 1 = 0%.
    # Day 2: 110 / (100 + 0) - 1 = 10%.
    # Day 3: 60 / (110 - 55) - 1 = 9.0909%; linked total = 20%.
    expected_daily = make_series(
        [0.0, 0.10, 60.0 / 55.0 - 1.0],
        dates,
        "daily_twr"
    )
    pd.testing.assert_series_equal(result["daily_twr"], expected_daily)
    assert np.isclose(result["total_twr"], 0.20)
