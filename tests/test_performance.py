import numpy as np
import pandas as pd
import pytest

from src.performance import (
    calculate_annualized_twr,
    calculate_calmar_ratio,
    calculate_daily_risk_free_rate,
    calculate_investor_performance_metrics,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_time_weighted_return,
    calculate_twr_drawdown
)
from src.transactions import reconstruct_accounting_history


def make_series(values, dates, name):
    return pd.Series(values, index=pd.DatetimeIndex(dates), name=name)


def calculate(values, flows, dates):
    portfolio_value = make_series(values, dates, "portfolio_value")
    external_flows = make_series(flows, dates, "external_cash_flow")
    return calculate_time_weighted_return(portfolio_value, external_flows)


def test_annualized_twr_uses_geometric_active_observations():
    daily_twr = pd.Series([0.01, np.nan, 0.02], name="daily_twr")
    original = daily_twr.copy(deep=True)

    result = calculate_annualized_twr(daily_twr)
    expected = (1.01 * 1.02) ** (252 / 2) - 1.0

    assert np.isclose(result, expected)
    pd.testing.assert_series_equal(daily_twr, original)


def test_annualized_twr_can_be_negative_and_rejects_zero_wealth():
    assert calculate_annualized_twr(
        pd.Series([-0.01, -0.02], name="daily_twr")
    ) < 0.0

    with pytest.raises(ValueError, match="linked wealth.*not positive"):
        calculate_annualized_twr(pd.Series([-1.0], name="daily_twr"))


@pytest.mark.parametrize("annual_rate", [0.03, -0.05])
def test_daily_risk_free_rate_uses_compounded_conversion(annual_rate):
    expected = (1.0 + annual_rate) ** (1.0 / 252) - 1.0
    assert np.isclose(
        calculate_daily_risk_free_rate(annual_rate),
        expected
    )


def test_sharpe_known_case_uses_sample_standard_deviation():
    daily_twr = pd.Series([0.01, -0.005, 0.02], name="daily_twr")
    expected = daily_twr.mean() / daily_twr.std(ddof=1) * np.sqrt(252)

    assert np.isclose(calculate_sharpe_ratio(daily_twr), expected)


def test_sharpe_applies_nonzero_compounded_risk_free_hurdle():
    daily_twr = pd.Series([0.01, -0.005, 0.02], name="daily_twr")
    risk_free_daily = (1.03 ** (1 / 252)) - 1.0
    excess = daily_twr - risk_free_daily
    expected = excess.mean() / excess.std(ddof=1) * np.sqrt(252)

    assert np.isclose(
        calculate_sharpe_ratio(daily_twr, risk_free_rate_annual=0.03),
        expected
    )


def test_sortino_known_case_uses_full_sample_lower_partial_moment():
    daily_twr = pd.Series([0.01, -0.02, 0.03], name="daily_twr")
    downside_deviation = np.sqrt(np.mean(np.array([0.0, -0.02, 0.0]) ** 2))
    expected = (
        daily_twr.mean() * 252
        / (downside_deviation * np.sqrt(252))
    )

    assert np.isclose(calculate_sortino_ratio(daily_twr), expected)


def test_calmar_uses_annualized_twr_and_actual_investor_drawdown():
    assert np.isclose(calculate_calmar_ratio(0.12, -0.20), 0.60)


def test_undefined_ratio_denominators_and_too_few_returns_are_nan():
    constant_positive = pd.Series([0.01, 0.01, 0.01], name="daily_twr")
    one_return = pd.Series([0.01], name="daily_twr")

    assert np.isnan(calculate_sharpe_ratio(constant_positive))
    assert np.isnan(calculate_sortino_ratio(constant_positive))
    assert np.isnan(calculate_calmar_ratio(0.10, 0.0))
    assert np.isnan(calculate_sharpe_ratio(one_return))
    assert np.isnan(calculate_sortino_ratio(one_return))


@pytest.mark.parametrize("risk_free_rate", [-0.20, 0.50, np.inf, True])
def test_risk_free_rate_validation_rejects_invalid_values(risk_free_rate):
    with pytest.raises(ValueError, match="risk_free_rate_annual"):
        calculate_daily_risk_free_rate(risk_free_rate)


def test_performance_metrics_exclude_inactive_nan_periods_and_do_not_mutate():
    daily_twr = pd.Series(
        [np.nan, 0.01, -0.02, np.nan, 0.03],
        name="daily_twr"
    )
    original = daily_twr.copy(deep=True)

    metrics = calculate_investor_performance_metrics(
        daily_twr=daily_twr,
        investor_max_drawdown=-0.02,
        risk_free_rate_annual=-0.01
    )
    active = daily_twr.dropna()

    assert metrics["performance_observations"] == 3
    assert np.isclose(
        metrics["annualized_twr"],
        (1.0 + active).prod() ** (252 / 3) - 1.0
    )
    assert metrics["risk_free_rate_annual"] == -0.01
    pd.testing.assert_series_equal(daily_twr, original)


def test_initial_deposit_with_no_performance_has_zero_return():
    result = calculate([100.0], [100.0], ["2024-01-02"])

    assert np.isclose(result["daily_twr"].iloc[0], 0.0)
    assert np.isclose(result["total_twr"], 0.0)
    assert np.isclose(
        calculate_twr_drawdown(result["daily_twr"])[
            "investor_max_drawdown"
        ],
        0.0
    )


def test_deposit_does_not_create_artificial_return():
    result = calculate(
        [100.0, 200.0],
        [100.0, 100.0],
        ["2024-01-02", "2024-01-03"]
    )

    assert np.isclose(result["daily_twr"].iloc[1], 0.0)
    assert np.isclose(
        calculate_twr_drawdown(result["daily_twr"])[
            "investor_max_drawdown"
        ],
        0.0
    )


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
    assert np.isclose(
        calculate_twr_drawdown(result["daily_twr"])[
            "investor_max_drawdown"
        ],
        0.0
    )


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


def test_twr_drawdown_known_wealth_and_drawdown_path():
    dates = pd.date_range("2024-01-02", periods=2)
    daily_twr = make_series([0.10, -0.10], dates, "daily_twr")

    result = calculate_twr_drawdown(daily_twr)

    expected_wealth = make_series(
        [1.10, 0.99],
        dates,
        "investor_wealth_index"
    )
    expected_drawdown = make_series(
        [0.0, -0.10],
        dates,
        "investor_drawdown"
    )
    pd.testing.assert_series_equal(
        result["investor_wealth_index"],
        expected_wealth
    )
    pd.testing.assert_series_equal(
        result["investor_drawdown"],
        expected_drawdown
    )
    assert np.isclose(result["investor_max_drawdown"], -0.10)


def test_twr_drawdown_includes_starting_wealth_in_high_water_mark():
    daily_twr = pd.Series([-0.10], name="daily_twr")

    result = calculate_twr_drawdown(daily_twr)

    assert np.isclose(result["investor_wealth_index"].iloc[0], 0.90)
    assert np.isclose(result["investor_drawdown"].iloc[0], -0.10)
    assert np.isclose(result["investor_max_drawdown"], -0.10)


def test_twr_drawdown_returns_to_zero_at_recovery_and_new_high():
    dates = pd.date_range("2024-01-02", periods=4)
    daily_twr = make_series(
        [0.10, -0.10, 1.10 / 0.99 - 1.0, 0.05],
        dates,
        "daily_twr"
    )

    result = calculate_twr_drawdown(daily_twr)

    assert np.isclose(result["investor_drawdown"].iloc[1], -0.10)
    assert np.isclose(result["investor_drawdown"].iloc[2], 0.0)
    assert np.isclose(result["investor_drawdown"].iloc[3], 0.0)


def test_twr_drawdown_leaves_initial_inactive_period_missing():
    dates = pd.date_range("2024-01-02", periods=3)
    daily_twr = make_series([np.nan, 0.0, 0.10], dates, "daily_twr")

    result = calculate_twr_drawdown(daily_twr)

    assert np.isnan(result["investor_wealth_index"].iloc[0])
    assert np.isnan(result["investor_drawdown"].iloc[0])
    assert np.isclose(result["investor_wealth_index"].iloc[1], 1.0)
    assert np.isclose(result["investor_drawdown"].iloc[1], 0.0)
    assert np.isclose(result["investor_max_drawdown"], 0.0)


@pytest.mark.parametrize(
    ("daily_twr", "message"),
    [
        (pd.Series(dtype="float64"), "must not be empty"),
        (pd.Series(["bad"]), "real numeric"),
        (pd.Series([np.inf]), "finite values"),
        (pd.Series([np.nan]), "active-period return"),
        (pd.Series([-1.01]), "below -100%")
    ]
)
def test_twr_drawdown_rejects_invalid_series(daily_twr, message):
    with pytest.raises(ValueError, match=message):
        calculate_twr_drawdown(daily_twr)


def test_twr_drawdown_does_not_mutate_caller_series():
    dates = pd.date_range("2024-01-02", periods=3)
    daily_twr = make_series([0.05, np.nan, -0.02], dates, "daily_twr")
    original_daily_twr = daily_twr.copy(deep=True)

    calculate_twr_drawdown(daily_twr)

    pd.testing.assert_series_equal(daily_twr, original_daily_twr)


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

    drawdown_result = calculate_twr_drawdown(result["daily_twr"])
    assert np.isclose(drawdown_result["investor_max_drawdown"], 0.0)
    assert not drawdown_result["investor_drawdown"].isna().any()


def test_twr_links_across_liquidation_withdrawal_and_later_refunding():
    dates = pd.bdate_range("2024-01-02", periods=6)
    prices = pd.DataFrame(
        {"AAA": [10.0, 12.0, 12.0, 12.0, 10.0, 11.0]},
        index=dates
    )
    transactions = pd.DataFrame({
        "date": [dates[0], dates[1], dates[4]],
        "ticker": ["AAA", "AAA", "AAA"],
        "side": ["BUY", "SELL", "BUY"],
        "quantity": [10, 10, 5],
        "price": [10.0, 12.0, 10.0]
    })
    cash_flows = pd.DataFrame({
        "date": [dates[0], dates[2], dates[4]],
        "type": ["DEPOSIT", "WITHDRAWAL", "DEPOSIT"],
        "amount": [100.0, 120.0, 50.0]
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

    assert np.isclose(result["daily_twr"].iloc[1], 0.20)
    assert np.isnan(result["daily_twr"].iloc[2])
    assert np.isnan(result["daily_twr"].iloc[3])
    assert np.isclose(result["daily_twr"].iloc[4], 0.0)
    assert np.isclose(result["daily_twr"].iloc[5], 0.10)
    assert np.isclose(result["total_twr"], 0.32)

    performance_metrics = calculate_investor_performance_metrics(
        result["daily_twr"],
        calculate_twr_drawdown(result["daily_twr"])[
            "investor_max_drawdown"
        ]
    )
    assert performance_metrics["performance_observations"] == 4
    assert np.isclose(
        performance_metrics["annualized_twr"],
        1.32 ** (252 / 4) - 1.0
    )

    drawdown_result = calculate_twr_drawdown(result["daily_twr"])
    assert drawdown_result["investor_drawdown"].iloc[2:4].isna().all()
    assert np.isclose(
        drawdown_result["investor_wealth_index"].iloc[4],
        1.20
    )
    assert np.isclose(drawdown_result["investor_max_drawdown"], 0.0)
