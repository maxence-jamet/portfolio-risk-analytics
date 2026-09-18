import numpy as np
import pandas as pd
import pytest

from src.benchmark import (
    normalize_benchmark_ticker,
    run_benchmark_analysis
)
from src.performance import calculate_daily_risk_free_rate


def make_benchmark_case(investor_returns, benchmark_returns, **overrides):
    dates = pd.bdate_range("2024-01-02", periods=len(benchmark_returns) + 1)
    benchmark_prices = pd.Series(
        np.concatenate((
            [100.0],
            100.0 * np.cumprod(1.0 + np.asarray(benchmark_returns))
        )),
        index=dates,
        name="SPY"
    )
    daily_twr = pd.Series(
        investor_returns,
        index=dates[1:],
        name="daily_twr"
    )
    arguments = {
        "daily_twr": daily_twr,
        "benchmark_prices": benchmark_prices,
        "benchmark_ticker": " spy ",
        "risk_free_rate_annual": 0.0
    }
    arguments.update(overrides)
    return run_benchmark_analysis(**arguments), daily_twr, benchmark_prices


def test_benchmark_ticker_is_trimmed_uppercase_and_empty_is_rejected():
    assert normalize_benchmark_ticker("  voo ") == "VOO"
    with pytest.raises(ValueError, match="must not be empty"):
        normalize_benchmark_ticker("   ")


def test_identical_returns_have_unit_beta_and_zero_tracking_error():
    returns = np.array([0.01, -0.02, 0.015, 0.005])
    result, _, _ = make_benchmark_case(returns, returns)

    assert np.isclose(result["beta"], 1.0)
    assert np.isclose(result["alpha_annualized"], 0.0)
    assert np.isclose(result["tracking_error"], 0.0)
    assert np.isnan(result["information_ratio"])


def test_double_benchmark_returns_have_beta_two():
    benchmark_returns = np.array([0.01, -0.02, 0.015, 0.005])
    result, _, _ = make_benchmark_case(
        2.0 * benchmark_returns,
        benchmark_returns
    )

    assert np.isclose(result["beta"], 2.0)
    assert np.isclose(result["alpha_annualized"], 0.0)


def test_jensen_alpha_uses_compounded_daily_risk_free_rate():
    benchmark_returns = np.array([0.01, -0.005, 0.02, 0.0])
    annual_risk_free_rate = 0.03
    risk_free_daily = calculate_daily_risk_free_rate(annual_risk_free_rate)
    expected_beta = 1.5
    expected_alpha_daily = 0.0004
    investor_returns = (
        risk_free_daily
        + expected_alpha_daily
        + expected_beta * (benchmark_returns - risk_free_daily)
    )
    result, _, _ = make_benchmark_case(
        investor_returns,
        benchmark_returns,
        risk_free_rate_annual=annual_risk_free_rate
    )

    assert np.isclose(result["beta"], expected_beta)
    assert np.isclose(
        result["alpha_annualized"],
        expected_alpha_daily * 252
    )


def test_tracking_error_and_information_ratio_known_case():
    benchmark_returns = np.array([0.01, 0.00, -0.01, 0.02])
    active_returns = np.array([0.01, -0.01, 0.02, 0.00])
    investor_returns = benchmark_returns + active_returns
    result, _, _ = make_benchmark_case(
        investor_returns,
        benchmark_returns
    )
    active_standard_deviation = active_returns.std(ddof=1)

    assert np.isclose(
        result["tracking_error"],
        active_standard_deviation * np.sqrt(252)
    )
    assert np.isclose(
        result["information_ratio"],
        active_returns.mean()
        / active_standard_deviation
        * np.sqrt(252)
    )


def test_constant_active_return_has_zero_tracking_error_and_no_ratio():
    benchmark_returns = np.array([0.01, -0.02, 0.015, 0.005])
    investor_returns = benchmark_returns + 0.001
    result, _, _ = make_benchmark_case(
        investor_returns,
        benchmark_returns
    )

    assert np.isclose(result["tracking_error"], 0.0)
    assert np.isnan(result["information_ratio"])


def test_constant_benchmark_return_makes_beta_and_alpha_unavailable():
    result, _, _ = make_benchmark_case(
        [0.01, 0.02, 0.03],
        [0.005, 0.005, 0.005]
    )

    assert np.isnan(result["beta"])
    assert np.isnan(result["alpha_annualized"])


@pytest.mark.parametrize("annual_rate", [0.03, -0.01])
def test_positive_and_negative_risk_free_rates_are_supported(annual_rate):
    result, _, _ = make_benchmark_case(
        [0.01, -0.005, 0.02],
        [0.008, -0.004, 0.015],
        risk_free_rate_annual=annual_rate
    )

    assert np.isfinite(result["alpha_annualized"])


def test_missing_benchmark_dates_reduce_alignment_without_filling():
    result, daily_twr, benchmark_prices = make_benchmark_case(
        [0.01, 0.02, 0.03, 0.04],
        [0.005, 0.006, 0.007, 0.008]
    )
    missing_prices = benchmark_prices.copy()
    missing_prices.iloc[2] = np.nan

    missing_result = run_benchmark_analysis(
        daily_twr,
        missing_prices,
        benchmark_ticker="SPY"
    )

    assert result["observations"] == 4
    assert missing_result["observations"] == 2
    assert missing_result["investor_observations_excluded"] == 2
    assert missing_result["aligned_returns"].index.tolist() == [
        benchmark_prices.index[1],
        benchmark_prices.index[4]
    ]


def test_inactive_investor_dates_are_excluded_from_exact_alignment():
    result, daily_twr, benchmark_prices = make_benchmark_case(
        [0.01, np.nan, 0.02, np.nan],
        [0.005, 0.006, 0.007, 0.008]
    )

    assert result["observations"] == 2
    assert result["performance_observations"] == 2
    assert result["aligned_returns"].index.equals(
        daily_twr.dropna().index
    )
    assert result["benchmark_returns"].index.equals(benchmark_prices.index)


def test_cumulative_paths_use_only_the_same_aligned_dates():
    result, daily_twr, benchmark_prices = make_benchmark_case(
        [0.10, np.nan, -0.05],
        [0.02, 0.03, -0.01]
    )
    expected_dates = daily_twr.dropna().index

    assert result["investor_cumulative_aligned"].index.equals(expected_dates)
    assert result["benchmark_cumulative_return"].index.equals(expected_dates)
    assert np.allclose(
        result["investor_cumulative_aligned"],
        [0.10, 0.10 * 0.95 + 0.95 - 1.0]
    )
    expected_benchmark = (
        (1.0 + benchmark_prices.pct_change(fill_method=None).loc[expected_dates])
        .cumprod()
        - 1.0
    )
    assert np.allclose(
        result["benchmark_cumulative_return"],
        expected_benchmark
    )


def test_benchmark_annualization_matches_active_observation_convention():
    benchmark_returns = np.array([0.01, -0.02, 0.03])
    result, _, _ = make_benchmark_case(
        [0.0, 0.0, 0.0],
        benchmark_returns
    )
    linked_wealth = np.prod(1.0 + benchmark_returns)

    assert np.isclose(
        result["benchmark_total_return"],
        linked_wealth - 1.0
    )
    assert np.isclose(
        result["benchmark_annualized_return"],
        linked_wealth ** (252 / 3) - 1.0
    )


def test_one_aligned_observation_keeps_return_but_ratios_are_unavailable():
    result, _, _ = make_benchmark_case([0.01], [0.02])

    assert result["observations"] == 1
    assert np.isclose(result["benchmark_total_return"], 0.02)
    for key in (
        "beta",
        "alpha_annualized",
        "tracking_error",
        "information_ratio"
    ):
        assert np.isnan(result[key])


def test_non_finite_inputs_are_rejected_and_inputs_are_not_mutated():
    result, daily_twr, benchmark_prices = make_benchmark_case(
        [0.01, -0.01, 0.02],
        [0.005, -0.004, 0.01]
    )
    original_twr = daily_twr.copy(deep=True)
    original_prices = benchmark_prices.copy(deep=True)

    run_benchmark_analysis(daily_twr, benchmark_prices, benchmark_ticker="SPY")

    pd.testing.assert_series_equal(daily_twr, original_twr)
    pd.testing.assert_series_equal(benchmark_prices, original_prices)
    invalid_twr = daily_twr.copy()
    invalid_twr.iloc[0] = np.inf
    with pytest.raises(ValueError, match="finite values"):
        run_benchmark_analysis(
            invalid_twr,
            benchmark_prices,
            benchmark_ticker="SPY"
        )
    assert result["price_basis"] == "adjusted_close_total_return_style"
