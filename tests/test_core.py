import numpy as np
import pandas as pd
import pytest

from src.data import calculate_returns
from src.portfolio import (
    calculate_portfolio_returns,
    validate_portfolio
)

from src.risk import (
    ewma_parametric_var_threshold,
    historical_var,
    expected_shortfall,
    portfolio_volatility_from_covariance
)
from src.backtesting import evaluate_var_backtest, kupiec_test


def test_portfolio_returns_known_case():

    returns = pd.DataFrame({
        "A": [0.10, -0.10],
        "B": [0.00, 0.20]
    })

    weights = pd.Series({
        "A": 0.50,
        "B": 0.50
    })

    portfolio_returns, _ = (
        calculate_portfolio_returns(
            returns,
            weights
        )
    )

    expected_returns = np.array([
        0.05,
        0.05
    ])

    assert np.allclose(
        portfolio_returns.to_numpy(),
        expected_returns
    )


def test_missing_price_is_not_forward_filled_for_returns():
    dates = pd.bdate_range("2024-01-02", periods=4)
    prices = pd.DataFrame(
        {"AAA": [100.0, np.nan, 110.0, 121.0]},
        index=dates
    )

    returns = calculate_returns(prices)

    assert returns.index.tolist() == [dates[3]]
    assert np.isclose(returns.loc[dates[3], "AAA"], 0.10)


def test_validate_portfolio_does_not_mutate_caller_dataframe():
    portfolio = pd.DataFrame({
        "ticker": [" aaa "],
        "quantity": ["10"],
        "purchase_price": ["90.0"]
    })
    original_portfolio = portfolio.copy(deep=True)

    validated = validate_portfolio(portfolio)

    pd.testing.assert_frame_equal(portfolio, original_portfolio)
    assert validated.loc[0, "ticker"] == "AAA"
    assert validated.loc[0, "quantity"] == 10
    assert validated.loc[0, "purchase_price"] == 90.0


def test_weights_must_sum_to_one():

    returns = pd.DataFrame({
        "A": [0.01],
        "B": [0.02]
    })

    weights = pd.Series({
        "A": 0.40,
        "B": 0.40
    })

    with pytest.raises(
        ValueError,
        match="must sum to 1"
    ):
        calculate_portfolio_returns(
            returns,
            weights
        )


def test_expected_shortfall_is_not_below_var():

    returns = pd.Series([
        -0.10,
        -0.05,
        -0.02,
        0.00,
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
        0.06
    ])

    var_95, threshold_95 = (
        historical_var(
            returns,
            confidence_level=0.95
        )
    )

    es_95 = expected_shortfall(
        returns,
        threshold_95
    )

    assert es_95 >= var_95


def test_covariance_volatility():

    covariance_matrix = pd.DataFrame(
        [
            [0.04, 0.00],
            [0.00, 0.01]
        ],
        index=["A", "B"],
        columns=["A", "B"]
    )

    weights = pd.Series({
        "A": 0.50,
        "B": 0.50
    })

    volatility = (
        portfolio_volatility_from_covariance(
            covariance_matrix,
            weights
        )
    )

    expected_volatility = np.sqrt(
        0.50**2 * 0.04
        +
        0.50**2 * 0.01
    )

    assert np.isclose(
        volatility,
        expected_volatility
    )

def test_ewma_volatility_known_case():

    returns = pd.Series(
        [0.01, -0.01] * 20
    )

    (
        var_threshold,
        daily_volatility,
        annualized_volatility
    ) = ewma_parametric_var_threshold(
        returns,
        confidence_level=0.95,
        decay_factor=0.94,
        min_periods=10
    )

    last_daily_volatility = (
        daily_volatility
        .dropna()
        .iloc[-1]
    )

    assert np.isclose(
        last_daily_volatility,
        0.01
    )

    assert (
        var_threshold
        .dropna()
        .iloc[-1]
        < 0
    )


def test_evaluate_var_backtest_known_case():

    backtest = pd.DataFrame({
        "Breach": [False, True, False, False]
    })

    results = evaluate_var_backtest(
        backtest,
        expected_breach_rate=0.25
    )

    assert results["observations"] == 4
    assert results["breaches"] == 1
    assert np.isclose(
        results["breach_rate"],
        0.25
    )
    assert np.isclose(
        results["expected_breach_rate"],
        0.25
    )
    assert np.isclose(
        results["kupiec_lr_statistic"],
        0.0
    )


def test_kupiec_test_with_zero_breaches():
    statistic, p_value = kupiec_test(
        number_of_breaches=0,
        number_of_observations=100,
        expected_breach_rate=0.05
    )

    expected_statistic = -2 * 100 * np.log(0.95)

    assert np.isfinite(statistic)
    assert np.isfinite(p_value)
    assert np.isclose(statistic, expected_statistic)


def test_kupiec_test_with_all_observations_breached():
    statistic, p_value = kupiec_test(
        number_of_breaches=100,
        number_of_observations=100,
        expected_breach_rate=0.05
    )

    expected_statistic = -2 * 100 * np.log(0.05)

    assert np.isfinite(statistic)
    assert np.isfinite(p_value)
    assert np.isclose(statistic, expected_statistic)
