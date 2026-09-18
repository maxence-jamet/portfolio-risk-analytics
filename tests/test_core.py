import numpy as np
import pandas as pd
import pytest

from src.data import (
    calculate_returns,
    calculate_returns_with_metadata,
    resolve_risk_window
)
from src.portfolio import (
    calculate_portfolio_returns,
    validate_portfolio
)

from src.risk import (
    calculate_risk_contributions,
    calculate_stress_test,
    ewma_parametric_var_threshold,
    historical_var,
    expected_shortfall,
    portfolio_volatility_from_covariance
)
from src.backtesting import (
    backtest_historical_var,
    christoffersen_conditional_coverage_test,
    christoffersen_independence_test,
    evaluate_var_backtest,
    kupiec_test,
    required_backtest_observations
)


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


def test_risk_window_presets_use_supplied_reference_date():
    reference_date = pd.Timestamp("2025-08-29")
    earliest_date = pd.Timestamp("2018-01-03")

    assert resolve_risk_window(
        "1 year", reference_date, earliest_date
    ) == pd.Timestamp("2024-08-29")
    assert resolve_risk_window(
        "3 years", reference_date, earliest_date
    ) == pd.Timestamp("2022-08-29")
    assert resolve_risk_window(
        "Maximum available", reference_date, earliest_date
    ) == earliest_date


def test_return_alignment_metadata_counts_missing_dates_without_zero_fill():
    dates = pd.bdate_range("2024-01-02", periods=6)
    prices = pd.DataFrame(
        {
            "AAA": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "BBB": [50.0, 51.0, np.nan, 53.0, 54.0, 55.0]
        },
        index=dates
    )
    original = prices.copy(deep=True)

    returns, metadata = calculate_returns_with_metadata(prices)

    assert returns.index.tolist() == [dates[1], dates[4], dates[5]]
    assert not (returns == 0.0).any().any()
    assert metadata == {
        "raw_price_observations": 6,
        "asset_return_observations_before_alignment": 5,
        "aligned_portfolio_return_observations": 3,
        "observations_dropped_during_alignment": 2,
        "usable_return_observations_by_asset": {"AAA": 5, "BBB": 3}
    }
    pd.testing.assert_frame_equal(prices, original)


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


def test_risk_contributions_reconcile_with_negative_covariance():
    covariance_matrix = pd.DataFrame(
        [[0.04, -0.015], [-0.015, 0.01]],
        index=["A", "B"],
        columns=["A", "B"]
    )
    weights = pd.Series({"A": 0.60, "B": 0.40})

    contributions = calculate_risk_contributions(
        covariance_matrix,
        weights
    )
    portfolio_volatility = portfolio_volatility_from_covariance(
        covariance_matrix,
        weights
    )

    assert np.isclose(
        contributions["Contribution_volatilite"].sum(),
        portfolio_volatility
    )
    assert np.isclose(
        contributions["Contribution_risque_pct"].sum(),
        1.0
    )
    assert contributions.loc["B", "Contribution_risque_pct"] < 0.0


def test_risk_contributions_reject_zero_volatility():
    covariance_matrix = pd.DataFrame(
        np.zeros((2, 2)),
        index=["A", "B"],
        columns=["A", "B"]
    )
    weights = pd.Series({"A": 0.50, "B": 0.50})

    with pytest.raises(ValueError, match="zero or numerically near zero"):
        calculate_risk_contributions(covariance_matrix, weights)


def test_stress_test_uses_security_weights_and_value_consistently():
    scenarios = pd.DataFrame(
        {"A": [-0.10], "B": [0.05]},
        index=["Mixed shock"]
    )
    weights = pd.Series({"A": 0.60, "B": 0.40})

    results = calculate_stress_test(
        scenarios,
        weights,
        portfolio_value=1000.0
    )

    assert np.isclose(results.loc["Mixed shock", "Portfolio_return"], -0.04)
    assert np.isclose(results.loc["Mixed shock", "P&L"], -40.0)


def test_historical_var_backtest_uses_only_prior_returns():
    dates = pd.bdate_range("2024-01-02", periods=6)
    returns = pd.Series(
        [-0.01, 0.00, 0.01, -0.50, 0.02, 0.03],
        index=dates
    )

    backtest, _, _, _ = backtest_historical_var(
        returns,
        window=3,
        confidence_level=0.95
    )

    expected_first_threshold = returns.iloc[:3].quantile(0.05)
    assert np.isclose(backtest.iloc[0]["VaR"], expected_first_threshold)
    assert bool(backtest.iloc[0]["Breach"])

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
        expected_breach_rate=0.25,
        base_minimum_observations=2,
        minimum_expected_breaches=1
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


@pytest.mark.parametrize(
    ("confidence_level", "expected_observations"),
    [(0.95, 250), (0.99, 1000)]
)
def test_required_backtest_observations_project_policy(
    confidence_level,
    expected_observations
):
    assert required_backtest_observations(confidence_level) == (
        expected_observations
    )


@pytest.mark.parametrize("confidence_level", [0.0, 1.0, -0.01, 1.01])
def test_required_backtest_observations_rejects_invalid_confidence(
    confidence_level
):
    with pytest.raises(ValueError, match="confidence_level"):
        required_backtest_observations(confidence_level)


@pytest.mark.parametrize(
    ("threshold_name", "threshold_value"),
    [
        ("base_minimum_observations", 0),
        ("base_minimum_observations", 250.0),
        ("minimum_expected_breaches", -1),
        ("minimum_expected_breaches", True)
    ]
)
def test_required_backtest_observations_rejects_invalid_thresholds(
    threshold_name,
    threshold_value
):
    arguments = {threshold_name: threshold_value}

    with pytest.raises(ValueError, match=threshold_name):
        required_backtest_observations(0.95, **arguments)


@pytest.mark.parametrize(
    ("observations", "expected_breach_rate", "is_sufficient"),
    [
        (249, 0.05, False),
        (250, 0.05, True),
        (999, 0.01, False),
        (1000, 0.01, True)
    ]
)
def test_model_validation_enforces_dynamic_sample_policy(
    observations,
    expected_breach_rate,
    is_sufficient
):
    backtest = pd.DataFrame({
        "Breach": np.zeros(observations, dtype=bool)
    })

    if not is_sufficient:
        with pytest.raises(
            ValueError,
            match=f"received {observations}"
        ):
            evaluate_var_backtest(backtest, expected_breach_rate)
        return

    result = evaluate_var_backtest(backtest, expected_breach_rate)
    assert result["model_validation_available"] is True
    assert result["validation_observations"] == observations
    assert result["minimum_required_observations"] == observations


def test_low_level_lr_tests_remain_reusable_on_small_known_sample():
    backtest = pd.DataFrame({
        "Breach": [False, True, False, False]
    })
    kupiec_statistic, kupiec_p_value = kupiec_test(1, 4, 0.25)
    independence = christoffersen_independence_test(backtest)
    conditional_statistic, conditional_p_value = (
        christoffersen_conditional_coverage_test(
            kupiec_statistic,
            independence[0]
        )
    )

    assert np.isclose(kupiec_statistic, 0.0)
    assert np.isfinite(kupiec_p_value)
    assert np.isfinite(independence[0])
    assert np.isfinite(independence[1])
    assert np.isfinite(conditional_statistic)
    assert np.isfinite(conditional_p_value)


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
