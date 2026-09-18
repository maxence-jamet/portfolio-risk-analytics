import numpy as np
import pandas as pd
import pytest

from src.account_risk import (
    build_account_risk_results,
    calculate_account_allocation
)
from src.backtesting import backtest_historical_var, backtest_var_threshold
from src.risk import (
    annualized_volatility,
    calculate_drawdown,
    distribution_statistics,
    ewma_parametric_var_threshold,
    expected_shortfall,
    historical_var,
    parametric_var,
    worst_historical_day
)


def make_security_risk_results(security_market_value):
    dates = pd.bdate_range("2023-01-02", periods=321)
    observations = np.arange(len(dates))
    security_returns = pd.Series(
        np.where(observations % 17 == 0, -0.04, 0.002),
        index=dates,
        name="security_return"
    )
    sleeve_volatility = annualized_volatility(security_returns)
    stress_returns = pd.Series(
        [-0.10, 0.05],
        index=["Sell-off", "Rally"]
    )
    stress_results = pd.DataFrame({
        "Portfolio_return": stress_returns,
        "P&L": stress_returns * security_market_value
    })
    risk_contribution_table = pd.DataFrame(
        {
            "Poids": [0.60, 0.40],
            "Contribution_volatilite": [
                0.60 * sleeve_volatility,
                0.40 * sleeve_volatility
            ],
            "Contribution_risque_pct": [0.60, 0.40]
        },
        index=["AAA", "BBB"]
    )
    historical_backtest = backtest_historical_var(
        security_returns,
        window=252,
        confidence_level=0.95
    )[0]
    ewma_threshold = ewma_parametric_var_threshold(
        security_returns,
        confidence_level=0.95,
        decay_factor=0.94,
        min_periods=30
    )[0]
    ewma_backtest = backtest_var_threshold(
        security_returns,
        ewma_threshold,
        expected_breach_rate=0.05
    )[0]

    return {
        "portfolio_returns": security_returns,
        "annual_volatility": sleeve_volatility,
        "stress_results": stress_results,
        "risk_contribution_table": risk_contribution_table,
        "historical_backtest": historical_backtest,
        "ewma_backtest": ewma_backtest
    }


def test_no_cash_account_risk_equals_security_sleeve_risk():
    security_risk = make_security_risk_results(100.0)
    account_risk = build_account_risk_results(
        security_risk,
        security_market_value=100.0,
        cash_balance=0.0,
        total_account_value=100.0
    )
    sleeve_var, sleeve_quantile = historical_var(
        security_risk["portfolio_returns"],
        confidence_level=0.95
    )

    pd.testing.assert_series_equal(
        account_risk["portfolio_returns"],
        security_risk["portfolio_returns"].rename("account_return")
    )
    assert account_risk["invested_ratio"] == 1.0
    assert account_risk["cash_weight"] == 0.0
    assert np.isclose(
        account_risk["annual_volatility"],
        security_risk["annual_volatility"]
    )
    assert np.isclose(account_risk["historical_var_95"], sleeve_var)
    assert np.isclose(
        account_risk["expected_shortfall_95"],
        expected_shortfall(
            security_risk["portfolio_returns"],
            sleeve_quantile
        )
    )


def test_material_cash_scales_returns_and_linear_risk_metrics():
    security_risk = make_security_risk_results(75.0)
    account_risk = build_account_risk_results(
        security_risk,
        security_market_value=75.0,
        cash_balance=25.0,
        total_account_value=100.0
    )
    sleeve_returns = security_risk["portfolio_returns"]
    sleeve_var, sleeve_quantile = historical_var(
        sleeve_returns,
        confidence_level=0.95
    )
    sleeve_es = expected_shortfall(sleeve_returns, sleeve_quantile)

    assert account_risk["invested_ratio"] == 0.75
    assert account_risk["cash_weight"] == 0.25
    pd.testing.assert_series_equal(
        account_risk["portfolio_returns"],
        (0.75 * sleeve_returns).rename("account_return")
    )
    assert np.isclose(
        account_risk["annual_volatility"],
        0.75 * security_risk["annual_volatility"]
    )
    assert np.isclose(account_risk["historical_var_95"], 0.75 * sleeve_var)
    assert np.isclose(account_risk["expected_shortfall_95"], 0.75 * sleeve_es)

    sleeve_var_99, sleeve_quantile_99 = historical_var(
        sleeve_returns,
        confidence_level=0.99
    )
    assert np.isclose(
        account_risk["historical_var_99"],
        0.75 * sleeve_var_99
    )
    assert np.isclose(
        account_risk["expected_shortfall_99"],
        0.75 * expected_shortfall(sleeve_returns, sleeve_quantile_99)
    )
    for confidence_level in (0.95, 0.99):
        assert np.isclose(
            account_risk[f"parametric_var_{int(confidence_level * 100)}"],
            0.75 * parametric_var(sleeve_returns, confidence_level)
        )

    sleeve_skewness, sleeve_kurtosis = distribution_statistics(sleeve_returns)
    assert np.isclose(account_risk["skewness"], sleeve_skewness)
    assert np.isclose(account_risk["excess_kurtosis"], sleeve_kurtosis)
    sleeve_worst_date, sleeve_worst_return = worst_historical_day(sleeve_returns)
    assert account_risk["worst_date"] == sleeve_worst_date
    assert np.isclose(account_risk["worst_return"], 0.75 * sleeve_worst_return)

    sleeve_ewma = ewma_parametric_var_threshold(sleeve_returns)
    for account_series, sleeve_series in zip(
        (
            account_risk["ewma_var_threshold_95"],
            account_risk["ewma_daily_volatility"],
            account_risk["ewma_annualized_volatility"]
        ),
        sleeve_ewma
    ):
        pd.testing.assert_series_equal(
            account_series,
            0.75 * sleeve_series,
            check_names=False
        )


def test_account_drawdown_stress_and_contributions_reconcile_without_mutation():
    security_risk = make_security_risk_results(75.0)
    original_returns = security_risk["portfolio_returns"].copy(deep=True)
    original_stress = security_risk["stress_results"].copy(deep=True)
    original_contributions = security_risk[
        "risk_contribution_table"
    ].copy(deep=True)

    account_risk = build_account_risk_results(
        security_risk,
        security_market_value=75.0,
        cash_balance=25.0,
        total_account_value=100.0
    )
    expected_drawdown = calculate_drawdown(0.75 * original_returns)[2]
    sleeve_drawdown = calculate_drawdown(original_returns)[2]

    assert np.isclose(account_risk["max_drawdown"], expected_drawdown)
    assert not np.isclose(account_risk["max_drawdown"], 0.75 * sleeve_drawdown)
    pd.testing.assert_series_equal(
        account_risk["stress_results"]["P&L"],
        original_stress["P&L"]
    )
    assert np.allclose(
        account_risk["stress_results"]["Portfolio_return"],
        original_stress["P&L"] / 100.0
    )
    contributions = account_risk["risk_contribution_table"]
    assert contributions.loc["Cash", "Poids"] == 0.25
    assert contributions.loc["Cash", "Contribution_volatilite"] == 0.0
    assert contributions.loc["Cash", "Contribution_risque_pct"] == 0.0
    assert np.isclose(
        contributions["Contribution_volatilite"].sum(),
        account_risk["annual_volatility"]
    )
    assert np.isclose(contributions["Contribution_risque_pct"].sum(), 1.0)
    assert account_risk["backtest_scale_invariant"] is True
    for backtest_name in ("historical_backtest", "ewma_backtest"):
        backtest = security_risk[backtest_name]
        scaled_breaches = (
            0.75 * backtest["Return"] < 0.75 * backtest["VaR"]
        )
        pd.testing.assert_series_equal(
            scaled_breaches,
            backtest["Breach"],
            check_names=False
        )

    pd.testing.assert_series_equal(
        security_risk["portfolio_returns"],
        original_returns
    )
    pd.testing.assert_frame_equal(security_risk["stress_results"], original_stress)
    pd.testing.assert_frame_equal(
        security_risk["risk_contribution_table"],
        original_contributions
    )


def test_account_allocation_rejects_zero_total_value():
    with pytest.raises(ValueError, match="total account value must be positive"):
        calculate_account_allocation(
            security_market_value=0.0,
            cash_balance=0.0,
            total_account_value=0.0
        )
