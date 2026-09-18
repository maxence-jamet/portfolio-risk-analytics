import numpy as np
import pandas as pd

from src.risk import (
    annualized_volatility,
    calculate_drawdown,
    distribution_statistics,
    ewma_parametric_var_threshold,
    expected_shortfall,
    historical_var,
    parametric_var,
    rolling_volatility,
    worst_historical_day
)


RATIO_TOLERANCE = 1e-10


def calculate_account_allocation(
    security_market_value,
    cash_balance,
    total_account_value,
    tolerance=RATIO_TOLERANCE
):
    """Return today's invested and cash proportions for account risk."""
    values = np.asarray(
        [security_market_value, cash_balance, total_account_value],
        dtype=float
    )
    if not np.isfinite(values).all():
        raise ValueError("Current account allocation values must be finite.")

    security_market_value = float(security_market_value)
    cash_balance = float(cash_balance)
    total_account_value = float(total_account_value)

    if total_account_value <= tolerance:
        raise ValueError(
            "Current total account value must be positive to calculate "
            "account-risk allocation ratios."
        )
    if security_market_value < -tolerance or cash_balance < -tolerance:
        raise ValueError(
            "Account risk requires non-negative security value and cash."
        )
    if not np.isclose(
        security_market_value + cash_balance,
        total_account_value,
        rtol=tolerance,
        atol=tolerance
    ):
        raise ValueError(
            "Current security value plus cash must equal total account value."
        )

    invested_ratio = security_market_value / total_account_value
    cash_weight = cash_balance / total_account_value
    if not np.isclose(
        invested_ratio + cash_weight,
        1.0,
        rtol=tolerance,
        atol=tolerance
    ):
        raise ValueError(
            "Invested ratio and cash weight must reconcile to 1."
        )

    if np.isclose(invested_ratio, 1.0, rtol=tolerance, atol=tolerance):
        invested_ratio = 1.0
    if np.isclose(cash_weight, 0.0, rtol=tolerance, atol=tolerance):
        cash_weight = 0.0

    return invested_ratio, cash_weight


def _build_account_stress_results(
    security_stress_results,
    invested_ratio,
    security_market_value,
    total_account_value
):
    """Convert sleeve stress percentages to a total-account denominator."""
    account_stress_results = security_stress_results.copy()
    account_stress_results["Portfolio_return"] = (
        account_stress_results["Portfolio_return"] * invested_ratio
    )

    account_dollar_impact = account_stress_results["P&L"]
    dollar_from_account_percentage = (
        account_stress_results["Portfolio_return"]
        * total_account_value
    )
    dollar_from_security_percentage = (
        security_stress_results["Portfolio_return"]
        * security_market_value
    )
    if not (
        np.allclose(account_dollar_impact, dollar_from_account_percentage)
        and np.allclose(account_dollar_impact, dollar_from_security_percentage)
    ):
        raise ValueError(
            "Account and invested-security stress impacts do not reconcile."
        )

    return account_stress_results


def _build_account_risk_contributions(
    security_contributions,
    invested_ratio,
    cash_weight,
    account_volatility
):
    """Scale absolute sleeve contributions and add zero-risk cash."""
    account_contributions = security_contributions.copy()
    account_contributions["Poids"] = (
        account_contributions["Poids"] * invested_ratio
    )
    account_contributions["Contribution_volatilite"] = (
        account_contributions["Contribution_volatilite"] * invested_ratio
    )
    account_contributions.loc["Cash"] = {
        "Poids": cash_weight,
        "Contribution_volatilite": 0.0,
        "Contribution_risque_pct": 0.0
    }

    if not np.isclose(
        account_contributions["Contribution_volatilite"].sum(),
        account_volatility
    ):
        raise ValueError(
            "Account component risk contributions must sum to account "
            "volatility."
        )

    return account_contributions


def _backtest_breaches_are_scale_invariant(backtest, invested_ratio):
    """Confirm strict VaR breaches survive a positive constant scaling."""
    scaled_breaches = (
        backtest["Return"] * invested_ratio
        < backtest["VaR"] * invested_ratio
    )
    return scaled_breaches.equals(backtest["Breach"])


def build_account_risk_results(
    security_risk_results,
    security_market_value,
    cash_balance,
    total_account_value
):
    """Build risk for today's total account with zero-return current cash."""
    invested_ratio, cash_weight = calculate_account_allocation(
        security_market_value,
        cash_balance,
        total_account_value
    )
    if invested_ratio <= 0:
        raise ValueError(
            "Account risk requires at least one currently invested security."
        )

    account_returns = (
        security_risk_results["portfolio_returns"]
        .mul(invested_ratio)
        .rename("account_return")
    )
    account_annual_volatility = annualized_volatility(account_returns)
    (
        cumulative_performance,
        drawdown,
        max_drawdown
    ) = calculate_drawdown(account_returns)

    var_95, quantile_95 = historical_var(
        account_returns,
        confidence_level=0.95
    )
    var_99, quantile_99 = historical_var(
        account_returns,
        confidence_level=0.99
    )
    es_95 = expected_shortfall(account_returns, quantile_95)
    es_99 = expected_shortfall(account_returns, quantile_99)
    parametric_var_95 = parametric_var(
        account_returns,
        confidence_level=0.95
    )
    parametric_var_99 = parametric_var(
        account_returns,
        confidence_level=0.99
    )
    skewness, excess_kurtosis = distribution_statistics(account_returns)
    worst_date, worst_return = worst_historical_day(account_returns)
    rolling_volatility_series = rolling_volatility(
        account_returns,
        window=252
    )
    (
        ewma_var_threshold_95,
        ewma_daily_volatility,
        ewma_annualized_volatility
    ) = ewma_parametric_var_threshold(
        account_returns,
        confidence_level=0.95,
        decay_factor=0.94,
        min_periods=30
    )

    account_stress_results = _build_account_stress_results(
        security_risk_results["stress_results"],
        invested_ratio,
        security_market_value,
        total_account_value
    )
    account_contributions = _build_account_risk_contributions(
        security_risk_results["risk_contribution_table"],
        invested_ratio,
        cash_weight,
        account_annual_volatility
    )

    backtest_scale_invariant = all(
        _backtest_breaches_are_scale_invariant(
            security_risk_results[backtest_key],
            invested_ratio
        )
        for backtest_key in ("historical_backtest", "ewma_backtest")
    )
    if not backtest_scale_invariant:
        raise ValueError(
            "Account and invested-security VaR breach sequences must match "
            "under positive constant scaling."
        )

    return {
        "scope": "total_account_current_allocation",
        "cash_return_assumption": 0.0,
        "portfolio_value": float(total_account_value),
        "security_market_value": float(security_market_value),
        "cash_balance": float(cash_balance),
        "invested_ratio": invested_ratio,
        "cash_weight": cash_weight,
        "weights": account_contributions["Poids"].copy(),
        "portfolio_returns": account_returns,
        "annual_volatility": account_annual_volatility,
        "cumulative_performance": cumulative_performance,
        "drawdown": drawdown,
        "max_drawdown": max_drawdown,
        "historical_var_95": var_95,
        "historical_var_99": var_99,
        "historical_quantile_95": quantile_95,
        "historical_quantile_99": quantile_99,
        "expected_shortfall_95": es_95,
        "expected_shortfall_99": es_99,
        "parametric_var_95": parametric_var_95,
        "parametric_var_99": parametric_var_99,
        "skewness": skewness,
        "excess_kurtosis": excess_kurtosis,
        "worst_date": worst_date,
        "worst_return": worst_return,
        "rolling_volatility": rolling_volatility_series,
        "ewma_var_threshold_95": ewma_var_threshold_95,
        "ewma_daily_volatility": ewma_daily_volatility,
        "ewma_annualized_volatility": ewma_annualized_volatility,
        "risk_contribution_table": account_contributions,
        "stress_results": account_stress_results,
        "backtest_scale_invariant": backtest_scale_invariant,
        "backtest_source": "invested_securities"
    }
