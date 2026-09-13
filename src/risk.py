import numpy as np
import pandas as pd
from scipy.stats import norm

from src.data import validate_stress_scenarios


def annualized_volatility(
    portfolio_returns,
    trading_days=252
):
    daily_volatility = portfolio_returns.std()

    annual_volatility = (
        daily_volatility * np.sqrt(trading_days)
    )

    return annual_volatility


def calculate_drawdown(portfolio_returns):

    cumulative_performance = (
        1 + portfolio_returns
    ).cumprod()

    running_max = cumulative_performance.cummax()

    drawdown = (
        cumulative_performance / running_max - 1
    )

    max_drawdown = drawdown.min()

    return (
        cumulative_performance,
        drawdown,
        max_drawdown
    )


def historical_var(
    portfolio_returns,
    confidence_level=0.95
):
    alpha = 1 - confidence_level

    quantile = portfolio_returns.quantile(alpha)

    var = -quantile

    return var, quantile


def expected_shortfall(
    portfolio_returns,
    var_threshold
):
    tail_returns = portfolio_returns[
        portfolio_returns <= var_threshold
    ]

    es = -tail_returns.mean()

    return es


def worst_historical_day(portfolio_returns):

    worst_return = portfolio_returns.min()
    worst_date = portfolio_returns.idxmin()

    return worst_date, worst_return

def annualized_covariance_matrix(
    returns,
    trading_days=252
):
    daily_covariance_matrix = returns.cov()

    annual_covariance_matrix = (
        daily_covariance_matrix * trading_days
    )

    return annual_covariance_matrix


def portfolio_volatility_from_covariance(
    covariance_matrix,
    weights
):
    portfolio_variance = (
        weights.T
        @ covariance_matrix
        @ weights
    )

    portfolio_volatility = np.sqrt(
        portfolio_variance
    )

    return portfolio_volatility


def calculate_risk_contributions(
    covariance_matrix,
    weights
):
    portfolio_volatility = (
        portfolio_volatility_from_covariance(
            covariance_matrix,
            weights
        )
    )

    marginal_risk = (
        covariance_matrix @ weights
    ) / portfolio_volatility

    component_risk = (
        weights * marginal_risk
    )

    risk_contribution_pct = (
        component_risk / portfolio_volatility
    )

    risk_contribution_table = pd.DataFrame({
        "Poids": weights,
        "Contribution_volatilite": component_risk,
        "Contribution_risque_pct": risk_contribution_pct
    })

    return risk_contribution_table

def parametric_var(
    portfolio_returns,
    confidence_level=0.95
):
    daily_mean = portfolio_returns.mean()
    daily_std = portfolio_returns.std()

    alpha = 1 - confidence_level

    z_score = norm.ppf(alpha)

    parametric_quantile = (
        daily_mean + z_score * daily_std
    )

    parametric_var = -parametric_quantile

    return parametric_var


def calculate_stress_test(
    stress_scenarios,
    weights,
    portfolio_value
):
    stress_scenarios = validate_stress_scenarios(
        stress_scenarios,
        required_assets=weights.index
    )

    stress_scenarios = stress_scenarios.reindex(
        columns=weights.index
    )

    stress_portfolio_returns = (
        stress_scenarios
        .mul(weights, axis=1)
        .sum(axis=1)
    )

    stress_pnl = (
        stress_portfolio_returns
        * portfolio_value
    )

    stress_results = pd.DataFrame({
        "Portfolio_return": stress_portfolio_returns,
        "P&L": stress_pnl
    })

    return stress_results


def rolling_volatility(
    portfolio_returns,
    window=252,
    trading_days=252
):
    rolling_vol = (
        portfolio_returns
        .rolling(window=window)
        .std()
        * np.sqrt(trading_days)
    )

    return rolling_vol

def distribution_statistics(
    portfolio_returns
):
    skewness = portfolio_returns.skew()

    excess_kurtosis = portfolio_returns.kurt()

    return skewness, excess_kurtosis

def ewma_volatility(
    portfolio_returns,
    decay_factor=0.94,
    min_periods=30,
    trading_days=252
):
    if not 0 < decay_factor < 1:
        raise ValueError(
            "decay_factor must be between 0 and 1."
        )

    squared_returns = (
        portfolio_returns
        .shift(1)
        .pow(2)
    )

    ewma_variance = (
        squared_returns
        .ewm(
            alpha=1 - decay_factor,
            adjust=False,
            min_periods=min_periods
        )
        .mean()
    )

    ewma_daily_volatility = np.sqrt(
        ewma_variance
    )

    ewma_annualized_volatility = (
        ewma_daily_volatility
        * np.sqrt(trading_days)
    )

    return (
        ewma_daily_volatility,
        ewma_annualized_volatility
    )


def ewma_parametric_var_threshold(
    portfolio_returns,
    confidence_level=0.95,
    decay_factor=0.94,
    min_periods=30
):
    (
        ewma_daily_volatility,
        ewma_annualized_volatility
    ) = ewma_volatility(
        portfolio_returns,
        decay_factor=decay_factor,
        min_periods=min_periods
    )

    alpha = 1 - confidence_level

    z_score = norm.ppf(alpha)

    var_threshold = (
        z_score
        * ewma_daily_volatility
    )

    return (
        var_threshold,
        ewma_daily_volatility,
        ewma_annualized_volatility
    )
