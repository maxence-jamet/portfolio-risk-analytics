import pandas as pd

from src.data import (
    calculate_returns,
    download_prices,
    validate_price_data,
    validate_price_history,
    validate_stress_scenarios
)
from src.portfolio import (
    calculate_portfolio_positions,
    calculate_portfolio_returns,
    validate_portfolio
)
from src.risk import (
    annualized_covariance_matrix,
    annualized_volatility,
    calculate_drawdown,
    calculate_risk_contributions,
    calculate_stress_test,
    distribution_statistics,
    ewma_parametric_var_threshold,
    expected_shortfall,
    historical_var,
    parametric_var,
    portfolio_volatility_from_covariance,
    rolling_volatility,
    worst_historical_day
)
from src.backtesting import (
    backtest_historical_var,
    backtest_var_threshold,
    christoffersen_independence_test,
    evaluate_var_backtest
)


def run_portfolio_analysis(
    portfolio,
    stress_scenarios,
    start_date="2022-01-01",
    prices=None
):
    """Run the portfolio calculations and return reusable analysis results.

    Supplying prices skips the market-data download. This keeps automated
    tests deterministic and lets future callers provide their own data.
    """
    portfolio = validate_portfolio(portfolio)
    portfolio_tickers = portfolio["ticker"].tolist()

    stress_scenarios = validate_stress_scenarios(
        stress_scenarios,
        required_assets=portfolio_tickers
    )

    if prices is None:
        prices = download_prices(
            portfolio_tickers,
            start_date
        )

    prices = validate_price_data(
        prices,
        required_assets=portfolio_tickers
    )

    returns = calculate_returns(prices)
    validate_price_history(
        returns,
        window=252,
        minimum_out_of_sample=2
    )

    positions, weights, portfolio_value = (
        calculate_portfolio_positions(
            portfolio,
            prices
        )
    )

    portfolio_returns, aligned_weights = (
        calculate_portfolio_returns(
            returns,
            weights
        )
    )

    annual_volatility = annualized_volatility(
        portfolio_returns
    )

    correlation_matrix = returns.corr()

    (
        cumulative_performance,
        drawdown,
        max_drawdown
    ) = calculate_drawdown(
        portfolio_returns
    )

    var_95, quantile_95 = historical_var(
        portfolio_returns,
        confidence_level=0.95
    )

    var_99, quantile_99 = historical_var(
        portfolio_returns,
        confidence_level=0.99
    )

    es_95 = expected_shortfall(
        portfolio_returns,
        quantile_95
    )

    es_99 = expected_shortfall(
        portfolio_returns,
        quantile_99
    )

    skewness, excess_kurtosis = (
        distribution_statistics(
            portfolio_returns
        )
    )

    worst_date, worst_return = (
        worst_historical_day(
            portfolio_returns
        )
    )

    covariance_matrix = annualized_covariance_matrix(
        returns
    )

    portfolio_volatility_matrix = (
        portfolio_volatility_from_covariance(
            covariance_matrix,
            aligned_weights
        )
    )

    risk_contribution_table = (
        calculate_risk_contributions(
            covariance_matrix,
            aligned_weights
        )
    )

    parametric_var_95 = parametric_var(
        portfolio_returns,
        confidence_level=0.95
    )

    parametric_var_99 = parametric_var(
        portfolio_returns,
        confidence_level=0.99
    )

    stress_results = calculate_stress_test(
        stress_scenarios,
        aligned_weights,
        portfolio_value
    )

    rolling_volatility_series = rolling_volatility(
        portfolio_returns,
        window=252
    )

    (
        ewma_var_threshold_95,
        ewma_daily_volatility_series,
        ewma_annualized_volatility_series
    ) = ewma_parametric_var_threshold(
        portfolio_returns,
        confidence_level=0.95,
        decay_factor=0.94,
        min_periods=30
    )

    (
        historical_backtest,
        historical_n_breaches,
        historical_breach_rate,
        historical_expected_rate
    ) = backtest_historical_var(
        portfolio_returns,
        window=252,
        confidence_level=0.95
    )

    historical_validation = evaluate_var_backtest(
        historical_backtest,
        historical_expected_rate
    )

    (
        _,
        _,
        n00,
        n01,
        n10,
        n11
    ) = christoffersen_independence_test(
        historical_backtest
    )

    historical_validation["transition_counts"] = {
        "n00": int(n00),
        "n01": int(n01),
        "n10": int(n10),
        "n11": int(n11)
    }

    (
        ewma_backtest,
        ewma_n_breaches,
        ewma_breach_rate,
        ewma_expected_rate
    ) = backtest_var_threshold(
        portfolio_returns,
        ewma_var_threshold_95,
        expected_breach_rate=0.05
    )

    ewma_validation = evaluate_var_backtest(
        ewma_backtest,
        ewma_expected_rate
    )

    common_backtest_dates = (
        historical_backtest.index
        .intersection(ewma_backtest.index)
    )

    historical_common_backtest = (
        historical_backtest.loc[common_backtest_dates]
    )

    ewma_common_backtest = (
        ewma_backtest.loc[common_backtest_dates]
    )

    historical_common_results = evaluate_var_backtest(
        historical_common_backtest,
        expected_breach_rate=0.05
    )

    ewma_common_results = evaluate_var_backtest(
        ewma_common_backtest,
        expected_breach_rate=0.05
    )

    model_comparison = pd.DataFrame(
        [
            {
                "model": "Historical VaR 95%",
                **historical_common_results
            },
            {
                "model": "EWMA VaR 95%",
                **ewma_common_results
            }
        ]
    )

    return {
        "prices": prices,
        "returns": returns,
        "positions": positions,
        "portfolio_value": portfolio_value,
        "weights": weights,
        "aligned_weights": aligned_weights,
        "portfolio_returns": portfolio_returns,
        "correlation_matrix": correlation_matrix,
        "annual_volatility": annual_volatility,
        "cumulative_performance": cumulative_performance,
        "drawdown": drawdown,
        "max_drawdown": max_drawdown,
        "historical_var_95": var_95,
        "historical_var_99": var_99,
        "historical_quantile_95": quantile_95,
        "historical_quantile_99": quantile_99,
        "expected_shortfall_95": es_95,
        "expected_shortfall_99": es_99,
        "skewness": skewness,
        "excess_kurtosis": excess_kurtosis,
        "worst_date": worst_date,
        "worst_return": worst_return,
        "annual_covariance_matrix": covariance_matrix,
        "portfolio_volatility_matrix": portfolio_volatility_matrix,
        "risk_contribution_table": risk_contribution_table,
        "parametric_var_95": parametric_var_95,
        "parametric_var_99": parametric_var_99,
        "stress_results": stress_results,
        "rolling_volatility": rolling_volatility_series,
        "ewma_var_threshold_95": ewma_var_threshold_95,
        "ewma_daily_volatility": ewma_daily_volatility_series,
        "ewma_annualized_volatility": ewma_annualized_volatility_series,
        "historical_backtest": historical_backtest,
        "historical_validation": historical_validation,
        "historical_n_breaches": historical_n_breaches,
        "historical_breach_rate": historical_breach_rate,
        "historical_expected_rate": historical_expected_rate,
        "ewma_backtest": ewma_backtest,
        "ewma_validation": ewma_validation,
        "ewma_n_breaches": ewma_n_breaches,
        "ewma_breach_rate": ewma_breach_rate,
        "ewma_expected_rate": ewma_expected_rate,
        "common_backtest_dates": common_backtest_dates,
        "historical_common_backtest": historical_common_backtest,
        "ewma_common_backtest": ewma_common_backtest,
        "model_comparison": model_comparison
    }
