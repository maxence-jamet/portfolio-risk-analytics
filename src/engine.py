import pandas as pd

from src.data import (
    DEFAULT_MAX_PRICE_STALENESS_BUSINESS_DAYS,
    calculate_returns_with_metadata,
    download_price_views,
    validate_price_data,
    validate_stress_scenarios,
    validate_valuation_price_freshness
)
from src.portfolio import (
    calculate_portfolio_positions,
    calculate_portfolio_returns,
    validate_portfolio
)
from src.risk import (
    annualized_covariance_matrix,
    annualized_volatility,
    calculate_diversification_diagnostics,
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
    DEFAULT_BASE_MINIMUM_OBSERVATIONS,
    DEFAULT_MINIMUM_EXPECTED_BREACHES,
    backtest_historical_var,
    backtest_var_threshold,
    christoffersen_independence_test,
    evaluate_var_backtest,
    required_backtest_observations
)


def _model_validation_result(backtest, expected_breach_rate):
    """Return validation statistics or explicit sample-policy unavailability."""
    confidence_level = 1 - expected_breach_rate
    observations = len(backtest)
    minimum_required = required_backtest_observations(confidence_level)

    if observations >= minimum_required:
        return evaluate_var_backtest(backtest, expected_breach_rate)

    breaches = int(backtest["Breach"].sum())
    reason = (
        "VaR model validation requires at least "
        f"{minimum_required} out-of-sample observations at "
        f"{confidence_level:.0%} confidence under the current validation "
        f"policy; received {observations}."
    )
    return {
        "model_validation_available": False,
        "model_validation_reason": reason,
        "confidence_level": confidence_level,
        "observations": observations,
        "validation_observations": observations,
        "minimum_required_observations": minimum_required,
        "base_minimum_observations_policy": (
            DEFAULT_BASE_MINIMUM_OBSERVATIONS
        ),
        "minimum_expected_breaches_policy": (
            DEFAULT_MINIMUM_EXPECTED_BREACHES
        ),
        "breaches": breaches,
        "breach_rate": (
            breaches / observations if observations else None
        ),
        "expected_breach_rate": expected_breach_rate
    }


def _resolve_price_views(
    portfolio_tickers,
    start_date,
    prices,
    valuation_prices,
    return_prices
):
    """Resolve explicit price channels or the legacy shared-price input."""
    explicit_view_supplied = (
        valuation_prices is not None
        or return_prices is not None
    )
    if prices is not None and explicit_view_supplied:
        raise ValueError(
            "Use either legacy prices= or both valuation_prices= and "
            "return_prices=, not a mixture."
        )

    if prices is not None:
        shared_prices = validate_price_data(
            prices,
            required_assets=portfolio_tickers
        )
        return shared_prices.copy(), shared_prices.copy()

    if (valuation_prices is None) != (return_prices is None):
        raise ValueError(
            "valuation_prices and return_prices must be supplied together."
        )

    if valuation_prices is None:
        price_views = download_price_views(
            portfolio_tickers,
            start_date
        )
        valuation_prices = price_views["valuation_prices"]
        return_prices = price_views["return_prices"]

    validated_valuation_prices = validate_price_data(
        valuation_prices,
        required_assets=portfolio_tickers
    )
    validated_return_prices = validate_price_data(
        return_prices,
        required_assets=portfolio_tickers
    )

    return validated_valuation_prices, validated_return_prices


def _resolve_optional_stress_scenarios(stress_scenarios, portfolio_tickers):
    """Return validated scenarios or an explicit optional-unavailable state."""
    unavailable_reason = (
        "Stress testing unavailable: define shocks for all current holdings."
    )
    if stress_scenarios is None:
        return None, False, unavailable_reason
    if isinstance(stress_scenarios, pd.DataFrame):
        missing_assets = set(portfolio_tickers) - set(stress_scenarios.columns)
        if stress_scenarios.empty or missing_assets:
            return None, False, unavailable_reason

    validated = validate_stress_scenarios(
        stress_scenarios,
        required_assets=portfolio_tickers
    )
    return validated, True, None


def run_portfolio_analysis(
    portfolio,
    stress_scenarios=None,
    start_date="2022-01-01",
    prices=None,
    valuation_prices=None,
    return_prices=None,
    max_price_staleness_business_days=(
        DEFAULT_MAX_PRICE_STALENESS_BUSINESS_DAYS
    )
):
    """Run the portfolio calculations and return reusable analysis results.

    ``valuation_prices`` values current positions and must represent raw market
    Close. ``return_prices`` drives every historical return/risk calculation
    and must represent adjusted Close. Supplying both skips the market-data
    download.

    The legacy ``prices`` parameter remains supported as one shared prepared
    view for deterministic callers. It cannot be combined with either explicit
    price channel. Current raw prices must be no more than the configured
    number of business days older than the valuation-data reference date.
    """
    portfolio = validate_portfolio(portfolio)
    portfolio_tickers = portfolio["ticker"].tolist()

    (
        stress_scenarios,
        stress_test_available,
        stress_test_reason
    ) = _resolve_optional_stress_scenarios(
        stress_scenarios,
        portfolio_tickers
    )

    valuation_prices, return_prices = _resolve_price_views(
        portfolio_tickers=portfolio_tickers,
        start_date=start_date,
        prices=prices,
        valuation_prices=valuation_prices,
        return_prices=return_prices
    )
    returns, return_observation_metadata = (
        calculate_returns_with_metadata(return_prices)
    )
    if len(returns) < 2:
        raise ValueError(
            "Core risk statistics require at least two fully aligned daily "
            f"return observations; received {len(returns)}."
        )

    valuation_prices, valuation_freshness = (
        validate_valuation_price_freshness(
            valuation_prices,
            required_assets=portfolio_tickers,
            max_price_staleness_business_days=(
                max_price_staleness_business_days
            )
        )
    )

    positions, weights, portfolio_value = (
        calculate_portfolio_positions(
            portfolio,
            valuation_prices
        )
    )
    positions["price_as_of"] = positions["ticker"].map(
        valuation_freshness["valuation_price_dates"]
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

    diversification = calculate_diversification_diagnostics(
        covariance_matrix=covariance_matrix,
        weights=aligned_weights,
        component_volatility_contributions=risk_contribution_table[
            "Contribution_volatilite"
        ]
    )

    parametric_var_95 = parametric_var(
        portfolio_returns,
        confidence_level=0.95
    )

    parametric_var_99 = parametric_var(
        portfolio_returns,
        confidence_level=0.99
    )

    if stress_test_available:
        stress_results = calculate_stress_test(
            stress_scenarios,
            aligned_weights,
            portfolio_value
        )
    else:
        stress_results = pd.DataFrame(
            columns=["Portfolio_return", "P&L"],
            dtype="float64"
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

    historical_backtest_required_returns = 254
    historical_expected_rate = 0.05
    historical_backtest_available = (
        len(portfolio_returns) >= historical_backtest_required_returns
    )
    historical_backtest_reason = None
    if historical_backtest_available:
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
    else:
        historical_backtest = pd.DataFrame(
            {
                "Return": pd.Series(dtype=float),
                "VaR": pd.Series(dtype=float),
                "Breach": pd.Series(dtype=bool)
            }
        )
        historical_n_breaches = 0
        historical_breach_rate = None
        historical_backtest_reason = (
            "The 252-day rolling Historical VaR backtest requires at least "
            "254 aligned portfolio return observations (252 estimation "
            "observations plus two out-of-sample observations); received "
            f"{len(portfolio_returns)}."
        )

    historical_validation = _model_validation_result(
        historical_backtest,
        historical_expected_rate
    )

    if len(historical_backtest) >= 2:
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
    else:
        n00 = n01 = n10 = n11 = 0

    historical_validation["transition_counts"] = {
        "n00": int(n00),
        "n01": int(n01),
        "n10": int(n10),
        "n11": int(n11)
    }

    ewma_backtest_available = bool(ewma_var_threshold_95.notna().any())
    ewma_backtest_reason = None
    ewma_expected_rate = 0.05
    if ewma_backtest_available:
        (
            ewma_backtest,
            ewma_n_breaches,
            ewma_breach_rate,
            ewma_expected_rate
        ) = backtest_var_threshold(
            portfolio_returns,
            ewma_var_threshold_95,
            expected_breach_rate=ewma_expected_rate
        )
    else:
        ewma_backtest = pd.DataFrame(
            {
                "Return": pd.Series(dtype=float),
                "VaR": pd.Series(dtype=float),
                "Breach": pd.Series(dtype=bool)
            }
        )
        ewma_n_breaches = 0
        ewma_breach_rate = None
        ewma_backtest_reason = (
            "The EWMA VaR backtest requires at least 31 aligned portfolio "
            "return observations (30 initialization observations plus one "
            "out-of-sample observation); received "
            f"{len(portfolio_returns)}."
        )

    ewma_validation = _model_validation_result(
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

    historical_common_results = _model_validation_result(
        historical_common_backtest,
        expected_breach_rate=0.05
    )

    ewma_common_results = _model_validation_result(
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
        "prices": return_prices,
        "valuation_prices": valuation_prices,
        "return_prices": return_prices,
        **valuation_freshness,
        "returns": returns,
        **return_observation_metadata,
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
        "diversification": diversification,
        "parametric_var_95": parametric_var_95,
        "parametric_var_99": parametric_var_99,
        "stress_results": stress_results,
        "stress_test_available": stress_test_available,
        "stress_test_reason": stress_test_reason,
        "rolling_volatility": rolling_volatility_series,
        "ewma_var_threshold_95": ewma_var_threshold_95,
        "ewma_daily_volatility": ewma_daily_volatility_series,
        "ewma_annualized_volatility": ewma_annualized_volatility_series,
        "historical_backtest": historical_backtest,
        "historical_backtest_available": historical_backtest_available,
        "historical_backtest_reason": historical_backtest_reason,
        "historical_backtest_required_returns": (
            historical_backtest_required_returns
        ),
        "historical_validation": historical_validation,
        "historical_n_breaches": historical_n_breaches,
        "historical_breach_rate": historical_breach_rate,
        "historical_expected_rate": historical_expected_rate,
        "ewma_backtest": ewma_backtest,
        "ewma_backtest_available": ewma_backtest_available,
        "ewma_backtest_reason": ewma_backtest_reason,
        "ewma_validation": ewma_validation,
        "ewma_n_breaches": ewma_n_breaches,
        "ewma_breach_rate": ewma_breach_rate,
        "ewma_expected_rate": ewma_expected_rate,
        "common_backtest_dates": common_backtest_dates,
        "historical_common_backtest": historical_common_backtest,
        "ewma_common_backtest": ewma_common_backtest,
        "model_comparison": model_comparison,
        "model_validation_available": historical_common_results[
            "model_validation_available"
        ],
        "model_validation_reason": historical_common_results.get(
            "model_validation_reason"
        ),
        "validation_observations": historical_common_results[
            "validation_observations"
        ],
        "minimum_required_observations": historical_common_results[
            "minimum_required_observations"
        ],
        "validation_confidence_level": historical_common_results[
            "confidence_level"
        ],
        "base_minimum_observations_policy": historical_common_results[
            "base_minimum_observations_policy"
        ],
        "minimum_expected_breaches_policy": historical_common_results[
            "minimum_expected_breaches_policy"
        ]
    }
