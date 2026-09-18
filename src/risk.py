import numpy as np
import pandas as pd
from scipy.stats import norm

from src.data import validate_stress_scenarios


DIVERSIFICATION_ZERO_TOLERANCE = 1e-12


def _validate_security_weights(weights):
    """Return finite long-only weights that sum to one."""
    if isinstance(weights, pd.Series) and weights.index.duplicated().any():
        raise ValueError("Security weights must not contain duplicate assets.")

    try:
        weight_values = np.asarray(weights, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Security weights must contain only numeric values."
        ) from error

    if weight_values.ndim != 1 or weight_values.size == 0:
        raise ValueError(
            "Security weights must be a non-empty one-dimensional vector."
        )
    if not np.isfinite(weight_values).all():
        raise ValueError("Security weights must contain only finite values.")
    if (weight_values < 0.0).any():
        raise ValueError(
            "Diversification diagnostics require long-only security weights."
        )
    if not np.isclose(weight_values.sum(), 1.0):
        raise ValueError(
            "Security weights must sum to 1. "
            f"Current sum: {weight_values.sum():.4f}"
        )

    return weight_values.copy()


def weight_concentration_hhi(weights):
    """Return the Herfindahl index of current security weights."""
    weight_values = _validate_security_weights(weights)
    return float(np.square(weight_values).sum())


def effective_number_of_holdings(weights):
    """Return the equal-weight holding count implied by weight HHI."""
    return 1.0 / weight_concentration_hhi(weights)


def _aligned_covariance_values(covariance_matrix, weights):
    """Validate and align a covariance matrix with security weights."""
    weight_values = _validate_security_weights(weights)

    if isinstance(covariance_matrix, pd.DataFrame):
        if covariance_matrix.index.duplicated().any():
            raise ValueError(
                "Covariance matrix row labels must not contain duplicates."
            )
        if covariance_matrix.columns.duplicated().any():
            raise ValueError(
                "Covariance matrix column labels must not contain duplicates."
            )
        if covariance_matrix.shape[0] != covariance_matrix.shape[1]:
            raise ValueError("Covariance matrix must be square.")

        covariance_assets = pd.Index(covariance_matrix.index)
        if set(covariance_assets) != set(covariance_matrix.columns):
            raise ValueError(
                "Covariance matrix row and column assets must match."
            )
        covariance_matrix = covariance_matrix.reindex(
            columns=covariance_assets
        )

        if isinstance(weights, pd.Series):
            missing_assets = weights.index.difference(covariance_assets)
            extra_assets = covariance_assets.difference(weights.index)
            if len(missing_assets) > 0 or len(extra_assets) > 0:
                raise ValueError(
                    "Security weights and covariance matrix assets must match."
                )
            covariance_matrix = covariance_matrix.reindex(
                index=weights.index,
                columns=weights.index
            )

        covariance_values = covariance_matrix.to_numpy(dtype=float, copy=True)
    else:
        try:
            covariance_values = np.asarray(
                covariance_matrix,
                dtype=float
            ).copy()
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Covariance matrix must contain only numeric values."
            ) from error

    if (
        covariance_values.ndim != 2
        or covariance_values.shape[0] != covariance_values.shape[1]
    ):
        raise ValueError("Covariance matrix must be square.")
    if covariance_values.shape[0] != weight_values.size:
        raise ValueError(
            "Covariance matrix dimensions must match the security weights."
        )
    if not np.isfinite(covariance_values).all():
        raise ValueError("Covariance matrix must contain only finite values.")
    if not np.allclose(covariance_values, covariance_values.T):
        raise ValueError("Covariance matrix must be symmetric.")

    diagonal = np.diag(covariance_values)
    if (diagonal < -DIVERSIFICATION_ZERO_TOLERANCE).any():
        raise ValueError("Covariance matrix variances must be non-negative.")

    return covariance_values, weight_values


def diversification_ratio(covariance_matrix, weights):
    """Return weighted standalone volatility divided by sleeve volatility."""
    covariance_values, weight_values = _aligned_covariance_values(
        covariance_matrix,
        weights
    )
    individual_volatilities = np.sqrt(
        np.maximum(np.diag(covariance_values), 0.0)
    )
    portfolio_variance = float(
        weight_values.T @ covariance_values @ weight_values
    )
    if portfolio_variance < -DIVERSIFICATION_ZERO_TOLERANCE:
        raise ValueError(
            "Covariance matrix produces a negative portfolio variance."
        )

    portfolio_volatility = np.sqrt(max(portfolio_variance, 0.0))
    if portfolio_volatility <= DIVERSIFICATION_ZERO_TOLERANCE:
        return np.nan

    weighted_standalone_volatility = float(
        weight_values @ individual_volatilities
    )
    return float(weighted_standalone_volatility / portfolio_volatility)


def risk_contribution_concentration(component_volatility_contributions):
    """Return HHI and effective count from absolute Euler components."""
    try:
        component_values = np.asarray(
            component_volatility_contributions,
            dtype=float
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Component volatility contributions must be numeric."
        ) from error

    if component_values.ndim != 1 or component_values.size == 0:
        raise ValueError(
            "Component volatility contributions must be a non-empty "
            "one-dimensional vector."
        )
    if not np.isfinite(component_values).all():
        raise ValueError(
            "Component volatility contributions must contain only finite "
            "values."
        )

    absolute_components = np.abs(component_values)
    absolute_total = float(absolute_components.sum())
    if absolute_total <= DIVERSIFICATION_ZERO_TOLERANCE:
        return np.nan, np.nan

    normalized_absolute_components = absolute_components / absolute_total
    risk_concentration_hhi = float(
        np.square(normalized_absolute_components).sum()
    )
    effective_risk_contributors = 1.0 / risk_concentration_hhi
    return risk_concentration_hhi, effective_risk_contributors


def calculate_diversification_diagnostics(
    covariance_matrix,
    weights,
    component_volatility_contributions
):
    """Return current invested-security diversification diagnostics."""
    weight_hhi = weight_concentration_hhi(weights)
    risk_hhi, effective_risk_contributors = (
        risk_contribution_concentration(component_volatility_contributions)
    )

    return {
        "weight_hhi": weight_hhi,
        "effective_number_of_holdings": 1.0 / weight_hhi,
        "diversification_ratio": diversification_ratio(
            covariance_matrix,
            weights
        ),
        "risk_concentration_hhi": risk_hhi,
        "effective_risk_contributors": effective_risk_contributors
    }


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

    if (
        not np.isfinite(portfolio_volatility)
        or portfolio_volatility <= 1e-12
    ):
        raise ValueError(
            "Risk contributions are undefined when annualized portfolio "
            "volatility is zero or numerically near zero."
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
