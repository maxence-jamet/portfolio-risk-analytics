import numpy as np
import pandas as pd


def validate_portfolio(portfolio):
    """Validate a copied portfolio without modifying the caller's DataFrame."""
    if not isinstance(portfolio, pd.DataFrame):
        raise ValueError("Portfolio input must be a pandas DataFrame.")

    if portfolio.empty:
        raise ValueError("Portfolio must contain at least one position.")

    required_columns = {
        "ticker",
        "quantity",
        "purchase_price"
    }

    missing_columns = (
        required_columns
        - set(portfolio.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing portfolio columns: {sorted(missing_columns)}"
        )

    portfolio = portfolio.copy()

    portfolio["ticker"] = (
        portfolio["ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    if (portfolio["ticker"] == "").any():
        raise ValueError("Portfolio tickers must not be blank.")

    try:
        portfolio["quantity"] = pd.to_numeric(
            portfolio["quantity"],
            errors="raise"
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Portfolio quantities must contain only numeric values."
        ) from error

    try:
        portfolio["purchase_price"] = pd.to_numeric(
            portfolio["purchase_price"],
            errors="raise"
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Portfolio purchase prices must contain only numeric values."
        ) from error

    if portfolio["ticker"].duplicated().any():
        raise ValueError(
            "Duplicate tickers found in portfolio."
        )

    if not np.isfinite(portfolio["quantity"]).all():
        raise ValueError(
            "Portfolio quantities must contain only finite numeric values."
        )

    if not np.isfinite(portfolio["purchase_price"]).all():
        raise ValueError(
            "Portfolio purchase prices must contain only finite numeric values."
        )

    if (portfolio["quantity"] <= 0).any():
        raise ValueError(
            "All quantities must be strictly positive."
        )

    if (portfolio["purchase_price"] <= 0).any():
        raise ValueError(
            "All purchase prices must be strictly positive."
        )

    return portfolio


def load_portfolio(file_path):
    portfolio = pd.read_csv(file_path)

    return validate_portfolio(portfolio)


def calculate_portfolio_positions(
    portfolio,
    prices
):
    portfolio_tickers = pd.Index(
        portfolio["ticker"]
    )

    missing_prices = (
        portfolio_tickers
        .difference(prices.columns)
    )

    if len(missing_prices) > 0:
        raise ValueError(
            f"Missing price data for: {list(missing_prices)}"
        )

    latest_prices = (
        prices
        .ffill()
        .iloc[-1]
        .reindex(portfolio_tickers)
    )

    positions = portfolio.copy()

    positions["current_price"] = (
        latest_prices.to_numpy()
    )

    positions["cost_basis"] = (
        positions["quantity"]
        * positions["purchase_price"]
    )

    positions["market_value"] = (
        positions["quantity"]
        * positions["current_price"]
    )

    positions["unrealized_pnl"] = (
        positions["market_value"]
        - positions["cost_basis"]
    )

    positions["unrealized_pnl_pct"] = (
        positions["unrealized_pnl"]
        / positions["cost_basis"]
    )

    portfolio_value = (
        positions["market_value"].sum()
    )

    positions["weight"] = (
        positions["market_value"]
        / portfolio_value
    )

    weights = (
        positions
        .set_index("ticker")["weight"]
    )

    return (
        positions,
        weights,
        portfolio_value
    )
def calculate_portfolio_returns(returns, weights):

    missing_weights = returns.columns.difference(weights.index)
    extra_weights = weights.index.difference(returns.columns)

    if len(missing_weights) > 0:
        raise ValueError(
            f"Missing weights for: {list(missing_weights)}"
        )

    if len(extra_weights) > 0:
        raise ValueError(
            f"Weights provided for unknown assets: {list(extra_weights)}"
        )

    if not np.isclose(weights.sum(), 1.0):
        raise ValueError(
            f"Portfolio weights must sum to 1. Current sum: {weights.sum():.4f}"
        )

    aligned_weights = weights.reindex(returns.columns)

    weighted_returns = returns.mul(
        aligned_weights,
        axis=1
    )

    portfolio_returns = weighted_returns.sum(axis=1)

    return portfolio_returns, aligned_weights
