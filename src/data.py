import numpy as np
import yfinance as yf
import pandas as pd


def validate_price_data(prices, required_assets=None):
    """Validate prices, allowing NaN for missing observations but not infinity."""
    if not isinstance(prices, pd.DataFrame):
        raise ValueError("Price data must be a pandas DataFrame.")

    if prices.empty:
        raise ValueError("Price data must not be empty.")

    if prices.columns.duplicated().any():
        duplicates = prices.columns[prices.columns.duplicated()].tolist()
        raise ValueError(
            f"Duplicate asset columns found in price data: {duplicates}"
        )

    non_numeric_columns = [
        column
        for column in prices.columns
        if not pd.api.types.is_numeric_dtype(prices[column])
    ]
    if non_numeric_columns:
        raise ValueError(
            "Price data must contain only numeric values. "
            f"Non-numeric columns: {non_numeric_columns}"
        )

    finite_values = prices.to_numpy()[~prices.isna().to_numpy()]
    if not np.isfinite(finite_values).all():
        raise ValueError("Price data must contain only finite numeric values.")

    if required_assets is not None:
        missing_assets = pd.Index(required_assets).difference(prices.columns)
        if len(missing_assets) > 0:
            raise ValueError(
                f"Missing price data for: {list(missing_assets)}"
            )

    return prices.copy()


def validate_price_history(
    returns,
    window=252,
    minimum_out_of_sample=2
):
    """Require enough usable returns for rolling VaR model validation."""
    minimum_returns = window + minimum_out_of_sample
    usable_returns = len(returns)

    if usable_returns < minimum_returns:
        raise ValueError(
            "Insufficient price history for the Historical VaR backtest: "
            f"at least {minimum_returns} usable daily return observations are "
            f"required ({window} for the rolling window and "
            f"{minimum_out_of_sample} out-of-sample observations); "
            f"received {usable_returns}."
        )

def download_prices(tickers, start_date):
    data = yf.download(
        tickers,
        start=start_date,
        auto_adjust=True,
        progress=False
    )

    prices = data["Close"]

    return validate_price_data(
        prices,
        required_assets=tickers
    )


def calculate_returns(prices):
    prices = validate_price_data(prices)
    returns = prices.pct_change(fill_method=None).dropna()

    if returns.empty:
        raise ValueError(
            "No usable return observations remain after cleaning price data."
        )

    return returns

def validate_stress_scenarios(stress_scenarios, required_assets=None):
    """Validate and standardize stress scenarios from any source."""
    if not isinstance(stress_scenarios, pd.DataFrame):
        raise ValueError("Stress scenarios must be a pandas DataFrame.")

    if stress_scenarios.empty:
        raise ValueError(
            "Stress scenarios must contain at least one scenario."
        )

    stress_scenarios = stress_scenarios.copy()
    stress_scenarios.columns = [
        str(column).strip().upper()
        for column in stress_scenarios.columns
    ]

    if stress_scenarios.columns.duplicated().any():
        raise ValueError(
            "Duplicate assets found in stress scenarios."
        )

    try:
        stress_scenarios = stress_scenarios.apply(
            pd.to_numeric,
            errors="raise"
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Stress scenario shocks must contain only numeric values."
        ) from error

    if not np.isfinite(stress_scenarios.to_numpy()).all():
        raise ValueError(
            "Stress scenario shocks must contain only finite numeric values."
        )

    if required_assets is not None:
        missing_assets = (
            pd.Index(required_assets)
            .difference(stress_scenarios.columns)
        )
        if len(missing_assets) > 0:
            raise ValueError(
                "Stress scenarios missing assets: "
                f"{list(missing_assets)}"
            )

    return stress_scenarios


def load_stress_scenarios(file_path):
    stress_scenarios = pd.read_csv(
        file_path,
        index_col="scenario"
    )

    return validate_stress_scenarios(stress_scenarios)
