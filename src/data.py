import numpy as np
import yfinance as yf
import pandas as pd


DEFAULT_MAX_PRICE_STALENESS_BUSINESS_DAYS = 3
DEFAULT_RISK_WINDOW_LABEL = "3 years"
RISK_WINDOW_PRESETS = {
    "1 year": 1,
    "2 years": 2,
    "3 years": 3,
    "5 years": 5,
    "Maximum available": None
}
MAXIMUM_HISTORY_DOWNLOAD_START = "1900-01-01"


def _normalize_market_date(value, input_name):
    """Return one timezone-naive normalized market date."""
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{input_name} must be a valid date.") from error

    if pd.isna(timestamp):
        raise ValueError(f"{input_name} must be a valid date.")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp.normalize()


def resolve_risk_window(
    risk_window_label,
    reference_date,
    earliest_available_date
):
    """Resolve a preset from supplied market dates, never the system clock.

    Calendar-year presets are measured backward from the latest relevant
    supplied/downloaded market date. ``Maximum available`` begins at the
    earliest relevant return-price date. The returned boundary is a desired
    price start; the first aligned portfolio return will normally be later.
    """
    if risk_window_label not in RISK_WINDOW_PRESETS:
        raise ValueError(
            "risk_window_label must be one of: "
            f"{list(RISK_WINDOW_PRESETS)}."
        )

    reference_date = _normalize_market_date(
        reference_date,
        "reference_date"
    )
    earliest_available_date = _normalize_market_date(
        earliest_available_date,
        "earliest_available_date"
    )
    if earliest_available_date > reference_date:
        raise ValueError(
            "earliest_available_date must not be later than reference_date."
        )

    years = RISK_WINDOW_PRESETS[risk_window_label]
    if years is None:
        return earliest_available_date
    return reference_date - pd.DateOffset(years=years)


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


def prepare_price_data(prices, required_assets=None):
    """Validate and return a chronological copy with normalized daily dates."""
    prepared_prices = validate_price_data(
        prices,
        required_assets=required_assets
    )

    price_dates = pd.to_datetime(
        prepared_prices.index,
        errors="coerce"
    )
    if price_dates.isna().any():
        raise ValueError("The price index must contain only valid dates.")

    if price_dates.tz is not None:
        price_dates = price_dates.tz_localize(None)

    normalized_dates = price_dates.normalize()
    prepared_prices.index = (
        price_dates
        if price_dates.equals(normalized_dates)
        else normalized_dates
    )
    if prepared_prices.index.duplicated().any():
        raise ValueError("The price index must not contain duplicate dates.")

    return prepared_prices.sort_index(kind="stable")


def validate_valuation_price_freshness(
    valuation_prices,
    required_assets,
    max_price_staleness_business_days=(
        DEFAULT_MAX_PRICE_STALENESS_BUSINESS_DAYS
    )
):
    """Validate current raw-price freshness and return concise as-of data.

    The market-data reference date is the latest date in the supplied raw
    valuation-price index. Price age is the number of Monday-to-Friday business
    days from each asset's last valid observation to that reference date.
    """
    required_assets = _ticker_list(required_assets)
    if (
        isinstance(max_price_staleness_business_days, bool)
        or not isinstance(
            max_price_staleness_business_days,
            (int, np.integer)
        )
        or max_price_staleness_business_days < 0
    ):
        raise ValueError(
            "max_price_staleness_business_days must be a non-negative "
            "integer."
        )

    valuation_prices = prepare_price_data(
        valuation_prices,
        required_assets=required_assets
    )
    reference_date = valuation_prices.index[-1]
    valuation_price_dates = {}
    valuation_price_age_business_days = {}

    for ticker in required_assets:
        available_prices = valuation_prices.loc[
            valuation_prices.index <= reference_date,
            ticker
        ].dropna()
        if available_prices.empty:
            raise ValueError(
                f"No current or earlier valuation price is available for "
                f"{ticker} on or before market-data reference date "
                f"{reference_date.date()}."
            )

        price_date = available_prices.index[-1]
        price_age = int(np.busday_count(
            np.datetime64(price_date.date()),
            np.datetime64(reference_date.date())
        ))
        if price_age > max_price_staleness_business_days:
            raise ValueError(
                f"{ticker} valuation price is stale: last price "
                f"{price_date.date()}, market-data reference date "
                f"{reference_date.date()}, age {price_age} business days, "
                f"maximum allowed "
                f"{max_price_staleness_business_days}."
            )

        valuation_price_dates[ticker] = price_date
        valuation_price_age_business_days[ticker] = price_age

    return valuation_prices, {
        "valuation_reference_date": reference_date,
        "valuation_price_dates": valuation_price_dates,
        "valuation_price_age_business_days": (
            valuation_price_age_business_days
        ),
        "max_price_staleness_business_days": (
            max_price_staleness_business_days
        )
    }


def _ticker_list(tickers):
    """Return ticker input as a reusable ordered list."""
    if isinstance(tickers, str):
        return [tickers]

    return list(tickers)


def _extract_yahoo_price_field(data, field_name, tickers):
    """Extract one required Yahoo price field without ambiguous fallbacks."""
    tickers = _ticker_list(tickers)

    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("Yahoo Finance returned no market-price data.")

    if isinstance(data.columns, pd.MultiIndex):
        matching_levels = [
            level
            for level in range(data.columns.nlevels)
            if field_name in data.columns.get_level_values(level)
        ]
        if len(matching_levels) != 1:
            raise ValueError(
                "Yahoo Finance response must contain one unambiguous "
                f"'{field_name}' column level."
            )

        prices = data.xs(
            field_name,
            axis=1,
            level=matching_levels[0],
            drop_level=True
        )
        if isinstance(prices.columns, pd.MultiIndex):
            raise ValueError(
                "Yahoo Finance returned an unsupported nested column "
                f"structure for '{field_name}'."
            )
    else:
        if field_name not in data.columns:
            raise ValueError(
                "Yahoo Finance response is missing required price field "
                f"'{field_name}'."
            )
        if len(tickers) != 1:
            raise ValueError(
                "A flat Yahoo Finance response cannot be mapped "
                "unambiguously to multiple tickers."
            )

        prices = data[[field_name]].rename(
            columns={field_name: tickers[0]}
        )

    prices = prices.copy()
    prices.columns = [str(column) for column in prices.columns]

    validated_prices = validate_price_data(
        prices,
        required_assets=tickers
    )
    return validated_prices.reindex(columns=tickers)


def download_price_views(tickers, start_date):
    """Download raw valuation prices and adjusted return prices once.

    Yahoo Finance is requested with ``auto_adjust=False`` explicitly. Raw
    ``Close`` values are used to value ledger quantities, while ``Adj Close``
    values are used to calculate historical economic returns. A missing or
    ambiguous field raises an error instead of substituting the wrong series.
    """
    tickers = _ticker_list(tickers)
    data = yf.download(
        tickers,
        start=start_date,
        auto_adjust=False,
        actions=False,
        group_by="column",
        progress=False
    )

    return {
        "valuation_prices": _extract_yahoo_price_field(
            data,
            "Close",
            tickers
        ),
        "return_prices": _extract_yahoo_price_field(
            data,
            "Adj Close",
            tickers
        )
    }


def download_prices(tickers, start_date):
    """Return adjusted prices for backward-compatible risk calculations."""
    return download_price_views(tickers, start_date)["return_prices"]


def calculate_returns_with_metadata(prices):
    """Calculate fully aligned returns and transparent observation counts.

    A candidate asset-return date is a row where at least one asset has a
    genuine return. Portfolio risk then retains only rows where every held
    asset has a return. Missing prices are never filled, so neither the
    missing date nor its following date can manufacture a zero return.
    """
    prices = validate_price_data(prices)
    unaligned_returns = prices.pct_change(fill_method=None)
    candidate_return_dates = unaligned_returns.notna().any(axis=1)
    aligned_returns = unaligned_returns.dropna(how="any")

    metadata = {
        "raw_price_observations": len(prices),
        "asset_return_observations_before_alignment": int(
            candidate_return_dates.sum()
        ),
        "aligned_portfolio_return_observations": len(aligned_returns),
        "observations_dropped_during_alignment": int(
            candidate_return_dates.sum() - len(aligned_returns)
        ),
        "usable_return_observations_by_asset": {
            str(asset): int(count)
            for asset, count in unaligned_returns.notna().sum().items()
        }
    }

    if aligned_returns.empty:
        raise ValueError(
            "No usable return observations remain after cleaning price data."
        )

    return aligned_returns, metadata


def calculate_returns(prices):
    """Return fully aligned asset returns for backward-compatible callers."""
    returns, _ = calculate_returns_with_metadata(prices)
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
