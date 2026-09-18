import numpy as np
import pandas as pd

from src.performance import (
    PERFORMANCE_DENOMINATOR_TOLERANCE,
    TRADING_DAYS_PER_YEAR,
    calculate_daily_risk_free_rate
)


def normalize_benchmark_ticker(benchmark_ticker):
    """Return one non-empty uppercase benchmark ticker."""
    if not isinstance(benchmark_ticker, str):
        raise ValueError("benchmark_ticker must be provided as text.")

    normalized_ticker = benchmark_ticker.strip().upper()
    if not normalized_ticker:
        raise ValueError("benchmark_ticker must not be empty.")
    return normalized_ticker


def _validate_investor_returns(daily_twr):
    """Return a chronological copy with inactive NaN observations retained."""
    if not isinstance(daily_twr, pd.Series) or daily_twr.empty:
        raise ValueError("daily_twr must be a non-empty pandas Series.")
    if (
        not pd.api.types.is_numeric_dtype(daily_twr.dtype)
        or pd.api.types.is_bool_dtype(daily_twr.dtype)
        or pd.api.types.is_complex_dtype(daily_twr.dtype)
    ):
        raise ValueError("daily_twr must contain only real numeric values.")
    if daily_twr.index.has_duplicates:
        raise ValueError("daily_twr index must not contain duplicate dates.")
    if not daily_twr.index.is_monotonic_increasing:
        raise ValueError("daily_twr index must be in chronological order.")

    investor_returns = daily_twr.astype("float64").copy()
    active_returns = investor_returns.dropna()
    if not np.isfinite(active_returns.to_numpy()).all():
        raise ValueError(
            "daily_twr must contain only finite values or NaN inactive "
            "periods."
        )
    if active_returns.lt(-1.0).any():
        raise ValueError("daily_twr returns must not be below -100%.")
    return investor_returns


def _validate_benchmark_prices(benchmark_prices, benchmark_ticker):
    """Return one chronological adjusted benchmark price Series."""
    if isinstance(benchmark_prices, pd.DataFrame):
        if benchmark_ticker not in benchmark_prices.columns:
            raise ValueError(
                f"Benchmark adjusted Close is missing ticker {benchmark_ticker}."
            )
        benchmark_prices = benchmark_prices[benchmark_ticker]
    if not isinstance(benchmark_prices, pd.Series) or benchmark_prices.empty:
        raise ValueError(
            "benchmark_prices must be a non-empty pandas Series or a "
            "DataFrame containing the benchmark ticker."
        )
    if (
        not pd.api.types.is_numeric_dtype(benchmark_prices.dtype)
        or pd.api.types.is_bool_dtype(benchmark_prices.dtype)
        or pd.api.types.is_complex_dtype(benchmark_prices.dtype)
    ):
        raise ValueError(
            "benchmark_prices must contain only real numeric values."
        )
    if benchmark_prices.index.has_duplicates:
        raise ValueError(
            "benchmark_prices index must not contain duplicate dates."
        )
    if not benchmark_prices.index.is_monotonic_increasing:
        raise ValueError(
            "benchmark_prices index must be in chronological order."
        )

    prices = benchmark_prices.astype("float64").copy()
    available_prices = prices.dropna()
    if available_prices.empty:
        raise ValueError(
            f"No adjusted-Close benchmark prices are available for "
            f"{benchmark_ticker}."
        )
    if not np.isfinite(available_prices.to_numpy()).all():
        raise ValueError(
            "benchmark_prices must contain only finite values or NaN "
            "missing observations."
        )
    if available_prices.le(0.0).any():
        raise ValueError("benchmark_prices must be strictly positive.")
    return prices


def _geometric_return_metrics(returns, trading_days):
    """Return total and annualized geometric return for one aligned sample."""
    if returns.empty:
        return float("nan"), float("nan")

    linked_wealth = float((1.0 + returns).prod())
    total_return = linked_wealth - 1.0
    if not np.isfinite(linked_wealth) or linked_wealth <= 0.0:
        return float(total_return), float("nan")

    with np.errstate(over="ignore", invalid="ignore"):
        annualized_return = np.power(
            linked_wealth,
            trading_days / len(returns)
        ) - 1.0
    if not np.isfinite(annualized_return):
        annualized_return = float("nan")
    return float(total_return), float(annualized_return)


def run_benchmark_analysis(
    daily_twr,
    benchmark_prices,
    risk_free_rate_annual=0.0,
    benchmark_ticker="SPY",
    trading_days=TRADING_DAYS_PER_YEAR
):
    """Compare investor TWR with aligned adjusted-Close benchmark returns."""
    ticker = normalize_benchmark_ticker(benchmark_ticker)
    investor_returns = _validate_investor_returns(daily_twr)
    prices = _validate_benchmark_prices(benchmark_prices, ticker)
    risk_free_daily = calculate_daily_risk_free_rate(
        risk_free_rate_annual,
        trading_days=trading_days
    )

    benchmark_returns = prices.pct_change(fill_method=None)
    benchmark_returns.name = "benchmark_return"
    aligned_returns = pd.concat(
        [
            investor_returns.rename("investor_return"),
            benchmark_returns
        ],
        axis=1,
        join="inner"
    ).dropna(how="any")

    investor_cumulative_aligned = (
        (1.0 + aligned_returns["investor_return"]).cumprod() - 1.0
    ).rename("Investor TWR")
    benchmark_cumulative_return = (
        (1.0 + aligned_returns["benchmark_return"]).cumprod() - 1.0
    ).rename(ticker)

    benchmark_total_return, benchmark_annualized_return = (
        _geometric_return_metrics(
            aligned_returns["benchmark_return"],
            trading_days
        )
    )
    investor_total_return, _ = _geometric_return_metrics(
        aligned_returns["investor_return"],
        trading_days
    )

    beta = float("nan")
    alpha_annualized = float("nan")
    tracking_error = float("nan")
    information_ratio = float("nan")
    if len(aligned_returns) >= 2:
        investor_aligned = aligned_returns["investor_return"]
        benchmark_aligned = aligned_returns["benchmark_return"]
        benchmark_variance = float(benchmark_aligned.var(ddof=1))
        if (
            np.isfinite(benchmark_variance)
            and benchmark_variance > PERFORMANCE_DENOMINATOR_TOLERANCE
        ):
            beta = float(
                investor_aligned.cov(benchmark_aligned)
                / benchmark_variance
            )
            alpha_daily = (
                (investor_aligned - risk_free_daily).mean()
                - beta * (benchmark_aligned - risk_free_daily).mean()
            )
            alpha_annualized = float(alpha_daily * trading_days)

        active_returns = investor_aligned - benchmark_aligned
        active_standard_deviation = float(active_returns.std(ddof=1))
        if np.isfinite(active_standard_deviation):
            if (
                active_standard_deviation
                > PERFORMANCE_DENOMINATOR_TOLERANCE
            ):
                tracking_error = float(
                    active_standard_deviation * np.sqrt(trading_days)
                )
                information_ratio = float(
                    active_returns.mean()
                    / active_standard_deviation
                    * np.sqrt(trading_days)
                )
            else:
                tracking_error = 0.0

    aligned_start_date = (
        aligned_returns.index[0] if not aligned_returns.empty else None
    )
    aligned_end_date = (
        aligned_returns.index[-1] if not aligned_returns.empty else None
    )
    performance_observations = int(investor_returns.notna().sum())

    return {
        "ticker": ticker,
        "price_basis": "adjusted_close_total_return_style",
        "observations": len(aligned_returns),
        "benchmark_observations": len(aligned_returns),
        "performance_observations": performance_observations,
        "investor_observations_excluded": (
            performance_observations - len(aligned_returns)
        ),
        "aligned_start_date": aligned_start_date,
        "aligned_end_date": aligned_end_date,
        "benchmark_returns": benchmark_returns,
        "aligned_returns": aligned_returns,
        "investor_total_return_aligned": investor_total_return,
        "benchmark_total_return": benchmark_total_return,
        "benchmark_annualized_return": benchmark_annualized_return,
        "beta": beta,
        "alpha_annualized": alpha_annualized,
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
        "investor_cumulative_aligned": investor_cumulative_aligned,
        "benchmark_cumulative_return": benchmark_cumulative_return
    }
