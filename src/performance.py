import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252
PERFORMANCE_DENOMINATOR_TOLERANCE = 1e-12
MIN_RISK_FREE_RATE_ANNUAL = -0.20
MAX_RISK_FREE_RATE_ANNUAL = 0.50


def _validate_daily_twr(daily_twr):
    """Return independent TWR copies including only valid active returns."""
    if not isinstance(daily_twr, pd.Series):
        raise ValueError("daily_twr must be provided as a pandas Series.")
    if daily_twr.empty:
        raise ValueError("daily_twr must not be empty.")
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

    twr = daily_twr.astype("float64").copy()
    valid_returns = twr.dropna()
    if valid_returns.empty:
        raise ValueError(
            "daily_twr must contain at least one active-period return."
        )
    if not np.isfinite(valid_returns.to_numpy()).all():
        raise ValueError(
            "daily_twr must contain only finite values or NaN inactive "
            "periods."
        )
    if valid_returns.lt(-1.0).any():
        raise ValueError("daily_twr returns must not be below -100%.")

    return twr, valid_returns


def validate_annual_risk_free_rate(risk_free_rate_annual):
    """Return a finite decimal annual rate inside the project UI range."""
    if isinstance(risk_free_rate_annual, (bool, np.bool_)):
        raise ValueError("risk_free_rate_annual must be a real number.")
    try:
        rate = float(risk_free_rate_annual)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "risk_free_rate_annual must be a real number."
        ) from error
    if not np.isfinite(rate):
        raise ValueError("risk_free_rate_annual must be finite.")
    if not MIN_RISK_FREE_RATE_ANNUAL < rate < MAX_RISK_FREE_RATE_ANNUAL:
        raise ValueError(
            "risk_free_rate_annual must be greater than -20% and less "
            "than 50%."
        )
    return rate


def calculate_daily_risk_free_rate(
    risk_free_rate_annual,
    trading_days=TRADING_DAYS_PER_YEAR
):
    """Convert an annual rate to an equivalent compounded daily rate."""
    annual_rate = validate_annual_risk_free_rate(risk_free_rate_annual)
    return (1.0 + annual_rate) ** (1.0 / trading_days) - 1.0


def calculate_annualized_twr(
    daily_twr,
    trading_days=TRADING_DAYS_PER_YEAR
):
    """Geometrically annualize valid active daily TWR observations."""
    _, valid_returns = _validate_daily_twr(daily_twr)
    linked_wealth = float((1.0 + valid_returns).prod())
    if not np.isfinite(linked_wealth) or linked_wealth <= 0.0:
        raise ValueError(
            "Annualized TWR is undefined because cumulative linked wealth "
            "is not positive and finite."
        )
    annualized_twr = linked_wealth ** (
        trading_days / len(valid_returns)
    ) - 1.0
    if not np.isfinite(annualized_twr):
        raise ValueError("Annualized TWR is not finite.")
    return float(annualized_twr)


def calculate_sharpe_ratio(
    daily_twr,
    risk_free_rate_annual=0.0,
    trading_days=TRADING_DAYS_PER_YEAR
):
    """Return annualized Sharpe from active daily excess TWR returns."""
    _, valid_returns = _validate_daily_twr(daily_twr)
    risk_free_daily = calculate_daily_risk_free_rate(
        risk_free_rate_annual,
        trading_days=trading_days
    )
    if len(valid_returns) < 2:
        return float("nan")
    excess_returns = valid_returns - risk_free_daily
    sample_standard_deviation = float(excess_returns.std(ddof=1))
    if (
        not np.isfinite(sample_standard_deviation)
        or sample_standard_deviation <= PERFORMANCE_DENOMINATOR_TOLERANCE
    ):
        return float("nan")
    return float(
        excess_returns.mean()
        / sample_standard_deviation
        * np.sqrt(trading_days)
    )


def calculate_sortino_ratio(
    daily_twr,
    risk_free_rate_annual=0.0,
    trading_days=TRADING_DAYS_PER_YEAR
):
    """Return annualized Sortino using a full-sample lower partial moment."""
    _, valid_returns = _validate_daily_twr(daily_twr)
    risk_free_daily = calculate_daily_risk_free_rate(
        risk_free_rate_annual,
        trading_days=trading_days
    )
    if len(valid_returns) < 2:
        return float("nan")
    excess_returns = valid_returns - risk_free_daily
    downside_returns = np.minimum(excess_returns.to_numpy(), 0.0)
    downside_deviation_daily = float(
        np.sqrt(np.mean(downside_returns ** 2))
    )
    if (
        not np.isfinite(downside_deviation_daily)
        or downside_deviation_daily <= PERFORMANCE_DENOMINATOR_TOLERANCE
    ):
        return float("nan")
    return float(
        excess_returns.mean() * trading_days
        / (downside_deviation_daily * np.sqrt(trading_days))
    )


def calculate_calmar_ratio(annualized_twr, investor_max_drawdown):
    """Return Annualized TWR divided by actual investor max drawdown."""
    values = np.asarray(
        [annualized_twr, investor_max_drawdown],
        dtype=float
    )
    if not np.isfinite(values).all():
        raise ValueError(
            "annualized_twr and investor_max_drawdown must be finite."
        )
    drawdown_magnitude = abs(float(investor_max_drawdown))
    if drawdown_magnitude <= PERFORMANCE_DENOMINATOR_TOLERANCE:
        return float("nan")
    return float(annualized_twr / drawdown_magnitude)


def calculate_investor_performance_metrics(
    daily_twr,
    investor_max_drawdown,
    risk_free_rate_annual=0.0
):
    """Calculate actual-history performance metrics from daily TWR only."""
    _, valid_returns = _validate_daily_twr(daily_twr)
    annual_rate = validate_annual_risk_free_rate(risk_free_rate_annual)
    annualized_twr = calculate_annualized_twr(daily_twr)
    return {
        "annualized_twr": annualized_twr,
        "sharpe_ratio": calculate_sharpe_ratio(daily_twr, annual_rate),
        "sortino_ratio": calculate_sortino_ratio(daily_twr, annual_rate),
        "calmar_ratio": calculate_calmar_ratio(
            annualized_twr,
            investor_max_drawdown
        ),
        "risk_free_rate_annual": annual_rate,
        "performance_observations": len(valid_returns)
    }


def calculate_twr_drawdown(daily_twr):
    """Calculate cash-flow-neutral investor drawdown from daily TWR.

    Missing daily TWR values represent zero-capital inactive periods. They are
    neutral factors for geometric linking, but remain missing in the published
    wealth-index and drawdown series so no investment exposure is implied.
    """
    twr, _ = _validate_daily_twr(daily_twr)

    linked_wealth = (1.0 + twr.fillna(0.0)).cumprod()
    inactive_periods = twr.isna()
    wealth_index = linked_wealth.mask(inactive_periods)
    wealth_index.name = "investor_wealth_index"

    # The normalized starting wealth of 1 is part of the high-water mark, so
    # an initial loss is immediately recognized as drawdown.
    running_max = wealth_index.cummax().clip(lower=1.0)
    drawdown = wealth_index.div(running_max).sub(1.0)
    drawdown.name = "investor_drawdown"

    return {
        "investor_wealth_index": wealth_index,
        "investor_drawdown": drawdown,
        "investor_max_drawdown": float(drawdown.min())
    }


def _validate_performance_inputs(
    daily_portfolio_value,
    external_cash_flows_by_day
):
    """Validate TWR inputs and return independent floating-point copies."""
    inputs = {
        "daily_portfolio_value": daily_portfolio_value,
        "external_cash_flows_by_day": external_cash_flows_by_day
    }

    for input_name, values in inputs.items():
        if not isinstance(values, pd.Series):
            raise ValueError(
                f"{input_name} must be provided as a pandas Series."
            )

        if values.empty:
            raise ValueError(f"{input_name} must not be empty.")

        if not pd.api.types.is_numeric_dtype(values.dtype):
            raise ValueError(
                f"{input_name} must contain only numeric values."
            )

        if (
            pd.api.types.is_bool_dtype(values.dtype)
            or pd.api.types.is_complex_dtype(values.dtype)
        ):
            raise ValueError(
                f"{input_name} must contain only real numeric values."
            )

        numeric_values = values.astype("float64").to_numpy(copy=True)
        if not np.isfinite(numeric_values).all():
            raise ValueError(
                f"{input_name} must contain only finite numeric values."
            )

    if not daily_portfolio_value.index.equals(
        external_cash_flows_by_day.index
    ):
        raise ValueError(
            "daily_portfolio_value and external_cash_flows_by_day indexes "
            "must align exactly."
        )

    index = daily_portfolio_value.index
    if index.has_duplicates:
        raise ValueError(
            "Performance input indexes must not contain duplicate dates."
        )

    if index.hasnans:
        raise ValueError("Performance input indexes must contain valid dates.")

    if not index.is_monotonic_increasing:
        raise ValueError(
            "Performance input indexes must be in chronological order."
        )

    return (
        daily_portfolio_value.astype("float64").copy(),
        external_cash_flows_by_day.astype("float64").copy()
    )


def calculate_time_weighted_return(
    daily_portfolio_value,
    external_cash_flows_by_day
):
    """Calculate daily and cumulative cash-flow-aware TWR.

    The project uses a beginning-of-day external cash-flow convention. For
    date t, the capital base is the previous end-of-day portfolio value plus
    date t's signed external flow, and the daily return is:

        portfolio_value_t / capital_base_t - 1

    TWR neutralizes investor DEPOSIT and WITHDRAWAL amounts so it measures
    portfolio performance independently of contribution timing and size.
    BUY and SELL transactions are internal portfolio activity, so they must
    not be included as external investor flows. TWR is not the investor's
    personal money-weighted return.

    The first previous portfolio value is zero. A zero capital base with a
    zero portfolio value is an inactive period and has an undefined (NaN)
    daily return; its growth factor is treated as one only when linking the
    cumulative TWR.
    """
    portfolio_value, external_cash_flows = _validate_performance_inputs(
        daily_portfolio_value,
        external_cash_flows_by_day
    )

    previous_portfolio_value = portfolio_value.shift(
        periods=1,
        fill_value=0.0
    )
    capital_base = previous_portfolio_value + external_cash_flows

    zero_capital = pd.Series(
        np.isclose(capital_base.to_numpy(), 0.0),
        index=capital_base.index
    )
    negative_capital = capital_base.lt(0.0) & ~zero_capital
    if negative_capital.any():
        invalid_date = negative_capital[negative_capital].index[0]
        raise ValueError(
            "TWR capital base cannot be negative under the no-margin/"
            f"no-borrowing convention. Invalid value on {invalid_date}: "
            f"{capital_base.loc[invalid_date]:g}."
        )

    nonzero_value_without_capital = zero_capital & ~pd.Series(
        np.isclose(portfolio_value.to_numpy(), 0.0),
        index=portfolio_value.index
    )
    if nonzero_value_without_capital.any():
        invalid_date = nonzero_value_without_capital[
            nonzero_value_without_capital
        ].index[0]
        raise ValueError(
            "Portfolio value must be zero when the TWR capital base is zero. "
            f"Inconsistent value on {invalid_date}: "
            f"{portfolio_value.loc[invalid_date]:g}."
        )

    daily_twr = pd.Series(
        np.nan,
        index=portfolio_value.index,
        name="daily_twr"
    )
    active_periods = ~zero_capital
    daily_twr.loc[active_periods] = (
        portfolio_value.loc[active_periods]
        / capital_base.loc[active_periods]
        - 1.0
    )

    # An inactive period contributes a neutral factor of 1 to geometric
    # linking while its daily return remains explicitly undefined.
    cumulative_twr = (
        (1.0 + daily_twr.fillna(0.0)).cumprod()
        - 1.0
    )
    cumulative_twr.name = "cumulative_twr"

    return {
        "daily_twr": daily_twr,
        "cumulative_twr": cumulative_twr,
        "total_twr": float(cumulative_twr.iloc[-1])
    }
