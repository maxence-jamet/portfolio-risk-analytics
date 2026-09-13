import numpy as np
import pandas as pd


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
