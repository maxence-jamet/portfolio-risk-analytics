from decimal import Decimal, ROUND_CEILING

import numpy as np
import pandas as pd

from scipy.stats import chi2
from scipy.special import xlogy


DEFAULT_BASE_MINIMUM_OBSERVATIONS = 250
DEFAULT_MINIMUM_EXPECTED_BREACHES = 10


def required_backtest_observations(
    confidence_level,
    base_minimum_observations=DEFAULT_BASE_MINIMUM_OBSERVATIONS,
    minimum_expected_breaches=DEFAULT_MINIMUM_EXPECTED_BREACHES
):
    """Return the project's minimum sample for asymptotic VaR validation.

    This is an explicit project safeguard, not a theorem that the resulting
    sample makes the asymptotic tests exact or statistically sufficient.
    """
    if isinstance(confidence_level, bool):
        raise ValueError("confidence_level must be between 0 and 1.")
    try:
        confidence_level = float(confidence_level)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "confidence_level must be between 0 and 1."
        ) from error
    if not np.isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")

    thresholds = {
        "base_minimum_observations": base_minimum_observations,
        "minimum_expected_breaches": minimum_expected_breaches
    }
    for threshold_name, threshold in thresholds.items():
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, np.integer))
            or threshold <= 0
        ):
            raise ValueError(
                f"{threshold_name} must be a positive integer."
            )

    expected_breach_rate = Decimal("1") - Decimal(str(confidence_level))
    minimum_from_expected_breaches = int(
        (
            Decimal(int(minimum_expected_breaches))
            / expected_breach_rate
        ).to_integral_value(rounding=ROUND_CEILING)
    )

    return int(max(
        base_minimum_observations,
        minimum_from_expected_breaches
    ))



def backtest_var_threshold(
    portfolio_returns,
    var_threshold,
    expected_breach_rate
):
    backtest = pd.DataFrame({
        "Return": portfolio_returns,
        "VaR": var_threshold
    })

    backtest = backtest.dropna()

    if backtest.empty:
        raise ValueError(
            "VaR backtest has no usable observations after aligning "
            "returns and VaR thresholds."
        )

    backtest["Breach"] = (
        backtest["Return"]
        < backtest["VaR"]
    )

    number_of_breaches = (
        backtest["Breach"].sum()
    )

    number_of_observations = len(
        backtest
    )

    breach_rate = (
        number_of_breaches
        / number_of_observations
    )

    return (
        backtest,
        number_of_breaches,
        breach_rate,
        expected_breach_rate
    )

def backtest_historical_var(
    portfolio_returns,
    window=252,
    confidence_level=0.95
):
    minimum_returns = window + 2
    usable_returns = portfolio_returns.dropna()

    if len(usable_returns) < minimum_returns:
        raise ValueError(
            "Insufficient price history for the Historical VaR backtest: "
            f"at least {minimum_returns} usable daily return observations "
            f"are required for a {window}-day rolling window and two "
            "out-of-sample observations."
        )

    alpha = 1 - confidence_level

    rolling_var = (
        portfolio_returns
        .shift(1)
        .rolling(window=window)
        .quantile(alpha)
    )

    results = backtest_var_threshold(
        portfolio_returns,
        rolling_var,
        expected_breach_rate=alpha
    )

    if len(results[0]) < 2:
        raise ValueError(
            "Insufficient price history for the Historical VaR backtest: "
            f"the {window}-day rolling window must leave at least two "
            "out-of-sample observations."
        )

    return results


def kupiec_test(
    number_of_breaches,
    number_of_observations,
    expected_breach_rate
):
    if number_of_observations <= 0:
        raise ValueError("number_of_observations must be strictly positive.")

    if not 0 <= number_of_breaches <= number_of_observations:
        raise ValueError(
            "number_of_breaches must be between zero and "
            "number_of_observations."
        )

    if not 0 < expected_breach_rate < 1:
        raise ValueError("expected_breach_rate must be between 0 and 1.")

    observed_breach_rate = (
        number_of_breaches
        / number_of_observations
    )

    log_likelihood_expected = (
        xlogy(
            number_of_observations - number_of_breaches,
            1 - expected_breach_rate
        )
        +
        xlogy(number_of_breaches, expected_breach_rate)
    )

    log_likelihood_observed = (
        xlogy(
            number_of_observations - number_of_breaches,
            1 - observed_breach_rate
        )
        +
        xlogy(number_of_breaches, observed_breach_rate)
    )

    lr_statistic = -2 * (
        log_likelihood_expected
        - log_likelihood_observed
    )

    lr_statistic = max(0.0, float(lr_statistic))

    p_value = 1 - chi2.cdf(
        lr_statistic,
        df=1
    )

    return lr_statistic, p_value

def christoffersen_independence_test(
    backtest
):
    breaches = (
        backtest["Breach"]
        .astype(int)
        .to_numpy()
    )

    previous = breaches[:-1]
    current = breaches[1:]

    n00 = np.sum(
        (previous == 0)
        & (current == 0)
    )

    n01 = np.sum(
        (previous == 0)
        & (current == 1)
    )

    n10 = np.sum(
        (previous == 1)
        & (current == 0)
    )

    n11 = np.sum(
        (previous == 1)
        & (current == 1)
    )

    pi_01 = (
        n01 / (n00 + n01)
        if (n00 + n01) > 0
        else 0.0
    )

    pi_11 = (
        n11 / (n10 + n11)
        if (n10 + n11) > 0
        else 0.0
    )

    total_transitions = (
        n00 + n01 + n10 + n11
    )

    pi = (
        (n01 + n11)
        / total_transitions
    )

    log_likelihood_independent = (
        xlogy(n00 + n10, 1 - pi)
        +
        xlogy(n01 + n11, pi)
    )

    log_likelihood_markov = (
        xlogy(n00, 1 - pi_01)
        +
        xlogy(n01, pi_01)
        +
        xlogy(n10, 1 - pi_11)
        +
        xlogy(n11, pi_11)
    )

    lr_independence = -2 * (
        log_likelihood_independent
        - log_likelihood_markov
    )

    lr_independence = max(
        0.0,
        float(lr_independence)
    )

    p_value = 1 - chi2.cdf(
        lr_independence,
        df=1
    )

    return (
        lr_independence,
        p_value,
        n00,
        n01,
        n10,
        n11
    )


def christoffersen_conditional_coverage_test(
    kupiec_statistic,
    independence_statistic
):
    lr_conditional_coverage = (
        kupiec_statistic
        + independence_statistic
    )

    p_value = 1 - chi2.cdf(
        lr_conditional_coverage,
        df=2
    )

    return (
        lr_conditional_coverage,
        p_value
    )


def evaluate_var_backtest(
    backtest,
    expected_breach_rate,
    base_minimum_observations=DEFAULT_BASE_MINIMUM_OBSERVATIONS,
    minimum_expected_breaches=DEFAULT_MINIMUM_EXPECTED_BREACHES
):
    """Calculate asymptotic coverage tests for an adequate backtest sample."""
    if "Breach" not in backtest.columns:
        raise ValueError(
            "Backtest DataFrame must contain a 'Breach' column."
        )

    number_of_observations = len(backtest)

    if not 0 < expected_breach_rate < 1:
        raise ValueError(
            "expected_breach_rate must be between 0 and 1."
        )

    confidence_level = 1 - expected_breach_rate
    minimum_required_observations = required_backtest_observations(
        confidence_level=confidence_level,
        base_minimum_observations=base_minimum_observations,
        minimum_expected_breaches=minimum_expected_breaches
    )
    if number_of_observations < minimum_required_observations:
        raise ValueError(
            "VaR model validation requires at least "
            f"{minimum_required_observations} out-of-sample observations "
            f"at {confidence_level:.0%} confidence under the current "
            f"validation policy; received {number_of_observations}."
        )

    number_of_breaches = int(
        backtest["Breach"].sum()
    )

    breach_rate = (
        number_of_breaches
        / number_of_observations
    )

    kupiec_statistic, kupiec_p_value = kupiec_test(
        number_of_breaches,
        number_of_observations,
        expected_breach_rate
    )

    (
        independence_statistic,
        independence_p_value,
        _,
        _,
        _,
        _
    ) = christoffersen_independence_test(
        backtest
    )

    (
        conditional_coverage_statistic,
        conditional_coverage_p_value
    ) = christoffersen_conditional_coverage_test(
        kupiec_statistic,
        independence_statistic
    )

    return {
        "model_validation_available": True,
        "confidence_level": confidence_level,
        "observations": number_of_observations,
        "validation_observations": number_of_observations,
        "minimum_required_observations": minimum_required_observations,
        "base_minimum_observations_policy": base_minimum_observations,
        "minimum_expected_breaches_policy": minimum_expected_breaches,
        "breaches": number_of_breaches,
        "breach_rate": breach_rate,
        "expected_breach_rate": expected_breach_rate,
        "kupiec_lr_statistic": kupiec_statistic,
        "kupiec_p_value": kupiec_p_value,
        "christoffersen_independence_lr_statistic": independence_statistic,
        "christoffersen_independence_p_value": independence_p_value,
        "conditional_coverage_lr_statistic": conditional_coverage_statistic,
        "conditional_coverage_p_value": conditional_coverage_p_value
    }
