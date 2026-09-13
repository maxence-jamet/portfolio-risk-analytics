import pandas as pd


def build_risk_summary(
    portfolio_value,
    annual_volatility,
    max_drawdown,
    var_95,
    var_99,
    es_95,
    es_99,
    parametric_var_95,
    parametric_var_99,
    skewness,
    excess_kurtosis,
    backtest_observations,
    backtest_breaches,
    breach_rate,
    expected_breach_rate,
    kupiec_p_value,
    christoffersen_p_value,
    conditional_p_value
):
    risk_summary = pd.DataFrame([{
        "portfolio_value": portfolio_value,
        "annual_volatility": annual_volatility,
        "max_drawdown": max_drawdown,
        "historical_var_95": var_95,
        "historical_var_99": var_99,
        "expected_shortfall_95": es_95,
        "expected_shortfall_99": es_99,
        "parametric_var_95": parametric_var_95,
        "parametric_var_99": parametric_var_99,
        "skewness": skewness,
        "excess_kurtosis": excess_kurtosis,
        "backtest_observations": backtest_observations,
        "backtest_breaches": backtest_breaches,
        "breach_rate": breach_rate,
        "expected_breach_rate": expected_breach_rate,
        "kupiec_p_value": kupiec_p_value,
        "christoffersen_p_value": christoffersen_p_value,
        "conditional_coverage_p_value": conditional_p_value
    }])

    return risk_summary


def export_results(
    positions,
    risk_summary,
    risk_contribution_table,
    stress_results,
    backtest,
    output_dir
):
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    positions.to_csv(
        output_dir / "positions.csv",
        index=False
    )

    risk_summary.to_csv(
        output_dir / "risk_summary.csv",
        index=False
    )

    risk_contribution_table.to_csv(
        output_dir / "risk_contributions.csv"
    )

    stress_results.to_csv(
        output_dir / "stress_results.csv"
    )

    backtest.to_csv(
        output_dir / "var_backtest_data.csv",
        index_label="Date"
    )