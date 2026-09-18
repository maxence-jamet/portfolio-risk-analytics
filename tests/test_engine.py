import numpy as np
import pandas as pd
import pytest

from src.data import calculate_returns
from src.engine import run_portfolio_analysis


def make_prices(tickers=("AAA",), periods=255):
    dates = pd.bdate_range("2023-01-02", periods=periods)
    return pd.DataFrame(
        {
            ticker: 100.0 * np.cumprod(
                np.full(periods, 1.001)
            )
            for ticker in tickers
        },
        index=dates
    )


def make_variable_prices(periods):
    dates = pd.bdate_range("2022-01-03", periods=periods)
    observations = np.arange(periods - 1)
    returns = np.where(observations % 17 == 0, -0.025, 0.001)
    return pd.DataFrame(
        {
            "AAA": np.concatenate((
                [100.0],
                100.0 * np.cumprod(1.0 + returns)
            ))
        },
        index=dates
    )


def make_portfolio(tickers=("AAA",)):
    return pd.DataFrame({
        "ticker": list(tickers),
        "quantity": [10] * len(tickers),
        "purchase_price": [90.0] * len(tickers)
    })


def make_stress_scenarios(tickers=("AAA",)):
    return pd.DataFrame(
        {
            ticker: [-0.10]
            for ticker in tickers
        },
        index=["Sell-off"]
    )


def test_engine_uses_injected_prices_and_returns_structured_results(
    monkeypatch
):
    dates = pd.bdate_range(
        "2023-01-02",
        periods=321
    )

    asset_a_returns = np.full(320, 0.001)
    asset_b_returns = np.full(320, 0.0005)
    asset_a_returns[::20] = -0.04
    asset_b_returns[::25] = -0.025

    prices = pd.DataFrame(
        {
            "AAA": np.concatenate(
                ([100.0], 100.0 * np.cumprod(1 + asset_a_returns))
            ),
            "BBB": np.concatenate(
                ([80.0], 80.0 * np.cumprod(1 + asset_b_returns))
            )
        },
        index=dates
    )

    portfolio = pd.DataFrame({
        "ticker": ["AAA", "BBB"],
        "quantity": [10, 20],
        "purchase_price": [90.0, 75.0]
    })

    stress_scenarios = pd.DataFrame(
        {
            "AAA": [-0.10, 0.05],
            "BBB": [-0.05, 0.02]
        },
        index=["Sell-off", "Rally"]
    )

    def fail_if_download_is_called(*args, **kwargs):
        raise AssertionError(
            "Injected prices should skip the market-data download."
        )

    monkeypatch.setattr(
        "src.engine.download_price_views",
        fail_if_download_is_called
    )

    results = run_portfolio_analysis(
        portfolio=portfolio,
        stress_scenarios=stress_scenarios,
        prices=prices
    )

    required_results = {
        "prices",
        "valuation_prices",
        "return_prices",
        "returns",
        "positions",
        "portfolio_value",
        "aligned_weights",
        "portfolio_returns",
        "correlation_matrix",
        "annual_volatility",
        "cumulative_performance",
        "drawdown",
        "max_drawdown",
        "historical_var_95",
        "historical_var_99",
        "expected_shortfall_95",
        "expected_shortfall_99",
        "skewness",
        "excess_kurtosis",
        "worst_date",
        "worst_return",
        "annual_covariance_matrix",
        "portfolio_volatility_matrix",
        "risk_contribution_table",
        "parametric_var_95",
        "parametric_var_99",
        "stress_results",
        "rolling_volatility",
        "ewma_daily_volatility",
        "ewma_annualized_volatility",
        "historical_backtest",
        "ewma_backtest",
        "historical_validation",
        "ewma_validation",
        "model_comparison"
    }

    assert required_results.issubset(results)
    pd.testing.assert_frame_equal(
        results["prices"],
        prices
    )
    assert np.isclose(
        results["annual_volatility"],
        results["portfolio_volatility_matrix"]
    )
    assert results["model_comparison"]["observations"].nunique() == 1
    assert results["model_comparison"]["model"].tolist() == [
        "Historical VaR 95%",
        "EWMA VaR 95%"
    ]
    assert results["historical_common_backtest"].index.equals(
        results["ewma_common_backtest"].index
    )
    assert results["stress_test_available"] is True
    assert results["stress_test_reason"] is None
    assert results["stress_results"].index.tolist() == ["Sell-off", "Rally"]


@pytest.mark.parametrize(
    "stress_scenarios",
    [
        None,
        pd.DataFrame(),
        pd.DataFrame({"BBB": [-0.10]}, index=["Sell-off"])
    ]
)
def test_optional_unavailable_stress_does_not_block_core_analysis(
    stress_scenarios
):
    results = run_portfolio_analysis(
        portfolio=make_portfolio(),
        stress_scenarios=stress_scenarios,
        prices=make_variable_prices(periods=321)
    )

    assert np.isfinite(results["annual_volatility"])
    assert results["stress_test_available"] is False
    assert results["stress_results"].empty
    assert results["stress_test_reason"] == (
        "Stress testing unavailable: define shocks for all current holdings."
    )


def test_engine_separates_raw_valuation_from_adjusted_returns():
    dates = pd.bdate_range("2023-01-02", periods=321)
    valuation_prices = pd.DataFrame(
        {
            "AAA": np.linspace(100.0, 200.0, len(dates)),
            "BBB": np.linspace(80.0, 100.0, len(dates))
        },
        index=dates
    )
    observations = np.arange(len(dates) - 1)
    adjusted_a_returns = np.where(observations % 17 == 0, -0.03, 0.001)
    adjusted_b_returns = np.where(observations % 29 == 0, -0.02, 0.0005)
    return_prices = pd.DataFrame(
        {
            "AAA": np.concatenate((
                [90.0],
                90.0 * np.cumprod(1.0 + adjusted_a_returns)
            )),
            "BBB": np.concatenate((
                [70.0],
                70.0 * np.cumprod(1.0 + adjusted_b_returns)
            ))
        },
        index=dates
    )
    original_valuation_prices = valuation_prices.copy(deep=True)
    original_return_prices = return_prices.copy(deep=True)
    portfolio = pd.DataFrame({
        "ticker": ["AAA", "BBB"],
        "quantity": [2.0, 5.0],
        "purchase_price": [90.0, 75.0]
    })

    results = run_portfolio_analysis(
        portfolio=portfolio,
        stress_scenarios=make_stress_scenarios(("AAA", "BBB")),
        valuation_prices=valuation_prices,
        return_prices=return_prices
    )

    assert results["positions"]["current_price"].tolist() == [200.0, 100.0]
    assert results["positions"]["market_value"].tolist() == [400.0, 500.0]
    assert np.allclose(
        results["weights"].to_numpy(),
        [400.0 / 900.0, 500.0 / 900.0]
    )
    pd.testing.assert_frame_equal(
        results["returns"],
        calculate_returns(return_prices)
    )
    assert not results["returns"].equals(
        calculate_returns(valuation_prices)
    )
    pd.testing.assert_frame_equal(
        valuation_prices,
        original_valuation_prices
    )
    pd.testing.assert_frame_equal(return_prices, original_return_prices)


@pytest.mark.parametrize(
    ("portfolio", "message"),
    [
        (
            pd.DataFrame(
                columns=["ticker", "quantity", "purchase_price"]
            ),
            "at least one position"
        ),
        (
            pd.DataFrame({"ticker": ["AAA"], "quantity": [10]}),
            "Missing portfolio columns"
        ),
        (
            pd.DataFrame({
                "ticker": ["AAA", "AAA"],
                "quantity": [10, 20],
                "purchase_price": [90.0, 95.0]
            }),
            "Duplicate tickers"
        ),
        (
            pd.DataFrame({
                "ticker": ["AAA"],
                "quantity": ["ten"],
                "purchase_price": [90.0]
            }),
            "quantities.*numeric"
        ),
        (
            pd.DataFrame({
                "ticker": ["AAA"],
                "quantity": [10],
                "purchase_price": ["ninety"]
            }),
            "purchase prices.*numeric"
        ),
        (
            pd.DataFrame({
                "ticker": ["AAA"],
                "quantity": [0],
                "purchase_price": [90.0]
            }),
            "quantities must be strictly positive"
        ),
        (
            pd.DataFrame({
                "ticker": ["AAA"],
                "quantity": [10],
                "purchase_price": [0]
            }),
            "purchase prices must be strictly positive"
        )
    ]
)
def test_engine_rejects_invalid_portfolios(portfolio, message):
    with pytest.raises(ValueError, match=message):
        run_portfolio_analysis(
            portfolio=portfolio,
            stress_scenarios=make_stress_scenarios(),
            prices=make_prices()
        )


def test_engine_rejects_empty_price_data():
    with pytest.raises(ValueError, match="Price data must not be empty"):
        run_portfolio_analysis(
            portfolio=make_portfolio(),
            stress_scenarios=make_stress_scenarios(),
            prices=pd.DataFrame(columns=["AAA"])
        )


def test_engine_rejects_missing_asset_prices():
    with pytest.raises(ValueError, match="Missing price data.*BBB"):
        run_portfolio_analysis(
            portfolio=make_portfolio(("AAA", "BBB")),
            stress_scenarios=make_stress_scenarios(("AAA", "BBB")),
            prices=make_prices(("AAA",))
        )


def test_engine_rejects_duplicate_price_columns():
    prices = pd.DataFrame(
        np.ones((255, 2)),
        columns=["AAA", "AAA"]
    )

    with pytest.raises(ValueError, match="Duplicate asset columns.*price"):
        run_portfolio_analysis(
            portfolio=make_portfolio(),
            stress_scenarios=make_stress_scenarios(),
            prices=prices
        )


def test_engine_rejects_non_numeric_prices():
    prices = make_prices().astype(str)

    with pytest.raises(ValueError, match="Price data.*numeric"):
        run_portfolio_analysis(
            portfolio=make_portfolio(),
            stress_scenarios=make_stress_scenarios(),
            prices=prices
        )


def test_engine_rejects_prices_without_usable_returns():
    prices = make_prices()
    prices.loc[:, "AAA"] = np.nan

    with pytest.raises(ValueError, match="No usable return observations"):
        run_portfolio_analysis(
            portfolio=make_portfolio(),
            stress_scenarios=make_stress_scenarios(),
            prices=prices
        )


def test_engine_rejects_insufficient_price_history():
    results = run_portfolio_analysis(
        portfolio=make_portfolio(),
        stress_scenarios=make_stress_scenarios(),
        prices=make_variable_prices(periods=254)
    )

    assert np.isfinite(results["historical_var_95"])
    assert results["historical_backtest_available"] is False
    assert "requires at least 254" in results[
        "historical_backtest_reason"
    ]
    assert results["model_validation_available"] is False
    assert "rejected" not in results["model_validation_reason"].lower()


def test_very_short_history_separates_core_and_both_backtest_levels():
    results = run_portfolio_analysis(
        portfolio=make_portfolio(),
        stress_scenarios=make_stress_scenarios(),
        prices=make_variable_prices(periods=20)
    )

    assert np.isfinite(results["historical_var_95"])
    assert results["historical_backtest_available"] is False
    assert results["ewma_backtest_available"] is False
    assert "requires at least 31" in results["ewma_backtest_reason"]
    assert results["model_validation_available"] is False


def test_historical_validation_uses_out_of_sample_not_total_observations():
    results = run_portfolio_analysis(
        portfolio=make_portfolio(),
        stress_scenarios=make_stress_scenarios(),
        prices=make_variable_prices(periods=502)
    )

    assert len(results["returns"]) == 501
    assert len(results["historical_backtest"]) == 249
    assert results["historical_validation"][
        "model_validation_available"
    ] is False
    assert results["historical_validation"][
        "validation_observations"
    ] == 249
    assert results["historical_validation"][
        "minimum_required_observations"
    ] == 250


def test_historical_and_common_validation_pass_at_250_observations():
    results = run_portfolio_analysis(
        portfolio=make_portfolio(),
        stress_scenarios=make_stress_scenarios(),
        prices=make_variable_prices(periods=503)
    )

    assert len(results["historical_backtest"]) == 250
    assert results["historical_validation"][
        "model_validation_available"
    ] is True
    assert results["model_validation_available"] is True
    assert results["validation_observations"] == 250
    assert "kupiec_p_value" in results["model_comparison"].columns


def test_ewma_validation_uses_same_sample_policy():
    results = run_portfolio_analysis(
        portfolio=make_portfolio(),
        stress_scenarios=make_stress_scenarios(),
        prices=make_variable_prices(periods=255)
    )

    assert results["ewma_validation"][
        "model_validation_available"
    ] is False
    assert results["ewma_validation"][
        "validation_observations"
    ] < 250
    assert results["ewma_validation"][
        "minimum_required_observations"
    ] == 250


def test_common_period_must_independently_meet_validation_policy():
    results = run_portfolio_analysis(
        portfolio=make_portfolio(),
        stress_scenarios=make_stress_scenarios(),
        prices=make_variable_prices(periods=502)
    )

    assert results["ewma_validation"][
        "model_validation_available"
    ] is True
    assert results["model_validation_available"] is False
    assert results["validation_observations"] == 249
    assert "kupiec_p_value" not in results["model_comparison"].columns
    assert "rejected" not in results["model_validation_reason"].lower()
    assert "requires more history" not in results[
        "model_validation_reason"
    ].lower()
    assert "requires at least 250" in results[
        "model_validation_reason"
    ]
    assert np.isfinite(results["historical_var_95"])


@pytest.mark.parametrize(
    ("stress_scenarios", "message"),
    [
        (
            pd.DataFrame({"AAA": ["large"]}, index=["Sell-off"]),
            "shocks.*numeric"
        ),
        (
            pd.DataFrame(
                [[-0.10, -0.20]],
                columns=["AAA", "AAA"],
                index=["Sell-off"]
            ),
            "Duplicate assets"
        )
    ]
)
def test_engine_rejects_invalid_stress_scenarios(
    stress_scenarios,
    message
):
    with pytest.raises(ValueError, match=message):
        run_portfolio_analysis(
            portfolio=make_portfolio(),
            stress_scenarios=stress_scenarios,
            prices=make_prices()
        )
