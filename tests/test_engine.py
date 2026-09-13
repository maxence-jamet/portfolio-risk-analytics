import numpy as np
import pandas as pd
import pytest

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
        "src.engine.download_prices",
        fail_if_download_is_called
    )

    results = run_portfolio_analysis(
        portfolio=portfolio,
        stress_scenarios=stress_scenarios,
        prices=prices
    )

    required_results = {
        "prices",
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
    with pytest.raises(ValueError, match="Insufficient price history.*252"):
        run_portfolio_analysis(
            portfolio=make_portfolio(),
            stress_scenarios=make_stress_scenarios(),
            prices=make_prices(periods=254)
        )


@pytest.mark.parametrize(
    ("stress_scenarios", "message"),
    [
        (pd.DataFrame(columns=["AAA"]), "at least one scenario"),
        (
            pd.DataFrame({"BBB": [-0.10]}, index=["Sell-off"]),
            "missing assets.*AAA"
        ),
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
