import numpy as np
import pandas as pd
import pytest

from src.data import calculate_returns
from src.investor import run_investor_performance_analysis
from src.portfolio import calculate_portfolio_returns
from src.risk import diversification_ratio
from src.unified import run_unified_portfolio_analysis


def make_unified_case(periods=700):
    dates = pd.bdate_range("2022-01-03", periods=periods)
    observations = np.arange(periods - 1)
    aaa_returns = np.where(observations % 19 == 0, -0.025, 0.001)
    bbb_returns = np.where(observations % 23 == 0, -0.018, 0.0007)
    spy_returns = np.where(observations % 29 == 0, -0.015, 0.0006)
    prices = pd.DataFrame(
        {
            "AAA": np.concatenate(
                ([100.0], 100.0 * np.cumprod(1.0 + aaa_returns))
            ),
            "BBB": np.concatenate(
                ([80.0], 80.0 * np.cumprod(1.0 + bbb_returns))
            ),
            "SPY": np.concatenate(
                ([120.0], 120.0 * np.cumprod(1.0 + spy_returns))
            )
        },
        index=dates
    )
    transactions = pd.DataFrame({
        "date": [dates[300], dates[310], dates[330], dates[350]],
        "ticker": ["AAA", "BBB", "AAA", "BBB"],
        "side": ["BUY", "BUY", "BUY", "SELL"],
        "quantity": [10, 5, 5, 5],
        "price": [100.0, 80.0, 120.0, 90.0]
    })
    cash_flows = pd.DataFrame({
        "date": [dates[295]],
        "type": ["DEPOSIT"],
        "amount": [5000.0]
    })
    stress_scenarios = pd.DataFrame(
        {
            "AAA": [-0.10, 0.05],
            "BBB": [-0.06, 0.03]
        },
        index=["Sell-off", "Rally"]
    )

    return dates, prices, transactions, cash_flows, stress_scenarios


def run_case(**overrides):
    dates, prices, transactions, cash_flows, stress_scenarios = (
        make_unified_case()
    )
    arguments = {
        "transactions": transactions,
        "cash_flows": cash_flows,
        "stress_scenarios": stress_scenarios,
        "prices": prices,
        "risk_start_date": dates[0]
    }
    arguments.update(overrides)
    return run_unified_portfolio_analysis(**arguments)


def test_transactions_feed_investor_and_current_risk_results():
    results = run_case()

    assert set(results) == {
        "investor",
        "risk",
        "account_risk",
        "diversification",
        "benchmark",
        "current_portfolio",
        "metadata"
    }
    assert np.isfinite(results["investor"]["total_twr"])
    assert "daily_portfolio_value" in results["investor"]
    assert "investor_wealth_index" in results["investor"]
    assert "investor_drawdown" in results["investor"]
    assert "investor_max_drawdown" in results["investor"]
    assert "investor_drawdown" not in results["risk"]
    assert "investor_max_drawdown" not in results["account_risk"]
    assert "historical_var_95" in results["risk"]
    assert "stress_results" in results["risk"]
    assert results["risk"]["stress_test_available"] is True
    assert results["risk"]["stress_test_reason"] is None
    assert "risk_contribution_table" in results["risk"]
    assert results["diversification"] is results["risk"]["diversification"]
    assert set(results["diversification"]) == {
        "weight_hhi",
        "effective_number_of_holdings",
        "diversification_ratio",
        "risk_concentration_hhi",
        "effective_risk_contributors"
    }
    assert "model_comparison" in results["risk"]
    assert results["risk"]["model_validation_available"] is True
    assert results["risk"]["validation_observations"] == 447
    assert results["risk"]["minimum_required_observations"] == 250
    assert results["benchmark"]["ticker"] == "SPY"
    assert results["benchmark"]["price_basis"] == (
        "adjusted_close_total_return_style"
    )
    assert results["benchmark"]["observations"] > 0

    expected_portfolio = pd.DataFrame({
        "ticker": ["AAA"],
        "quantity": [15.0],
        "purchase_price": [1600.0 / 15.0]
    })
    pd.testing.assert_frame_equal(
        results["current_portfolio"],
        expected_portfolio
    )
    assert results["risk"]["positions"]["ticker"].tolist() == ["AAA"]


def test_unified_accepts_one_normalized_ledger_without_separate_cash_flows():
    dates, prices, transactions, cash_flows, stress_scenarios = (
        make_unified_case()
    )
    trades = transactions.rename(columns={"side": "type"}).assign(
        amount=np.nan,
        fees=0.0,
        taxes=0.0
    )
    flows = cash_flows.assign(
        ticker=np.nan,
        quantity=np.nan,
        price=np.nan,
        fees=0.0,
        taxes=0.0
    )
    ledger = pd.concat([flows, trades], ignore_index=True, sort=False)
    original_ledger = ledger.copy(deep=True)

    results = run_unified_portfolio_analysis(
        transactions=ledger,
        stress_scenarios=stress_scenarios,
        prices=prices,
        risk_start_date=dates[0]
    )

    assert results["current_portfolio"]["ticker"].tolist() == ["AAA"]
    assert np.isclose(results["investor"]["current_cash_balance"], 3450.0)
    pd.testing.assert_frame_equal(ledger, original_ledger)


def test_diversification_uses_the_invested_risk_sample_and_covariance():
    results = run_case()
    risk = results["risk"]
    expected_covariance = risk["returns"].cov() * 252

    pd.testing.assert_frame_equal(
        risk["annual_covariance_matrix"],
        expected_covariance
    )
    assert np.isclose(
        results["diversification"]["diversification_ratio"],
        diversification_ratio(expected_covariance, risk["aligned_weights"])
    )
    assert risk["returns"].index.equals(risk["portfolio_returns"].index)
    assert len(risk["returns"]) == results["metadata"]["risk_observations"]


def test_cash_balance_does_not_change_security_sleeve_diversification():
    dates, prices, transactions, cash_flows, stress_scenarios = (
        make_unified_case()
    )
    more_cash_flows = cash_flows.copy(deep=True)
    more_cash_flows.loc[0, "amount"] += 1000.0

    baseline = run_unified_portfolio_analysis(
        transactions=transactions,
        cash_flows=cash_flows,
        stress_scenarios=stress_scenarios,
        prices=prices,
        risk_start_date=dates[0]
    )
    more_cash = run_unified_portfolio_analysis(
        transactions=transactions,
        cash_flows=more_cash_flows,
        stress_scenarios=stress_scenarios,
        prices=prices,
        risk_start_date=dates[0]
    )

    assert more_cash["investor"]["current_cash_balance"] > (
        baseline["investor"]["current_cash_balance"]
    )
    assert more_cash["metadata"]["cash_weight"] > (
        baseline["metadata"]["cash_weight"]
    )
    assert more_cash["diversification"] == baseline["diversification"]


def test_risk_free_rate_updates_performance_but_not_risk_results():
    baseline = run_case(risk_free_rate_annual=0.0)
    configured = run_case(risk_free_rate_annual=0.03)

    assert configured["investor"]["risk_free_rate_annual"] == 0.03
    assert configured["investor"]["performance_observations"] == len(
        configured["investor"]["daily_twr"].dropna()
    )
    assert not np.isclose(
        configured["investor"]["sharpe_ratio"],
        baseline["investor"]["sharpe_ratio"]
    )
    assert not np.isclose(
        configured["investor"]["sortino_ratio"],
        baseline["investor"]["sortino_ratio"]
    )
    assert not np.isclose(
        configured["benchmark"]["alpha_annualized"],
        baseline["benchmark"]["alpha_annualized"]
    )
    pd.testing.assert_series_equal(
        configured["risk"]["portfolio_returns"],
        baseline["risk"]["portfolio_returns"]
    )
    pd.testing.assert_series_equal(
        configured["account_risk"]["portfolio_returns"],
        baseline["account_risk"]["portfolio_returns"]
    )


@pytest.mark.parametrize(
    ("window_label", "years"),
    [("1 year", 1), ("3 years", 3)]
)
def test_risk_window_resolves_from_supplied_market_endpoint(
    window_label,
    years
):
    dates, prices, transactions, cash_flows, stress_scenarios = (
        make_unified_case(periods=1000)
    )

    results = run_unified_portfolio_analysis(
        transactions=transactions,
        cash_flows=cash_flows,
        stress_scenarios=stress_scenarios,
        prices=prices,
        risk_window_label=window_label
    )

    expected_requested_start = dates[-1] - pd.DateOffset(years=years)
    metadata = results["metadata"]
    assert metadata["risk_reference_date"] == dates[-1]
    assert metadata["risk_requested_start_date"] == expected_requested_start
    assert metadata["risk_window_label"] == window_label
    assert metadata["risk_start_date"] > expected_requested_start
    assert metadata["risk_end_date"] == dates[-1]


def test_maximum_available_uses_earliest_relevant_supplied_price_date():
    dates, prices, transactions, cash_flows, stress_scenarios = (
        make_unified_case()
    )

    results = run_unified_portfolio_analysis(
        transactions=transactions,
        cash_flows=cash_flows,
        stress_scenarios=stress_scenarios,
        prices=prices,
        risk_window_label="Maximum available"
    )

    metadata = results["metadata"]
    assert metadata["earliest_available_return_price_date"] == dates[0]
    assert metadata["risk_requested_start_date"] == dates[0]
    assert metadata["risk_start_date"] == dates[1]


def test_unified_exposes_shared_constant_weight_risk_sample_metadata():
    results = run_case()
    metadata = results["metadata"]

    assert metadata["constant_weight_assumption"] is True
    assert metadata["constant_weight_methodology"] == (
        "Historical risk uses today's portfolio weights held constant "
        "through the selected risk window (equivalent to daily rebalancing)."
    )
    assert metadata["risk_observations"] == len(
        results["risk"]["portfolio_returns"]
    )
    assert metadata["risk_observations"] == len(
        results["account_risk"]["portfolio_returns"]
    )
    assert results["risk"]["portfolio_returns"].index.equals(
        results["account_risk"]["portfolio_returns"].index
    )


def test_unified_reports_missing_return_alignment_without_mutation():
    dates, prices, transactions, cash_flows, stress_scenarios = (
        make_unified_case()
    )
    transactions = transactions.iloc[:3].copy()
    prices = prices.copy(deep=True)
    prices.loc[dates[100], "BBB"] = np.nan
    original_prices = prices.copy(deep=True)

    results = run_unified_portfolio_analysis(
        transactions=transactions,
        cash_flows=cash_flows,
        stress_scenarios=stress_scenarios,
        prices=prices,
        risk_start_date=dates[0]
    )

    metadata = results["metadata"]
    assert metadata["raw_price_observations"] == 700
    assert metadata["asset_return_observations_before_alignment"] == 699
    assert metadata["aligned_portfolio_return_observations"] == 697
    assert metadata["observations_dropped_during_alignment"] == 2
    assert metadata["usable_return_observations_by_asset"] == {
        "AAA": 699,
        "BBB": 697
    }
    assert dates[100] not in results["risk"]["returns"].index
    assert dates[101] not in results["risk"]["returns"].index
    assert not (results["risk"]["returns"] == 0.0).any().any()
    pd.testing.assert_frame_equal(prices, original_prices)


def test_cash_is_separate_from_invested_security_risk():
    results = run_case()

    investor = results["investor"]
    risk = results["risk"]
    assert np.isclose(investor["current_cash_balance"], 3450.0)
    assert np.isclose(
        investor["current_portfolio_value"],
        investor["current_security_value"]
        + investor["current_cash_balance"]
    )
    assert np.isclose(
        risk["portfolio_value"],
        investor["current_security_value"]
    )
    assert results["metadata"]["risk_scope"] == "invested_securities_only"
    assert results["metadata"]["risk_value_basis"] == (
        "current_security_market_value"
    )
    assert results["metadata"]["default_risk_view"] == "account"
    assert results["metadata"]["available_risk_views"] == [
        "account",
        "invested_securities"
    ]
    invested_ratio = (
        investor["current_security_value"]
        / investor["current_portfolio_value"]
    )
    assert np.isclose(results["metadata"]["invested_ratio"], invested_ratio)
    assert np.isclose(
        results["metadata"]["cash_weight"],
        investor["current_cash_balance"]
        / investor["current_portfolio_value"]
    )
    pd.testing.assert_series_equal(
        results["account_risk"]["portfolio_returns"],
        (risk["portfolio_returns"] * invested_ratio).rename("account_return")
    )


def test_no_cash_makes_account_and_security_risk_equal():
    dates, prices, transactions, _, stress_scenarios = make_unified_case()
    open_transactions = transactions.iloc[:3].copy()
    cash_flows = pd.DataFrame({
        "date": [dates[295]],
        "type": ["DEPOSIT"],
        "amount": [2000.0]
    })

    results = run_unified_portfolio_analysis(
        transactions=open_transactions,
        cash_flows=cash_flows,
        stress_scenarios=stress_scenarios,
        prices=prices,
        risk_start_date=dates[0]
    )

    assert np.isclose(results["investor"]["current_cash_balance"], 0.0)
    assert results["metadata"]["invested_ratio"] == 1.0
    assert results["metadata"]["cash_weight"] == 0.0
    for metric in (
        "annual_volatility",
        "historical_var_95",
        "historical_var_99",
        "expected_shortfall_95",
        "expected_shortfall_99",
        "parametric_var_95",
        "parametric_var_99",
        "max_drawdown"
    ):
        assert np.isclose(
            results["account_risk"][metric],
            results["risk"][metric]
        )


def test_accounting_and_pnl_identities_reconcile():
    dates, _, _, cash_flows, _ = make_unified_case()
    cash_flows = pd.concat(
        [
            cash_flows,
            pd.DataFrame({
                "date": [dates[400]],
                "type": ["WITHDRAWAL"],
                "amount": [200.0]
            })
        ],
        ignore_index=True
    )
    results = run_case(cash_flows=cash_flows)
    investor = results["investor"]
    risk = results["risk"]

    pd.testing.assert_series_equal(
        investor["daily_portfolio_value"],
        (
            investor["daily_security_value"]
            + investor["daily_cash"]
        ).rename("portfolio_value")
    )

    realized_pnl = investor["holdings"]["realized_pnl"].sum()
    unrealized_pnl = risk["positions"]["unrealized_pnl"].sum()
    total_pnl = realized_pnl + unrealized_pnl
    signed_external_flows = cash_flows["amount"].where(
        cash_flows["type"].eq("DEPOSIT"),
        -cash_flows["amount"]
    )
    net_external_contributions = signed_external_flows.sum()

    assert np.isclose(
        total_pnl,
        investor["current_portfolio_value"] - net_external_contributions
    )


def test_closed_positions_are_excluded_from_current_risk():
    results = run_case()

    assert results["investor"]["holdings"]["ticker"].tolist() == [
        "AAA",
        "BBB"
    ]
    assert results["metadata"]["transaction_tickers"] == ["AAA", "BBB"]
    assert results["metadata"]["current_risk_tickers"] == ["AAA"]
    assert results["risk"]["prices"].columns.tolist() == ["AAA"]
    assert results["risk"]["weights"].index.tolist() == ["AAA"]


def test_prepared_prices_skip_all_live_downloads(monkeypatch):
    def fail_if_download_is_called(*args, **kwargs):
        raise AssertionError("Prepared prices must prevent live downloads.")

    monkeypatch.setattr(
        "src.unified.download_price_views",
        fail_if_download_is_called
    )
    monkeypatch.setattr(
        "src.investor.download_price_views",
        fail_if_download_is_called
    )
    monkeypatch.setattr(
        "src.engine.download_price_views",
        fail_if_download_is_called
    )

    results = run_case()

    assert results["metadata"]["price_source"] == "injected"


def test_shared_download_uses_all_tickers_and_earliest_start(monkeypatch):
    dates, prepared_prices, transactions, cash_flows, stress_scenarios = (
        make_unified_case()
    )
    download_calls = []

    def fake_download(tickers, start_date):
        download_calls.append((tickers, start_date))
        return {
            "valuation_prices": prepared_prices,
            "return_prices": prepared_prices
        }

    monkeypatch.setattr("src.unified.download_price_views", fake_download)

    results = run_unified_portfolio_analysis(
        transactions=transactions,
        cash_flows=cash_flows,
        stress_scenarios=stress_scenarios,
        risk_start_date="2023-06-01"
    )

    assert download_calls == [(["AAA", "BBB", "SPY"], "2023-02-20")]
    assert results["metadata"]["price_source"] == "yahoo_finance"
    assert results["investor"]["prices"].index.min() == dates[295]
    assert results["risk"]["prices"].index.min() >= pd.Timestamp(
        "2023-06-01"
    )


def test_unified_workflow_keeps_price_channels_separate_without_mutation():
    dates, valuation_prices, transactions, cash_flows, stress_scenarios = (
        make_unified_case()
    )
    transactions = transactions.iloc[:3].copy()
    observations = np.arange(len(dates) - 1)
    return_prices = pd.DataFrame(
        {
            "AAA": np.concatenate((
                [75.0],
                75.0 * np.cumprod(
                    1.0 + np.where(observations % 13 == 0, -0.04, 0.0015)
                )
            )),
            "BBB": np.concatenate((
                [60.0],
                60.0 * np.cumprod(
                    1.0 + np.where(observations % 31 == 0, -0.02, 0.0004)
                )
            )),
            "SPY": np.concatenate((
                [110.0],
                110.0 * np.cumprod(
                    1.0 + np.where(observations % 27 == 0, -0.015, 0.0008)
                )
            ))
        },
        index=dates
    )
    originals = [
        frame.copy(deep=True)
        for frame in (
            transactions,
            cash_flows,
            stress_scenarios,
            valuation_prices,
            return_prices
        )
    ]

    results = run_unified_portfolio_analysis(
        transactions=transactions,
        cash_flows=cash_flows,
        stress_scenarios=stress_scenarios,
        valuation_prices=valuation_prices,
        return_prices=return_prices,
        risk_start_date=dates[0]
    )
    expected_investor = run_investor_performance_analysis(
        transactions=transactions,
        external_cash_flows=cash_flows,
        valuation_prices=valuation_prices.loc[dates[295]:]
    )
    adjusted_path_investor = run_investor_performance_analysis(
        transactions=transactions,
        external_cash_flows=cash_flows,
        valuation_prices=return_prices.loc[dates[295]:]
    )
    expected_asset_returns = calculate_returns(return_prices[["AAA", "BBB"]])
    expected_portfolio_returns, _ = calculate_portfolio_returns(
        expected_asset_returns,
        results["risk"]["weights"]
    )

    pd.testing.assert_series_equal(
        results["investor"]["daily_portfolio_value"],
        expected_investor["daily_portfolio_value"]
    )
    assert np.isclose(
        results["investor"]["total_twr"],
        expected_investor["total_twr"]
    )
    assert not results["investor"]["daily_portfolio_value"].equals(
        adjusted_path_investor["daily_portfolio_value"]
    )
    assert not np.isclose(
        results["investor"]["total_twr"],
        adjusted_path_investor["total_twr"]
    )
    pd.testing.assert_frame_equal(
        results["risk"]["returns"],
        expected_asset_returns
    )
    pd.testing.assert_series_equal(
        results["risk"]["portfolio_returns"],
        expected_portfolio_returns
    )
    latest_raw_values = pd.Series({
        "AAA": 15.0 * valuation_prices.iloc[-1]["AAA"],
        "BBB": 5.0 * valuation_prices.iloc[-1]["BBB"]
    })
    expected_weights = latest_raw_values / latest_raw_values.sum()
    expected_weights.index.name = "ticker"
    expected_weights.name = "weight"
    pd.testing.assert_series_equal(results["risk"]["weights"], expected_weights)
    assert results["metadata"]["valuation_price_basis"] == "raw_close"
    assert results["metadata"]["return_price_basis"] == "adjusted_close"
    assert results["metadata"]["investor_income_scope"] == (
        "ledger_income_and_price_return"
    )

    for frame, original in zip(
        (
            transactions,
            cash_flows,
            stress_scenarios,
            valuation_prices,
            return_prices
        ),
        originals
    ):
        pd.testing.assert_frame_equal(frame, original)


def test_benchmark_uses_adjusted_return_prices_not_raw_close():
    dates, valuation_prices, transactions, cash_flows, stress_scenarios = (
        make_unified_case()
    )
    return_prices = valuation_prices.copy(deep=True)
    observations = np.arange(len(dates) - 1)
    return_prices["SPY"] = np.concatenate((
        [90.0],
        90.0 * np.cumprod(
            1.0 + np.where(observations % 11 == 0, -0.03, 0.0015)
        )
    ))

    results = run_unified_portfolio_analysis(
        transactions=transactions,
        cash_flows=cash_flows,
        stress_scenarios=stress_scenarios,
        valuation_prices=valuation_prices,
        return_prices=return_prices,
        risk_start_date=dates[0],
        benchmark_ticker=" spy "
    )

    expected_raw_returns = (
        valuation_prices["SPY"]
        .pct_change(fill_method=None)
        .rename("benchmark_return")
    )
    adjusted_returns = (
        return_prices["SPY"]
        .pct_change(fill_method=None)
        .rename("benchmark_return")
    )
    pd.testing.assert_series_equal(
        results["benchmark"]["benchmark_returns"],
        adjusted_returns
    )
    assert not results["benchmark"]["benchmark_returns"].equals(
        expected_raw_returns
    )
    assert results["benchmark"]["ticker"] == "SPY"
    assert results["metadata"]["benchmark_price_basis"] == (
        "adjusted_close_total_return_style"
    )


def test_benchmark_ticker_does_not_change_investor_or_risk_analytics():
    dates, prices, transactions, cash_flows, stress_scenarios = (
        make_unified_case()
    )
    observations = np.arange(len(dates) - 1)
    prices = prices.assign(
        QQQ=np.concatenate((
            [130.0],
            130.0 * np.cumprod(
                1.0 + np.where(observations % 17 == 0, -0.02, 0.0009)
            )
        ))
    )

    spy_results = run_unified_portfolio_analysis(
        transactions=transactions,
        cash_flows=cash_flows,
        stress_scenarios=stress_scenarios,
        prices=prices,
        risk_start_date=dates[0],
        benchmark_ticker="SPY"
    )
    qqq_results = run_unified_portfolio_analysis(
        transactions=transactions,
        cash_flows=cash_flows,
        stress_scenarios=stress_scenarios,
        prices=prices,
        risk_start_date=dates[0],
        benchmark_ticker="QQQ"
    )

    assert spy_results["benchmark"]["ticker"] == "SPY"
    assert qqq_results["benchmark"]["ticker"] == "QQQ"
    assert "SPY" not in spy_results["risk"]["weights"].index
    assert "QQQ" not in qqq_results["risk"]["weights"].index
    pd.testing.assert_series_equal(
        spy_results["investor"]["daily_twr"],
        qqq_results["investor"]["daily_twr"]
    )
    pd.testing.assert_series_equal(
        spy_results["risk"]["portfolio_returns"],
        qqq_results["risk"]["portfolio_returns"]
    )
    pd.testing.assert_series_equal(
        spy_results["account_risk"]["portfolio_returns"],
        qqq_results["account_risk"]["portfolio_returns"]
    )
    pd.testing.assert_frame_equal(
        spy_results["risk"]["annual_covariance_matrix"],
        qqq_results["risk"]["annual_covariance_matrix"]
    )
    assert spy_results["diversification"] == qqq_results["diversification"]


def test_empty_benchmark_ticker_is_rejected():
    with pytest.raises(ValueError, match="benchmark_ticker must not be empty"):
        run_case(benchmark_ticker="   ")


def test_caller_dataframes_are_not_mutated():
    dates, prices, transactions, cash_flows, stress_scenarios = (
        make_unified_case()
    )
    originals = [
        frame.copy(deep=True)
        for frame in (transactions, cash_flows, stress_scenarios, prices)
    ]

    run_unified_portfolio_analysis(
        transactions=transactions,
        cash_flows=cash_flows,
        stress_scenarios=stress_scenarios,
        prices=prices,
        risk_start_date=dates[0]
    )

    for frame, original in zip(
        (transactions, cash_flows, stress_scenarios, prices),
        originals
    ):
        pd.testing.assert_frame_equal(frame, original)


def test_unified_exposes_freshness_metadata_and_position_price_dates():
    dates, valuation_prices, transactions, cash_flows, stress_scenarios = (
        make_unified_case()
    )
    transactions = transactions.iloc[:3].copy()
    return_prices = valuation_prices.copy(deep=True)
    valuation_prices = valuation_prices.copy(deep=True)
    valuation_prices.loc[dates[-1], "AAA"] = np.nan
    original_valuation_prices = valuation_prices.copy(deep=True)
    original_return_prices = return_prices.copy(deep=True)

    results = run_unified_portfolio_analysis(
        transactions=transactions,
        cash_flows=cash_flows,
        stress_scenarios=stress_scenarios,
        valuation_prices=valuation_prices,
        return_prices=return_prices,
        risk_start_date=dates[0]
    )

    expected_price_dates = {
        "AAA": dates[-2],
        "BBB": dates[-1]
    }
    assert results["metadata"]["valuation_reference_date"] == dates[-1]
    assert results["metadata"]["valuation_price_dates"] == (
        expected_price_dates
    )
    assert results["metadata"]["valuation_price_age_business_days"] == {
        "AAA": 1,
        "BBB": 0
    }
    assert results["metadata"]["max_price_staleness_business_days"] == 3
    position_dates = (
        results["risk"]["positions"]
        .set_index("ticker")["price_as_of"]
        .to_dict()
    )
    assert position_dates == expected_price_dates
    assert results["risk"]["positions"]["current_price"].tolist() == [
        valuation_prices.loc[dates[-2], "AAA"],
        valuation_prices.loc[dates[-1], "BBB"]
    ]
    pd.testing.assert_frame_equal(
        valuation_prices,
        original_valuation_prices
    )
    pd.testing.assert_frame_equal(return_prices, original_return_prices)


def test_unified_rejects_stale_current_valuation_price():
    dates, valuation_prices, transactions, cash_flows, stress_scenarios = (
        make_unified_case()
    )
    return_prices = valuation_prices.copy(deep=True)
    valuation_prices = valuation_prices.copy(deep=True)
    valuation_prices.loc[dates[-4]:, "AAA"] = np.nan

    with pytest.raises(
        ValueError,
        match=(
            "AAA valuation price is stale: last price .*age 4 business "
            "days, maximum allowed 3"
        )
    ):
        run_unified_portfolio_analysis(
            transactions=transactions,
            cash_flows=cash_flows,
            stress_scenarios=stress_scenarios,
            valuation_prices=valuation_prices,
            return_prices=return_prices,
            risk_start_date=dates[0]
        )

    configured_results = run_unified_portfolio_analysis(
        transactions=transactions,
        cash_flows=cash_flows,
        stress_scenarios=stress_scenarios,
        valuation_prices=valuation_prices,
        return_prices=return_prices,
        risk_start_date=dates[0],
        max_price_staleness_business_days=4
    )
    assert configured_results["metadata"][
        "valuation_price_age_business_days"
    ]["AAA"] == 4
    assert configured_results["metadata"][
        "max_price_staleness_business_days"
    ] == 4


@pytest.mark.parametrize(
    "stress_scenarios",
    [
        None,
        pd.DataFrame(),
        pd.DataFrame({"SPY": [-0.10]}, index=["Sell-off"])
    ]
)
def test_optional_or_incomplete_stress_does_not_block_unified_analysis(
    stress_scenarios
):
    results = run_case(stress_scenarios=stress_scenarios)

    assert np.isfinite(results["risk"]["annual_volatility"])
    assert results["risk"]["stress_test_available"] is False
    assert results["risk"]["stress_results"].empty
    assert results["risk"]["stress_test_reason"] == (
        "Stress testing unavailable: define shocks for all current holdings."
    )


def test_no_open_securities_raises_clear_current_risk_error():
    dates, prices, _, cash_flows, stress_scenarios = make_unified_case()
    closed_transactions = pd.DataFrame({
        "date": [dates[300], dates[301]],
        "ticker": ["AAA", "AAA"],
        "side": ["BUY", "SELL"],
        "quantity": [10, 10],
        "price": [100.0, 110.0]
    })

    with pytest.raises(
        ValueError,
        match=(
            "Current-risk analysis requires at least one open security"
            ".*Cash-only"
        )
    ):
        run_unified_portfolio_analysis(
            transactions=closed_transactions,
            cash_flows=cash_flows,
            stress_scenarios=stress_scenarios,
            prices=prices,
            risk_start_date=dates[0]
        )


def test_short_risk_history_keeps_core_risk_but_not_rolling_validation():
    dates, prices, transactions, cash_flows, stress_scenarios = (
        make_unified_case(periods=553)
    )
    late_risk_start = dates[-254]

    results = run_unified_portfolio_analysis(
        transactions=transactions,
        cash_flows=cash_flows,
        stress_scenarios=stress_scenarios,
        prices=prices,
        risk_start_date=late_risk_start
    )

    assert np.isfinite(results["risk"]["historical_var_95"])
    assert results["risk"]["historical_backtest_available"] is False
    assert "received 253" in results["risk"][
        "historical_backtest_reason"
    ]
    assert results["risk"]["model_validation_available"] is False


def test_missing_transaction_price_and_late_events_raise_clear_errors():
    dates, prices, transactions, cash_flows, stress_scenarios = (
        make_unified_case()
    )

    with pytest.raises(ValueError, match="Missing price data.*BBB"):
        run_unified_portfolio_analysis(
            transactions=transactions,
            cash_flows=cash_flows,
            stress_scenarios=stress_scenarios,
            prices=prices[["AAA"]],
            risk_start_date=dates[0]
        )

    late_transactions = transactions.copy()
    late_transactions.loc[0, "date"] = dates[-1] + pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="must not be later than.*price date"):
        run_unified_portfolio_analysis(
            transactions=late_transactions,
            cash_flows=cash_flows,
            stress_scenarios=stress_scenarios,
            prices=prices,
            risk_start_date=dates[0]
        )
