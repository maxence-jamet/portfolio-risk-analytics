from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest


APP_FILE = Path(__file__).resolve().parents[1] / "app.py"


def make_dashboard_results(validation_available=True):
    dates = pd.bdate_range("2024-01-02", periods=3)
    investor = {
        "current_portfolio_value": 100.0,
        "current_security_value": 75.0,
        "current_cash_balance": 25.0,
        "dividend_income": 20.0,
        "interest_income": 3.0,
        "fees_paid": 4.0,
        "taxes_paid": 6.0,
        "net_external_contributions": 70.0,
        "economic_total_pnl": 30.0,
        "total_twr": 0.08,
        "annualized_twr": 0.12,
        "sharpe_ratio": 0.85,
        "sortino_ratio": 1.10,
        "calmar_ratio": 2.40,
        "risk_free_rate_annual": 0.0,
        "performance_observations": 3,
        "cumulative_twr": pd.Series(
            [0.0, -0.05, 0.08],
            index=dates
        ),
        "investor_drawdown": pd.Series(
            [0.0, -0.05, 0.0],
            index=dates,
            name="investor_drawdown"
        ),
        "investor_max_drawdown": -0.05,
        "daily_portfolio_value": pd.Series(
            [95.0, 98.0, 100.0],
            index=dates
        ),
        "holdings": pd.DataFrame({
            "ticker": ["AAA"],
            "quantity": [3.0],
            "average_cost": [20.0],
            "total_cost_basis": [60.0],
            "realized_pnl": [2.0]
        })
    }
    security_risk = {
        "annual_volatility": 0.20,
        "historical_var_95": 0.02,
        "expected_shortfall_95": 0.03,
        "max_drawdown": -0.20,
        "positions": pd.DataFrame({
            "ticker": ["AAA"],
            "quantity": [3.0],
            "current_price": [25.0],
            "price_as_of": [dates[-1]],
            "market_value": [75.0],
            "weight": [1.0],
            "purchase_price": [20.0],
            "unrealized_pnl": [15.0]
        }),
        "weights": pd.Series({"AAA": 1.0}),
        "risk_contribution_table": pd.DataFrame(
            {
                "Contribution_risque_pct": [1.0]
            },
            index=["AAA"]
        ),
        "stress_results": pd.DataFrame(
            {"Portfolio_return": [-0.10], "P&L": [-7.50]},
            index=["Sell-off"]
        ),
        "skewness": -0.25,
        "excess_kurtosis": 1.50,
        "worst_date": dates[0],
        "worst_return": -0.04,
        "model_validation_available": validation_available,
        "historical_backtest_available": True,
        "historical_backtest_reason": None,
        "model_validation_reason": (
            None
            if validation_available
            else (
                "VaR model validation requires at least 250 out-of-sample "
                "observations at 95% confidence under the current validation "
                "policy; received 249."
            )
        ),
        "validation_confidence_level": 0.95,
        "validation_observations": 250 if validation_available else 249,
        "minimum_required_observations": 250,
        "model_comparison": pd.DataFrame([
            {
                "model": model,
                "observations": 250 if validation_available else 249,
                "breaches": 12,
                "breach_rate": 0.048,
                "expected_breach_rate": 0.05,
                **(
                    {
                        "kupiec_p_value": 0.80,
                        "christoffersen_independence_p_value": 0.70,
                        "conditional_coverage_p_value": 0.60
                    }
                    if validation_available
                    else {}
                )
            }
            for model in ("Historical VaR 95%", "EWMA VaR 95%")
        ])
    }
    account_risk = {
        "annual_volatility": 0.15,
        "historical_var_95": 0.015,
        "expected_shortfall_95": 0.0225,
        "max_drawdown": -0.14,
        "risk_contribution_table": pd.DataFrame(
            {"Contribution_risque_pct": [1.0, 0.0]},
            index=["AAA", "Cash"]
        ),
        "stress_results": pd.DataFrame(
            {"Portfolio_return": [-0.075], "P&L": [-7.50]},
            index=["Sell-off"]
        )
    }
    return {
        "analysis_marker": "precomputed-no-rerun",
        "investor": investor,
        "risk": security_risk,
        "account_risk": account_risk,
        "diversification": {
            "weight_hhi": 1.0,
            "effective_number_of_holdings": 1.0,
            "diversification_ratio": 1.0,
            "risk_concentration_hhi": 1.0,
            "effective_risk_contributors": 1.0
        },
        "benchmark": {
            "ticker": "SPY",
            "observations": 3,
            "benchmark_total_return": 0.05,
            "benchmark_annualized_return": 0.10,
            "beta": 0.80,
            "alpha_annualized": 0.03,
            "tracking_error": 0.12,
            "information_ratio": 0.40,
            "investor_cumulative_aligned": pd.Series(
                [0.0, -0.05, 0.08],
                index=dates,
                name="Investor TWR"
            ),
            "benchmark_cumulative_return": pd.Series(
                [0.0, 0.01, 0.05],
                index=dates,
                name="SPY"
            )
        },
        "metadata": {
            "risk_window_label": "3 years",
            "risk_start_date": pd.Timestamp("2022-01-01"),
            "risk_end_date": dates[-1],
            "risk_observations": 756,
            "raw_price_observations": 760,
            "asset_return_observations_before_alignment": 759,
            "observations_dropped_during_alignment": 3,
            "usable_return_observations_by_asset": {"AAA": 759},
            "valuation_reference_date": dates[-1],
            "cash_weight": 0.25
        }
    }


def load_page(page, results=None):
    app = AppTest.from_file(str(APP_FILE), default_timeout=30).run()
    if results is None:
        results = make_dashboard_results()
    app.session_state["unified_analysis"] = results
    app.session_state["navigation"] = page
    return app.run()


def metric_value(app, label):
    return next(metric.value for metric in app.metric if metric.label == label)


def open_transactions_page():
    app = AppTest.from_file(str(APP_FILE), default_timeout=30).run()
    app.session_state["navigation"] = "Transactions"
    return app.run()


def click_button(app, label, key=None):
    button = next(
        item
        for item in app.button
        if item.label == label and (key is None or item.key == key)
    )
    return button.click().run()


def top_navigation(app):
    return next(
        item
        for item in app.segmented_control
        if item.label == "Navigation"
    )


def make_cash_ledger(amounts=(100.0, 200.0, 300.0)):
    dates = pd.to_datetime([
        "2025-01-01", "2025-01-02", "2025-01-03"
    ])[:len(amounts)]
    return pd.DataFrame({
        "date": dates,
        "type": ["DEPOSIT"] * len(amounts),
        "ticker": [np.nan] * len(amounts),
        "quantity": [np.nan] * len(amounts),
        "price": [np.nan] * len(amounts),
        "amount": list(amounts),
        "fees": [0.0] * len(amounts),
        "taxes": [0.0] * len(amounts)
    })


def open_transactions_with_ledger(ledger, results=None):
    app = open_transactions_page()
    app.session_state["transaction_ledger_input"] = ledger.copy(deep=True)
    if results is not None:
        app.session_state["unified_analysis"] = results
    return app.run()


def request_selected_transaction_deletion(app):
    app = click_button(app, "Manage transaction")
    return click_button(app, "Delete transaction")


def test_transactions_replaces_data_and_starts_empty():
    app = open_transactions_page()

    assert len(app.exception) == 0
    assert top_navigation(app).options == [
        "Overview", "Performance", "Risk", "Holdings", "Transactions",
        "Advanced"
    ]
    assert top_navigation(app).value == "Transactions"
    assert len(app.sidebar.radio) == 0
    assert app.session_state["transaction_ledger_input"].empty
    assert app.session_state["stress_scenarios_input"].empty
    assert "Advanced stress testing (optional)" in [
        item.label for item in app.expander
    ]
    assert "No transactions yet" in [item.value for item in app.subheader]
    assert [item.label for item in app.button[:4]] == [
        "Add transaction",
        "Import BoursoBank CSV",
        "Load demo portfolio",
        "Clear portfolio"
    ]
    analyze = next(
        item for item in app.button if item.label == "Analyze portfolio"
    )
    assert analyze.disabled is True


def test_fresh_session_uses_canonical_analysis_defaults():
    app = AppTest.from_file(str(APP_FILE), default_timeout=30).run()

    assert app.session_state["risk_free_rate_percent_input"] == 0.0
    assert app.session_state["benchmark_ticker_input"] == "SPY"
    assert app.session_state["risk_window_input"] == "3 years"


def test_ordinary_rerun_preserves_changed_analysis_settings():
    app = AppTest.from_file(str(APP_FILE), default_timeout=30).run()
    app.session_state["risk_free_rate_percent_input"] = 2.5
    app.session_state["benchmark_ticker_input"] = "QQQ"
    app.session_state["risk_window_input"] = "1 year"

    app = app.run()

    assert app.session_state["risk_free_rate_percent_input"] == 2.5
    assert app.session_state["benchmark_ticker_input"] == "QQQ"
    assert app.session_state["risk_window_input"] == "1 year"


def test_top_navigation_changes_page_and_preserves_ledger_and_analysis():
    ledger = make_cash_ledger()
    results = make_dashboard_results()
    app = open_transactions_with_ledger(ledger, results=results)
    risk_free_input = next(
        item
        for item in app.number_input
        if item.label == "Annual risk-free rate (%)"
    )
    app = risk_free_input.set_value(2.5).run()
    benchmark_input = next(
        item for item in app.text_input if item.label == "Benchmark ticker"
    )
    app = benchmark_input.set_value("QQQ").run()
    risk_history = next(
        item for item in app.selectbox if item.label == "Risk history"
    )
    app = risk_history.set_value("1 year").run()

    app = top_navigation(app).set_value("Risk").run()

    assert len(app.exception) == 0
    assert top_navigation(app).value == "Risk"
    assert "Risk" in [item.value for item in app.header]
    pd.testing.assert_frame_equal(
        app.session_state["transaction_ledger_input"],
        ledger
    )
    assert app.session_state["unified_analysis"]["analysis_marker"] == (
        "precomputed-no-rerun"
    )
    assert app.session_state["risk_free_rate_percent_input"] == 2.5
    assert app.session_state["benchmark_ticker_input"] == "QQQ"
    assert app.session_state["risk_window_input"] == "1 year"
    assert len(app.sidebar.radio) == 0


def test_load_demo_portfolio_is_explicit_and_preserves_normalized_ledger():
    app = open_transactions_page()
    app.session_state["risk_free_rate_percent_input"] = 2.5
    app.session_state["benchmark_ticker_input"] = "QQQ"
    app.session_state["risk_window_input"] = "1 year"
    app = click_button(app, "Load demo portfolio")

    ledger = app.session_state["transaction_ledger_input"]
    assert len(ledger) > 0
    assert not app.session_state["stress_scenarios_input"].empty
    assert app.session_state["display_currency"] == "USD"
    assert app.session_state["risk_free_rate_percent_input"] == 0.0
    assert app.session_state["benchmark_ticker_input"] == "SPY"
    assert app.session_state["risk_window_input"] == "3 years"
    assert ledger.columns.tolist() == [
        "date", "type", "ticker", "quantity", "price", "amount",
        "fees", "taxes"
    ]
    assert "Transaction history" in [item.value for item in app.subheader]
    assert not any(
        item.value == "No transactions yet" for item in app.subheader
    )


def test_deletion_requires_confirmation_before_ledger_changes():
    ledger = make_cash_ledger()
    app = open_transactions_with_ledger(ledger)
    app = request_selected_transaction_deletion(app)

    pd.testing.assert_frame_equal(
        app.session_state["transaction_ledger_input"],
        ledger
    )
    assert any(item.label == "Confirm deletion" for item in app.button)
    assert any(item.label == "Cancel" for item in app.button)


def test_confirmed_delete_removes_only_newest_selected_underlying_row():
    ledger = make_cash_ledger()
    app = open_transactions_with_ledger(ledger)
    displayed_history = app.dataframe[0].value
    assert displayed_history.iloc[0]["Price or Amount"] == 300.0

    app = request_selected_transaction_deletion(app)
    app = click_button(app, "Confirm deletion")

    remaining = app.session_state["transaction_ledger_input"]
    assert remaining["amount"].tolist() == [100.0, 200.0]
    assert remaining["date"].is_monotonic_increasing
    assert app.session_state["show_transaction_manager"] is False
    assert any(item.value == "Transaction deleted." for item in app.success)


def test_delete_confirmation_cancel_preserves_ledger_and_closes_panel():
    ledger = make_cash_ledger()
    app = open_transactions_with_ledger(ledger)
    app = request_selected_transaction_deletion(app)
    app = click_button(app, "Cancel", key="cancel_transaction_deletion")

    pd.testing.assert_frame_equal(
        app.session_state["transaction_ledger_input"],
        ledger
    )
    assert app.session_state["show_transaction_manager"] is False
    assert "Manage transaction" not in [item.value for item in app.subheader]


def test_deleting_transaction_invalidates_existing_analysis():
    app = open_transactions_with_ledger(
        make_cash_ledger(),
        results=make_dashboard_results()
    )
    app = request_selected_transaction_deletion(app)
    app = click_button(app, "Confirm deletion")

    assert "unified_analysis" not in app.session_state
    assert len(app.session_state["transaction_ledger_input"]) == 2


def test_deleting_final_transaction_returns_to_empty_state():
    app = open_transactions_with_ledger(make_cash_ledger((100.0,)))
    app = request_selected_transaction_deletion(app)
    app = click_button(app, "Confirm deletion")

    assert app.session_state["transaction_ledger_input"].empty
    assert "No transactions yet" in [item.value for item in app.subheader]
    assert not any(
        item.value == "Transaction history" for item in app.subheader
    )


def test_add_buy_transaction_uses_relevant_fields_and_stores_it():
    app = open_transactions_page()
    app = click_button(app, "Add transaction")

    assert "Add transaction" in [item.value for item in app.subheader]
    assert app.session_state["show_add_transaction"] is True
    assert {item.label for item in app.number_input} >= {
        "Quantity", "Price", "Fees", "Taxes"
    }
    next(item for item in app.text_input if item.label == "Ticker").set_value(
        " aaa "
    )
    next(
        item for item in app.number_input if item.label == "Quantity"
    ).set_value(3.0)
    next(
        item for item in app.number_input if item.label == "Price"
    ).set_value(12.5)
    submit = next(
        item
        for item in app.button
        if item.key == "FormSubmitter:add_transaction_form-Add transaction"
    )
    app = submit.click().run()

    ledger = app.session_state["transaction_ledger_input"]
    assert len(ledger) == 1
    assert ledger.iloc[0]["type"] == "BUY"
    assert ledger.iloc[0]["ticker"] == "AAA"
    assert np.isclose(ledger.iloc[0]["quantity"], 3.0)
    assert np.isclose(ledger.iloc[0]["price"], 12.5)
    assert app.session_state["show_add_transaction"] is False
    assert "Add transaction" not in [item.value for item in app.subheader]
    assert any(item.value == "Transaction added." for item in app.success)


def test_cancel_closes_add_form_without_changing_ledger():
    app = open_transactions_page()
    app = click_button(app, "Add transaction")
    next(item for item in app.text_input if item.label == "Ticker").set_value(
        "AAA"
    )
    cancel = next(
        item
        for item in app.button
        if item.key == "FormSubmitter:add_transaction_form-Cancel"
    )
    app = cancel.click().run()

    assert app.session_state["transaction_ledger_input"].empty
    assert app.session_state["show_add_transaction"] is False
    assert "Add transaction" not in [item.value for item in app.subheader]
    assert "No transactions yet" in [item.value for item in app.subheader]


def test_add_and_import_panels_are_mutually_exclusive():
    app = open_transactions_page()
    app = click_button(app, "Import BoursoBank CSV")

    assert app.session_state["show_boursobank_import"] is True
    assert app.session_state["show_add_transaction"] is False
    assert len(app.file_uploader) == 1

    app = click_button(app, "Add transaction")

    assert app.session_state["show_add_transaction"] is True
    assert app.session_state["show_boursobank_import"] is False
    assert len(app.file_uploader) == 0

    app = click_button(app, "Import BoursoBank CSV")

    assert app.session_state["show_boursobank_import"] is True
    assert app.session_state["show_add_transaction"] is False
    assert len(app.file_uploader) == 1
    assert "Add transaction" not in [item.value for item in app.subheader]


def test_add_deposit_changes_fields_and_stores_amount():
    app = open_transactions_page()
    app = click_button(app, "Add transaction")
    transaction_type = next(
        item for item in app.selectbox if item.label == "Type"
    )
    app = transaction_type.set_value("DEPOSIT").run()

    number_labels = {item.label for item in app.number_input}
    assert "Amount" in number_labels
    assert "Quantity" not in number_labels
    assert "Price" not in number_labels
    assert not any(item.label == "Ticker" for item in app.text_input)

    next(
        item for item in app.number_input if item.label == "Amount"
    ).set_value(500.0)
    submit = next(
        item
        for item in app.button
        if item.key == "FormSubmitter:add_transaction_form-Add transaction"
    )
    app = submit.click().run()

    deposit = app.session_state["transaction_ledger_input"].iloc[0]
    assert deposit["type"] == "DEPOSIT"
    assert np.isclose(deposit["amount"], 500.0)


def test_boursobank_upload_replaces_ledger_without_running_analysis():
    app = open_transactions_page()
    app = click_button(app, "Load demo portfolio")
    assert not app.session_state["stress_scenarios_input"].empty
    app = click_button(app, "Import BoursoBank CSV")
    csv_content = (
        "Date;Type;Valeur;Devise de l'opération;Frais;Impôts / Taxes;Parts;"
        "ISIN;Symbole boursier;Nom du titre\n"
        "2025-01-02;Dépôt;1 000,00;EUR;;;;;;"
    ).encode("utf-8")
    app = app.file_uploader[0].upload(
        "boursobank.csv", csv_content, "text/csv"
    ).run()
    app = click_button(app, "Import uploaded CSV")

    ledger = app.session_state["transaction_ledger_input"]
    assert ledger["type"].tolist() == ["DEPOSIT"]
    assert ledger["amount"].tolist() == [1000.0]
    assert app.session_state["stress_scenarios_input"].empty
    assert app.session_state["display_currency"] == "EUR"
    assert "unified_analysis" not in app.session_state
    assert app.session_state["show_boursobank_import"] is False
    assert len(app.file_uploader) == 0
    assert "Transaction history" in [item.value for item in app.subheader]
    assert any("Imported 1 transactions" in item.value for item in app.success)


def test_boursobank_import_errors_are_shown_without_crashing():
    app = open_transactions_page()
    app = click_button(app, "Import BoursoBank CSV")
    bad_csv = (
        "Date;Type;Valeur;Devise de l'opération;Frais;Impôts / Taxes;Parts;"
        "ISIN;Symbole boursier;Nom du titre\n"
        "2025-01-02;Unsupported;10,00;EUR;;;;;;"
    ).encode("utf-8")
    app = app.file_uploader[0].upload(
        "bad.csv", bad_csv, "text/csv"
    ).run()
    app = click_button(app, "Import uploaded CSV")

    assert len(app.exception) == 0
    assert app.session_state["transaction_ledger_input"].empty
    assert any(
        "Could not import BoursoBank CSV" in item.value
        for item in app.error
    )


def test_adding_transaction_invalidates_existing_analysis():
    app = load_page("Transactions")
    app = click_button(app, "Add transaction")
    app = next(
        item for item in app.selectbox if item.label == "Type"
    ).set_value("DEPOSIT").run()
    next(
        item for item in app.number_input if item.label == "Amount"
    ).set_value(100.0)
    submit = next(
        item
        for item in app.button
        if item.key == "FormSubmitter:add_transaction_form-Add transaction"
    )
    app = submit.click().run()

    assert "unified_analysis" not in app.session_state
    assert len(app.session_state["transaction_ledger_input"]) == 1


@pytest.mark.parametrize(
    "page",
    [
        "Overview", "Performance", "Risk", "Holdings", "Transactions",
        "Advanced"
    ]
)
def test_all_six_navigation_pages_render_without_rerunning_analysis(page):
    app = load_page(page)

    assert len(app.exception) == 0
    assert app.session_state["unified_analysis"]["analysis_marker"] == (
        "precomputed-no-rerun"
    )


def test_overview_uses_account_risk_metrics():
    app = load_page("Overview")

    assert len(app.exception) == 0
    subheaders = [item.value for item in app.subheader]
    assert "Portfolio Summary" in subheaders
    assert "Investor Account Value" in subheaders
    assert "Current Allocation" in subheaders
    assert "Account Snapshot" in subheaders
    assert "Account Risk" in subheaders
    assert {metric.label for metric in app.metric[:4]} == {
        "Total Account Value",
        "Total P&L",
        "Time-Weighted Return",
        "Cash"
    }
    assert metric_value(app, "Annualized Volatility") == "15.00%"
    assert metric_value(app, "Historical VaR 95%") == "1.50%"
    assert metric_value(app, "Expected Shortfall 95%") == "2.25%"
    assert metric_value(app, "Max Drawdown") == "-14.00%"
    assert metric_value(app, "Total P&L") == "+€30.00"
    assert len(app.get("vega_lite_chart")) >= 2
    assert {item.value for item in app.caption} >= {
        "Largest Position",
        "Largest Risk Contributor",
        "Worst Stress Scenario",
        "Cash Weight"
    }
    assert any(
        "Account Risk · 3-year window · 756 daily observations" in item.value
        for item in app.caption
    )


def test_risk_selector_defaults_to_account_and_switches_without_analysis():
    app = load_page("Risk")
    risk_view = next(
        item
        for item in app.segmented_control
        if item.label == "Risk view"
    )

    assert len(app.exception) == 0
    assert risk_view.value == "Account"
    assert metric_value(app, "Annualized Volatility") == "15.00%"
    assert list(app.dataframe[0].value.columns) == [
        "Scenario",
        "Account Impact (%)",
        "Account Impact (EUR)"
    ]
    assert app.expander[0].label == "Methodology details"

    app = risk_view.set_value("Invested Securities").run()
    risk_view = next(
        item
        for item in app.segmented_control
        if item.label == "Risk view"
    )

    assert len(app.exception) == 0
    assert risk_view.value == "Invested Securities"
    assert metric_value(app, "Annualized Volatility") == "20.00%"
    assert list(app.dataframe[0].value.columns) == [
        "Scenario",
        "Security-Sleeve Impact (%)",
        "Security-Sleeve Impact (EUR)"
    ]
    assert app.session_state["unified_analysis"]["analysis_marker"] == (
        "precomputed-no-rerun"
    )


def test_risk_page_explains_unavailable_optional_stress_testing():
    results = make_dashboard_results()
    empty_stress = pd.DataFrame(columns=["Portfolio_return", "P&L"])
    results["risk"]["stress_results"] = empty_stress
    results["risk"]["stress_test_available"] = False
    results["risk"]["stress_test_reason"] = (
        "Stress testing unavailable: define shocks for all current holdings."
    )
    results["account_risk"]["stress_results"] = empty_stress.copy()

    app = load_page("Risk", results=results)

    assert len(app.exception) == 0
    assert any(
        item.value == (
            "Stress testing unavailable: define shocks for all current "
            "holdings."
        )
        for item in app.info
    )


def test_holdings_displays_invested_security_diversification_diagnostics():
    app = load_page("Holdings")

    assert len(app.exception) == 0
    assert "Diversification" in [item.value for item in app.subheader]
    assert metric_value(app, "Effective Holdings") == "1.00"
    assert metric_value(app, "Concentration HHI") == "1.000"
    assert metric_value(app, "Diversification Ratio") == "1.00"
    assert metric_value(app, "Effective Risk Contributors") == "1.00"
    assert any(
        "invested securities only; cash is excluded" in caption.value
        for caption in app.caption
    )
    assert app.session_state["unified_analysis"]["analysis_marker"] == (
        "precomputed-no-rerun"
    )

    unavailable_results = make_dashboard_results()
    unavailable_results["diversification"]["diversification_ratio"] = np.nan
    unavailable_results["diversification"][
        "effective_risk_contributors"
    ] = np.nan
    unavailable_app = load_page("Holdings", results=unavailable_results)

    assert metric_value(unavailable_app, "Diversification Ratio") == "N/A"
    assert metric_value(
        unavailable_app,
        "Effective Risk Contributors"
    ) == "N/A"


def test_risk_history_change_marks_existing_analysis_for_rerun():
    app = load_page("Transactions")

    risk_history = next(
        item for item in app.selectbox if item.label == "Risk history"
    )
    assert risk_history.value == "3 years"
    risk_history.set_value("1 year").run()

    assert len(app.exception) == 0
    assert app.session_state["unified_analysis"]["analysis_marker"] == (
        "precomputed-no-rerun"
    )
    assert any(
        "Existing results still use" in warning.value
        for warning in app.warning
    )
    assert any(
        caption.value == "Analysis status: requires rerun"
        for caption in app.caption
    )


def test_risk_free_rate_change_marks_existing_analysis_for_rerun():
    app = load_page("Transactions")

    risk_free_input = next(
        item
        for item in app.number_input
        if item.label == "Annual risk-free rate (%)"
    )
    assert risk_free_input.value == 0.0
    risk_free_input.set_value(3.0).run()

    assert len(app.exception) == 0
    assert app.session_state["unified_analysis"]["analysis_marker"] == (
        "precomputed-no-rerun"
    )
    assert any(
        "Analysis settings changed" in warning.value
        for warning in app.warning
    )
    assert any(
        caption.value == "Analysis status: requires rerun"
        for caption in app.caption
    )


def test_benchmark_change_marks_existing_analysis_stale_without_rerun():
    app = load_page("Transactions")

    benchmark_input = next(
        item for item in app.text_input if item.label == "Benchmark ticker"
    )
    assert benchmark_input.value == "SPY"
    benchmark_input.set_value(" qqq ").run()

    assert len(app.exception) == 0
    assert app.session_state["unified_analysis"]["analysis_marker"] == (
        "precomputed-no-rerun"
    )
    assert any(
        "Analysis settings changed" in warning.value
        for warning in app.warning
    )
    assert any(
        caption.value == "Analysis status: requires rerun"
        for caption in app.caption
    )


def test_analyze_uses_settings_and_stores_results_without_navigation():
    expected_results = make_dashboard_results()
    with patch(
        "src.unified.run_unified_portfolio_analysis",
        return_value=expected_results
    ) as run_analysis:
        app = AppTest.from_file(str(APP_FILE), default_timeout=30).run()
        app.session_state["navigation"] = "Transactions"
        app = app.run()
        load_demo_button = next(
            button
            for button in app.button
            if button.label == "Load demo portfolio"
        )
        app = load_demo_button.click().run()
        analyze_button = next(
            button
            for button in app.button
            if button.label == "Analyze portfolio"
        )
        app = analyze_button.click().run()

    assert len(app.exception) == 0
    assert run_analysis.call_count == 1
    call_arguments = run_analysis.call_args.kwargs
    assert "cash_flows" not in call_arguments
    assert call_arguments["transactions"].columns.tolist() == [
        "date", "type", "ticker", "quantity", "price", "amount",
        "fees", "taxes"
    ]
    assert call_arguments["risk_window_label"] == "3 years"
    assert call_arguments["risk_free_rate_annual"] == 0.0
    assert call_arguments["benchmark_ticker"] == "SPY"
    assert app.session_state["unified_analysis"] is expected_results
    assert app.session_state["navigation"] == "Transactions"
    assert any(
        message.value == "Analysis completed successfully."
        for message in app.success
    )


def test_clear_portfolio_returns_to_empty_and_clears_analysis():
    app = load_page("Transactions")
    load_demo_button = next(
        button
        for button in app.button
        if button.label == "Load demo portfolio"
    )
    app = load_demo_button.click().run()
    app.session_state["unified_analysis"] = make_dashboard_results()
    app.session_state["risk_free_rate_percent_input"] = 2.5
    app.session_state["benchmark_ticker_input"] = "QQQ"
    app.session_state["risk_window_input"] = "1 year"
    clear_button = next(
        button for button in app.button if button.label == "Clear portfolio"
    )
    app = clear_button.click().run()

    assert len(app.exception) == 0
    assert "unified_analysis" not in app.session_state
    assert app.session_state["transaction_ledger_input"].empty
    assert app.session_state["risk_free_rate_percent_input"] == 0.0
    assert app.session_state["benchmark_ticker_input"] == "SPY"
    assert app.session_state["risk_window_input"] == "3 years"
    assert any(
        item.value == "No transactions yet"
        for item in app.subheader
    )


def test_performance_shows_investor_drawdown_without_changing_risk_drawdown():
    app = load_page("Performance")

    assert len(app.exception) == 0
    assert metric_value(app, "Time-Weighted Return") == "+8.00%"
    assert metric_value(app, "Annualized TWR") == "+12.00%"
    assert metric_value(app, "Investor Max Drawdown") == "-5.00%"
    assert metric_value(app, "Sharpe Ratio") == "0.85"
    assert metric_value(app, "Sortino Ratio") == "1.10"
    assert metric_value(app, "Calmar Ratio") == "2.40"
    assert len(app.metric) == 13
    assert metric_value(app, "Total P&L") == "+€30.00"
    assert metric_value(app, "Benchmark Return") == "+5.00%"
    assert metric_value(app, "Benchmark Annualized Return") == "+10.00%"
    assert metric_value(app, "Beta") == "0.80"
    assert metric_value(app, "Annualized Alpha") == "+3.00%"
    assert metric_value(app, "Tracking Error") == "12.00%"
    assert metric_value(app, "Information Ratio") == "0.40"
    assert "Benchmark Comparison · SPY" in [
        item.value for item in app.subheader
    ]
    assert "Investor vs Benchmark" in [
        item.value for item in app.subheader
    ]
    assert "Accounting Breakdown" in [
        item.value for item in app.subheader
    ]
    assert "Investor History" in [item.value for item in app.subheader]
    assert [item.label for item in app.tabs] == ["Drawdown", "Account Value"]
    accounting = app.dataframe[0].value
    assert accounting["Accounting component"].tolist() == [
        "Realized Market P&L",
        "Unrealized Market P&L",
        "Dividend Income",
        "Interest Income",
        "Fees Paid",
        "Taxes Paid",
        "Net External Contributions",
        "Economic Total P&L"
    ]
    assert accounting["Amount"].tolist() == [
        "+€2.00", "+€15.00", "€20.00", "€3.00", "€4.00", "€6.00",
        "+€70.00", "+€30.00"
    ]
    assert any(
        "3 active daily TWR observations" in item.value
        for item in app.caption
    )
    assert any(
        "3 exact-date aligned active observations" in item.value
        for item in app.caption
    )
    assert len(app.get("vega_lite_chart")) == 3
    account_value_chart = app.get("vega_lite_chart")[-1]
    assert len(account_value_chart.proto.datasets) == 1
    assert account_value_chart.proto.datasets[0].data.data
    assert '"field": "Account value"' in account_value_chart.proto.spec
    assert '"title": "Account value (EUR)"' in account_value_chart.proto.spec
    assert not any(
        "dividends are not modeled" in item.value.lower()
        for item in app.caption
    )

    overview = load_page("Overview")
    assert metric_value(overview, "Max Drawdown") == "-14.00%"


def test_display_currency_changes_formatting_only():
    results = make_dashboard_results()
    app = load_page("Transactions", results=results)
    currency = next(
        item
        for item in app.selectbox
        if item.label == "Portfolio display currency"
    )

    app = currency.set_value("USD").run()
    app = top_navigation(app).set_value("Overview").run()

    assert metric_value(app, "Total Account Value") == "$100.00"
    assert metric_value(app, "Total P&L") == "+$30.00"
    assert app.session_state["unified_analysis"]["investor"][
        "economic_total_pnl"
    ] == 30.0


def test_performance_explains_unavailable_benchmark_overlap():
    results = make_dashboard_results()
    results["benchmark"]["observations"] = 0
    results["benchmark"]["benchmark_total_return"] = np.nan
    results["benchmark"]["benchmark_annualized_return"] = np.nan
    results["benchmark"]["beta"] = np.nan
    results["benchmark"]["alpha_annualized"] = np.nan
    results["benchmark"]["tracking_error"] = np.nan
    results["benchmark"]["information_ratio"] = np.nan
    results["benchmark"]["investor_cumulative_aligned"] = pd.Series(
        dtype=float,
        name="Investor TWR"
    )
    results["benchmark"]["benchmark_cumulative_return"] = pd.Series(
        dtype=float,
        name="SPY"
    )

    app = load_page("Performance", results=results)

    assert len(app.exception) == 0
    assert metric_value(app, "Benchmark Return") == "N/A"
    messages = [message.value for message in app.info]
    assert any("At least two exact-date" in message for message in messages)
    assert any("No exact-date overlap" in message for message in messages)


def test_advanced_displays_validation_policy_and_inferential_table():
    app = load_page("Advanced")

    assert len(app.exception) == 0
    assert any(
        "requires at least 250 observations" in caption.value
        for caption in app.caption
    )
    assert len(app.dataframe) == 2
    assert "5% Result" in app.dataframe[1].value.columns


def test_advanced_keeps_risk_results_when_validation_is_unavailable():
    results = make_dashboard_results(validation_available=False)
    app = load_page("Advanced", results=results)

    assert len(app.exception) == 0
    assert len(app.dataframe) == 1
    messages = [message.value for message in app.info]
    assert any("requires more history" in message for message in messages)
    assert any("received 249" in message for message in messages)
    assert all("rejected" not in message.lower() for message in messages)
