from pathlib import Path

import pandas as pd
import streamlit as st

from src.data import load_stress_scenarios
from src.engine import run_portfolio_analysis
from src.investor import run_investor_performance_analysis


PROJECT_DIR = Path(__file__).resolve().parent
STRESS_SCENARIOS_FILE = PROJECT_DIR / "inputs" / "stress_scenarios.csv"
TRANSACTIONS_EXAMPLE_FILE = PROJECT_DIR / "inputs" / "transactions_example.csv"
CASH_FLOWS_EXAMPLE_FILE = PROJECT_DIR / "inputs" / "cash_flows_example.csv"
START_DATE = "2022-01-01"

EXAMPLE_PORTFOLIO = pd.DataFrame({
    "ticker": ["SPY", "TLT", "GLD", "AAPL"],
    "quantity": [10, 50, 30, 25],
    "purchase_price": [500.0, 95.0, 190.0, 180.0]
})
EXAMPLE_TRANSACTIONS = pd.read_csv(
    TRANSACTIONS_EXAMPLE_FILE,
    parse_dates=["date"]
)
EXAMPLE_CASH_FLOWS = pd.read_csv(
    CASH_FLOWS_EXAMPLE_FILE,
    parse_dates=["date"]
)


def display_overview(results):
    """Display headline metrics, positions and current allocation."""
    st.subheader("Key metrics")
    metric_columns = st.columns(5)
    metric_columns[0].metric(
        "Current portfolio value",
        f"${results['portfolio_value']:,.2f}",
        border=True
    )
    metric_columns[1].metric(
        "Annualized volatility",
        f"{results['annual_volatility']:.2%}",
        border=True
    )
    metric_columns[2].metric(
        "Maximum drawdown",
        f"{results['max_drawdown']:.2%}",
        border=True
    )
    metric_columns[3].metric(
        "Historical VaR 95%",
        f"{results['historical_var_95']:.2%}",
        border=True
    )
    metric_columns[4].metric(
        "Historical ES 95%",
        f"{results['expected_shortfall_95']:.2%}",
        border=True
    )

    st.divider()
    st.subheader("Current positions")
    positions = (
        results["positions"]
        [[
            "ticker",
            "quantity",
            "purchase_price",
            "current_price",
            "market_value",
            "weight",
            "unrealized_pnl",
            "unrealized_pnl_pct"
        ]]
        .rename(columns={
            "ticker": "Ticker",
            "quantity": "Quantity",
            "purchase_price": "Purchase price",
            "current_price": "Current price",
            "market_value": "Market value",
            "weight": "Portfolio weight",
            "unrealized_pnl": "Unrealized P&L",
            "unrealized_pnl_pct": "Unrealized P&L %"
        })
    )
    positions_style = positions.style.format({
        "Quantity": "{:,.2f}",
        "Purchase price": "${:,.2f}",
        "Current price": "${:,.2f}",
        "Market value": "${:,.2f}",
        "Portfolio weight": "{:.2%}",
        "Unrealized P&L": "${:,.2f}",
        "Unrealized P&L %": "{:.2%}"
    })
    st.dataframe(
        positions_style,
        width="stretch",
        hide_index=True
    )

    st.divider()
    st.subheader("Current portfolio allocation")
    st.caption("Market-value weights calculated by the analysis engine.")
    allocation_percent = (
        results["weights"]
        .mul(100.0)
        .rename("Portfolio weight (%)")
    )
    st.bar_chart(
        allocation_percent,
        y_label="Weight (%)",
        height=320
    )


def display_risk_analysis(results):
    """Display existing historical performance and drawdown results."""
    st.subheader("Historical performance of current allocation")
    st.caption(
        "Backtested using today's portfolio weights. This does not "
        "reconstruct the investor's actual historical holdings or "
        "transaction history."
    )
    cumulative_performance = results["cumulative_performance"].rename(
        "Growth of $1"
    )
    st.line_chart(
        cumulative_performance,
        x_label="Date",
        y_label="Growth of $1",
        height=360
    )

    st.divider()
    st.subheader("Drawdown")
    drawdown_metric, _ = st.columns([1, 4])
    drawdown_metric.metric(
        "Maximum drawdown",
        f"{results['max_drawdown']:.2%}"
    )
    drawdown_percent = (
        results["drawdown"]
        .mul(100.0)
        .rename("Drawdown (%)")
    )
    st.line_chart(
        drawdown_percent,
        x_label="Date",
        y_label="Drawdown (%)",
        height=340
    )


def display_stress_and_risk_contribution(results):
    """Display existing stress-test and risk-contribution results."""
    st.subheader("Stress tests")
    st.caption(
        "Impact of the predefined scenarios in inputs/stress_scenarios.csv."
    )
    stress_results = (
        results["stress_results"]
        .reset_index()
        .rename(columns={
            "scenario": "Scenario",
            "Portfolio_return": "Portfolio return",
            "P&L": "P&L impact"
        })
    )
    stress_results_style = stress_results.style.format({
        "Portfolio return": "{:.2%}",
        "P&L impact": "${:,.2f}"
    })
    st.dataframe(
        stress_results_style,
        width="stretch",
        hide_index=True
    )

    st.divider()
    st.subheader("Risk contributions")
    risk_contributions = results["risk_contribution_table"].reset_index()
    risk_contributions = risk_contributions.rename(columns={
        risk_contributions.columns[0]: "Ticker",
        "Poids": "Portfolio weight",
        "Contribution_volatilite": "Contribution to volatility",
        "Contribution_risque_pct": "Contribution to total risk"
    })
    risk_contributions_style = risk_contributions.style.format({
        "Portfolio weight": "{:.2%}",
        "Contribution to volatility": "{:.2%}",
        "Contribution to total risk": "{:.2%}"
    })
    st.dataframe(
        risk_contributions_style,
        width="stretch",
        hide_index=True
    )

    risk_contribution_percent = (
        risk_contributions
        .set_index("Ticker")["Contribution to total risk"]
        .mul(100.0)
        .rename("Contribution to total risk (%)")
    )
    st.bar_chart(
        risk_contribution_percent,
        y_label="Share of total risk (%)",
        height=320
    )


def display_model_validation(results):
    """Display the engine's common-period VaR model comparison."""
    st.subheader("Common-period VaR model comparison")
    st.write(
        "Historical and EWMA VaR are compared over the same evaluation dates."
    )

    comparison_columns = [
        "observations",
        "breaches",
        "breach_rate",
        "expected_breach_rate",
        "kupiec_p_value",
        "christoffersen_independence_p_value",
        "conditional_coverage_p_value"
    ]
    model_comparison = (
        results["model_comparison"]
        .set_index("model")[comparison_columns]
        .transpose()
        .rename(index={
            "observations": "Observations",
            "breaches": "Breaches",
            "breach_rate": "Breach rate",
            "expected_breach_rate": "Expected breach rate",
            "kupiec_p_value": "Kupiec p-value",
            "christoffersen_independence_p_value": "Independence p-value",
            "conditional_coverage_p_value": "Conditional coverage p-value"
        })
    )
    model_comparison.index.name = "Statistic"
    model_comparison_style = (
        model_comparison.style
        .format(
            "{:,.0f}",
            subset=pd.IndexSlice[["Observations", "Breaches"], :]
        )
        .format(
            "{:.2%}",
            subset=pd.IndexSlice[
                ["Breach rate", "Expected breach rate"],
                :
            ]
        )
        .format(
            "{:.4f}",
            subset=pd.IndexSlice[
                [
                    "Kupiec p-value",
                    "Independence p-value",
                    "Conditional coverage p-value"
                ],
                :
            ]
        )
    )
    st.dataframe(model_comparison_style, width="stretch")
    st.caption(
        "At the 5% significance level, p-values greater than or equal to "
        "0.05 mean the test does not reject the model; p-values below 0.05 "
        "indicate rejection."
    )


def display_analysis(results):
    """Organize engine results into the Streamlit dashboard."""
    overview_tab, risk_tab, stress_tab, validation_tab = st.tabs([
        "Overview",
        "Risk analysis",
        "Stress & risk contribution",
        "Model validation"
    ])

    with overview_tab:
        display_overview(results)

    with risk_tab:
        display_risk_analysis(results)

    with stress_tab:
        display_stress_and_risk_contribution(results)

    with validation_tab:
        display_model_validation(results)


def display_investor_analysis(results):
    """Display reconstructed accounting history and validated TWR results."""
    st.subheader("Investor performance summary")
    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "Current total portfolio value",
        f"${results['current_portfolio_value']:,.2f}",
        border=True
    )
    metric_columns[1].metric(
        "Current security market value",
        f"${results['current_security_value']:,.2f}",
        border=True
    )
    metric_columns[2].metric(
        "Current cash balance",
        f"${results['current_cash_balance']:,.2f}",
        border=True
    )
    metric_columns[3].metric(
        "Time-Weighted Return",
        f"{results['total_twr']:.2%}",
        border=True
    )
    st.caption(
        "Time-Weighted Return (TWR) measures portfolio performance "
        "independently of the timing and size of external deposits and "
        "withdrawals. It is not the investor's money-weighted IRR/XIRR; "
        "IRR and XIRR are not yet implemented."
    )

    st.divider()
    st.subheader("Actual historical portfolio value")
    st.caption(
        "This history reconstructs actual BUY/SELL transactions and "
        "DEPOSIT/WITHDRAWAL records. Unlike the Current Portfolio Risk "
        "chart, it does not apply today's allocation to the past."
    )
    portfolio_value = results["daily_portfolio_value"].rename(
        "Portfolio value"
    )
    st.line_chart(
        portfolio_value,
        x_label="Date",
        y_label="Portfolio value ($)",
        height=360
    )

    st.divider()
    st.subheader("Cumulative Time-Weighted Return")
    cumulative_twr_percent = (
        results["cumulative_twr"]
        .mul(100.0)
        .rename("Cumulative TWR (%)")
    )
    st.line_chart(
        cumulative_twr_percent,
        x_label="Date",
        y_label="Cumulative TWR (%)",
        height=360
    )
    st.caption(
        "The chart is shown in percentage points. TWR neutralizes external "
        "DEPOSIT and WITHDRAWAL flows; BUY and SELL remain internal trades."
    )

    st.divider()
    current_holdings_column, accounting_column = st.columns([2, 3])
    current_holdings = (
        results["current_holdings"]
        [["ticker", "quantity", "average_cost"]]
        .rename(columns={
            "ticker": "Ticker",
            "quantity": "Quantity",
            "average_cost": "Average cost"
        })
    )
    holdings_accounting = (
        results["holdings"]
        [[
            "ticker",
            "quantity",
            "average_cost",
            "total_cost_basis",
            "realized_pnl"
        ]]
        .rename(columns={
            "ticker": "Ticker",
            "quantity": "Quantity",
            "average_cost": "Average cost",
            "total_cost_basis": "Total cost basis",
            "realized_pnl": "Realized P&L"
        })
    )
    with current_holdings_column:
        st.subheader("Current open holdings")
        st.dataframe(
            current_holdings.style.format({
                "Quantity": "{:,.2f}",
                "Average cost": "${:,.2f}"
            }),
            width="stretch",
            hide_index=True
        )

    with accounting_column:
        st.subheader("Holdings accounting")
        st.dataframe(
            holdings_accounting.style.format({
                "Quantity": "{:,.2f}",
                "Average cost": "${:,.2f}",
                "Total cost basis": "${:,.2f}",
                "Realized P&L": "${:,.2f}"
            }),
            width="stretch",
            hide_index=True
        )

    st.divider()
    st.subheader("Cash and external flows")
    cash_and_flows = pd.concat(
        [
            results["daily_cash"].rename("Cash balance"),
            results["external_cash_flows_by_day"].rename(
                "External cash flow"
            )
        ],
        axis=1
    )
    cash_and_flows.index.name = "Date"
    st.dataframe(
        cash_and_flows.style.format({
            "Cash balance": "${:,.2f}",
            "External cash flow": "${:,.2f}"
        }),
        width="stretch",
        height=300
    )


def display_current_portfolio_risk_workflow():
    """Display the existing current-allocation risk workflow."""
    st.header("Current Portfolio Risk")
    st.caption(
        "Point-in-time risk analysis for today's positions. Historical "
        "results apply current weights and are not reconstructed investor "
        "performance."
    )

    st.subheader("Portfolio input")
    st.caption(
        "Add, remove or edit positions. Purchase price is used for "
        "unrealized P&L, while historical risk and performance are based on "
        "today's portfolio allocation rather than actual transaction history."
    )
    portfolio_input = st.data_editor(
        EXAMPLE_PORTFOLIO,
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        column_config={
            "ticker": st.column_config.TextColumn("Ticker"),
            "quantity": st.column_config.NumberColumn(
                "Quantity",
                format="%.2f"
            ),
            "purchase_price": st.column_config.NumberColumn(
                "Purchase price",
                format="$%.2f"
            )
        },
        key="portfolio_editor"
    )

    if st.button("Run analysis", type="primary", key="run_risk_analysis"):
        st.session_state.pop("analysis_results", None)

        try:
            stress_scenarios = load_stress_scenarios(STRESS_SCENARIOS_FILE)

            with st.spinner("Downloading market data and calculating risk..."):
                analysis_results = run_portfolio_analysis(
                    portfolio=portfolio_input,
                    stress_scenarios=stress_scenarios,
                    start_date=START_DATE
                )
        except ValueError as error:
            st.error(str(error))
        else:
            st.session_state["analysis_results"] = analysis_results

    if "analysis_results" in st.session_state:
        st.caption("Dashboard results reflect the last successful analysis run.")
        display_analysis(st.session_state["analysis_results"])


def display_investor_performance_workflow():
    """Display inputs and results for actual investor accounting history."""
    st.header("Investor Performance")
    st.caption(
        "Transaction-ledger accounting and cash-flow-aware performance. "
        "BUY/SELL are internal trades; DEPOSIT/WITHDRAWAL are external "
        "investor flows."
    )

    transaction_column, cash_flow_column = st.columns([3, 2])
    with transaction_column:
        st.subheader("Transactions")
        transactions_input = st.data_editor(
            EXAMPLE_TRANSACTIONS,
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            column_config={
                "date": st.column_config.DateColumn(
                    "Date",
                    format="YYYY-MM-DD"
                ),
                "ticker": st.column_config.TextColumn("Ticker"),
                "side": st.column_config.SelectboxColumn(
                    "Side",
                    options=["BUY", "SELL"]
                ),
                "quantity": st.column_config.NumberColumn(
                    "Quantity",
                    format="%.2f"
                ),
                "price": st.column_config.NumberColumn(
                    "Price",
                    format="$%.2f"
                )
            },
            key="transactions_editor"
        )

    with cash_flow_column:
        st.subheader("External cash flows")
        cash_flows_input = st.data_editor(
            EXAMPLE_CASH_FLOWS,
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            column_config={
                "date": st.column_config.DateColumn(
                    "Date",
                    format="YYYY-MM-DD"
                ),
                "type": st.column_config.SelectboxColumn(
                    "Type",
                    options=["DEPOSIT", "WITHDRAWAL"]
                ),
                "amount": st.column_config.NumberColumn(
                    "Amount",
                    format="$%.2f"
                )
            },
            key="cash_flows_editor"
        )

    if st.button(
        "Run investor analysis",
        type="primary",
        key="run_investor_analysis"
    ):
        try:
            with st.spinner(
                "Downloading market data and reconstructing investor history..."
            ):
                investor_results = run_investor_performance_analysis(
                    transactions=transactions_input,
                    external_cash_flows=cash_flows_input
                )
        except ValueError as error:
            st.error(str(error))
        else:
            st.session_state["investor_analysis_results"] = investor_results

    if "investor_analysis_results" in st.session_state:
        st.caption(
            "Results reflect the most recent successful investor analysis."
        )
        display_investor_analysis(
            st.session_state["investor_analysis_results"]
        )


st.set_page_config(
    page_title="Portfolio Risk Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("Portfolio Risk Analytics")
st.caption(
    "Two validated workflows for current-allocation risk and reconstructed "
    "investor performance."
)

current_risk_tab, investor_performance_tab = st.tabs([
    "Current Portfolio Risk",
    "Investor Performance"
])

with current_risk_tab:
    display_current_portfolio_risk_workflow()

with investor_performance_tab:
    display_investor_performance_workflow()
