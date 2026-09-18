from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.data import RISK_WINDOW_PRESETS
from src.importers.boursobank import import_boursobank_csv
from src.transactions import (
    NORMALIZED_LEDGER_COLUMNS,
    normalize_transaction_ledger,
    reconstruct_holdings,
    validate_transaction_ledger
)
from src.unified import run_unified_portfolio_analysis


PROJECT_DIR = Path(__file__).resolve().parent
TRANSACTIONS_EXAMPLE_FILE = PROJECT_DIR / "inputs" / "transactions_example.csv"
CASH_FLOWS_EXAMPLE_FILE = PROJECT_DIR / "inputs" / "cash_flows_example.csv"
STRESS_SCENARIOS_FILE = PROJECT_DIR / "inputs" / "stress_scenarios.csv"
NAVIGATION_SECTIONS = [
    "Overview",
    "Performance",
    "Risk",
    "Holdings",
    "Transactions",
    "Advanced"
]
RISK_VIEW_OPTIONS = ["Account", "Invested Securities"]
DEFAULT_RISK_WINDOW_INPUT = "3 years"
DEFAULT_RISK_FREE_RATE_PERCENT_INPUT = 0.0
DEFAULT_BENCHMARK_TICKER_INPUT = "SPY"

EXAMPLE_TRANSACTIONS = pd.read_csv(
    TRANSACTIONS_EXAMPLE_FILE,
    parse_dates=["date"]
)
EXAMPLE_CASH_FLOWS = pd.read_csv(
    CASH_FLOWS_EXAMPLE_FILE,
    parse_dates=["date"]
)
EXAMPLE_STRESS_SCENARIOS = pd.read_csv(
    STRESS_SCENARIOS_FILE,
    index_col="scenario"
)
EDITOR_STATE_KEYS = [
    "stress_scenarios_editor"
]


def empty_transaction_ledger():
    """Return an independent empty normalized transaction ledger."""
    return pd.DataFrame(columns=NORMALIZED_LEDGER_COLUMNS)


def empty_stress_scenarios():
    """Return an empty stress table compatible with the scenario editor."""
    return pd.DataFrame(index=pd.Index([], dtype="object", name="scenario"))


def invalidate_analysis():
    """Remove results and provenance after portfolio activity changes."""
    for key in (
        "unified_analysis",
        "analysis_completed_message",
        "analyzed_risk_window_label",
        "analyzed_risk_free_rate_percent",
        "analyzed_benchmark_ticker"
    ):
        st.session_state.pop(key, None)


def reset_analysis_settings():
    """Restore the canonical settings used for a fresh analysis."""
    st.session_state["risk_window_input"] = DEFAULT_RISK_WINDOW_INPUT
    st.session_state["risk_free_rate_percent_input"] = (
        DEFAULT_RISK_FREE_RATE_PERCENT_INPUT
    )
    st.session_state["benchmark_ticker_input"] = (
        DEFAULT_BENCHMARK_TICKER_INPUT
    )


def initialize_session_state():
    """Initialize a persistent empty portfolio and analysis settings."""
    if "transaction_ledger_input" not in st.session_state:
        st.session_state["transaction_ledger_input"] = (
            empty_transaction_ledger()
        )
    if "stress_scenarios_input" not in st.session_state:
        st.session_state["stress_scenarios_input"] = empty_stress_scenarios()
    if "risk_window_input" not in st.session_state:
        st.session_state["risk_window_input"] = DEFAULT_RISK_WINDOW_INPUT
    if "risk_free_rate_percent_input" not in st.session_state:
        st.session_state["risk_free_rate_percent_input"] = (
            DEFAULT_RISK_FREE_RATE_PERCENT_INPUT
        )
    if "benchmark_ticker_input" not in st.session_state:
        st.session_state["benchmark_ticker_input"] = (
            DEFAULT_BENCHMARK_TICKER_INPUT
        )
    # Keep widget-backed settings when their controls are not rendered on the
    # selected page. Self-assignment detaches them from Streamlit's widget
    # cleanup without changing their current values.
    for key in (
        "risk_window_input",
        "risk_free_rate_percent_input",
        "benchmark_ticker_input"
    ):
        st.session_state[key] = st.session_state[key]
    if "display_currency" not in st.session_state:
        st.session_state["display_currency"] = "EUR"
    if "show_add_transaction" not in st.session_state:
        st.session_state["show_add_transaction"] = False
    if "show_boursobank_import" not in st.session_state:
        st.session_state["show_boursobank_import"] = False
    if "show_transaction_manager" not in st.session_state:
        st.session_state["show_transaction_manager"] = False
    if "pending_delete_transaction_position" not in st.session_state:
        st.session_state["pending_delete_transaction_position"] = None
    if "transaction_manager_generation" not in st.session_state:
        st.session_state["transaction_manager_generation"] = 0


def navigate_to_transactions():
    """Move the top navigation to the portfolio-input page."""
    st.session_state["navigation"] = "Transactions"


def load_demo_portfolio():
    """Load the existing examples into one normalized working ledger."""
    st.session_state["transaction_ledger_input"] = (
        normalize_transaction_ledger(
            EXAMPLE_TRANSACTIONS,
            EXAMPLE_CASH_FLOWS
        )
        .sort_values("date", kind="stable")
        .reset_index(drop=True)
    )
    invalidate_analysis()
    reset_analysis_settings()
    st.session_state["show_add_transaction"] = False
    st.session_state["show_boursobank_import"] = False
    st.session_state["show_transaction_manager"] = False
    st.session_state["pending_delete_transaction_position"] = None
    st.session_state["stress_scenarios_input"] = (
        EXAMPLE_STRESS_SCENARIOS.copy(deep=True)
    )
    st.session_state["display_currency"] = "USD"
    st.session_state.pop("stress_scenarios_editor", None)
    st.session_state["transaction_action_message"] = (
        "Demo portfolio loaded. Run the analysis when ready."
    )


def clear_portfolio():
    """Clear the ledger and all analysis state without restoring examples."""
    st.session_state["transaction_ledger_input"] = empty_transaction_ledger()
    invalidate_analysis()
    reset_analysis_settings()
    st.session_state["show_add_transaction"] = False
    st.session_state["show_boursobank_import"] = False
    st.session_state["show_transaction_manager"] = False
    st.session_state["pending_delete_transaction_position"] = None
    st.session_state["stress_scenarios_input"] = empty_stress_scenarios()
    st.session_state["display_currency"] = "EUR"
    st.session_state.pop("stress_scenarios_editor", None)
    st.session_state["transaction_action_message"] = "Portfolio cleared."


def show_add_transaction():
    st.session_state["show_add_transaction"] = True
    st.session_state["show_boursobank_import"] = False
    st.session_state["show_transaction_manager"] = False
    st.session_state["pending_delete_transaction_position"] = None


def show_boursobank_import():
    st.session_state["show_boursobank_import"] = True
    st.session_state["show_add_transaction"] = False
    st.session_state["show_transaction_manager"] = False
    st.session_state["pending_delete_transaction_position"] = None


def show_transaction_manager():
    st.session_state["show_transaction_manager"] = True
    st.session_state["show_add_transaction"] = False
    st.session_state["show_boursobank_import"] = False
    st.session_state["pending_delete_transaction_position"] = None
    st.session_state["transaction_manager_generation"] += 1


def display_pre_analysis_state():
    """Show a neutral prompt instead of empty analytical components."""
    st.info(
        "Add or review your portfolio activity in Transactions, then run "
        "the analysis."
    )
    st.button(
        "Go to Transactions",
        on_click=navigate_to_transactions,
        key="go_to_transactions"
    )


def display_currency_symbol():
    """Return the symbol for the session-level display currency."""
    return {"EUR": "€", "USD": "$"}[st.session_state["display_currency"]]


def format_currency(value, show_positive_sign=False):
    """Format positive and negative monetary values consistently."""
    if pd.isna(value):
        return "N/A"
    if value < 0:
        sign = "-"
    elif show_positive_sign and value > 0:
        sign = "+"
    else:
        sign = ""
    return f"{sign}{display_currency_symbol()}{abs(value):,.2f}"


def format_percentage(value, signed=False, decimal_places=2):
    """Format a decimal percentage with consistent sign and N/A behavior."""
    if pd.isna(value):
        return "N/A"
    sign = "+" if signed and value > 0 else ""
    return f"{sign}{value:.{decimal_places}%}"


def format_ratio(value):
    """Format a performance ratio or show an explicit unavailable state."""
    return "N/A" if pd.isna(value) else f"{value:.2f}"


def format_hhi(value):
    """Format a concentration index without presenting it as a percentage."""
    return "N/A" if pd.isna(value) else f"{value:.3f}"


def summarize_pnl(investor, risk):
    """Return realized, unrealized and economic total P&L outputs."""
    realized_pnl = investor["holdings"]["realized_pnl"].sum()
    unrealized_pnl = risk["positions"]["unrealized_pnl"].sum()
    return realized_pnl, unrealized_pnl, investor["economic_total_pnl"]


def summarize_concentration(risk):
    """Return the largest existing position and risk-contribution values."""
    largest_position = risk["weights"].idxmax()
    largest_position_weight = risk["weights"].loc[largest_position]
    risk_contributions = risk["risk_contribution_table"][
        "Contribution_risque_pct"
    ]
    largest_risk_contributor = risk_contributions.idxmax()
    largest_risk_contribution = risk_contributions.loc[
        largest_risk_contributor
    ]
    return (
        largest_position,
        largest_position_weight,
        largest_risk_contributor,
        largest_risk_contribution
    )


def risk_results_for_view(results, risk_view):
    """Return the precomputed result for one presentation-only risk view."""
    if risk_view == "Account":
        return results["account_risk"]
    if risk_view == "Invested Securities":
        return results["risk"]

    raise ValueError(f"Unsupported risk view: {risk_view}")


def analysis_requires_rerun():
    """Return whether any configurable analysis setting has changed."""
    results = st.session_state.get("unified_analysis")
    if results is None:
        return False
    analyzed_window = st.session_state.get(
        "analyzed_risk_window_label",
        results["metadata"].get("risk_window_label")
    )
    analyzed_risk_free_rate_percent = st.session_state.get(
        "analyzed_risk_free_rate_percent",
        results["investor"].get("risk_free_rate_annual", 0.0) * 100.0
    )
    analyzed_benchmark_ticker = st.session_state.get(
        "analyzed_benchmark_ticker",
        results.get("benchmark", {}).get("ticker", "SPY")
    )
    current_benchmark_ticker = str(
        st.session_state["benchmark_ticker_input"]
    ).strip().upper()
    return (
        analyzed_window != st.session_state["risk_window_input"]
        or abs(
            analyzed_risk_free_rate_percent
            - st.session_state["risk_free_rate_percent_input"]
        ) > 1e-12
        or analyzed_benchmark_ticker != current_benchmark_ticker
    )


def _risk_window_display(label):
    """Return compact prose for one backend risk-window label."""
    if label.endswith(" years"):
        return f"{label.removesuffix(' years')}-year window"
    if label == "1 year":
        return "1-year window"
    if label == "Maximum available":
        return "maximum-available window"
    return "explicit-date window"


def risk_history_caption(results, risk_view):
    """Describe the current-allocation method and exact sample size."""
    metadata = results["metadata"]
    window = _risk_window_display(metadata["risk_window_label"])
    observations = metadata["risk_observations"]
    if risk_view == "Account":
        return (
            f"Account Risk · {window} · {observations:,} daily observations "
            "· constant current weights; cash has zero return."
        )

    return (
        f"Invested Securities · {window} · {observations:,} daily "
        "observations · constant current weights; cash is excluded."
    )


def portfolio_bar_chart_height(asset_count):
    """Return a compact chart height that grows with the asset count."""
    return max(260, min(520, 220 + 30 * asset_count))


def display_security_percentage_bars(
    values,
    value_label,
    height=None,
    show_security_axis_title=True
):
    """Display descending security percentages as horizontal bars."""
    chart_values = (
        values
        .mul(100.0)
        .sort_values(ascending=False)
        .rename(value_label)
    )
    st.bar_chart(
        chart_values,
        horizontal=True,
        # Streamlit swaps the configured axes for horizontal bars.
        x_label="Security" if show_security_axis_title else "",
        y_label=value_label,
        height=(
            height
            if height is not None
            else portfolio_bar_chart_height(len(chart_values))
        )
    )


def display_performance_chart(series, value_label, height, value_format):
    """Display a performance series within its exact observed date range."""
    chart_data = (
        series
        .rename("Value")
        .rename_axis("Date")
        .reset_index()
    )
    date_domain = [series.index.min(), series.index.max()]
    chart = (
        alt.Chart(chart_data)
        .mark_line()
        .encode(
            x=alt.X(
                "Date:T",
                title="Date",
                scale=alt.Scale(domain=date_domain, nice=False),
                axis=alt.Axis(
                    format="%b %Y",
                    tickCount=5,
                    labelAngle=0
                )
            ),
            y=alt.Y(
                "Value:Q",
                title=value_label,
                scale=alt.Scale(zero=False),
                axis=alt.Axis(format=value_format)
            ),
            tooltip=[
                alt.Tooltip("Date:T", title="Date", format="%Y-%m-%d"),
                alt.Tooltip(
                    "Value:Q",
                    title=value_label,
                    format=value_format
                )
            ]
        )
        .properties(height=height)
    )
    st.altair_chart(chart, width="stretch")


def display_benchmark_comparison_chart(benchmark):
    """Display aligned investor and benchmark cumulative percentage returns."""
    chart_data = (
        pd.concat(
            [
                benchmark["investor_cumulative_aligned"],
                benchmark["benchmark_cumulative_return"]
            ],
            axis=1
        )
        .mul(100.0)
        .rename_axis("Date")
        .reset_index()
        .melt(
            id_vars="Date",
            var_name="Series",
            value_name="Cumulative Return (%)"
        )
    )
    chart = (
        alt.Chart(chart_data)
        .mark_line()
        .encode(
            x=alt.X(
                "Date:T",
                title="Date",
                axis=alt.Axis(
                    format="%b %Y",
                    tickCount=5,
                    labelAngle=0
                )
            ),
            y=alt.Y(
                "Cumulative Return (%):Q",
                title="Cumulative Return (%)",
                axis=alt.Axis(format=".2f")
            ),
            color=alt.Color("Series:N", title=None),
            tooltip=[
                alt.Tooltip("Date:T", title="Date", format="%Y-%m-%d"),
                alt.Tooltip("Series:N", title="Series"),
                alt.Tooltip(
                    "Cumulative Return (%):Q",
                    title="Cumulative Return (%)",
                    format=".2f"
                )
            ]
        )
        .properties(height=340)
    )
    st.altair_chart(chart, width="stretch")


def apply_dashboard_styles():
    """Apply restrained layout and metric-card styling to dashboard pages."""
    st.markdown(
        """
        <style>
        [data-testid="stMainBlockContainer"] {
            max-width: 1180px;
            margin: 0 auto;
            padding-top: 2.25rem;
        }
        div[data-testid="stMetric"] {
            min-height: 98px;
            padding: 0.75rem 0.9rem;
            border-radius: 0.7rem;
        }
        div[data-testid="stMetricValue"] {
            font-size: clamp(1.45rem, 2vw, 1.9rem);
            font-weight: 650;
            letter-spacing: -0.02em;
        }
        @media (max-width: 700px) {
            [data-testid="stMainBlockContainer"] {
                padding-left: 1rem;
                padding-right: 1rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def display_accounting_breakdown(investor, realized_pnl, unrealized_pnl):
    """Display the existing accounting summary as one compact table."""
    accounting_rows = [
        ("Realized Market P&L", realized_pnl, True),
        ("Unrealized Market P&L", unrealized_pnl, True),
        ("Dividend Income", investor["dividend_income"], False),
        ("Interest Income", investor["interest_income"], False),
        ("Fees Paid", investor["fees_paid"], False),
        ("Taxes Paid", investor["taxes_paid"], False),
        (
            "Net External Contributions",
            investor["net_external_contributions"],
            True
        ),
        ("Economic Total P&L", investor["economic_total_pnl"], True)
    ]
    breakdown = pd.DataFrame({
        "Accounting component": [row[0] for row in accounting_rows],
        "Amount": [
            format_currency(value, show_positive_sign=signed)
            for _, value, signed in accounting_rows
        ]
    })
    st.dataframe(
        breakdown,
        hide_index=True,
        width="stretch",
        height=318
    )


def display_overview(results):
    """Display the investor-readable portfolio dashboard."""
    investor = results["investor"]
    security_risk = results["risk"]
    account_risk = results["account_risk"]

    _, _, total_pnl = summarize_pnl(investor, security_risk)
    (
        largest_position,
        largest_position_weight,
        largest_risk_contributor,
        largest_risk_contribution
    ) = summarize_concentration(security_risk)

    if security_risk.get("stress_test_available", True):
        worst_stress_scenario = account_risk["stress_results"]["P&L"].idxmin()
        worst_stress_pnl = account_risk["stress_results"].loc[
            worst_stress_scenario,
            "P&L"
        ]
        worst_stress_return = account_risk["stress_results"].loc[
            worst_stress_scenario,
            "Portfolio_return"
        ]
        worst_stress_display = (
            f"{worst_stress_scenario} · "
            f"{format_percentage(worst_stress_return, signed=True, decimal_places=1)} "
            f"({format_currency(worst_stress_pnl, show_positive_sign=True)})"
        )
    else:
        worst_stress_display = "Unavailable"

    total_account_value = investor["current_portfolio_value"]
    cash_weight = results["metadata"]["cash_weight"]

    apply_dashboard_styles()

    st.header("Overview")
    st.subheader("Portfolio Summary")
    account_columns = st.columns(4, gap="medium")
    account_columns[0].metric(
        "Total Account Value",
        format_currency(total_account_value),
        border=True
    )
    account_columns[1].metric(
        "Total P&L",
        format_currency(total_pnl, show_positive_sign=True),
        help=(
            "Economic P&L: realized and unrealized market P&L plus recorded "
            "dividend and interest income, less recorded fees and taxes."
        ),
        border=True
    )
    account_columns[2].metric(
        "Time-Weighted Return",
        format_percentage(investor["total_twr"], signed=True),
        help=(
            "Geometrically linked daily account returns using a "
            "beginning-of-day external cash-flow convention. BUY and SELL "
            "transactions are internal flows; valuations use raw Close."
        ),
        border=True
    )
    account_columns[3].metric(
        "Cash",
        format_currency(investor["current_cash_balance"]),
        border=True
    )
    valuation_reference_date = results["metadata"][
        "valuation_reference_date"
    ]
    st.caption(
        f"Valuation as of {valuation_reference_date:%Y-%m-%d}. Account value "
        "and investor TWR use raw security closing prices plus the recorded "
        "cash ledger. Income and costs are included; deposits and withdrawals "
        "are neutralized. Splits, mergers and spin-offs are not supported."
    )

    st.subheader("Investor Account Value")
    st.line_chart(
        investor["daily_portfolio_value"].rename("Account value"),
        x_label="Date",
        y_label=f"Account value ({st.session_state['display_currency']})",
        height=340
    )
    st.caption("Reconstructed account value from actual investor history.")

    allocation_column, snapshot_column = st.columns(
        [3, 2],
        gap="large"
    )
    with allocation_column:
        st.subheader("Current Allocation")
        st.caption("Invested securities only; cash is excluded.")
        display_security_percentage_bars(
            security_risk["weights"],
            value_label="Weight (%)",
            height=280
        )

    with snapshot_column:
        st.subheader("Account Snapshot")
        with st.container(border=True):
            snapshot_items = [
                (
                    "Largest Position",
                    f"{largest_position} · "
                    f"{format_percentage(largest_position_weight, decimal_places=1)}"
                ),
                (
                    "Largest Risk Contributor",
                    f"{largest_risk_contributor} · "
                    f"{format_percentage(largest_risk_contribution, decimal_places=1)}"
                ),
                (
                    "Worst Stress Scenario",
                    worst_stress_display
                ),
                (
                    "Cash Weight",
                    format_percentage(cash_weight, decimal_places=1)
                )
            ]
            for label, value in snapshot_items:
                label_column, value_column = st.columns([3, 2])
                label_column.caption(label)
                value_column.markdown(f"**{value}**")

    st.subheader("Account Risk")
    risk_columns = st.columns(4, gap="medium")
    risk_columns[0].metric(
        "Annualized Volatility",
        format_percentage(account_risk["annual_volatility"]),
        help=(
            "Sample volatility of daily current-allocation returns, "
            "annualized with sqrt(252)."
        )
    )
    risk_columns[1].metric(
        "Historical VaR 95%",
        format_percentage(account_risk["historical_var_95"]),
        help=(
            "Negative of the 5th percentile of daily returns, shown as a "
            "one-trading-day loss measure."
        )
    )
    risk_columns[2].metric(
        "Expected Shortfall 95%",
        format_percentage(account_risk["expected_shortfall_95"]),
        help=(
            "Negative mean daily return at or below the Historical VaR 95% "
            "return threshold."
        )
    )
    risk_columns[3].metric(
        "Max Drawdown",
        format_percentage(account_risk["max_drawdown"]),
        help=(
            "Largest peak-to-trough decline in the fixed-weight historical "
            "simulation of today's total account allocation, including "
            "zero-return current cash; this is not actual historical account "
            "drawdown."
        )
    )
    st.caption(risk_history_caption(results, "Account"))


def display_performance(results):
    """Display existing actual investor-history performance outputs."""
    investor = results["investor"]
    risk = results["risk"]
    benchmark = results["benchmark"]
    realized_pnl, unrealized_pnl, total_pnl = summarize_pnl(investor, risk)

    apply_dashboard_styles()

    st.header("Performance")
    st.caption(
        "Ledger-based investor TWR includes recorded income and costs while "
        "neutralizing deposits and withdrawals. It is not IRR or XIRR; "
        "unsupported corporate actions remain a limitation."
    )

    st.subheader("Primary Performance")
    primary_metric_columns = st.columns(4, gap="medium")
    primary_metric_columns[0].metric(
        "Time-Weighted Return",
        format_percentage(investor["total_twr"], signed=True),
        help=(
            "Geometrically linked daily account returns using a "
            "beginning-of-day external cash-flow convention and raw closing "
            "prices."
        ),
        border=True
    )
    primary_metric_columns[1].metric(
        "Annualized TWR",
        format_percentage(investor["annualized_twr"], signed=True),
        help=(
            "Geometrically annualized performance over active TWR "
            "observations using 252 trading days per year."
        ),
        border=True
    )
    primary_metric_columns[2].metric(
        "Investor Max Drawdown",
        format_percentage(investor["investor_max_drawdown"]),
        help=(
            "Largest peak-to-trough decline in actual cash-flow-neutralized "
            "investor performance, derived from the TWR wealth index. This "
            "is distinct from current-allocation risk drawdown."
        ),
        border=True
    )
    primary_metric_columns[3].metric(
        "Total P&L",
        format_currency(total_pnl, show_positive_sign=True),
        border=True
    )

    st.subheader("Risk-Adjusted Performance")
    ratio_columns = st.columns(3, gap="medium")
    ratio_columns[0].metric(
        "Sharpe Ratio",
        format_ratio(investor["sharpe_ratio"]),
        help="Annualized excess return per unit of total volatility."
    )
    ratio_columns[1].metric(
        "Sortino Ratio",
        format_ratio(investor["sortino_ratio"]),
        help=(
            "Annualized excess return relative to full-sample downside "
            "deviation."
        )
    )
    ratio_columns[2].metric(
        "Calmar Ratio",
        format_ratio(investor["calmar_ratio"]),
        help=(
            "Annualized TWR relative to actual investor maximum drawdown."
        )
    )
    st.caption(
        f"{investor['performance_observations']:,} active daily TWR "
        "observations | 252-observation annualization | annual risk-free "
        f"rate {investor['risk_free_rate_annual']:.2%}. Inactive zero-capital "
        "dates are excluded."
    )

    st.subheader(f"Benchmark Comparison · {benchmark['ticker']}")
    benchmark_primary_columns = st.columns(3, gap="medium")
    benchmark_primary_columns[0].metric(
        "Benchmark Return",
        format_percentage(
            benchmark["benchmark_total_return"],
            signed=True
        ),
        help="Geometrically linked adjusted-Close benchmark return.",
        border=True
    )
    benchmark_primary_columns[1].metric(
        "Benchmark Annualized Return",
        format_percentage(
            benchmark["benchmark_annualized_return"],
            signed=True
        ),
        help=(
            "Geometrically annualized benchmark price return over aligned "
            "active observations using 252 trading days."
        ),
        border=True
    )
    benchmark_primary_columns[2].metric(
        "Beta",
        format_ratio(benchmark["beta"]),
        help=(
            "Sample covariance of investor and benchmark returns divided by "
            "benchmark sample variance."
        ),
        border=True
    )

    benchmark_secondary_columns = st.columns(3, gap="medium")
    benchmark_secondary_columns[0].metric(
        "Annualized Alpha",
        format_percentage(
            benchmark["alpha_annualized"],
            signed=True
        ),
        help=(
            "Daily Jensen-alpha intercept, using the configured compounded "
            "daily risk-free rate, arithmetically annualized by 252."
        ),
        border=True
    )
    benchmark_secondary_columns[1].metric(
        "Tracking Error",
        format_percentage(benchmark["tracking_error"]),
        help=(
            "Sample standard deviation of active returns annualized by "
            "square root of 252."
        ),
        border=True
    )
    benchmark_secondary_columns[2].metric(
        "Information Ratio",
        format_ratio(benchmark["information_ratio"]),
        help="Annualized mean active return per unit of tracking error.",
        border=True
    )
    st.caption(
        f"{benchmark['observations']:,} exact-date aligned active "
        "observations. The benchmark uses adjusted Close as a total-return-"
        "style comparison. Investor performance uses the actual ledger and "
        "is net of recorded fees and taxes; unrecorded distributions or "
        "corporate actions can still limit comparability."
    )
    if benchmark["observations"] < 2:
        st.info(
            "At least two exact-date aligned active observations are "
            "required for Beta, Annualized Alpha, Tracking Error and "
            "Information Ratio."
        )

    st.subheader("Investor vs Benchmark")
    if benchmark["observations"] == 0:
        st.info(
            "No exact-date overlap is available between active Investor "
            "History TWR and benchmark returns."
        )
    else:
        display_benchmark_comparison_chart(benchmark)

    st.subheader("Accounting Breakdown")
    display_accounting_breakdown(investor, realized_pnl, unrealized_pnl)

    st.subheader("Investor History")
    drawdown_tab, account_value_tab = st.tabs([
        "Drawdown", "Account Value"
    ])
    with drawdown_tab:
        investor_drawdown_percent = (
            investor["investor_drawdown"]
            .mul(100.0)
            .rename("Investor Drawdown (%)")
        )
        display_performance_chart(
            investor_drawdown_percent,
            value_label="Investor Drawdown (%)",
            height=220,
            value_format=".2f"
        )
        st.caption(
            "Cash-flow-neutralized drawdown from actual investor TWR history."
        )
    with account_value_tab:
        st.line_chart(
            investor["daily_portfolio_value"].rename("Account value"),
            x_label="Date",
            y_label=(
                f"Account value ({st.session_state['display_currency']})"
            ),
            height=240
        )


def display_risk(results):
    """Display a selected precomputed current-allocation risk view."""

    apply_dashboard_styles()

    st.header("Risk")
    risk_view = st.segmented_control(
        "Risk view",
        RISK_VIEW_OPTIONS,
        default="Account",
        key="risk_view"
    )
    if risk_view is None:
        risk_view = "Account"
    risk = risk_results_for_view(results, risk_view)
    st.caption(risk_history_caption(results, risk_view))
    metadata = results["metadata"]
    st.caption(
        "Historical Account Risk and Invested Securities risk hold today's "
        "selected allocation constant through the risk window (equivalent "
        "to daily rebalancing)."
    )
    with st.expander("Methodology details", expanded=False):
        st.write(
            "Risk sample: "
            f"{metadata['risk_start_date']:%Y-%m-%d} to "
            f"{metadata['risk_end_date']:%Y-%m-%d}."
        )
        st.write(
            "Alignment: "
            f"{metadata['raw_price_observations']:,} price rows; "
            f"{metadata['asset_return_observations_before_alignment']:,} "
            "candidate return dates; "
            f"{metadata['observations_dropped_during_alignment']:,} dates "
            "excluded because at least one held security lacked a usable "
            "return."
        )
        usable_counts = metadata["usable_return_observations_by_asset"]
        st.write(
            "Usable returns by security: "
            + ", ".join(
                f"{ticker} {count:,}"
                for ticker, count in usable_counts.items()
            )
            + "."
        )

    metric_columns = st.columns(4, gap="medium")
    metric_columns[0].metric(
        "Annualized Volatility",
        format_percentage(risk["annual_volatility"]),
        help=(
            "Sample volatility of daily current-allocation returns, "
            "annualized with sqrt(252)."
        ),
        border=True
    )
    metric_columns[1].metric(
        "Historical VaR 95%",
        format_percentage(risk["historical_var_95"]),
        help=(
            "Negative of the 5th percentile of daily returns, shown as a "
            "one-trading-day loss measure."
        ),
        border=True
    )
    metric_columns[2].metric(
        "Expected Shortfall 95%",
        format_percentage(risk["expected_shortfall_95"]),
        help=(
            "Negative mean daily return at or below the Historical VaR 95% "
            "return threshold."
        ),
        border=True
    )
    metric_columns[3].metric(
        "Max Drawdown",
        format_percentage(risk["max_drawdown"]),
        help=(
            "Largest peak-to-trough decline in the fixed-weight historical "
            "simulation of today's selected current allocation; this is not "
            "actual investor-history drawdown."
        ),
        border=True
    )

    st.subheader("Risk Contribution")
    display_security_percentage_bars(
        risk["risk_contribution_table"]["Contribution_risque_pct"],
        value_label="Risk Contribution (%)",
        show_security_axis_title=False
    )
    if risk_view == "Account":
        st.caption(
            "Cash retains its current account weight but contributes 0% of "
            "risk under the zero-return assumption."
        )
    else:
        st.caption(
            "Percentage contributions are within invested securities only."
        )

    st.subheader("Stress Testing")
    stress_status = results["risk"]
    if not stress_status.get("stress_test_available", True):
        st.info(stress_status["stress_test_reason"])
        return
    if risk_view == "Account":
        percentage_label = "Account Impact (%)"
        dollar_label = (
            f"Account Impact ({st.session_state['display_currency']})"
        )
        st.caption(
            "Existing security scenarios relative to total account value; "
            "cash is unaffected."
        )
    else:
        percentage_label = "Security-Sleeve Impact (%)"
        dollar_label = (
            f"Security-Sleeve Impact ({st.session_state['display_currency']})"
        )
        st.caption("Existing scenarios applied to invested securities only.")
    stress_results = (
        risk["stress_results"]
        .rename_axis("Scenario")
        .reset_index()
        .rename(columns={
            "Portfolio_return": percentage_label,
            "P&L": dollar_label
        })
    )
    stress_results[percentage_label] = stress_results[
        percentage_label
    ].map(lambda value: format_percentage(value, signed=True))
    stress_results[dollar_label] = stress_results[
        dollar_label
    ].map(lambda value: format_currency(value, show_positive_sign=True))
    st.dataframe(stress_results, width="stretch", hide_index=True)


def display_holdings(results):
    """Display one aligned table for current holdings and calculated risk."""
    investor = results["investor"]
    risk = results["risk"]
    diversification = results["diversification"]
    (
        largest_position,
        largest_position_weight,
        largest_risk_contributor,
        largest_risk_contribution
    ) = summarize_concentration(risk)

    apply_dashboard_styles()

    st.header("Holdings")
    st.caption(
        "Open security positions reconstructed from the transaction ledger."
    )

    summary_columns = st.columns(4, gap="medium")
    summary_columns[0].metric(
        "Securities Value",
        format_currency(investor["current_security_value"]),
        border=True
    )
    summary_columns[1].metric(
        "Cash",
        format_currency(investor["current_cash_balance"]),
        border=True
    )
    summary_columns[2].metric(
        "Largest Position",
        f"{largest_position} · "
        f"{format_percentage(largest_position_weight, decimal_places=1)}",
        border=True
    )
    summary_columns[3].metric(
        "Largest Risk Contributor",
        f"{largest_risk_contributor} · "
        f"{format_percentage(largest_risk_contribution, decimal_places=1)}",
        border=True
    )

    st.subheader("Diversification")
    st.caption(
        "Diversification diagnostics apply to invested securities only; "
        "cash is excluded."
    )
    diversification_columns = st.columns(4, gap="medium")
    diversification_columns[0].metric(
        "Effective Holdings",
        format_ratio(diversification["effective_number_of_holdings"]),
        help=(
            "Number of equally weighted positions corresponding to current "
            "weight concentration."
        ),
        border=True
    )
    diversification_columns[1].metric(
        "Concentration HHI",
        format_hhi(diversification["weight_hhi"]),
        help="Sum of squared security weights; higher means more concentrated.",
        border=True
    )
    diversification_columns[2].metric(
        "Diversification Ratio",
        format_ratio(diversification["diversification_ratio"]),
        help=(
            "Weighted average standalone volatility divided by portfolio "
            "volatility."
        ),
        border=True
    )
    diversification_columns[3].metric(
        "Effective Risk Contributors",
        format_ratio(diversification["effective_risk_contributors"]),
        help=(
            "Effective number of contributors based on absolute Euler "
            "volatility contributions."
        ),
        border=True
    )

    risk_contribution = (
        risk["risk_contribution_table"]
        [["Contribution_risque_pct"]]
        .rename(columns={
            "Contribution_risque_pct": "Risk Contribution"
        })
    )
    holdings = (
        risk["positions"]
        .merge(
            risk_contribution,
            left_on="ticker",
            right_index=True,
            how="left",
            validate="one_to_one"
        )
        [[
            "ticker",
            "quantity",
            "current_price",
            "price_as_of",
            "market_value",
            "weight",
            "purchase_price",
            "unrealized_pnl",
            "Risk Contribution"
        ]]
        .rename(columns={
            "ticker": "Ticker",
            "quantity": "Quantity",
            "current_price": "Current Price",
            "price_as_of": "Price As Of",
            "market_value": "Market Value",
            "weight": "Security Weight",
            "purchase_price": "Average Cost",
            "unrealized_pnl": "Unrealized P&L"
        })
    )
    st.subheader("Current Holdings")
    st.dataframe(
        holdings.style.format({
            "Quantity": "{:,.2f}",
            "Current Price": format_currency,
            "Price As Of": lambda value: value.strftime("%Y-%m-%d"),
            "Market Value": format_currency,
            "Security Weight": "{:.2%}",
            "Average Cost": format_currency,
            "Unrealized P&L": lambda value: format_currency(
                value,
                show_positive_sign=True
            ),
            "Risk Contribution": "{:.2%}"
        }),
        width="stretch",
        hide_index=True
    )

    st.subheader("Current Allocation")
    st.caption("Invested securities only; cash is excluded.")
    display_security_percentage_bars(
        risk["weights"],
        value_label="Weight (%)",
        show_security_axis_title=False
    )

    with st.expander("Accounting details"):
        st.caption(
            "Average-cost accounting for all ledger assets, including any "
            "closed positions."
        )
        accounting = (
            investor["holdings"]
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
                "average_cost": "Average Cost",
                "total_cost_basis": "Total Cost Basis",
                "realized_pnl": "Realized P&L"
            })
        )
        st.dataframe(
            accounting.style.format({
                "Quantity": "{:,.2f}",
                "Average Cost": format_currency,
                "Total Cost Basis": format_currency,
                "Realized P&L": lambda value: format_currency(
                    value,
                    show_positive_sign=True
                )
            }),
            width="stretch",
            hide_index=True
        )


TRANSACTION_TYPE_LABELS = {
    "BUY": "Buy",
    "SELL": "Sell",
    "DIVIDEND": "Dividend",
    "DEPOSIT": "Deposit",
    "WITHDRAWAL": "Withdrawal",
    "FEE": "Fee",
    "TAX": "Tax",
    "INTEREST": "Interest"
}


def _store_transaction_ledger(ledger):
    """Validate, sort and store activity while invalidating old results."""
    validated = validate_transaction_ledger(ledger)
    reconstruct_holdings(validated)
    st.session_state["transaction_ledger_input"] = (
        validated.sort_values("date", kind="stable").reset_index(drop=True)
    )
    invalidate_analysis()


def _transaction_history_for_display(ledger):
    """Return a newest-first, human-readable transaction history."""
    history = ledger.sort_values("date", ascending=False, kind="stable").copy()
    price_or_amount = history["price"].where(
        history["type"].isin(["BUY", "SELL"]),
        history["amount"]
    )
    return pd.DataFrame({
        "Date": pd.to_datetime(history["date"]).dt.date,
        "Type": history["type"].map(TRANSACTION_TYPE_LABELS),
        "Security / ticker": history["ticker"].fillna("—"),
        "Quantity": history["quantity"],
        "Price or Amount": price_or_amount,
        "Fees": history["fees"],
        "Taxes": history["taxes"]
    }).reset_index(drop=True)


def _transaction_management_label(row):
    """Describe one underlying ledger row without using display row numbers."""
    date_label = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
    type_label = TRANSACTION_TYPE_LABELS[row["type"]]
    security_label = row["ticker"] if pd.notna(row["ticker"]) else "Cash"
    if row["type"] in {"BUY", "SELL"}:
        value_label = f"quantity {row['quantity']:g}"
    else:
        value_label = f"amount €{row['amount']:,.2f}"
    return " | ".join([
        date_label,
        type_label,
        str(security_label),
        value_label
    ])


def _delete_transaction(position):
    """Delete one underlying chronological ledger position safely."""
    ledger = st.session_state["transaction_ledger_input"]
    remaining = ledger.drop(index=ledger.index[position]).reset_index(drop=True)
    if remaining.empty:
        st.session_state["transaction_ledger_input"] = (
            empty_transaction_ledger()
        )
        invalidate_analysis()
    else:
        _store_transaction_ledger(remaining)


def _display_transaction_manager(ledger):
    """Select and confirm deletion of exactly one underlying ledger row."""
    newest_first_positions = list(reversed(range(len(ledger))))
    generation = st.session_state["transaction_manager_generation"]
    selected_position = st.selectbox(
        "Transaction to manage",
        options=newest_first_positions,
        format_func=lambda position: _transaction_management_label(
            ledger.iloc[position]
        ),
        key=f"manage_transaction_position_{generation}"
    )
    selected_label = _transaction_management_label(
        ledger.iloc[selected_position]
    )

    if st.button("Delete transaction", key="request_transaction_deletion"):
        st.session_state["pending_delete_transaction_position"] = (
            selected_position
        )

    pending_position = st.session_state[
        "pending_delete_transaction_position"
    ]
    if pending_position == selected_position:
        st.warning(f"Delete this transaction? {selected_label}")
        confirmation_columns = st.columns(2, gap="small")
        confirm = confirmation_columns[0].button(
            "Confirm deletion",
            type="primary",
            key="confirm_transaction_deletion"
        )
        cancel = confirmation_columns[1].button(
            "Cancel",
            key="cancel_transaction_deletion"
        )

        if cancel:
            st.session_state["show_transaction_manager"] = False
            st.session_state["pending_delete_transaction_position"] = None
            st.rerun()

        if confirm:
            try:
                _delete_transaction(selected_position)
            except ValueError as error:
                st.error(str(error))
            else:
                st.session_state["show_transaction_manager"] = False
                st.session_state["pending_delete_transaction_position"] = None
                st.session_state["transaction_action_message"] = (
                    "Transaction deleted."
                )
                st.rerun()


def _display_add_transaction_form():
    """Render type-specific controls and append one validated ledger row."""
    with st.form("add_transaction_form", clear_on_submit=False):
        transaction_type = st.selectbox(
            "Type",
            options=list(TRANSACTION_TYPE_LABELS),
            key="new_transaction_type"
        )
        transaction_date = st.date_input("Date", key="new_transaction_date")
        row = {
            "date": transaction_date,
            "type": transaction_type,
            "ticker": None,
            "quantity": None,
            "price": None,
            "amount": None,
            "fees": 0.0,
            "taxes": 0.0
        }

        if transaction_type in {"BUY", "SELL"}:
            row["ticker"] = st.text_input("Ticker", key="new_trade_ticker")
            row["quantity"] = st.number_input(
                "Quantity", value=1.0, min_value=0.0,
                key="new_trade_quantity"
            )
            row["price"] = st.number_input(
                "Price", value=1.0, min_value=0.0,
                key="new_trade_price"
            )
            row["fees"] = st.number_input(
                "Fees", value=0.0, min_value=0.0,
                key="new_trade_fees"
            )
            row["taxes"] = st.number_input(
                "Taxes", value=0.0, min_value=0.0,
                key="new_trade_taxes"
            )
        elif transaction_type == "DIVIDEND":
            row["ticker"] = st.text_input(
                "Ticker", key="new_dividend_ticker"
            )
            row["amount"] = st.number_input(
                "Gross amount", value=1.0, min_value=0.0,
                key="new_dividend_amount"
            )
            row["fees"] = st.number_input(
                "Fees", value=0.0, min_value=0.0,
                key="new_dividend_fees"
            )
            row["taxes"] = st.number_input(
                "Taxes", value=0.0, min_value=0.0,
                key="new_dividend_taxes"
            )
        elif transaction_type == "INTEREST":
            row["amount"] = st.number_input(
                "Gross amount", value=1.0, min_value=0.0,
                key="new_interest_amount"
            )
            row["fees"] = st.number_input(
                "Fees", value=0.0, min_value=0.0,
                key="new_interest_fees"
            )
            row["taxes"] = st.number_input(
                "Taxes", value=0.0, min_value=0.0,
                key="new_interest_taxes"
            )
        else:
            row["amount"] = st.number_input(
                "Amount", value=1.0, min_value=0.0,
                key="new_cash_amount"
            )

        form_actions = st.columns(2, gap="small")
        submitted = form_actions[0].form_submit_button(
            "Add transaction",
            type="primary"
        )
        cancelled = form_actions[1].form_submit_button("Cancel")

    if cancelled:
        st.session_state["show_add_transaction"] = False
        st.rerun()

    if submitted:
        current_ledger = st.session_state["transaction_ledger_input"]
        candidate = pd.concat(
            [current_ledger, pd.DataFrame([row])],
            ignore_index=True
        )
        try:
            _store_transaction_ledger(candidate)
        except ValueError as error:
            st.error(str(error))
        else:
            st.session_state["show_add_transaction"] = False
            st.session_state["transaction_action_message"] = (
                "Transaction added."
            )
            st.rerun()


def _display_boursobank_import():
    """Render the BoursoBank uploader and replace the working ledger."""
    uploaded_file = st.file_uploader(
        "BoursoBank CSV",
        type=["csv"],
        key="boursobank_csv_upload"
    )
    import_clicked = st.button(
        "Import uploaded CSV",
        disabled=uploaded_file is None,
        key="import_boursobank_csv"
    )
    if import_clicked:
        try:
            imported_ledger = import_boursobank_csv(uploaded_file)
            _store_transaction_ledger(imported_ledger)
        except (ValueError, OSError) as error:
            st.error(f"Could not import BoursoBank CSV: {error}")
        else:
            st.session_state["show_boursobank_import"] = False
            st.session_state["display_currency"] = "EUR"
            st.session_state["stress_scenarios_input"] = empty_stress_scenarios()
            st.session_state.pop("stress_scenarios_editor", None)
            st.session_state["transaction_action_message"] = (
                f"Imported {len(imported_ledger)} transactions from "
                "BoursoBank."
            )
            st.rerun()


def display_transactions():
    """Display transaction entry, history and the existing analysis inputs."""
    show_analysis_success = st.session_state.pop(
        "analysis_completed_message",
        False
    )
    apply_dashboard_styles()

    st.header("Transactions")
    st.caption("Build or import your portfolio activity, then analyze it.")

    ledger = st.session_state["transaction_ledger_input"]
    if ledger.empty:
        st.subheader("No transactions yet")
        st.caption(
            "Add activity manually, import a BoursoBank export, or start "
            "from the optional demo portfolio."
        )

    action_columns = st.columns(4, gap="small")
    action_columns[0].button(
        "Add transaction",
        on_click=show_add_transaction,
        key="show_add_transaction_button"
    )
    action_columns[1].button(
        "Import BoursoBank CSV",
        on_click=show_boursobank_import,
        key="show_boursobank_import_button"
    )
    action_columns[2].button(
        "Load demo portfolio",
        on_click=load_demo_portfolio,
        key="load_demo_portfolio"
    )
    action_columns[3].button(
        "Clear portfolio",
        on_click=clear_portfolio,
        disabled=ledger.empty,
        key="clear_portfolio"
    )

    action_message = st.session_state.pop("transaction_action_message", None)
    if action_message:
        st.success(action_message)

    if st.session_state.get("show_add_transaction", False):
        st.subheader("Add transaction")
        _display_add_transaction_form()
    if st.session_state.get("show_boursobank_import", False):
        st.subheader("Import BoursoBank CSV")
        _display_boursobank_import()

    ledger = st.session_state["transaction_ledger_input"]
    if not ledger.empty:
        st.subheader("Transaction history")
        st.caption(f"{len(ledger)} transactions · newest first")
        st.dataframe(
            _transaction_history_for_display(ledger),
            hide_index=True,
            width="stretch",
            column_config={
                "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
                "Quantity": st.column_config.NumberColumn(
                    "Quantity", format="%.4f"
                ),
                "Price or Amount": st.column_config.NumberColumn(
                    "Price or Amount",
                    format=f"{display_currency_symbol()}%.2f"
                ),
                "Fees": st.column_config.NumberColumn(
                    "Fees", format=f"{display_currency_symbol()}%.2f"
                ),
                "Taxes": st.column_config.NumberColumn(
                    "Taxes", format=f"{display_currency_symbol()}%.2f"
                )
            }
        )
        st.button(
            "Manage transaction",
            on_click=show_transaction_manager,
            key="show_transaction_manager_button"
        )
        if st.session_state["show_transaction_manager"]:
            st.subheader("Manage transaction")
            _display_transaction_manager(ledger)

    with st.expander("Analysis settings", expanded=False):
        performance_setting_columns = st.columns(2, gap="medium")
        performance_setting_columns[0].number_input(
            "Annual risk-free rate (%)",
            min_value=-19.9,
            max_value=49.9,
            step=0.1,
            format="%.2f",
            key="risk_free_rate_percent_input"
        )
        performance_setting_columns[1].text_input(
            "Benchmark ticker",
            key="benchmark_ticker_input"
        )
        st.selectbox(
            "Risk history",
            options=list(RISK_WINDOW_PRESETS),
            key="risk_window_input"
        )
        st.selectbox(
            "Portfolio display currency",
            options=["EUR", "USD"],
            key="display_currency",
            help="Formatting only; no foreign-exchange conversion is applied."
        )
        st.caption(
            "Risk metrics apply today's allocation to historical returns."
        )

    with st.expander("Advanced stress testing (optional)", expanded=False):
        st.caption(
            "Stress testing runs only when every current holding has a "
            "defined shock. Missing holdings are not assumed to have zero "
            "shock."
        )
        stress_scenarios_input = st.data_editor(
            st.session_state["stress_scenarios_input"],
            num_rows="dynamic",
            width="stretch",
            column_config={
                "_index": st.column_config.TextColumn("Scenario")
            },
            key="stress_scenarios_editor"
        )
        st.session_state["stress_scenarios_input"] = (
            stress_scenarios_input.copy()
        )

    if analysis_requires_rerun():
        st.warning(
            "Analysis settings changed. Existing results still use the "
            "previously analyzed settings; run Analyze portfolio to refresh "
            "them."
        )

    st.subheader("Analyze Portfolio")
    if ledger.empty:
        st.caption("Add or import transactions before running analysis.")
    if st.button(
        "Analyze portfolio",
        type="primary",
        disabled=ledger.empty,
        key="analyze_portfolio"
    ):
        try:
            with st.spinner(
                "Downloading market data and running unified analysis..."
            ):
                unified_analysis = run_unified_portfolio_analysis(
                    transactions=ledger,
                    stress_scenarios=(
                        st.session_state["stress_scenarios_input"]
                    ),
                    risk_window_label=st.session_state["risk_window_input"],
                    risk_free_rate_annual=(
                        st.session_state["risk_free_rate_percent_input"]
                        / 100.0
                    ),
                    benchmark_ticker=st.session_state[
                        "benchmark_ticker_input"
                    ]
                )
        except ValueError as error:
            st.error(str(error))
        else:
            st.session_state["unified_analysis"] = unified_analysis
            st.session_state["analyzed_risk_window_label"] = (
                st.session_state["risk_window_input"]
            )
            st.session_state["analyzed_risk_free_rate_percent"] = (
                st.session_state["risk_free_rate_percent_input"]
            )
            st.session_state["analyzed_benchmark_ticker"] = (
                unified_analysis["benchmark"]["ticker"]
            )
            st.session_state["analysis_completed_message"] = True
            st.rerun()

    if show_analysis_success:
        st.success("Analysis completed successfully.")

    if "unified_analysis" in st.session_state:
        st.caption(
            "View Overview, Performance, Risk or Holdings above."
        )


def display_advanced(results):
    """Display technical validation of Invested Securities risk models."""
    risk = results["risk"]

    apply_dashboard_styles()

    st.header("Advanced")
    st.caption(
        "Statistical validation of Invested Securities risk models—not "
        "Investor History performance."
    )
    st.caption(
        "Account and Invested Securities views have identical VaR breach "
        "dates and validation conclusions because returns and thresholds are "
        "scaled by the same positive invested ratio."
    )
    st.caption(
        "Statistical validation is reported only when the common "
        "out-of-sample sample meets the project minimum; at "
        f"{risk['validation_confidence_level']:.0%} confidence this requires "
        f"at least {risk['minimum_required_observations']:,} observations."
    )
    st.caption(
        f"Invested Securities return observations: "
        f"{results['metadata']['risk_observations']:,} · common "
        f"out-of-sample VaR validation observations: "
        f"{risk['validation_observations']:,}."
    )
    if not risk["historical_backtest_available"]:
        st.info(risk["historical_backtest_reason"])
    if not risk.get("ewma_backtest_available", True):
        st.info(risk["ewma_backtest_reason"])

    st.subheader("VaR Model Comparison")
    comparison_columns = [
        "model",
        "observations",
        "breaches",
        "breach_rate",
        "expected_breach_rate"
    ]
    model_comparison = (
        risk["model_comparison"][comparison_columns]
        .rename(columns={
            "model": "Model",
            "observations": "Observations",
            "breaches": "Breaches",
            "breach_rate": "Breach Rate",
            "expected_breach_rate": "Expected Rate"
        })
    )
    st.dataframe(
        model_comparison.style.format({
            "Observations": "{:,.0f}",
            "Breaches": "{:,.0f}",
            "Breach Rate": "{:.2%}",
            "Expected Rate": "{:.2%}"
        }),
        width="stretch",
        hide_index=True
    )
    st.caption(
        "Both models are compared over the same evaluation dates."
    )

    st.subheader("Backtesting Tests")
    if not risk["model_validation_available"]:
        availability_summary = (
            "Core risk metrics remain available; the EWMA descriptive "
            "backtest remains available when its 30-observation minimum is "
            "met. "
            if not risk["historical_backtest_available"]
            else "Risk metrics and descriptive VaR backtests remain available. "
        )
        st.info(
            availability_summary + (
                "Statistical VaR model validation requires more history. "
                f"{risk['model_validation_reason']}"
            )
        )
    else:
        validation_tests = [
            ("Kupiec Coverage", "kupiec_p_value"),
            (
                "Christoffersen Independence",
                "christoffersen_independence_p_value"
            ),
            (
                "Conditional Coverage",
                "conditional_coverage_p_value"
            )
        ]
        validation_rows = []
        for model in risk["model_comparison"].to_dict("records"):
            for test_name, p_value_key in validation_tests:
                p_value = model[p_value_key]
                validation_rows.append({
                    "Test": test_name,
                    "Model": model["model"],
                    "p-value": p_value,
                    "5% Result": (
                        "Not rejected" if p_value >= 0.05 else "Rejected"
                    )
                })
        backtesting_tests = pd.DataFrame(validation_rows)
        st.dataframe(
            backtesting_tests.style.format({"p-value": "{:.4f}"}),
            width="stretch",
            hide_index=True
        )
        st.caption(
            "At 5% significance, p-value ≥ 0.05 means the test does not "
            "reject its null hypothesis; p-value < 0.05 means the null is "
            "rejected. A p-value is not the probability that a model is "
            "correct. Meeting the sample minimum does not prove validity; "
            "the tests remain asymptotic approximations."
        )

    st.subheader("Distribution Diagnostics")
    diagnostic_columns = st.columns(4, gap="medium")
    diagnostic_columns[0].metric(
        "Skewness",
        f"{risk['skewness']:.2f}",
        border=True
    )
    diagnostic_columns[1].metric(
        "Excess Kurtosis",
        f"{risk['excess_kurtosis']:.2f}",
        border=True
    )
    diagnostic_columns[2].metric(
        "Current-Allocation Worst Day",
        pd.Timestamp(risk["worst_date"]).strftime("%Y-%m-%d"),
        help=(
            "Worst day in the fixed-weight historical simulation of today's "
            "invested-security allocation."
        ),
        border=True
    )
    diagnostic_columns[3].metric(
        "Worst Daily Return",
        format_percentage(risk["worst_return"], signed=True),
        border=True
    )


st.set_page_config(
    page_title="Portfolio Analytics",
    layout="wide",
    initial_sidebar_state="collapsed"
)
initialize_session_state()

st.title("Portfolio Analytics")
st.caption("Portfolio performance and risk analytics")

selected_page = st.segmented_control(
    "Navigation",
    options=NAVIGATION_SECTIONS,
    default="Overview",
    key="navigation",
    label_visibility="collapsed"
)

if analysis_requires_rerun():
    st.caption("Analysis status: requires rerun")
elif "unified_analysis" in st.session_state:
    st.caption("Analysis status: ready")
else:
    st.caption("Analysis status: not run")

results = st.session_state.get("unified_analysis")

if (
    results is not None
    and analysis_requires_rerun()
    and selected_page != "Transactions"
):
    st.warning(
        "Analysis settings changed. The displayed results use the previously "
        "analyzed settings; return to Transactions and run Analyze portfolio."
    )

if selected_page == "Transactions":
    display_transactions()
elif results is None:
    st.header(selected_page)
    display_pre_analysis_state()
elif selected_page == "Overview":
    display_overview(results)
elif selected_page == "Performance":
    display_performance(results)
elif selected_page == "Risk":
    display_risk(results)
elif selected_page == "Holdings":
    display_holdings(results)
elif selected_page == "Advanced":
    display_advanced(results)
