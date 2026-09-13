import matplotlib.pyplot as plt


def plot_cumulative_performance(
    cumulative_performance,
    output_dir
):
    plt.figure(figsize=(10, 5))

    plt.plot(
        cumulative_performance.index,
        cumulative_performance
    )

    plt.title(
        "Portfolio Cumulative Performance"
    )

    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        output_dir / "portfolio_performance.png",
        dpi=300
    )

    plt.close()


def plot_drawdown(
    drawdown,
    output_dir
):
    plt.figure(figsize=(10, 5))

    plt.plot(
        drawdown.index,
        drawdown
    )

    plt.title("Portfolio Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")

    plt.gca().yaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda x, _: f"{x:.0%}"
        )
    )

    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_dir / "drawdown.png",
        dpi=300
    )

    plt.close()


def plot_rolling_volatility(
    rolling_volatility_series,
    output_dir
):
    plt.figure(figsize=(10, 5))

    plt.plot(
        rolling_volatility_series.index,
        rolling_volatility_series
    )

    plt.title(
        "252-Day Rolling Volatility"
    )

    plt.xlabel("Date")
    plt.ylabel("Annualized volatility")

    plt.gca().yaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda x, _: f"{x:.0%}"
        )
    )

    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_dir / "rolling_volatility.png",
        dpi=300
    )

    plt.close()


def plot_var_backtest(
    backtest,
    output_dir
):
    breaches = backtest[
        backtest["Breach"]
    ]

    plt.figure(figsize=(12, 6))

    plt.plot(
        backtest.index,
        backtest["Return"],
        label="Portfolio Return",
        linewidth=0.8
    )

    plt.plot(
        backtest.index,
        backtest["VaR"],
        label="Historical VaR 95%",
        linewidth=1.2
    )

    plt.scatter(
        breaches.index,
        breaches["Return"],
        label="VaR Breach",
        zorder=3
    )

    plt.title(
        "Historical VaR 95% Backtest"
    )

    plt.xlabel("Date")
    plt.ylabel("Daily Return")

    plt.gca().yaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda x, _: f"{x:.1%}"
        )
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_dir / "var_backtest.png",
        dpi=300
    )

    plt.close()


def plot_return_distribution(
    portfolio_returns,
    quantile_95,
    quantile_99,
    output_dir
):
    plt.figure(figsize=(10, 5))

    plt.hist(
        portfolio_returns,
        bins=50,
        alpha=0.8
    )

    plt.axvline(
        quantile_95,
        linestyle="--",
        linewidth=2,
        label="Historical VaR 95%"
    )

    plt.axvline(
        quantile_99,
        linestyle="--",
        linewidth=2,
        label="Historical VaR 99%"
    )

    plt.title(
        "Distribution of Portfolio Daily Returns"
    )

    plt.xlabel("Daily Return")
    plt.ylabel("Frequency")

    plt.gca().xaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda x, _: f"{x:.1%}"
        )
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_dir / "return_distribution.png",
        dpi=300
    )

    plt.close()

def plot_volatility_comparison(
    rolling_volatility_series,
    ewma_annualized_volatility_series,
    output_dir
):
    plt.figure(figsize=(12, 6))

    plt.plot(
        rolling_volatility_series.index,
        rolling_volatility_series,
        label="252-Day Rolling Volatility"
    )

    plt.plot(
        ewma_annualized_volatility_series.index,
        ewma_annualized_volatility_series,
        label="EWMA Volatility"
    )

    plt.title(
        "Rolling vs EWMA Volatility"
    )

    plt.xlabel("Date")
    plt.ylabel("Annualized Volatility")

    plt.gca().yaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda x, _: f"{x:.0%}"
        )
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_dir
        / "volatility_comparison.png",
        dpi=300
    )

    plt.close()


def plot_ewma_var_backtest(
    backtest,
    output_dir
):
    breaches = backtest[
        backtest["Breach"]
    ]

    plt.figure(figsize=(12, 6))

    plt.plot(
        backtest.index,
        backtest["Return"],
        label="Portfolio Return",
        linewidth=0.8
    )

    plt.plot(
        backtest.index,
        backtest["VaR"],
        label="EWMA Parametric VaR 95%",
        linewidth=1.2
    )

    plt.scatter(
        breaches.index,
        breaches["Return"],
        label="VaR Breach",
        zorder=3
    )

    plt.title(
        "EWMA Parametric VaR 95% Backtest"
    )

    plt.xlabel("Date")
    plt.ylabel("Daily Return")

    plt.gca().yaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda x, _: f"{x:.1%}"
        )
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_dir
        / "ewma_var_backtest.png",
        dpi=300
    )

    plt.close()