from pathlib import Path

from src.data import load_stress_scenarios
from src.engine import run_portfolio_analysis
from src.portfolio import load_portfolio
from src.plots import (
    plot_cumulative_performance,
    plot_drawdown,
    plot_ewma_var_backtest,
    plot_return_distribution,
    plot_rolling_volatility,
    plot_var_backtest,
    plot_volatility_comparison
)
from src.reporting import (
    build_risk_summary,
    export_results
)


# -------------------------
# 1. Configuration
# -------------------------

start_date = "2022-01-01"

project_dir = Path(__file__).resolve().parent
portfolio_file = project_dir / "inputs" / "portfolio.csv"
stress_file = project_dir / "inputs" / "stress_scenarios.csv"
output_dir = project_dir / "outputs"

portfolio = load_portfolio(portfolio_file)
stress_scenarios = load_stress_scenarios(stress_file)


# -------------------------
# 2. Portfolio analysis
# -------------------------

results = run_portfolio_analysis(
    portfolio=portfolio,
    stress_scenarios=stress_scenarios,
    start_date=start_date
)

historical_validation = results["historical_validation"]
historical_transitions = historical_validation["transition_counts"]
ewma_validation = results["ewma_validation"]


# -------------------------
# 3. Terminal output
# -------------------------

print("\nPRIX :")
print(results["prices"].head())

print("\nPOSITIONS DU PORTEFEUILLE :")
print(
    results["positions"][
        [
            "ticker",
            "quantity",
            "purchase_price",
            "current_price",
            "market_value",
            "weight",
            "unrealized_pnl",
            "unrealized_pnl_pct"
        ]
    ]
)

print("\nVALEUR ACTUELLE DU PORTEFEUILLE :")
print(f"{results['portfolio_value']:,.2f}")

print("\nRENDEMENTS :")
print(results["returns"].head())

print("\nRENDEMENTS DU PORTEFEUILLE :")
print(results["portfolio_returns"].head())

print("\nVOLATILITÉ ANNUALISÉE :")
print(f"{results['annual_volatility']:.2%}")

print("\nMATRICE DE CORRÉLATION :")
print(results["correlation_matrix"])

print("\nMAXIMUM DRAWDOWN :")
print(f"{results['max_drawdown']:.2%}")

print("\nVALUE AT RISK HISTORIQUE :")
print(f"VaR 95% : {results['historical_var_95']:.2%}")
print(f"VaR 99% : {results['historical_var_99']:.2%}")

print("\nEXPECTED SHORTFALL :")
print(f"ES 95% : {results['expected_shortfall_95']:.2%}")
print(f"ES 99% : {results['expected_shortfall_99']:.2%}")

print("\nSTATISTIQUES DE DISTRIBUTION :")
print(f"Skewness : {results['skewness']:.4f}")
print(
    f"Excess kurtosis : "
    f"{results['excess_kurtosis']:.4f}"
)

print("\nPIRE JOURNÉE HISTORIQUE :")
print(f"Date : {results['worst_date'].date()}")
print(f"Rendement : {results['worst_return']:.2%}")

print("\nMATRICE DE COVARIANCE ANNUALISÉE :")
print(results["annual_covariance_matrix"])

print("\nVOLATILITÉ PAR CALCUL MATRICIEL :")
print(f"{results['portfolio_volatility_matrix']:.2%}")

print("\nÉCART ENTRE LES DEUX MÉTHODES :")
print(
    f"{abs(results['annual_volatility'] - results['portfolio_volatility_matrix']):.8%}"
)

print("\nCONTRIBUTION AU RISQUE :")
print(results["risk_contribution_table"])

print("\nVALUE AT RISK PARAMÉTRIQUE :")
print(
    f"VaR paramétrique 95% : "
    f"{results['parametric_var_95']:.2%}"
)
print(
    f"VaR paramétrique 99% : "
    f"{results['parametric_var_99']:.2%}"
)

print("\nSTRESS TESTS :")
print(results["stress_results"])

print("\nVOLATILITÉ GLISSANTE :")
print(results["rolling_volatility"].dropna().tail())

print("\nVOLATILITÉ EWMA :")
print(
    f"Dernière volatilité annualisée : "
    f"{results['ewma_annualized_volatility'].dropna().iloc[-1]:.2%}"
)

output_dir.mkdir(parents=True, exist_ok=True)

print("\nDossier de sortie :")
print(output_dir)


# -------------------------
# 4. Plots
# -------------------------

plot_cumulative_performance(
    results["cumulative_performance"],
    output_dir
)

plot_drawdown(
    results["drawdown"],
    output_dir
)

plot_rolling_volatility(
    results["rolling_volatility"],
    output_dir
)


# -------------------------
# 5. Backtest output
# -------------------------

print("\nBACKTEST VaR HISTORIQUE 95% :")
print(
    f"Observations : "
    f"{historical_validation['observations']}"
)
print(
    f"Dépassements : "
    f"{historical_validation['breaches']}"
)
print(
    f"Taux de dépassement : "
    f"{historical_validation['breach_rate']:.2%}"
)
print(
    f"Taux théorique attendu : "
    f"{historical_validation['expected_breach_rate']:.2%}"
)

print("\nTEST DE KUPIEC :")
print(
    f"Statistique LR : "
    f"{historical_validation['kupiec_lr_statistic']:.4f}"
)
print(
    f"P-value : "
    f"{historical_validation['kupiec_p_value']:.4f}"
)

if historical_validation["kupiec_p_value"] < 0.05:
    print("Résultat : rejet du modèle VaR au seuil de 5%.")
else:
    print("Résultat : le modèle VaR n'est pas rejeté au seuil de 5%.")

print("\nTEST D'INDÉPENDANCE DE CHRISTOFFERSEN :")
print(f"Transitions 0->0 : {historical_transitions['n00']}")
print(f"Transitions 0->1 : {historical_transitions['n01']}")
print(f"Transitions 1->0 : {historical_transitions['n10']}")
print(f"Transitions 1->1 : {historical_transitions['n11']}")
print(
    f"Statistique LR : "
    f"{historical_validation['christoffersen_independence_lr_statistic']:.4f}"
)
print(
    f"P-value : "
    f"{historical_validation['christoffersen_independence_p_value']:.4f}"
)

if historical_validation["christoffersen_independence_p_value"] < 0.05:
    print(
        "Résultat : rejet de l'indépendance "
        "des dépassements au seuil de 5%."
    )
else:
    print(
        "Résultat : l'indépendance des "
        "dépassements n'est pas rejetée "
        "au seuil de 5%."
    )

print("\nTEST DE COUVERTURE CONDITIONNELLE :")
print(
    f"Statistique LR : "
    f"{historical_validation['conditional_coverage_lr_statistic']:.4f}"
)
print(
    f"P-value : "
    f"{historical_validation['conditional_coverage_p_value']:.4f}"
)

if historical_validation["conditional_coverage_p_value"] < 0.05:
    print("Résultat : rejet du modèle VaR au seuil de 5%.")
else:
    print("Résultat : le modèle VaR n'est pas rejeté au seuil de 5%.")

print("\nBACKTEST VaR EWMA 95% :")
print(f"Observations : {ewma_validation['observations']}")
print(f"Dépassements : {ewma_validation['breaches']}")
print(
    f"Taux de dépassement : "
    f"{ewma_validation['breach_rate']:.2%}"
)
print(
    f"Taux théorique attendu : "
    f"{ewma_validation['expected_breach_rate']:.2%}"
)
print(
    f"Kupiec p-value : "
    f"{ewma_validation['kupiec_p_value']:.4f}"
)
print(
    f"Christoffersen p-value : "
    f"{ewma_validation['christoffersen_independence_p_value']:.4f}"
)
print(
    f"Conditional coverage p-value : "
    f"{ewma_validation['conditional_coverage_p_value']:.4f}"
)

print("\nCOMPARAISON DES MODÈLES SUR LA PÉRIODE COMMUNE :")
print(
    results["model_comparison"].to_string(
        index=False,
        formatters={
            "breach_rate": "{:.2%}".format,
            "expected_breach_rate": "{:.2%}".format,
            "kupiec_lr_statistic": "{:.4f}".format,
            "kupiec_p_value": "{:.4f}".format,
            "christoffersen_independence_lr_statistic": "{:.4f}".format,
            "christoffersen_independence_p_value": "{:.4f}".format,
            "conditional_coverage_lr_statistic": "{:.4f}".format,
            "conditional_coverage_p_value": "{:.4f}".format
        }
    )
)

results["model_comparison"].to_csv(
    output_dir / "model_comparison.csv",
    index=False
)

plot_var_backtest(
    results["historical_backtest"],
    output_dir
)

print(
    "\nGraphiques et fichiers de résultats "
    "enregistrés dans le dossier outputs."
)

plot_return_distribution(
    results["portfolio_returns"],
    results["historical_quantile_95"],
    results["historical_quantile_99"],
    output_dir
)

plot_volatility_comparison(
    results["rolling_volatility"],
    results["ewma_annualized_volatility"],
    output_dir
)

plot_ewma_var_backtest(
    results["ewma_backtest"],
    output_dir
)


# -------------------------
# 6. CSV exports
# -------------------------

risk_summary = build_risk_summary(
    portfolio_value=results["portfolio_value"],
    annual_volatility=results["annual_volatility"],
    max_drawdown=results["max_drawdown"],
    var_95=results["historical_var_95"],
    var_99=results["historical_var_99"],
    es_95=results["expected_shortfall_95"],
    es_99=results["expected_shortfall_99"],
    parametric_var_95=results["parametric_var_95"],
    parametric_var_99=results["parametric_var_99"],
    skewness=results["skewness"],
    excess_kurtosis=results["excess_kurtosis"],
    backtest_observations=historical_validation["observations"],
    backtest_breaches=historical_validation["breaches"],
    breach_rate=historical_validation["breach_rate"],
    expected_breach_rate=historical_validation["expected_breach_rate"],
    kupiec_p_value=historical_validation["kupiec_p_value"],
    christoffersen_p_value=(
        historical_validation[
            "christoffersen_independence_p_value"
        ]
    ),
    conditional_p_value=(
        historical_validation[
            "conditional_coverage_p_value"
        ]
    )
)

export_results(
    positions=results["positions"],
    risk_summary=risk_summary,
    risk_contribution_table=results["risk_contribution_table"],
    stress_results=results["stress_results"],
    backtest=results["historical_backtest"],
    output_dir=output_dir
)
