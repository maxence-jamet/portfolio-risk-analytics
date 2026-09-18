import numpy as np
import pandas as pd
import pytest

from src.risk import (
    calculate_diversification_diagnostics,
    diversification_ratio,
    effective_number_of_holdings,
    risk_contribution_concentration,
    weight_concentration_hhi
)


def test_four_equal_weights_have_expected_hhi_and_effective_holdings():
    weights = pd.Series([0.25, 0.25, 0.25, 0.25])

    assert np.isclose(weight_concentration_hhi(weights), 0.25)
    assert np.isclose(effective_number_of_holdings(weights), 4.0)


def test_single_asset_has_unit_weight_diagnostics_and_ratio():
    weights = pd.Series({"A": 1.0})
    covariance = pd.DataFrame([[0.04]], index=["A"], columns=["A"])
    diagnostics = calculate_diversification_diagnostics(
        covariance,
        weights,
        pd.Series({"A": 0.20})
    )

    assert diagnostics == {
        "weight_hhi": 1.0,
        "effective_number_of_holdings": 1.0,
        "diversification_ratio": 1.0,
        "risk_concentration_hhi": 1.0,
        "effective_risk_contributors": 1.0
    }


def test_ninety_ten_weights_have_expected_concentration():
    weights = np.array([0.90, 0.10])
    expected_hhi = 0.90**2 + 0.10**2

    assert np.isclose(weight_concentration_hhi(weights), expected_hhi)
    assert np.isclose(
        effective_number_of_holdings(weights),
        1.0 / expected_hhi
    )


def test_perfectly_correlated_assets_have_diversification_ratio_one():
    covariance = np.array([
        [0.04, 0.06],
        [0.06, 0.09]
    ])
    weights = np.array([0.50, 0.50])

    assert np.isclose(diversification_ratio(covariance, weights), 1.0)


def test_uncorrelated_equal_volatility_assets_have_sqrt_two_ratio():
    covariance = np.array([
        [0.04, 0.00],
        [0.00, 0.04]
    ])
    weights = np.array([0.50, 0.50])

    assert np.isclose(
        diversification_ratio(covariance, weights),
        np.sqrt(2.0)
    )


def test_known_absolute_risk_contributions_have_expected_effective_count():
    risk_hhi, effective_contributors = risk_contribution_concentration(
        np.array([3.0, 1.0])
    )

    expected_hhi = 0.75**2 + 0.25**2
    assert np.isclose(risk_hhi, expected_hhi)
    assert np.isclose(effective_contributors, 1.0 / expected_hhi)


def test_negative_component_contribution_uses_absolute_normalization():
    risk_hhi, effective_contributors = risk_contribution_concentration(
        pd.Series([0.06, -0.02])
    )

    assert np.isclose(risk_hhi, 0.75**2 + 0.25**2)
    assert np.isclose(effective_contributors, 1.6)


def test_near_zero_denominators_return_unavailable_diagnostics():
    assert np.isnan(
        diversification_ratio(np.zeros((2, 2)), np.array([0.50, 0.50]))
    )
    risk_hhi, effective_contributors = risk_contribution_concentration(
        np.array([1e-14, -1e-14])
    )
    assert np.isnan(risk_hhi)
    assert np.isnan(effective_contributors)


@pytest.mark.parametrize(
    ("weights", "match"),
    [
        (np.array([0.60, 0.30]), "sum to 1"),
        (np.array([1.10, -0.10]), "long-only"),
        (np.array([0.50, np.nan]), "finite")
    ]
)
def test_weight_diagnostics_reject_invalid_inputs(weights, match):
    with pytest.raises(ValueError, match=match):
        weight_concentration_hhi(weights)


def test_diversification_inputs_are_not_mutated():
    covariance = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.09]],
        index=["B", "A"],
        columns=["B", "A"]
    )
    weights = pd.Series({"A": 0.40, "B": 0.60})
    contributions = pd.Series({"A": 0.03, "B": -0.01})
    covariance_original = covariance.copy(deep=True)
    weights_original = weights.copy(deep=True)
    contributions_original = contributions.copy(deep=True)

    calculate_diversification_diagnostics(
        covariance,
        weights,
        contributions
    )

    pd.testing.assert_frame_equal(covariance, covariance_original)
    pd.testing.assert_series_equal(weights, weights_original)
    pd.testing.assert_series_equal(contributions, contributions_original)
