import numpy as np
import pandas as pd
import pytest

from src.data import (
    download_price_views,
    download_prices,
    validate_valuation_price_freshness
)


def test_valuation_freshness_uses_business_days_and_returns_asset_dates():
    dates = pd.to_datetime(["2024-01-05", "2024-01-08"])
    valuation_prices = pd.DataFrame(
        {
            "AAA": [100.0, np.nan],
            "BBB": [200.0, 201.0]
        },
        index=dates
    )
    original_prices = valuation_prices.copy(deep=True)

    prepared_prices, freshness = validate_valuation_price_freshness(
        valuation_prices,
        required_assets=["AAA", "BBB"],
        max_price_staleness_business_days=1
    )

    assert freshness["valuation_reference_date"] == pd.Timestamp(
        "2024-01-08"
    )
    assert freshness["valuation_price_dates"] == {
        "AAA": pd.Timestamp("2024-01-05"),
        "BBB": pd.Timestamp("2024-01-08")
    }
    assert freshness["valuation_price_age_business_days"] == {
        "AAA": 1,
        "BBB": 0
    }
    assert freshness["max_price_staleness_business_days"] == 1
    pd.testing.assert_frame_equal(prepared_prices, valuation_prices)
    pd.testing.assert_frame_equal(valuation_prices, original_prices)


def test_valuation_freshness_allows_price_at_default_threshold():
    dates = pd.to_datetime(["2024-01-03", "2024-01-08"])
    valuation_prices = pd.DataFrame(
        {"AAA": [100.0, np.nan]},
        index=dates
    )

    _, freshness = validate_valuation_price_freshness(
        valuation_prices,
        required_assets=["AAA"]
    )

    assert freshness["valuation_price_age_business_days"]["AAA"] == 3


def test_valuation_freshness_rejects_stale_price_with_full_context():
    dates = pd.to_datetime(["2024-01-02", "2024-01-08"])
    valuation_prices = pd.DataFrame(
        {"AAA": [100.0, np.nan]},
        index=dates
    )

    with pytest.raises(
        ValueError,
        match=(
            "AAA valuation price is stale: last price 2024-01-02, "
            "market-data reference date 2024-01-08, age 4 business days, "
            "maximum allowed 3"
        )
    ):
        validate_valuation_price_freshness(
            valuation_prices,
            required_assets=["AAA"]
        )


def test_valuation_freshness_rejects_asset_without_any_earlier_price():
    valuation_prices = pd.DataFrame(
        {"AAA": [np.nan, np.nan]},
        index=pd.to_datetime(["2024-01-05", "2024-01-08"])
    )

    with pytest.raises(
        ValueError,
        match=(
            "No current or earlier valuation price is available for AAA.*"
            "2024-01-08"
        )
    ):
        validate_valuation_price_freshness(
            valuation_prices,
            required_assets=["AAA"]
        )


def test_download_price_views_extracts_raw_and_adjusted_prices_once(
    monkeypatch
):
    dates = pd.bdate_range("2024-01-02", periods=3)
    columns = pd.MultiIndex.from_product(
        [["Adj Close", "Close"], ["AAA", "BBB"]],
        names=["Price", "Ticker"]
    )
    yahoo_data = pd.DataFrame(
        [
            [90.0, 180.0, 100.0, 200.0],
            [99.0, 189.0, 110.0, 210.0],
            [108.0, 198.0, 120.0, 220.0]
        ],
        index=dates,
        columns=columns
    )
    original_yahoo_data = yahoo_data.copy(deep=True)
    calls = []

    def fake_download(tickers, **kwargs):
        calls.append((tickers, kwargs))
        return yahoo_data

    monkeypatch.setattr("src.data.yf.download", fake_download)

    views = download_price_views(["AAA", "BBB"], "2024-01-01")

    assert len(calls) == 1
    assert calls[0][0] == ["AAA", "BBB"]
    assert calls[0][1] == {
        "start": "2024-01-01",
        "auto_adjust": False,
        "actions": False,
        "group_by": "column",
        "progress": False
    }
    expected_valuation_prices = pd.DataFrame(
        {"AAA": [100.0, 110.0, 120.0], "BBB": [200.0, 210.0, 220.0]},
        index=dates
    )
    expected_return_prices = pd.DataFrame(
        {"AAA": [90.0, 99.0, 108.0], "BBB": [180.0, 189.0, 198.0]},
        index=dates
    )
    pd.testing.assert_frame_equal(
        views["valuation_prices"],
        expected_valuation_prices
    )
    pd.testing.assert_frame_equal(
        views["return_prices"],
        expected_return_prices
    )
    pd.testing.assert_frame_equal(yahoo_data, original_yahoo_data)


def test_download_price_views_rejects_missing_adjusted_close(monkeypatch):
    dates = pd.bdate_range("2024-01-02", periods=2)
    yahoo_data = pd.DataFrame(
        {
            ("Close", "AAA"): [100.0, 101.0]
        },
        index=dates
    )

    monkeypatch.setattr(
        "src.data.yf.download",
        lambda *args, **kwargs: yahoo_data
    )

    with pytest.raises(ValueError, match="Adj Close"):
        download_price_views(["AAA"], "2024-01-01")


def test_download_price_views_handles_flat_single_ticker_response(
    monkeypatch
):
    dates = pd.bdate_range("2024-01-02", periods=2)
    yahoo_data = pd.DataFrame(
        {
            "Close": [100.0, 110.0],
            "Adj Close": [90.0, 99.0]
        },
        index=dates
    )
    monkeypatch.setattr(
        "src.data.yf.download",
        lambda *args, **kwargs: yahoo_data
    )

    views = download_price_views("AAA", "2024-01-01")

    assert views["valuation_prices"]["AAA"].tolist() == [100.0, 110.0]
    assert views["return_prices"]["AAA"].tolist() == [90.0, 99.0]


def test_download_prices_compatibility_helper_returns_adjusted_view(
    monkeypatch
):
    valuation_prices = pd.DataFrame({"AAA": [100.0]})
    return_prices = pd.DataFrame({"AAA": [90.0]})

    monkeypatch.setattr(
        "src.data.download_price_views",
        lambda *args, **kwargs: {
            "valuation_prices": valuation_prices,
            "return_prices": return_prices
        }
    )

    result = download_prices(["AAA"], "2024-01-01")

    pd.testing.assert_frame_equal(result, return_prices)
