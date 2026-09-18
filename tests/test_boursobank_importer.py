from io import StringIO

import numpy as np
import pandas as pd
import pytest

from src.importers.boursobank import (
    import_boursobank_csv,
    resolve_market_ticker
)
from src.transactions import reconstruct_accounting_history


HEADER = (
    "Date;Type;Valeur;Devise de l'opération;Frais;Impôts / Taxes;Parts;"
    "ISIN;Symbole boursier;Nom du titre"
)


def csv_fixture(*rows):
    return StringIO("\n".join([HEADER, *rows]))


@pytest.mark.parametrize(
    ("isin", "broker_symbol", "expected"),
    [
        ("IE0002XZSHO1", "AO00.DU", "WPEA.PA"),
        ("FR0013258662", "3AL.DU", "AYV.PA"),
        ("FR0000053225", "MMT.DU", "MMT.PA"),
        ("NL00150001Q9", "8TI.DE", "STLAP.PA"),
        ("UNKNOWN", "ORIGINAL.DE", "ORIGINAL.DE"),
        (" fr0013258662 ", "3AL.DU", "AYV.PA")
    ]
)
def test_market_ticker_resolution(isin, broker_symbol, expected):
    assert resolve_market_ticker(isin, broker_symbol) == expected


def test_real_examples_parse_french_numbers_and_reconstruct_buy_prices():
    source = csv_fixture(
        "2023-11-01T00:00;Dépôt;20\u202f000,00;EUR;;;;;;",
        "2024-04-05T16:57:50;Achat;-101,85;EUR;0,51;0,30;16;"
        "FR0013258662;3AL.DU;ALD",
        "2024-06-17T16:20:50;Achat;-2\u00a0013,50;EUR;9,99;5,99;164;"
        "FR0000053225;MMT.DU;METROPOLE TELEVISION"
    )

    ledger = import_boursobank_csv(source)

    deposit, ald_buy, m6_buy = list(ledger.itertuples(index=False))
    assert deposit.type == "DEPOSIT"
    assert np.isclose(deposit.amount, 20000.0)
    assert np.isclose(ald_buy.price, 6.315)
    assert ald_buy.ticker == "AYV.PA"
    assert np.isclose(m6_buy.price, 12.18)
    assert m6_buy.ticker == "MMT.PA"
    assert np.isclose(m6_buy.fees, 9.99)
    assert np.isclose(m6_buy.taxes, 5.99)


def test_all_french_thousands_space_variants_are_supported():
    ledger = import_boursobank_csv(csv_fixture(
        "2025-01-01;Dépôt;1 000,00;EUR;;;;;;",
        "2025-01-02;Dépôt;2\u00a0000,00;EUR;;;;;;",
        "2025-01-03;Dépôt;3\u202f000,00;EUR;;;;;;"
    ))

    assert ledger["amount"].tolist() == [1000.0, 2000.0, 3000.0]


def test_sell_reconstructs_gross_price_from_net_credit():
    ledger = import_boursobank_csv(csv_fixture(
        "2025-01-02T10:00;Vente;118,00;EUR;1,00;1,00;10;"
        "FR0000000001;AAA.PA;AAA"
    ))

    sale = ledger.iloc[0]
    assert sale["type"] == "SELL"
    assert np.isclose(sale["price"], 12.0)
    assert np.isclose(
        sale["quantity"] * sale["price"] - sale["fees"] - sale["taxes"],
        118.0
    )


def test_dividend_and_interest_convert_net_cash_to_gross_income():
    ledger = import_boursobank_csv(csv_fixture(
        "2025-04-23T00:00;Dividendes;187,85;EUR;;33,15;325;"
        "NL00150001Q9;8TI.DE;STELLANTIS",
        "2025-05-01T00:00;Intérêts créditeurs;9,50;EUR;0,25;0,25;;;;"
    ))

    dividend, interest = list(ledger.itertuples(index=False))
    assert np.isclose(dividend.amount, 221.0)
    assert dividend.ticker == "STLAP.PA"
    assert np.isclose(dividend.amount - dividend.fees - dividend.taxes, 187.85)
    assert np.isclose(interest.amount, 10.0)
    assert np.isclose(interest.amount - interest.fees - interest.taxes, 9.5)


def test_all_cash_types_map_and_rows_sort_chronologically():
    ledger = import_boursobank_csv(csv_fixture(
        "2025-04-04;Frais;-2,00;eur;;;;;;",
        "2025-04-01;dÉpÔt;100,00;EUR;;;;;;",
        "2025-04-03;Impôts / Taxes;-3,00;EUR;;;;;;",
        "2025-04-02;Retrait;-10,00;EUR;;;;;;"
    ))

    assert ledger["type"].tolist() == [
        "DEPOSIT", "WITHDRAWAL", "TAX", "FEE"
    ]
    assert ledger["amount"].tolist() == [100.0, 10.0, 3.0, 2.0]
    assert ledger["date"].is_monotonic_increasing


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (
            "2025-01-02;Fusion;10,00;EUR;;;;;;",
            "Unsupported BoursoBank transaction type"
        ),
        (
            "2025-01-02;Dépôt;10,00;USD;;;;;;",
            "supports EUR only"
        ),
        (
            "not-a-date;Dépôt;10,00;EUR;;;;;;",
            "Invalid BoursoBank transaction date"
        ),
        (
            "2025-01-02;Dépôt;many;EUR;;;;;;",
            "Invalid French-formatted number"
        ),
        (
            "2025-01-02;Achat;-10,00;EUR;;;1;;;",
            "BUY requires ticker"
        )
    ]
)
def test_invalid_rows_raise_clear_errors(row, message):
    with pytest.raises(ValueError, match=message):
        import_boursobank_csv(csv_fixture(row))


def test_missing_columns_and_empty_file_are_rejected():
    missing_ticker_column = StringIO(
        "Date;Type;Valeur;Devise de l'opération;Frais;Impôts / Taxes;Parts\n"
        "2025-01-02;Dépôt;10,00;EUR;;;"
    )

    with pytest.raises(ValueError, match="missing required columns"):
        import_boursobank_csv(missing_ticker_column)
    with pytest.raises(ValueError, match="Malformed or empty"):
        import_boursobank_csv(StringIO(""))
    with pytest.raises(ValueError, match="Malformed BoursoBank CSV"):
        import_boursobank_csv(csv_fixture(
            "2025-01-02;Dépôt;10,00;EUR;;;;;;;"
        ))


def test_imported_ledger_reproduces_boursobank_cash_movements_once():
    source = csv_fixture(
        "2023-11-01;Dépôt;200,00;EUR;;;;;;",
        "2024-04-05;Achat;-101,85;EUR;0,51;0,30;16;"
        "FR0013258662;3AL.DU;ALD",
        "2025-04-23;Dividendes;187,85;EUR;;33,15;16;"
        "FR0013258662;3AL.DU;ALD"
    )
    ledger = import_boursobank_csv(source)
    prices = pd.DataFrame(
        {"AYV.PA": [6.315, 6.315, 6.315]},
        index=pd.to_datetime(["2023-11-01", "2024-04-05", "2025-04-23"])
    )

    history = reconstruct_accounting_history(ledger, prices=prices)

    assert history["daily_cash"].tolist() == [200.0, 98.15, 286.0]
    assert history["external_cash_flows_by_day"].tolist() == [200.0, 0.0, 0.0]
    assert np.isclose(history["dividend_income"], 221.0)
    assert np.isclose(history["fees_paid"], 0.51)
    assert np.isclose(history["taxes_paid"], 33.45)


def test_real_multi_row_fixture_uses_canonical_market_tickers():
    ledger = import_boursobank_csv(csv_fixture(
        "2025-01-01;Achat;-100,00;EUR;;;1;IE0002XZSHO1;AO00.DU;WPEA",
        "2025-01-02;Achat;-100,00;EUR;;;1;FR0013258662;3AL.DU;ALD",
        "2025-01-03;Achat;-100,00;EUR;;;1;FR0000053225;MMT.DU;M6",
        "2025-01-04;Achat;-100,00;EUR;;;1;NL00150001Q9;8TI.DE;STELLANTIS",
        "2025-01-05;Achat;-100,00;EUR;;;1;UNKNOWN;KEEP.DE;UNKNOWN"
    ))

    assert set(ledger["ticker"]) == {
        "WPEA.PA", "AYV.PA", "MMT.PA", "STLAP.PA", "KEEP.DE"
    }
