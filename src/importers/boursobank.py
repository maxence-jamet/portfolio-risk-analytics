"""Convert a BoursoBank activity CSV to the normalized V4 ledger."""

import unicodedata

import numpy as np
import pandas as pd

from src.transactions import (
    NORMALIZED_LEDGER_COLUMNS,
    validate_transaction_ledger
)


SOURCE_COLUMNS = {
    "date": "Date",
    "type": "Type",
    "value": "Valeur",
    "currency": "Devise de l'opération",
    "fees": "Frais",
    "taxes": "Impôts / Taxes",
    "quantity": "Parts",
    "isin": "ISIN",
    "ticker": "Symbole boursier"
}

ISIN_MARKET_TICKERS = {
    "FR0013258662": "AYV.PA",
    "IE0002XZSHO1": "WPEA.PA",
    "FR0000053225": "MMT.PA",
    "NL00150001Q9": "STLAP.PA"
}

TYPE_MAPPING = {
    "depot": "DEPOSIT",
    "achat": "BUY",
    "vente": "SELL",
    "retrait": "WITHDRAWAL",
    "dividendes": "DIVIDEND",
    "interets": "INTEREST",
    "interets crediteurs": "INTEREST",
    "frais": "FEE",
    "impots / taxes": "TAX"
}


def _normalize_french_text(value):
    """Return a case- and accent-insensitive comparison key."""
    decomposed = unicodedata.normalize("NFKD", str(value))
    without_accents = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    return " ".join(without_accents.casefold().strip().split())


def resolve_market_ticker(isin, broker_symbol):
    """Resolve a known ISIN, otherwise preserve the normalized broker symbol."""
    normalized_isin = "" if pd.isna(isin) else str(isin).strip().upper()
    if normalized_isin in ISIN_MARKET_TICKERS:
        return ISIN_MARKET_TICKERS[normalized_isin]
    if pd.isna(broker_symbol):
        return pd.NA
    normalized_symbol = str(broker_symbol).strip().upper()
    return normalized_symbol if normalized_symbol else pd.NA


def _resolve_source_columns(source_columns):
    normalized_columns = {
        _normalize_french_text(column): column
        for column in source_columns
    }
    resolved = {}
    missing = []

    for field, expected_column in SOURCE_COLUMNS.items():
        source_column = normalized_columns.get(
            _normalize_french_text(expected_column)
        )
        if source_column is None:
            missing.append(expected_column)
        else:
            resolved[field] = source_column

    if missing:
        raise ValueError(
            f"BoursoBank CSV is missing required columns: {sorted(missing)}"
        )

    return resolved


def _parse_french_numbers(values, column_name, default=None):
    """Parse decimal commas and French thousands separators."""
    text = values.astype("string").str.strip()
    empty = text.isna() | text.eq("")
    normalized = (
        text
        .str.replace("\u00a0", "", regex=False)
        .str.replace("\u202f", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
        .mask(empty)
    )
    numeric = pd.to_numeric(normalized, errors="coerce")
    invalid = ~empty & numeric.isna()
    if invalid.any():
        row_number = int(invalid[invalid].index[0]) + 2
        raise ValueError(
            f"Invalid French-formatted number in {column_name!r} at CSV "
            f"row {row_number}: {text.loc[invalid[invalid].index[0]]!r}."
        )

    if default is not None:
        numeric = numeric.fillna(default)
    return numeric.astype("float64")


def _read_source(source):
    try:
        source_data = pd.read_csv(
            source,
            sep=";",
            dtype=str,
            keep_default_na=False
        )
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        raise ValueError("Malformed or empty BoursoBank CSV.") from error

    if not isinstance(source_data.index, pd.RangeIndex):
        raise ValueError(
            "Malformed BoursoBank CSV: data rows do not match the header."
        )
    if source_data.empty:
        raise ValueError("BoursoBank CSV must contain at least one data row.")
    return source_data


def import_boursobank_csv(source):
    """Import a path or pandas-compatible file object as a V4 ledger.

    BoursoBank ``Valeur`` is an actual net cash movement. Trade prices and
    gross income are reconstructed so the V4 engine reproduces that movement
    after applying the separately recorded fees and taxes exactly once.
    """
    source_data = _read_source(source)
    columns = _resolve_source_columns(source_data.columns)

    dates = pd.to_datetime(
        source_data[columns["date"]],
        errors="coerce",
        format="mixed"
    )
    if dates.isna().any():
        row_number = int(dates[dates.isna()].index[0]) + 2
        raise ValueError(
            f"Invalid BoursoBank transaction date at CSV row {row_number}."
        )

    raw_types = source_data[columns["type"]].astype("string").str.strip()
    normalized_types = raw_types.map(_normalize_french_text)
    mapped_types = normalized_types.map(TYPE_MAPPING)
    if mapped_types.isna().any():
        invalid_index = mapped_types[mapped_types.isna()].index[0]
        invalid_type = raw_types.loc[invalid_index]
        raise ValueError(
            "Unsupported BoursoBank transaction type at CSV row "
            f"{int(invalid_index) + 2}: {invalid_type!r}."
        )

    currencies = (
        source_data[columns["currency"]]
        .astype("string")
        .str.strip()
        .str.upper()
    )
    non_eur = currencies.ne("EUR")
    if non_eur.any():
        invalid_index = non_eur[non_eur].index[0]
        raise ValueError(
            "BoursoBank importer currently supports EUR only; CSV row "
            f"{int(invalid_index) + 2} uses {currencies.loc[invalid_index]!r}."
        )

    values = _parse_french_numbers(
        source_data[columns["value"]],
        SOURCE_COLUMNS["value"]
    )
    fees = _parse_french_numbers(
        source_data[columns["fees"]],
        SOURCE_COLUMNS["fees"],
        default=0.0
    )
    taxes = _parse_french_numbers(
        source_data[columns["taxes"]],
        SOURCE_COLUMNS["taxes"],
        default=0.0
    )
    quantities = _parse_french_numbers(
        source_data[columns["quantity"]],
        SOURCE_COLUMNS["quantity"]
    )
    tickers = pd.Series(
        [
            resolve_market_ticker(isin, broker_symbol)
            for isin, broker_symbol in zip(
                source_data[columns["isin"]],
                source_data[columns["ticker"]]
            )
        ],
        index=source_data.index,
        dtype="string"
    )

    ledger_rows = []
    for index in source_data.index:
        transaction_type = mapped_types.loc[index]
        value = values.loc[index]
        row_fees = fees.loc[index]
        row_taxes = taxes.loc[index]
        row_number = int(index) + 2

        if pd.isna(value) or value == 0.0:
            raise ValueError(
                f"BoursoBank Valeur must be non-zero at CSV row {row_number}."
            )

        supports_separate_costs = transaction_type in {
            "BUY", "SELL", "DIVIDEND", "INTEREST"
        }
        if (
            not supports_separate_costs
            and (row_fees != 0.0 or row_taxes != 0.0)
        ):
            raise ValueError(
                "Separate Frais or Impôts / Taxes are supported only for "
                "Achat, Vente, Dividendes and Intérêts; unexpected cost at "
                f"CSV row {row_number}."
            )

        row = {
            "date": dates.loc[index],
            "type": transaction_type,
            "ticker": np.nan,
            "quantity": np.nan,
            "price": np.nan,
            "amount": np.nan,
            "fees": 0.0,
            "taxes": 0.0
        }

        if transaction_type in {"BUY", "SELL"}:
            quantity = quantities.loc[index]
            ticker = tickers.loc[index]
            if pd.isna(ticker) or pd.isna(quantity) or quantity <= 0:
                raise ValueError(
                    f"{transaction_type} requires ticker and positive Parts "
                    f"at CSV row {row_number}."
                )
            if transaction_type == "BUY":
                gross_value = abs(value) - row_fees - row_taxes
            else:
                gross_value = abs(value) + row_fees + row_taxes
            if gross_value <= 0:
                raise ValueError(
                    f"{transaction_type} gross security value must be positive "
                    f"at CSV row {row_number}."
                )
            row.update({
                "ticker": ticker,
                "quantity": quantity,
                "price": gross_value / quantity,
                "fees": row_fees,
                "taxes": row_taxes
            })
        elif transaction_type in {"DIVIDEND", "INTEREST"}:
            if value <= 0:
                raise ValueError(
                    f"{transaction_type} net Valeur must be positive at CSV "
                    f"row {row_number}."
                )
            if transaction_type == "DIVIDEND":
                ticker = tickers.loc[index]
                if pd.isna(ticker):
                    raise ValueError(
                        "DIVIDEND requires ticker at CSV row "
                        f"{row_number}."
                    )
                row["ticker"] = ticker
            row.update({
                "amount": value + row_fees + row_taxes,
                "fees": row_fees,
                "taxes": row_taxes
            })
        else:
            row["amount"] = abs(value)

        ledger_rows.append(row)

    ledger = pd.DataFrame(ledger_rows, columns=NORMALIZED_LEDGER_COLUMNS)
    ledger = validate_transaction_ledger(ledger)
    return ledger.sort_values("date", kind="stable").reset_index(drop=True)
