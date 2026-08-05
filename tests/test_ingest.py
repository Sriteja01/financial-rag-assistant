import pandas as pd
import pytest

from ingest import (
    FinancialDataIngestor,
    SchemaValidationError,
    map_to_canonical_column,
    normalize_column_name,
    parse_currency_amount,
)


def test_normalize_column_name():
    assert normalize_column_name(" Business Unit ($) ") == "business_unit"


def test_map_to_canonical_column():
    assert map_to_canonical_column("Line Item") == "metric"
    assert map_to_canonical_column("Value") == "amount"
    assert map_to_canonical_column("Quarter") == "period"


@pytest.mark.parametrize(
    ("raw_value", "expected_amount"),
    [
        ("$2.15M", 2_150_000.0),
        ("(300,000)", -300_000.0),
        ("450K", 450_000.0),
        ("N/A", None),
    ],
)
def test_parse_currency_amount(raw_value, expected_amount):
    assert parse_currency_amount(raw_value) == expected_amount


def test_prepare_dataframe_standardizes_headers_and_adds_provenance():
    raw_data = pd.DataFrame(
        {
            "Line Item": ["Revenue"],
            "Value": [2_150_000],
            "Quarter": ["Q1 2024"],
        }
    )

    result = FinancialDataIngestor().prepare_dataframe(
        raw_data,
        "financials.csv",
    )

    assert {"metric", "amount", "period"}.issubset(result.columns)
    assert result.loc[0, "source_file"] == "financials.csv"
    assert result.loc[0, "source_row"] == 2


def test_clean_dataframe_normalizes_and_removes_bad_rows():
    raw_data = pd.DataFrame(
        {
            "Line Item": [" Revenue ", "Net Income", "Revenue", None],
            "Value": ["$2.15M", "(300,000)", "$2.15M", "$100"],
            "Report Date": [
                "03/31/2024",
                "invalid date",
                "03/31/2024",
                "03/31/2024",
            ],
            "Currency": ["usd", None, "USD", "USD"],
            "Business Unit": ["Retail", "Retail", "Retail", "Retail"],
        }
    )

    result = FinancialDataIngestor().clean_dataframe(
        raw_data,
        "financials.csv",
    )

    assert len(result) == 2

    revenue = result.loc[result["metric"] == "Revenue"].iloc[0]
    net_income = result.loc[result["metric"] == "Net Income"].iloc[0]

    assert revenue["amount"] == 2_150_000.0
    assert revenue["date"] == "2024-03-31"
    assert revenue["currency"] == "USD"
    assert revenue["source_row"] == 2

    assert net_income["amount"] == -300_000.0
    assert pd.isna(net_income["date"])
    assert net_income["currency"] == "USD"


def test_prepare_dataframe_rejects_missing_required_columns():
    raw_data = pd.DataFrame({"Quarter": ["Q1 2024"]})

    with pytest.raises(SchemaValidationError, match="missing required columns"):
        FinancialDataIngestor().prepare_dataframe(raw_data, "bad_file.csv")


def test_load_file_reads_csv(tmp_path):
    csv_file = tmp_path / "financials.csv"
    csv_file.write_text(
        "Line Item,Value,Quarter\nRevenue,2150000,Q1 2024\n",
        encoding="utf-8",
    )

    result = FinancialDataIngestor().load_file(csv_file)

    assert len(result) == 1
    assert result.loc[0, "metric"] == "Revenue"
    assert result.loc[0, "amount"] == 2_150_000.0


def test_load_many_merges_multiple_financial_files(tmp_path):
    first_file = tmp_path / "income_statement.csv"
    first_file.write_text(
        "Line Item,Value\nRevenue,1000\n",
        encoding="utf-8",
    )

    second_file = tmp_path / "balance_sheet.csv"
    second_file.write_text(
        "Line Item,Value\nCash,500\n",
        encoding="utf-8",
    )

    result = FinancialDataIngestor().load_many([first_file, second_file])

    assert len(result) == 2
    assert set(result["metric"]) == {"Revenue", "Cash"}
    assert set(result["source_file"]) == {
        "income_statement.csv",
        "balance_sheet.csv",
    }


def test_load_file_rejects_unsupported_extension(tmp_path):
    text_file = tmp_path / "financials.txt"
    text_file.write_text("Revenue", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        FinancialDataIngestor().load_file(text_file)