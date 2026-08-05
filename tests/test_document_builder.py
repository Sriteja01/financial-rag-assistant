import pandas as pd
import pytest

from document_builder import (
    FinancialDocumentBuilder,
    format_amount,
)


@pytest.mark.parametrize(
    ("amount", "currency", "expected"),
    [
        (2_150_000, "USD", "$2.15 million"),
        (-300_000, "USD", "-$300.00 thousand"),
        (1_200, "INR", "₹1.20 thousand"),
        (500, "EUR", "€500.00"),
    ],
)
def test_format_amount(amount, currency, expected):
    assert format_amount(amount, currency) == expected


def test_build_document_creates_readable_financial_text_and_citation():
    row = pd.Series(
        {
            "statement_type": "Income Statement",
            "metric": "Revenue",
            "period": "Q1 2024",
            "date": "2024-03-31",
            "amount": 2_150_000,
            "currency": "USD",
            "business_unit": "Retail",
            "source_file": "financials.csv",
            "source_row": 12,
        }
    )

    document = FinancialDocumentBuilder().build_document(row)

    assert document.page_content == (
        "Retail reported Revenue of $2.15 million during Q1 2024 "
        "as of 2024-03-31 from the Income Statement."
    )
    assert document.metadata["source_file"] == "financials.csv"
    assert document.metadata["source_row"] == 12
    assert document.metadata["amount"] == 2_150_000


def test_build_document_uses_company_when_business_unit_is_missing():
    row = pd.Series(
        {
            "metric": "Cash",
            "amount": 500_000,
            "currency": "USD",
            "business_unit": pd.NA,
            "source_file": "balance_sheet.csv",
            "source_row": 8,
        }
    )

    document = FinancialDocumentBuilder().build_document(row)

    assert document.page_content.startswith(
        "The company reported Cash of $500.00 thousand"
    )


def test_build_documents_creates_one_document_per_financial_row():
    dataframe = pd.DataFrame(
        [
            {
                "metric": "Revenue",
                "amount": 1_000_000,
                "currency": "USD",
                "source_file": "income.csv",
                "source_row": 2,
            },
            {
                "metric": "Net Income",
                "amount": 200_000,
                "currency": "USD",
                "source_file": "income.csv",
                "source_row": 3,
            },
        ]
    )

    documents = FinancialDocumentBuilder().build_documents(dataframe)

    assert len(documents) == 2
    assert documents[0].metadata["source_row"] == 2
    assert "Revenue" in documents[0].page_content
    assert "Net Income" in documents[1].page_content


def test_build_documents_rejects_incomplete_dataframe():
    incomplete_dataframe = pd.DataFrame(
        {
            "metric": ["Revenue"],
            "amount": [100],
        }
    )

    with pytest.raises(ValueError, match="Missing columns"):
        FinancialDocumentBuilder().build_documents(incomplete_dataframe)