"""Convert cleaned financial rows into retrieval-ready text documents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


CURRENCY_SYMBOLS = {
    "USD": "$",
    "INR": "₹",
    "EUR": "€",
    "GBP": "£",
}


@dataclass(frozen=True)
class FinancialDocument:
    """A searchable financial fact with traceable source metadata."""

    page_content: str
    metadata: dict[str, Any]


def has_value(value: object) -> bool:
    """Return True only for non-empty, non-null values."""
    if value is None or pd.isna(value):
        return False

    return str(value).strip() != ""


def format_amount(amount: float, currency: str) -> str:
    """Format an amount in readable financial language.

    Examples:
        2150000, USD -> $2.15 million
        -300000, USD -> -$300,000
        1200, INR -> ₹1,200
    """
    symbol = CURRENCY_SYMBOLS.get(currency.upper(), f"{currency.upper()} ")

    absolute_amount = abs(float(amount))

    if absolute_amount >= 1_000_000_000:
        value = f"{absolute_amount / 1_000_000_000:.2f} billion"
    elif absolute_amount >= 1_000_000:
        value = f"{absolute_amount / 1_000_000:.2f} million"
    elif absolute_amount >= 1_000:
        value = f"{absolute_amount / 1_000:.2f} thousand"
    else:
        value = f"{absolute_amount:,.2f}"

    prefix = "-" if amount < 0 else ""
    return f"{prefix}{symbol}{value}"


class FinancialDocumentBuilder:
    """Create searchable text documents from cleaned financial rows."""

    def build_document(self, row: pd.Series) -> FinancialDocument:
        """Convert one canonical financial row into text and metadata."""
        metric = str(row["metric"])
        amount = float(row["amount"])
        currency = str(row["currency"])

        subject = (
            str(row["business_unit"])
            if has_value(row.get("business_unit"))
            else "The company"
        )

        formatted_amount = format_amount(amount, currency)

        text_parts = [f"{subject} reported {metric} of {formatted_amount}"]

        if has_value(row.get("period")):
            text_parts.append(f"during {row['period']}")

        if has_value(row.get("date")):
            text_parts.append(f"as of {row['date']}")

        if has_value(row.get("statement_type")):
            text_parts.append(f"from the {row['statement_type']}")

        page_content = " ".join(text_parts) + "."

        metadata = {
            "metric": metric,
            "amount": amount,
            "currency": currency,
            "statement_type": row.get("statement_type"),
            "period": row.get("period"),
            "date": row.get("date"),
            "business_unit": row.get("business_unit"),
            "source_file": row["source_file"],
            "source_row": int(row["source_row"]),
        }

        return FinancialDocument(
            page_content=page_content,
            metadata=metadata,
        )

    def build_documents(
        self,
        dataframe: pd.DataFrame,
    ) -> list[FinancialDocument]:
        """Convert every cleaned financial row into one searchable document."""
        required_columns = {
            "metric",
            "amount",
            "currency",
            "source_file",
            "source_row",
        }

        missing_columns = required_columns - set(dataframe.columns)

        if missing_columns:
            raise ValueError(
                "Cannot build documents. Missing columns: "
                f"{sorted(missing_columns)}"
            )

        return [
            self.build_document(row)
            for _, row in dataframe.iterrows()
        ]