"""Load, validate, clean, and merge financial CSV and Excel files."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from config import settings
from utils import logger


CANONICAL_COLUMNS = [
    "statement_type",
    "metric",
    "period",
    "date",
    "amount",
    "currency",
    "business_unit",
    "source_file",
    "source_row",
]

CANONICAL_COLUMN_ALIASES = {
    "statement_type": {
        "statement",
        "statement_type",
        "financial_statement",
        "report_type",
        "category",
    },
    "metric": {
        "metric",
        "line_item",
        "account",
        "account_name",
        "description",
    },
    "amount": {
        "amount",
        "value",
        "balance",
        "financial_value",
    },
    "period": {
        "period",
        "quarter",
        "fiscal_period",
        "reporting_period",
        "year",
    },
    "date": {
        "date",
        "report_date",
        "transaction_date",
        "as_of_date",
    },
    "currency": {
        "currency",
        "currency_code",
        "ccy",
    },
    "business_unit": {
        "business_unit",
        "business",
        "company",
        "division",
        "segment",
        "department",
    },
}


class SchemaValidationError(ValueError):
    """Raised when a source file lacks the required financial fields."""


def normalize_column_name(column_name: object) -> str:
    """Convert a source header into a consistent snake_case representation."""
    normalized = re.sub(r"[^a-z0-9]+", "_", str(column_name).strip().lower())
    return normalized.strip("_")


def map_to_canonical_column(column_name: object) -> str:
    """Map known source-header aliases to canonical field names."""
    normalized_name = normalize_column_name(column_name)

    for canonical_name, aliases in CANONICAL_COLUMN_ALIASES.items():
        if normalized_name in aliases:
            return canonical_name

    return normalized_name


def parse_currency_amount(value: object) -> float | None:
    """Convert common financial value formats into a numeric amount.

    Examples:
        "$2.15M"    -> 2150000.0
        "(300,000)" -> -300000.0
        "450K"      -> 450000.0
        "N/A"       -> None
    """
    if pd.isna(value):
        return None

    if isinstance(value, int | float):
        return float(value)

    text = str(value).strip()

    if text.lower() in {"", "-", "n/a", "na", "none", "null"}:
        return None

    is_negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()").replace(",", "")

    suffix_match = re.search(r"(?i)([kmb])\s*$", text)
    multiplier = 1

    if suffix_match:
        multiplier = {
            "k": 1_000,
            "m": 1_000_000,
            "b": 1_000_000_000,
        }[suffix_match.group(1).lower()]
        text = re.sub(r"(?i)[kmb]\s*$", "", text)

    numeric_text = re.sub(r"[^0-9.\-]", "", text)

    try:
        amount = float(numeric_text) * multiplier
    except ValueError:
        return None

    return -abs(amount) if is_negative else amount


class FinancialDataIngestor:
    """Read, validate, clean, and merge financial-statement files."""

    required_columns = {"metric", "amount"}

    def load_file(
        self,
        file_path: str | Path,
        sheet_name: str | int = 0,
    ) -> pd.DataFrame:
        """Load one CSV/XLSX/XLS file and return its cleaned data."""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        try:
            if path.suffix.lower() == ".csv":
                dataframe = pd.read_csv(path)
            elif path.suffix.lower() in {".xlsx", ".xls"}:
                dataframe = pd.read_excel(path, sheet_name=sheet_name)
            else:
                raise ValueError(
                    "Unsupported file type. Use CSV, XLSX, or XLS files."
                )
        except (OSError, UnicodeDecodeError, pd.errors.ParserError) as error:
            logger.exception("Failed to read source file: %s", path.name)
            raise ValueError(
                f"Could not read '{path.name}': {error}"
            ) from error

        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("Select one Excel sheet at a time.")

        return self.clean_dataframe(dataframe, source_file=path.name)

    def prepare_dataframe(
        self,
        dataframe: pd.DataFrame,
        source_file: str,
    ) -> pd.DataFrame:
        """Standardize source headers and attach citation provenance."""
        prepared = dataframe.copy()

        prepared.columns = [
            map_to_canonical_column(column)
            for column in prepared.columns
        ]

        duplicate_columns = prepared.columns[
            prepared.columns.duplicated()
        ].unique()

        if len(duplicate_columns) > 0:
            raise SchemaValidationError(
                "Multiple source columns map to the same field: "
                f"{duplicate_columns.tolist()}"
            )

        if not self.required_columns.issubset(prepared.columns):
            prepared = self.reshape_wide_dataframe(prepared, source_file)

        self.validate_schema(prepared, source_file)

        prepared["source_file"] = source_file
        if "source_row" not in prepared.columns:
            prepared["source_row"] = prepared.index + 2

        return prepared

    def reshape_wide_dataframe(
        self,
        dataframe: pd.DataFrame,
        source_file: str,
    ) -> pd.DataFrame:
        """Convert arbitrary wide financial tables into metric/value rows."""
        canonical_metadata = {
            "statement_type",
            "period",
            "date",
            "currency",
            "business_unit",
        }
        measure_columns = [
            column
            for column in dataframe.columns
            if column not in canonical_metadata
            and dataframe[column].map(parse_currency_amount).notna().any()
        ]

        if not measure_columns:
            return dataframe

        source_rows = pd.Series(
            dataframe.index + 2,
            index=dataframe.index,
            name="_source_row",
        )
        dataframe = dataframe.assign(_source_row=source_rows)
        identifier_columns = [
            column
            for column in dataframe.columns
            if column not in measure_columns
        ]
        reshaped = dataframe.melt(
            id_vars=identifier_columns,
            value_vars=measure_columns,
            var_name="metric",
            value_name="amount",
        )
        reshaped = reshaped.dropna(subset=["amount"])
        reshaped["source_row"] = reshaped.pop("_source_row")

        logger.info(
            "Reshaped wide source %s: %d measure columns into %d records.",
            source_file,
            len(measure_columns),
            len(reshaped),
        )
        return reshaped

    def clean_dataframe(
        self,
        dataframe: pd.DataFrame,
        source_file: str,
    ) -> pd.DataFrame:
        """Clean a financial table and return the canonical schema."""
        cleaned = self.prepare_dataframe(dataframe, source_file)

        # Add optional canonical fields missing from the uploaded statement.
        for column in CANONICAL_COLUMNS:
            if column not in cleaned.columns:
                cleaned[column] = pd.NA

        # Normalize meaningful text without inventing missing information.
        for column in [
            "statement_type",
            "metric",
            "period",
            "currency",
            "business_unit",
        ]:
            cleaned[column] = cleaned[column].astype("string").str.strip()

        # Blank metric names cannot support a factual financial answer.
        cleaned.loc[cleaned["metric"].eq(""), "metric"] = pd.NA

        # Convert raw values to consistent numeric amounts.
        cleaned["amount"] = cleaned["amount"].map(parse_currency_amount)

        # Standardize date representation. Invalid dates become missing values.
        parsed_dates = pd.to_datetime(
            cleaned["date"],
            errors="coerce",
            utc=True,
        )
        cleaned["date"] = parsed_dates.dt.strftime("%Y-%m-%d")

        # Use the configured default only when the source omitted a currency.
        cleaned["currency"] = cleaned["currency"].fillna(
            settings.default_currency
        ).str.upper()

        # Rows with no metric or usable numeric amount are not retrievable facts.
        rows_before_missing_value_cleanup = len(cleaned)
        cleaned = cleaned.dropna(subset=["metric", "amount"])

        # Source information is excluded from duplicate detection intentionally:
        # identical facts from repeated exports should appear only once.
        duplicate_keys = [
            "statement_type",
            "metric",
            "period",
            "date",
            "amount",
            "currency",
            "business_unit",
        ]
        rows_before_deduplication = len(cleaned)
        cleaned = cleaned.drop_duplicates(subset=duplicate_keys, keep="first")

        result = cleaned.loc[:, CANONICAL_COLUMNS].reset_index(drop=True)

        logger.info(
            "Cleaned %s: input_rows=%d, removed_missing=%d, "
            "removed_duplicates=%d, output_rows=%d",
            source_file,
            len(dataframe),
            rows_before_missing_value_cleanup - rows_before_deduplication,
            rows_before_deduplication - len(result),
            len(result),
        )

        return result

    def load_many(self, file_paths: Iterable[str | Path]) -> pd.DataFrame:
        """Load and merge multiple financial statements into one table."""
        tables = [self.load_file(path) for path in file_paths]

        if not tables:
            raise ValueError("Provide at least one CSV or Excel file.")

        merged = pd.concat(tables, ignore_index=True)

        logger.info(
            "Merged %d files into %d financial rows.",
            len(tables),
            len(merged),
        )

        return merged

    def validate_schema(
        self,
        dataframe: pd.DataFrame,
        source_file: str,
    ) -> None:
        """Fail early when essential financial fields are absent."""
        missing_columns = self.required_columns - set(dataframe.columns)

        if missing_columns:
            available_columns = ", ".join(dataframe.columns)
            raise SchemaValidationError(
                f"{source_file} is missing required columns: "
                f"{sorted(missing_columns)}. "
                f"Available columns: {available_columns}"
            )