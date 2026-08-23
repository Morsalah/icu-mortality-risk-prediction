"""Raw-data validation utilities for the ICU mortality project."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from icu_mortality.data import (
    TARGET_COLUMN,
    load_training_data,
)

REPORTS_DIR = Path("reports")
TABLES_DIR = REPORTS_DIR / "tables"
METRICS_DIR = REPORTS_DIR / "metrics"

HIGH_MISSING_THRESHOLD = 0.50

def create_report_directories() -> None:
    """Create validation-report directories."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

def summarize_schema(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a summary of columns, dtypes and missingness."""
    rows: list[dict[str, object]] = []
    total_rows = len(dataframe)

    for column in dataframe.columns:
        missing_count = int(dataframe[column].isna().sum())
        rows.append(
            {
                "column": column,
                "dtype": str(dataframe[column].dtype),
                "missing_count": missing_count,
                "missing_percent": missing_count / total_rows * 100,
                "unique_values": int(dataframe[column].nunique(dropna=True)),
            }
        )

    return pd.DataFrame(rows)

def summarize_data_types(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Summarize technical dataframe dtypes with counts and percentages."""
    dtype_counts = dataframe.dtypes.astype(str).value_counts()
    summary = dtype_counts.rename("count").reset_index()
    summary.columns = ["dtype", "count"]
    summary["percent"] = summary["count"] / len(dataframe.columns) * 100
    return summary

def summarize_non_numeric_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Summarize all non-numeric columns."""
    rows: list[dict[str, object]] = []

    for column in dataframe.select_dtypes(exclude="number").columns:
        rows.append(
            {
                "column": column,
                "dtype": str(dataframe[column].dtype),
                "unique_values": int(dataframe[column].nunique(dropna=True)),
                "missing_percent": dataframe[column].isna().mean() * 100,
                "high_missing": (
                    dataframe[column].isna().mean() >= HIGH_MISSING_THRESHOLD
                ),
            }
        )

    return pd.DataFrame(rows)

def validate_target(dataframe: pd.DataFrame) -> None:
    """Validate the mortality target column."""
    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is missing.")

    if dataframe[TARGET_COLUMN].isna().any():
        raise ValueError("Target column contains missing values.")

    target_values = set(dataframe[TARGET_COLUMN].unique())
    if not target_values.issubset({0, 1}):
        raise ValueError("Target column must contain only 0 and 1.")


def find_duplicate_rows(dataframe: pd.DataFrame) -> int:
    """Return the number of fully duplicated rows."""
    return int(dataframe.duplicated().sum())


def find_constant_columns(dataframe: pd.DataFrame) -> list[str]:
    """Return columns containing at most one non-missing value."""
    return [
        column
        for column in dataframe.columns
        if dataframe[column].nunique(dropna=True) <= 1
    ]


def find_high_missing_columns(
    dataframe: pd.DataFrame,
    threshold: float = HIGH_MISSING_THRESHOLD,
) -> pd.DataFrame:
    """Return columns whose missing fraction exceeds a threshold."""
    if not 0 <= threshold <= 1:
        raise ValueError("Missing-value threshold must be between 0 and 1.")

    missing_fraction = dataframe.isna().mean()
    result = (
        missing_fraction[missing_fraction >= threshold]
        .sort_values(ascending=False)
        .rename("missing_fraction")
        .reset_index()
        .rename(columns={"index": "column"})
    )
    result["missing_percent"] = result["missing_fraction"] * 100
    return result


def find_potential_identifier_columns(
    dataframe: pd.DataFrame,
    uniqueness_threshold: float = 0.95,
) -> pd.DataFrame:
    """Identify columns that behave like row identifiers."""
    if not 0 < uniqueness_threshold <= 1:
        raise ValueError("Uniqueness threshold must be in (0, 1].")

    rows: list[dict[str, object]] = []
    total_rows = len(dataframe)

    for column in dataframe.columns:
        if column == TARGET_COLUMN:
            continue

        unique_count = dataframe[column].nunique(dropna=True)
        uniqueness_ratio = unique_count / total_rows

        if uniqueness_ratio >= uniqueness_threshold:
            rows.append(
                {
                    "column": column,
                    "unique_values": unique_count,
                    "uniqueness_ratio": uniqueness_ratio,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=["column", "unique_values", "uniqueness_ratio"]
        )

    return (
        pd.DataFrame(rows)
        .sort_values(by="uniqueness_ratio", ascending=False)
        .reset_index(drop=True)
    )


def build_constant_columns_table(constant_columns: list[str]) -> pd.DataFrame:
    """Convert constant-column names into a table."""
    return pd.DataFrame({"column": constant_columns})


def build_audit_summary(
    dataframe: pd.DataFrame,
    duplicate_rows: int,
    constant_columns: list[str],
    high_missing_columns: pd.DataFrame,
    identifier_candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Build a one-row summary of the raw-data audit."""
    target_counts = dataframe[TARGET_COLUMN].value_counts()
    negative_count = int(target_counts.get(0, 0))
    positive_count = int(target_counts.get(1, 0))
    positive_rate = positive_count / len(dataframe)

    return pd.DataFrame(
        [
            {
                "rows": len(dataframe),
                "columns": len(dataframe.columns),
                "duplicate_rows": duplicate_rows,
                "constant_columns": len(constant_columns),
                "high_missing_columns": len(high_missing_columns),
                "identifier_candidates": len(identifier_candidates),
                "target_negative_count": negative_count,
                "target_positive_count": positive_count,
                "target_positive_rate": positive_rate,
            }
        ]
    )


def save_audit_artifacts(
    schema: pd.DataFrame,
    data_type_summary: pd.DataFrame,
    non_numeric_summary: pd.DataFrame,
    constant_columns: list[str],
    high_missing_columns: pd.DataFrame,
    identifier_candidates: pd.DataFrame,
    audit_summary: pd.DataFrame,
) -> None:
    """Save raw-data audit results as CSV artifacts."""
    create_report_directories()

    schema.to_csv(TABLES_DIR / "raw_data_schema.csv", index=False)
    data_type_summary.to_csv(TABLES_DIR / "data_type_summary.csv", index=False)
    non_numeric_summary.to_csv(TABLES_DIR / "non_numeric_columns.csv", index=False)
    non_numeric_summary.loc[non_numeric_summary["high_missing"]].to_csv(
        TABLES_DIR / "high_missing_non_numeric_columns.csv", index=False
    )
    build_constant_columns_table(constant_columns).to_csv(
        TABLES_DIR / "constant_columns.csv",
        index=False,
    )
    high_missing_columns.to_csv(
        TABLES_DIR / "high_missing_columns.csv",
        index=False,
    )
    identifier_candidates.to_csv(
        TABLES_DIR / "identifier_candidates.csv",
        index=False,
    )
    audit_summary.to_csv(
        METRICS_DIR / "raw_data_audit_summary.csv",
        index=False,
    )


def run_raw_data_audit() -> None:
    """Run and persist the initial raw-data quality audit."""
    dataframe = load_training_data()
    validate_target(dataframe)

    schema = summarize_schema(dataframe)
    data_type_summary = summarize_data_types(dataframe)
    non_numeric_summary = summarize_non_numeric_columns(dataframe)
    high_missing_non_numeric = non_numeric_summary.loc[
        non_numeric_summary["high_missing"]
    ].copy()
    duplicates = find_duplicate_rows(dataframe)
    constant_columns = find_constant_columns(dataframe)
    high_missing_columns = find_high_missing_columns(
        dataframe, threshold=HIGH_MISSING_THRESHOLD
    )
    identifier_candidates = find_potential_identifier_columns(dataframe)

    audit_summary = build_audit_summary(
        dataframe=dataframe,
        duplicate_rows=duplicates,
        constant_columns=constant_columns,
        high_missing_columns=high_missing_columns,
        identifier_candidates=identifier_candidates,
    )

    save_audit_artifacts(
        schema=schema,
        data_type_summary=data_type_summary,
        non_numeric_summary=non_numeric_summary,
        constant_columns=constant_columns,
        high_missing_columns=high_missing_columns,
        identifier_candidates=identifier_candidates,
        audit_summary=audit_summary,
    )

    print("=" * 70)
    print("RAW DATA AUDIT")
    print("=" * 70)
    print(f"Rows: {len(dataframe)}")
    print(f"Columns: {len(dataframe.columns)}")
    print(f"Duplicate rows: {duplicates}")
    print(f"Constant columns: {len(constant_columns)}")
    print(f"Columns with >= 50% missing values: {len(high_missing_columns)}")
    print(f"Non-numeric columns: {len(non_numeric_summary)}")
    print(
        f"Non-numeric columns with >= {HIGH_MISSING_THRESHOLD:.0%} missing values: "
        f"{len(high_missing_non_numeric)}"
    )
    print(f"Potential identifier columns: {len(identifier_candidates)}")

    print("\n" + "=" * 70)
    print("CONSTANT COLUMNS")
    print("=" * 70)
    if constant_columns:
        for column in constant_columns:
            print(column)
    else:
        print("None")

    print("\n" + "=" * 70)
    print("HIGH-MISSING COLUMNS")
    print("=" * 70)
    if high_missing_columns.empty:
        print("None")
    else:
        print(
            high_missing_columns[["column", "missing_percent"]]
            .to_string(index=False)
        )

    print("\n" + "=" * 70)
    print("POTENTIAL IDENTIFIER COLUMNS")
    print("=" * 70)
    if identifier_candidates.empty:
        print("None")
    else:
        print(identifier_candidates.to_string(index=False))

    print("\n" + "=" * 70)
    print("DATA TYPE PROFILE")
    print("=" * 70)
    print(data_type_summary.to_string(index=False))

    print("\n" + "=" * 70)
    print("NON-NUMERIC COLUMNS")
    print("=" * 70)
    if non_numeric_summary.empty:
        print("None")
    else:
        print(non_numeric_summary.to_string(index=False))

    print("\n" + "=" * 70)
    print("HIGH-MISSING NON-NUMERIC COLUMNS")
    print("=" * 70)
    if high_missing_non_numeric.empty:
        print("None")
    else:
        print(high_missing_non_numeric.to_string(index=False))

    print("\n" + "=" * 70)
    print("TOP 20 COLUMNS BY MISSINGNESS")
    print("=" * 70)
    print(
        schema
        .sort_values(by="missing_percent", ascending=False)
        .head(20)[
            ["column", "dtype", "missing_percent", "unique_values"]
        ]
        .to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("AUDIT ARTIFACTS SAVED")
    print("=" * 70)
    print(TABLES_DIR / "raw_data_schema.csv")
    print(TABLES_DIR / "data_type_summary.csv")
    print(TABLES_DIR / "non_numeric_columns.csv")
    print(TABLES_DIR / "high_missing_non_numeric_columns.csv")
    print(TABLES_DIR / "constant_columns.csv")
    print(TABLES_DIR / "high_missing_columns.csv")
    print(TABLES_DIR / "identifier_candidates.csv")
    print(METRICS_DIR / "raw_data_audit_summary.csv")


if __name__ == "__main__":
    run_raw_data_audit()