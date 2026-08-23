"""High-level feature audit utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from icu_mortality.data import (
    TARGET_COLUMN,
    load_training_data,
)


DATA_REFERENCE_DIR = Path("data") / "reference"

DICTIONARY_PATH = (
    DATA_REFERENCE_DIR
    / "WiDS Datathon 2020 Dictionary.csv"
)

REPORTS_TABLES_DIR = Path("reports") / "tables"

RAW_SCHEMA_PATH = (
    REPORTS_TABLES_DIR
    / "raw_data_schema.csv"
)

FEATURE_CATALOG_PATH = (
    REPORTS_TABLES_DIR
    / "feature_catalog.csv"
)


HIGH_MISSING_THRESHOLD = 0.50
HIGH_CARDINALITY_THRESHOLD = 0.95


KNOWN_DROP_COLUMNS = {
    "encounter_id",
    "patient_id",
    "readmission_status",
}


def load_data_dictionary(
    path: Path = DICTIONARY_PATH,
) -> pd.DataFrame:
    """Load the WiDS feature dictionary."""

    if not path.exists():
        raise FileNotFoundError(
            "WiDS data dictionary was not found: "
            f"{path.resolve()}"
        )

    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise ValueError(
            "WiDS data dictionary is empty."
        )

    return dataframe


def load_raw_schema(
    path: Path = RAW_SCHEMA_PATH,
) -> pd.DataFrame:
    """Load the schema created during raw-data validation."""

    if not path.exists():
        raise FileNotFoundError(
            "Raw-data schema was not found. "
            "Run the validation phase first: "
            f"{path.resolve()}"
        )

    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise ValueError(
            "Raw-data schema is empty."
        )

    return dataframe


def normalize_dictionary_columns(
    dictionary: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize dictionary column names."""

    renamed = dictionary.rename(
        columns={
            "Category": "dictionary_category",
            "Variable Name": "feature",
            "Unit of Measure": "unit",
            "Data Type": "dictionary_data_type",
            "Description": "description",
            "Example": "example",
        }
    )

    required_columns = {
        "feature",
        "dictionary_category",
        "dictionary_data_type",
        "description",
    }

    missing_columns = (
        required_columns
        - set(renamed.columns)
    )

    if missing_columns:
        raise ValueError(
            "Data dictionary is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    return renamed


def calculate_uniqueness_ratio(
    unique_values: int | float,
    total_rows: int,
) -> float:
    """Calculate the fraction of unique values in a feature."""

    if total_rows <= 0:
        raise ValueError(
            "Total rows must be greater than zero."
        )

    return (
        float(unique_values)
        / total_rows
    )


def assign_decision(
    feature: str,
    high_missing_flag: bool,
    high_cardinality_flag: bool,
) -> str:
    """Assign a high-level feature decision."""

    if feature == TARGET_COLUMN:
        return "TARGET"

    if feature in KNOWN_DROP_COLUMNS:
        return "DROP"

    if high_missing_flag or high_cardinality_flag:
        return "INVESTIGATE"

    return "KEEP"


def assign_decision_reason(
    feature: str,
    high_missing_flag: bool,
    high_cardinality_flag: bool,
) -> str:
    """Explain the high-level feature decision."""

    if feature == TARGET_COLUMN:
        return "Target variable."

    if feature in KNOWN_DROP_COLUMNS:
        return (
            "Known identifier or constant feature."
        )

    if (
        high_missing_flag
        and high_cardinality_flag
    ):
        return (
            "High missingness and high cardinality."
        )

    if high_missing_flag:
        return (
            "High missingness."
        )

    if high_cardinality_flag:
        return (
            "High cardinality."
        )

    return (
        "No high-level exclusion or investigation rule triggered."
    )


def build_feature_catalog() -> pd.DataFrame:
    """Build the high-level feature catalog."""

    training_data = load_training_data()

    schema = load_raw_schema()

    dictionary = normalize_dictionary_columns(
        load_data_dictionary()
    )

    training_columns = pd.DataFrame(
        {
            "feature": training_data.columns,
        }
    )

    catalog = training_columns.merge(
        schema.rename(
            columns={
                "column": "feature",
            }
        ),
        on="feature",
        how="left",
        validate="one_to_one",
    )

    dictionary_columns_to_keep = [
        column
        for column in [
            "feature",
            "dictionary_category",
            "unit",
            "dictionary_data_type",
            "description",
            "example",
        ]
        if column in dictionary.columns
    ]

    catalog = catalog.merge(
        dictionary[
            dictionary_columns_to_keep
        ],
        on="feature",
        how="left",
        validate="one_to_one",
    )

    total_rows = len(
        training_data
    )

    catalog["uniqueness_ratio"] = (
        catalog["unique_values"]
        .apply(
            lambda value: calculate_uniqueness_ratio(
                unique_values=value,
                total_rows=total_rows,
            )
        )
    )

    catalog["high_missing_flag"] = (
        catalog["missing_percent"]
        > HIGH_MISSING_THRESHOLD * 100
    )

    catalog["high_cardinality_flag"] = (
        catalog["uniqueness_ratio"]
        >= HIGH_CARDINALITY_THRESHOLD
    )

    catalog["decision"] = (
        catalog.apply(
            lambda row: assign_decision(
                feature=row["feature"],
                high_missing_flag=bool(
                    row["high_missing_flag"]
                ),
                high_cardinality_flag=bool(
                    row["high_cardinality_flag"]
                ),
            ),
            axis=1,
        )
    )

    catalog["decision_reason"] = (
        catalog.apply(
            lambda row: assign_decision_reason(
                feature=row["feature"],
                high_missing_flag=bool(
                    row["high_missing_flag"]
                ),
                high_cardinality_flag=bool(
                    row["high_cardinality_flag"]
                ),
            ),
            axis=1,
        )
    )

    catalog["deep_review_status"] = (
        "NOT_REVIEWED"
    )

    catalog["final_decision"] = ""
    catalog["notes"] = ""

    return catalog


def summarize_feature_catalog(
    catalog: pd.DataFrame,
) -> None:
    """Print a concise summary of the high-level feature audit."""

    print("=" * 70)
    print("HIGH-LEVEL FEATURE AUDIT")
    print("=" * 70)

    print(
        f"Features in training data: "
        f"{len(catalog)}"
    )

    print(
        f"Features matched to dictionary: "
        f"{catalog['description'].notna().sum()}"
    )

    print(
        f"Features missing from dictionary: "
        f"{catalog['description'].isna().sum()}"
    )

    print(
        f"Features with > "
        f"{HIGH_MISSING_THRESHOLD:.0%} missing values: "
        f"{catalog['high_missing_flag'].sum()}"
    )

    print(
        f"Features with >= "
        f"{HIGH_CARDINALITY_THRESHOLD:.0%} cardinality: "
        f"{catalog['high_cardinality_flag'].sum()}"
    )

    print("\n" + "=" * 70)
    print("HIGH-LEVEL DECISIONS")
    print("=" * 70)

    print(
        catalog["decision"]
        .value_counts()
        .to_string()
    )

    print("\n" + "=" * 70)
    print("TARGET")
    print("=" * 70)

    target_features = catalog[
        catalog["decision"] == "TARGET"
    ]

    print(
        target_features[
            [
                "feature",
                "decision_reason",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("DROP")
    print("=" * 70)

    drop_features = catalog[
        catalog["decision"] == "DROP"
    ]

    if drop_features.empty:
        print("None")
    else:
        print(
            drop_features[
                [
                    "feature",
                    "dictionary_category",
                    "decision_reason",
                ]
            ]
            .to_string(
                index=False
            )
        )

    print("\n" + "=" * 70)
    print("INVESTIGATE")
    print("=" * 70)

    investigate_features = catalog[
        catalog["decision"]
        == "INVESTIGATE"
    ]

    if investigate_features.empty:
        print("None")
    else:
        print(
            investigate_features[
                [
                    "feature",
                    "dictionary_category",
                    "missing_percent",
                    "uniqueness_ratio",
                    "high_missing_flag",
                    "high_cardinality_flag",
                    "decision_reason",
                ]
            ]
            .sort_values(
                by=[
                    "high_missing_flag",
                    "missing_percent",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
            .to_string(
                index=False
            )
        )

    print("\n" + "=" * 70)
    print("KEEP")
    print("=" * 70)

    keep_features = catalog[
        catalog["decision"] == "KEEP"
    ]

    print(
        f"Features kept for further analysis: "
        f"{len(keep_features)}"
    )


def save_feature_catalog(
    catalog: pd.DataFrame,
) -> Path:
    """Save the high-level feature catalog."""

    REPORTS_TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    catalog.to_csv(
        FEATURE_CATALOG_PATH,
        index=False,
    )

    return FEATURE_CATALOG_PATH


def run_feature_audit() -> pd.DataFrame:
    """Build, summarize and persist the high-level feature audit."""

    catalog = build_feature_catalog()

    summarize_feature_catalog(
        catalog
    )

    output_path = save_feature_catalog(
        catalog
    )

    print("\n" + "=" * 70)
    print("FEATURE CATALOG SAVED")
    print("=" * 70)

    print(
        output_path
    )

    return catalog


if __name__ == "__main__":
    run_feature_audit()