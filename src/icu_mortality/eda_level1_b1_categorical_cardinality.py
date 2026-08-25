"""EDA Level 1 - B.1: Categorical feature cardinality analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype

from icu_mortality.data import (
    TARGET_COLUMN,
    load_training_data,
)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

REPORTS_TABLES_DIR = Path("reports") / "tables"

SUMMARY_OUTPUT_PATH = (
    REPORTS_TABLES_DIR
    / "eda_level1_b1_categorical_cardinality_summary.csv"
)

DICTIONARY_PATH = (
    Path("data")
    / "reference"
    / "WiDS Datathon 2020 Dictionary.csv"
)


# ---------------------------------------------------------------------
# Feature dictionary
# ---------------------------------------------------------------------

def load_feature_dictionary() -> dict[
    str,
    dict[str, str],
]:
    """
    Load categorical feature metadata from the WiDS data dictionary.

    Mapping:

        Variable Name ->
        {
            "description": ...,
            "data_type": ...
        }
    """

    if not DICTIONARY_PATH.exists():
        raise FileNotFoundError(
            "WiDS data dictionary was not found: "
            f"{DICTIONARY_PATH.resolve()}"
        )

    dictionary = pd.read_csv(
        DICTIONARY_PATH
    )

    required_columns = {
        "Variable Name",
        "Description",
        "Data Type",
    }

    missing_columns = (
        required_columns
        - set(dictionary.columns)
    )

    if missing_columns:
        raise ValueError(
            "Dictionary is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    feature_metadata: dict[
        str,
        dict[str, str],
    ] = {}

    for _, row in dictionary.iterrows():

        feature = row[
            "Variable Name"
        ]

        if pd.isna(feature):
            continue

        description = (
            ""
            if pd.isna(
                row["Description"]
            )
            else str(
                row["Description"]
            )
        )

        data_type = (
            ""
            if pd.isna(
                row["Data Type"]
            )
            else str(
                row["Data Type"]
            )
        )

        feature_metadata[
            str(feature)
        ] = {
            "description": description,
            "data_type": data_type,
        }

    return feature_metadata


# ---------------------------------------------------------------------
# Categorical feature selection
# ---------------------------------------------------------------------

def get_categorical_features(
    dataframe: pd.DataFrame,
) -> list[str]:
    """
    Return categorical predictor features for B.1 analysis.

    The target column is excluded.
    """

    features: list[str] = []

    for feature in dataframe.columns:

        if feature == TARGET_COLUMN:
            continue

        if is_numeric_dtype(
            dataframe[feature]
        ):
            continue

        features.append(
            feature
        )

    return features


# ---------------------------------------------------------------------
# Cardinality classification
# ---------------------------------------------------------------------

def classify_cardinality(
    unique_categories: int,
) -> str:
    """
    Classify categorical cardinality for descriptive EDA.

    Working thresholds:

        <= 2   -> binary
        3-10   -> low
        11-20  -> moderate
        > 20   -> high

    These are descriptive EDA thresholds and are not automatic
    preprocessing or encoding rules.
    """

    if unique_categories <= 2:
        return "binary"

    if unique_categories <= 10:
        return "low"

    if unique_categories <= 20:
        return "moderate"

    return "high"


# ---------------------------------------------------------------------
# Feature analysis
# ---------------------------------------------------------------------

def analyze_categorical_cardinality(
    dataframe: pd.DataFrame,
    feature: str,
    feature_metadata: dict[
        str,
        dict[str, str],
    ],
) -> dict[str, object]:
    """
    Analyze cardinality of one categorical feature.
    """

    series = dataframe[
        feature
    ]

    observed = (
        series
        .dropna()
    )

    missing_percent = (
        series.isna()
        .mean()
        * 100
    )

    unique_categories = int(
        observed.nunique()
    )

    cardinality_level = (
        classify_cardinality(
            unique_categories
        )
    )

    dictionary_data_type = ""
    feature_description = ""

    # Only show metadata when cardinality deserves
    # additional manual attention.
    if cardinality_level in {
        "moderate",
        "high",
    }:

        metadata = (
            feature_metadata.get(
                feature
            )
        )

        if metadata is None:

            dictionary_data_type = (
                "Data type not found in dictionary"
            )

            feature_description = (
                "Description not found in dictionary"
            )

        else:

            dictionary_data_type = (
                metadata[
                    "data_type"
                ]
            )

            feature_description = (
                metadata[
                    "description"
                ]
            )

    return {
        "feature": feature,
        "cardinality_level": (
            cardinality_level
        ),
        "dictionary_data_type": (
            dictionary_data_type
        ),
        "feature_description": (
            feature_description
        ),
        "missing_percent": (
            missing_percent
        ),
        "unique_categories": (
            unique_categories
        ),
    }


# ---------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------

def build_cardinality_summary(
    dataframe: pd.DataFrame,
    feature_metadata: dict[
        str,
        dict[str, str],
    ],
) -> pd.DataFrame:
    """
    Build B.1 cardinality summary for categorical features.
    """

    features = (
        get_categorical_features(
            dataframe
        )
    )

    rows: list[
        dict[str, object]
    ] = []

    for feature in features:

        rows.append(
            analyze_categorical_cardinality(
                dataframe=dataframe,
                feature=feature,
                feature_metadata=(
                    feature_metadata
                ),
            )
        )

    results = pd.DataFrame(
        rows
    )

    if results.empty:
        return results

    return (
        results
        .sort_values(
            by="unique_categories",
            ascending=False,
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

def save_summary(
    results: pd.DataFrame,
) -> None:
    """
    Save B.1 categorical cardinality summary.
    """

    REPORTS_TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        SUMMARY_OUTPUT_PATH,
        index=False,
    )


def print_results(
    results: pd.DataFrame,
) -> None:
    """
    Print B.1 categorical cardinality results.
    """

    print("=" * 160)

    print(
        "EDA LEVEL 1 — B.1 CATEGORICAL FEATURE CARDINALITY"
    )

    print("=" * 160)

    print(
        f"Categorical features investigated: "
        f"{len(results)}"
    )

    print("\n" + "=" * 160)

    print(
        "CARDINALITY SUMMARY — "
        "SORTED BY NUMBER OF UNIQUE CATEGORIES"
    )

    print("=" * 160)

    if results.empty:

        print(
            "No categorical features were found."
        )

    else:

        print(
            results.to_string(
                index=False,
                float_format=lambda value: (
                    f"{value:.2f}"
                ),
            )
        )

    print("\n" + "=" * 160)

    print(
        "CARDINALITY LEVEL COUNTS"
    )

    print("=" * 160)

    if not results.empty:

        print(
            results[
                "cardinality_level"
            ]
            .value_counts()
            .to_string()
        )

    print("\n" + "=" * 160)

    print(
        "MODERATE / HIGH CARDINALITY FEATURES"
    )

    print("=" * 160)

    if not results.empty:

        review_features = (
            results[
                results[
                    "cardinality_level"
                ].isin(
                    [
                        "moderate",
                        "high",
                    ]
                )
            ]
        )

        if review_features.empty:

            print(
                "No moderate/high cardinality "
                "categorical features detected."
            )

        else:

            print(
                review_features[
                    [
                        "feature",
                        "cardinality_level",
                        "unique_categories",
                        "missing_percent",
                        "dictionary_data_type",
                        "feature_description",
                    ]
                ]
                .to_string(
                    index=False,
                    float_format=lambda value: (
                        f"{value:.2f}"
                    ),
                )
            )

    print("\n" + "=" * 160)

    print(
        "ARTIFACT SAVED"
    )

    print("=" * 160)

    print(
        SUMMARY_OUTPUT_PATH
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def run_b1_categorical_cardinality() -> pd.DataFrame:
    """
    Run EDA Level 1 B.1.
    """

    dataframe = (
        load_training_data()
    )

    feature_metadata = (
        load_feature_dictionary()
    )

    results = (
        build_cardinality_summary(
            dataframe=dataframe,
            feature_metadata=(
                feature_metadata
            ),
        )
    )

    save_summary(
        results
    )

    print_results(
        results
    )

    return results


if __name__ == "__main__":
    run_b1_categorical_cardinality()