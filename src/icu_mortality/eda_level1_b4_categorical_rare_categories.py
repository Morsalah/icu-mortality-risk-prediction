"""EDA Level 1 - B.4: Categorical rare-category analysis."""

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
    / "eda_level1_b4_categorical_rare_categories_summary.csv"
)

DETAIL_OUTPUT_PATH = (
    REPORTS_TABLES_DIR
    / "eda_level1_b4_categorical_rare_categories_detail.csv"
)

DICTIONARY_PATH = (
    Path("data")
    / "reference"
    / "WiDS Datathon 2020 Dictionary.csv"
)


# ---------------------------------------------------------------------
# Working EDA threshold
# ---------------------------------------------------------------------

RARE_CATEGORY_THRESHOLD_PERCENT = 1.0


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
    Return categorical predictor features for B.4 analysis.

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
# Rare-category analysis
# ---------------------------------------------------------------------

def analyze_rare_categories(
    dataframe: pd.DataFrame,
    feature: str,
    feature_metadata: dict[
        str,
        dict[str, str],
    ],
) -> tuple[
    dict[str, object],
    pd.DataFrame,
]:
    """
    Analyze rare categories for one categorical feature.

    Missing values are excluded from the rare-category calculation.

    A category is considered rare when its frequency among observed
    values is below RARE_CATEGORY_THRESHOLD_PERCENT.

    The threshold is a working EDA threshold only. A rare category
    is not automatically considered unimportant and should not be
    automatically merged or removed.
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

    # -------------------------------------------------------------
    # No observed values
    # -------------------------------------------------------------

    if observed.empty:

        summary = {
            "feature": feature,
            "dictionary_data_type": "",
            "feature_description": "",
            "missing_percent": missing_percent,
            "unique_categories": unique_categories,
            "rare_categories_count": 0,
            "rare_categories_percent": float("nan"),
            "rare_observations_percent": float("nan"),
        }

        detail = pd.DataFrame(
            columns=[
                "feature",
                "category_value",
                "count",
                "percent_observed",
            ]
        )

        return (
            summary,
            detail,
        )

    # -------------------------------------------------------------
    # Category frequencies
    # -------------------------------------------------------------

    category_counts = (
        observed
        .value_counts()
    )

    category_percentages = (
        category_counts
        / len(observed)
        * 100
    )

    # -------------------------------------------------------------
    # Identify rare categories
    # -------------------------------------------------------------

    rare_mask = (
        category_percentages
        < RARE_CATEGORY_THRESHOLD_PERCENT
    )

    rare_counts = (
        category_counts[
            rare_mask
        ]
    )

    rare_percentages = (
        category_percentages[
            rare_mask
        ]
    )

    rare_categories_count = int(
        len(
            rare_counts
        )
    )

    # -------------------------------------------------------------
    # Percentage of category TYPES that are rare
    # -------------------------------------------------------------

    rare_categories_percent = (
        rare_categories_count
        / unique_categories
        * 100
        if unique_categories > 0
        else float("nan")
    )

    # -------------------------------------------------------------
    # Percentage of OBSERVATIONS belonging to rare categories
    # -------------------------------------------------------------

    rare_observations_percent = (
        rare_counts.sum()
        / len(observed)
        * 100
    )

    # -------------------------------------------------------------
    # Dictionary metadata
    # -------------------------------------------------------------

    dictionary_data_type = ""
    feature_description = ""

    # Only display dictionary information when rare categories exist.
    if rare_categories_count > 0:

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

    # -------------------------------------------------------------
    # Summary row
    # -------------------------------------------------------------

    summary = {
        "feature": feature,
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
        "rare_categories_count": (
            rare_categories_count
        ),
        "rare_categories_percent": (
            rare_categories_percent
        ),
        "rare_observations_percent": (
            rare_observations_percent
        ),
    }

    # -------------------------------------------------------------
    # Detail table
    # -------------------------------------------------------------

    detail = pd.DataFrame(
        {
            "feature": feature,
            "category_value": (
                rare_counts.index
            ),
            "count": (
                rare_counts.values
            ),
            "percent_observed": (
                rare_percentages.values
            ),
        }
    )

    return (
        summary,
        detail,
    )


# ---------------------------------------------------------------------
# Build tables
# ---------------------------------------------------------------------

def build_rare_category_tables(
    dataframe: pd.DataFrame,
    feature_metadata: dict[
        str,
        dict[str, str],
    ],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Build B.4 summary and detail tables.
    """

    features = (
        get_categorical_features(
            dataframe
        )
    )

    summary_rows: list[
        dict[str, object]
    ] = []

    detail_tables: list[
        pd.DataFrame
    ] = []

    for feature in features:

        (
            summary,
            detail,
        ) = analyze_rare_categories(
            dataframe=dataframe,
            feature=feature,
            feature_metadata=(
                feature_metadata
            ),
        )

        summary_rows.append(
            summary
        )

        if not detail.empty:

            detail_tables.append(
                detail
            )

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------

    summary_results = pd.DataFrame(
        summary_rows
    )

    if not summary_results.empty:

        summary_results = (
            summary_results
            .sort_values(
                by=[
                    "rare_observations_percent",
                    "rare_categories_percent",
                ],
                ascending=[
                    False,
                    False,
                ],
                na_position="last",
            )
            .reset_index(
                drop=True
            )
        )

    # -------------------------------------------------------------
    # Detail
    # -------------------------------------------------------------

    if detail_tables:

        detail_results = (
            pd.concat(
                detail_tables,
                ignore_index=True,
            )
            .sort_values(
                by=[
                    "feature",
                    "percent_observed",
                ],
                ascending=[
                    True,
                    True,
                ],
            )
            .reset_index(
                drop=True
            )
        )

    else:

        detail_results = pd.DataFrame(
            columns=[
                "feature",
                "category_value",
                "count",
                "percent_observed",
            ]
        )

    return (
        summary_results,
        detail_results,
    )


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

def save_results(
    summary_results: pd.DataFrame,
    detail_results: pd.DataFrame,
) -> None:
    """
    Save B.4 summary and detail tables.
    """

    REPORTS_TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_results.to_csv(
        SUMMARY_OUTPUT_PATH,
        index=False,
    )

    detail_results.to_csv(
        DETAIL_OUTPUT_PATH,
        index=False,
    )


def print_results(
    summary_results: pd.DataFrame,
    detail_results: pd.DataFrame,
) -> None:
    """
    Print B.4 rare-category results.
    """

    print("=" * 170)

    print(
        "EDA LEVEL 1 — B.4 CATEGORICAL RARE CATEGORIES"
    )

    print("=" * 170)

    print(
        f"Categorical features investigated: "
        f"{len(summary_results)}"
    )

    print(
        f"Working rare-category threshold: "
        f"< {RARE_CATEGORY_THRESHOLD_PERCENT:.2f}% "
        f"of observed values"
    )

    # -------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------

    print("\n" + "=" * 170)

    print(
        "RARE CATEGORY SUMMARY"
    )

    print("=" * 170)

    if summary_results.empty:

        print(
            "No categorical features were found."
        )

    else:

        print(
            summary_results.to_string(
                index=False,
                float_format=lambda value: (
                    f"{value:.2f}"
                ),
            )
        )

    # -------------------------------------------------------------
    # Detail table
    # -------------------------------------------------------------

    print("\n" + "=" * 170)

    print(
        "RARE CATEGORY DETAILS"
    )

    print("=" * 170)

    if detail_results.empty:

        print(
            "No rare categories were detected."
        )

    else:

        print(
            detail_results.to_string(
                index=False,
                float_format=lambda value: (
                    f"{value:.2f}"
                ),
            )
        )

    # -------------------------------------------------------------
    # Dataset-level summary
    # -------------------------------------------------------------

    print("\n" + "=" * 170)

    print(
        "SUMMARY"
    )

    print("=" * 170)

    if not summary_results.empty:

        features_with_rare = int(
            (
                summary_results[
                    "rare_categories_count"
                ]
                > 0
            )
            .sum()
        )

        total_rare_categories = int(
            summary_results[
                "rare_categories_count"
            ]
            .sum()
        )

        print(
            f"Features containing rare categories: "
            f"{features_with_rare}"
        )

        print(
            f"Total rare categories detected: "
            f"{total_rare_categories}"
        )

    # -------------------------------------------------------------
    # Saved artifacts
    # -------------------------------------------------------------

    print("\n" + "=" * 170)

    print(
        "ARTIFACTS SAVED"
    )

    print("=" * 170)

    print(
        SUMMARY_OUTPUT_PATH
    )

    print(
        DETAIL_OUTPUT_PATH
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def run_b4_categorical_rare_categories() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Run EDA Level 1 B.4.
    """

    dataframe = (
        load_training_data()
    )

    feature_metadata = (
        load_feature_dictionary()
    )

    (
        summary_results,
        detail_results,
    ) = build_rare_category_tables(
        dataframe=dataframe,
        feature_metadata=(
            feature_metadata
        ),
    )

    save_results(
        summary_results,
        detail_results,
    )

    print_results(
        summary_results,
        detail_results,
    )

    return (
        summary_results,
        detail_results,
    )


if __name__ == "__main__":
    run_b4_categorical_rare_categories()