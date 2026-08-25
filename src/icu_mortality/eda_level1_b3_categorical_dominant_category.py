"""EDA Level 1 - B.3: Categorical dominant-category analysis."""

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
    / "eda_level1_b3_categorical_dominant_category_summary.csv"
)

DICTIONARY_PATH = (
    Path("data")
    / "reference"
    / "WiDS Datathon 2020 Dictionary.csv"
)


# ---------------------------------------------------------------------
# Review thresholds
# ---------------------------------------------------------------------

DOMINANCE_LOW_THRESHOLD_PERCENT = 80.0
DOMINANCE_MEDIUM_THRESHOLD_PERCENT = 90.0
DOMINANCE_HIGH_THRESHOLD_PERCENT = 95.0


# ---------------------------------------------------------------------
# Feature dictionary
# ---------------------------------------------------------------------

def load_feature_dictionary() -> dict[
    str,
    dict[str, str],
]:
    """
    Load feature metadata from the WiDS data dictionary.

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
    Return categorical predictor features for B.3 analysis.

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
# Review policy
# ---------------------------------------------------------------------

def assign_review_information(
    dominant_category_percent: float,
) -> tuple[str, str]:
    """
    Assign B.3 review priority and reason.

    Working EDA thresholds:

        < 80%   -> NO_REVIEW
        80-90%  -> LOW
        >90-95% -> MEDIUM
        >95%    -> HIGH

    These thresholds are used only for investigation priority.
    They do not represent automatic feature-removal rules.
    """

    if pd.isna(
        dominant_category_percent
    ):
        return (
            "NO_REVIEW",
            "NO_OBSERVED_VALUES",
        )

    if (
        dominant_category_percent
        < DOMINANCE_LOW_THRESHOLD_PERCENT
    ):
        return (
            "NO_REVIEW",
            "",
        )

    if (
        dominant_category_percent
        <= DOMINANCE_MEDIUM_THRESHOLD_PERCENT
    ):
        return (
            "LOW",
            "HIGH_CATEGORY_DOMINANCE",
        )

    if (
        dominant_category_percent
        <= DOMINANCE_HIGH_THRESHOLD_PERCENT
    ):
        return (
            "MEDIUM",
            "VERY_HIGH_CATEGORY_DOMINANCE",
        )

    return (
        "HIGH",
        "EXTREME_CATEGORY_DOMINANCE",
    )


# ---------------------------------------------------------------------
# Dominant-category analysis
# ---------------------------------------------------------------------

def analyze_dominant_category(
    dataframe: pd.DataFrame,
    feature: str,
    feature_metadata: dict[
        str,
        dict[str, str],
    ],
) -> dict[str, object]:
    """
    Analyze the dominant observed category of one categorical feature.

    Missing values are excluded from the dominant-category percentage.
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

    if observed.empty:

        return {
            "feature": feature,
            "review_priority": "NO_REVIEW",
            "review_reason": "NO_OBSERVED_VALUES",
            "dictionary_data_type": "",
            "feature_description": "",
            "missing_percent": missing_percent,
            "unique_categories": unique_categories,
            "dominant_category": None,
            "dominant_category_percent": float("nan"),
        }

    value_counts = (
        observed
        .value_counts()
    )

    dominant_category = (
        value_counts.index[0]
    )

    dominant_category_count = int(
        value_counts.iloc[0]
    )

    dominant_category_percent = (
        dominant_category_count
        / len(observed)
        * 100
    )

    (
        review_priority,
        review_reason,
    ) = assign_review_information(
        dominant_category_percent
    )

    dictionary_data_type = ""
    feature_description = ""

    # Only show dictionary metadata for features that
    # deserve meaningful manual investigation.
    if review_priority in {
        "MEDIUM",
        "HIGH",
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
        "review_priority": (
            review_priority
        ),
        "review_reason": (
            review_reason
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
        "dominant_category": (
            dominant_category
        ),
        "dominant_category_percent": (
            dominant_category_percent
        ),
    }


# ---------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------

def build_dominant_category_summary(
    dataframe: pd.DataFrame,
    feature_metadata: dict[
        str,
        dict[str, str],
    ],
) -> pd.DataFrame:
    """
    Build B.3 dominant-category summary for all categorical features.
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
            analyze_dominant_category(
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
            by="dominant_category_percent",
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
    Save B.3 dominant-category summary.
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
    Print B.3 categorical dominant-category results.
    """

    print("=" * 170)

    print(
        "EDA LEVEL 1 — B.3 CATEGORICAL DOMINANT CATEGORY"
    )

    print("=" * 170)

    print(
        f"Categorical features investigated: "
        f"{len(results)}"
    )

    print(
        "Review thresholds:"
    )

    print(
        f"  LOW:    "
        f"{DOMINANCE_LOW_THRESHOLD_PERCENT:.2f}% "
        f"to {DOMINANCE_MEDIUM_THRESHOLD_PERCENT:.2f}%"
    )

    print(
        f"  MEDIUM: >"
        f"{DOMINANCE_MEDIUM_THRESHOLD_PERCENT:.2f}% "
        f"to {DOMINANCE_HIGH_THRESHOLD_PERCENT:.2f}%"
    )

    print(
        f"  HIGH:   >"
        f"{DOMINANCE_HIGH_THRESHOLD_PERCENT:.2f}%"
    )

    if not results.empty:

        print(
            "\nREVIEW PRIORITY COUNTS"
        )

        print(
            results[
                "review_priority"
            ]
            .value_counts()
            .to_string()
        )

    print("\n" + "=" * 170)

    print(
        "DOMINANT CATEGORY SUMMARY — "
        "SORTED BY DOMINANT CATEGORY PERCENT"
    )

    print("=" * 170)

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

    print("\n" + "=" * 170)

    print(
        "FEATURES SELECTED FOR MEDIUM / HIGH REVIEW"
    )

    print("=" * 170)

    if not results.empty:

        review_features = (
            results[
                results[
                    "review_priority"
                ].isin(
                    [
                        "MEDIUM",
                        "HIGH",
                    ]
                )
            ]
        )

        if review_features.empty:

            print(
                "No medium/high priority "
                "dominant-category features."
            )

        else:

            print(
                review_features[
                    [
                        "feature",
                        "review_priority",
                        "review_reason",
                        "dictionary_data_type",
                        "unique_categories",
                        "dominant_category",
                        "dominant_category_percent",
                        "missing_percent",
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

    print("\n" + "=" * 170)

    print(
        "ARTIFACT SAVED"
    )

    print("=" * 170)

    print(
        SUMMARY_OUTPUT_PATH
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def run_b3_categorical_dominant_category() -> pd.DataFrame:
    """
    Run EDA Level 1 B.3.
    """

    dataframe = (
        load_training_data()
    )

    feature_metadata = (
        load_feature_dictionary()
    )

    results = (
        build_dominant_category_summary(
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
    run_b3_categorical_dominant_category()