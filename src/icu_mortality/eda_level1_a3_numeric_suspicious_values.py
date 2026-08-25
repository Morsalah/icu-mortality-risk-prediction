"""EDA Level 1 - A.3: Numeric suspicious values / range analysis."""

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
    / "eda_level1_a3_numeric_suspicious_values_summary.csv"
)

DICTIONARY_PATH = (
    Path("data")
    / "reference"
    / "WiDS Datathon 2020 Dictionary.csv"
)


KNOWN_IDENTIFIER_COLUMNS = {
    "encounter_id",
    "patient_id",
}


LOW_UNIQUE_VALUES_THRESHOLD = 10


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
            "data_type": ...,
            "unit_of_measure": ...
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
        "Unit of Measure",
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

        unit_of_measure = (
            ""
            if pd.isna(
                row["Unit of Measure"]
            )
            else str(
                row["Unit of Measure"]
            )
        )

        feature_metadata[
            str(feature)
        ] = {
            "description": description,
            "data_type": data_type,
            "unit_of_measure": unit_of_measure,
        }

    return feature_metadata


# ---------------------------------------------------------------------
# Numeric feature selection
# ---------------------------------------------------------------------

def get_numeric_features(
    dataframe: pd.DataFrame,
) -> list[str]:
    """
    Return numeric predictor features for A.3 analysis.

    Target and known identifiers are excluded.
    """

    features: list[str] = []

    for feature in dataframe.columns:

        if feature == TARGET_COLUMN:
            continue

        if feature in KNOWN_IDENTIFIER_COLUMNS:
            continue

        if not is_numeric_dtype(
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

def assign_review_reason(
    unique_values: int,
    negative_values_percent: float,
    zero_values_percent: float,
) -> str:
    """
    Determine why a numeric feature requires A.3 review.

    A review flag does not mean the values are invalid.
    It only indicates that the feature deserves inspection
    using its clinical/logical meaning and data dictionary.
    """

    reasons: list[str] = []

    if unique_values < LOW_UNIQUE_VALUES_THRESHOLD:
        reasons.append(
            "LOW_UNIQUE_VALUES"
        )

    if (
        not pd.isna(
            negative_values_percent
        )
        and
        negative_values_percent > 0
    ):
        reasons.append(
            "HAS_NEGATIVE_VALUES"
        )

    if (
        not pd.isna(
            zero_values_percent
        )
        and
        zero_values_percent > 0
    ):
        reasons.append(
            "HAS_ZERO_VALUES"
        )

    return (
        " | ".join(
            reasons
        )
        if reasons
        else ""
    )


# ---------------------------------------------------------------------
# Suspicious-value analysis
# ---------------------------------------------------------------------

def analyze_numeric_range(
    dataframe: pd.DataFrame,
    feature: str,
    feature_metadata: dict[
        str,
        dict[str, str],
    ],
) -> dict[str, object]:
    """
    Analyze the observed range of one numeric feature.

    The function identifies values that may deserve manual review,
    but it does not automatically label them as medically invalid.
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

    unique_values = int(
        observed.nunique()
    )

    if observed.empty:

        return {
            "feature": feature,
            "review_reason": "NO_OBSERVED_VALUES",
            "dictionary_data_type": "",
            "unit_of_measure": "",
            "feature_description": "",
            "missing_percent": missing_percent,
            "unique_values": unique_values,
            "min_value": float("nan"),
            "max_value": float("nan"),
            "negative_values_percent": float("nan"),
            "zero_values_percent": float("nan"),
        }

    min_value = float(
        observed.min()
    )

    max_value = float(
        observed.max()
    )

    negative_values_percent = (
        (observed < 0)
        .mean()
        * 100
    )

    zero_values_percent = (
        (observed == 0)
        .mean()
        * 100
    )

    review_reason = (
        assign_review_reason(
            unique_values=unique_values,
            negative_values_percent=(
                negative_values_percent
            ),
            zero_values_percent=(
                zero_values_percent
            ),
        )
    )

    dictionary_data_type = ""
    unit_of_measure = ""
    feature_description = ""

    # Only display dictionary metadata for features
    # that require manual review.
    if review_reason:

        metadata = (
            feature_metadata.get(
                feature
            )
        )

        if metadata is None:

            dictionary_data_type = (
                "Data type not found in dictionary"
            )

            unit_of_measure = (
                "Unit not found in dictionary"
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

            unit_of_measure = (
                metadata[
                    "unit_of_measure"
                ]
            )

            feature_description = (
                metadata[
                    "description"
                ]
            )

    return {
        "feature": feature,
        "review_reason": (
            review_reason
        ),
        "dictionary_data_type": (
            dictionary_data_type
        ),
        "unit_of_measure": (
            unit_of_measure
        ),
        "feature_description": (
            feature_description
        ),
        "missing_percent": (
            missing_percent
        ),
        "unique_values": (
            unique_values
        ),
        "min_value": (
            min_value
        ),
        "max_value": (
            max_value
        ),
        "negative_values_percent": (
            negative_values_percent
        ),
        "zero_values_percent": (
            zero_values_percent
        ),
    }


# ---------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------

def build_range_summary(
    dataframe: pd.DataFrame,
    feature_metadata: dict[
        str,
        dict[str, str],
    ],
) -> pd.DataFrame:
    """
    Build A.3 range summary for all numeric predictor features.
    """

    features = (
        get_numeric_features(
            dataframe
        )
    )

    rows: list[
        dict[str, object]
    ] = []

    for feature in features:

        rows.append(
            analyze_numeric_range(
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

    # Put features that require review first.
    results = (
        results
        .assign(
            _requires_review=(
                results[
                    "review_reason"
                ] != ""
            )
        )
        .sort_values(
            by=[
                "_requires_review",
                "negative_values_percent",
                "zero_values_percent",
            ],
            ascending=[
                False,
                False,
                False,
            ],
            na_position="last",
        )
        .drop(
            columns=[
                "_requires_review"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return results


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

def save_summary(
    results: pd.DataFrame,
) -> None:
    """
    Save A.3 suspicious-value / range summary.
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
    Print A.3 suspicious-value / range results.
    """

    print("=" * 190)

    print(
        "EDA LEVEL 1 — A.3 NUMERIC SUSPICIOUS VALUES / RANGE"
    )

    print("=" * 190)

    print(
        f"Numeric features investigated: "
        f"{len(results)}"
    )

    if not results.empty:

        review_features = (
            results[
                results[
                    "review_reason"
                ] != ""
            ]
        )

        print(
            f"Features requiring A.3 review: "
            f"{len(review_features)}"
        )

    print("\n" + "=" * 190)

    print(
        "SUSPICIOUS VALUE / RANGE SUMMARY"
    )

    print("=" * 190)

    if results.empty:

        print(
            "No numeric features were found."
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

    print("\n" + "=" * 190)

    print(
        "FEATURES SELECTED FOR MANUAL REVIEW"
    )

    print("=" * 190)

    if not results.empty:

        review_features = (
            results[
                results[
                    "review_reason"
                ] != ""
            ]
        )

        if review_features.empty:

            print(
                "No features require manual A.3 review."
            )

        else:

            print(
                review_features[
                    [
                        "feature",
                        "review_reason",
                        "dictionary_data_type",
                        "unit_of_measure",
                        "unique_values",
                        "min_value",
                        "max_value",
                        "negative_values_percent",
                        "zero_values_percent",
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

    print("\n" + "=" * 190)

    print(
        "ARTIFACT SAVED"
    )

    print("=" * 190)

    print(
        SUMMARY_OUTPUT_PATH
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def run_a3_numeric_suspicious_values() -> pd.DataFrame:
    """
    Run EDA Level 1 A.3.
    """

    dataframe = (
        load_training_data()
    )

    feature_metadata = (
        load_feature_dictionary()
    )

    results = (
        build_range_summary(
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
    run_a3_numeric_suspicious_values()