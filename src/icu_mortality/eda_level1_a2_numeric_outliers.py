"""EDA Level 1 - A.2: Numeric feature outlier analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
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

REPORTS_FIGURES_DIR = (
    Path("reports")
    / "figures"
    / "eda_level1"
    / "numeric_outliers"
)

SUMMARY_OUTPUT_PATH = (
    REPORTS_TABLES_DIR
    / "eda_level1_a2_numeric_outliers_summary.csv"
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


# ---------------------------------------------------------------------
# Review thresholds
# ---------------------------------------------------------------------

LOW_UNIQUE_VALUES_THRESHOLD = 10

OUTLIER_LOW_THRESHOLD_PERCENT = 1.0

OUTLIER_HIGH_THRESHOLD_PERCENT = 5.0


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
# Numeric feature selection
# ---------------------------------------------------------------------

def get_numeric_features(
    dataframe: pd.DataFrame,
) -> list[str]:
    """
    Return numeric predictor features for A.2 outlier analysis.

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
# IQR calculation
# ---------------------------------------------------------------------

def calculate_iqr_bounds(
    observed: pd.Series,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
]:
    """
    Calculate Q1, Q3, IQR and IQR-based outlier bounds.

    Outlier rule:

        value < Q1 - 1.5 * IQR

        OR

        value > Q3 + 1.5 * IQR
    """

    q1 = observed.quantile(
        0.25
    )

    q3 = observed.quantile(
        0.75
    )

    iqr = (
        q3 - q1
    )

    lower_bound = (
        q1
        - 1.5 * iqr
    )

    upper_bound = (
        q3
        + 1.5 * iqr
    )

    return (
        q1,
        q3,
        iqr,
        lower_bound,
        upper_bound,
    )


# ---------------------------------------------------------------------
# Review policy
# ---------------------------------------------------------------------

def assign_review_information(
    unique_values: int,
    outlier_percent_observed: float,
) -> tuple[str, str]:
    """
    Assign A.2 review priority and reason.

    Priority is primarily based on the percentage of observed values
    classified as IQR outliers.

    Low-cardinality numeric features are explicitly flagged because
    IQR-based outlier interpretation may be less meaningful for
    binary, ordinal or discrete variables.
    """

    reasons: list[str] = []

    if unique_values < LOW_UNIQUE_VALUES_THRESHOLD:
        reasons.append(
            "LOW_UNIQUE_VALUES"
        )

    if pd.isna(
        outlier_percent_observed
    ):
        return (
            "NO_REVIEW",
            "NO_OBSERVED_VALUES",
        )

    if outlier_percent_observed == 0:

        priority = (
            "NO_REVIEW"
        )

    elif (
        outlier_percent_observed
        <= OUTLIER_LOW_THRESHOLD_PERCENT
    ):

        priority = (
            "LOW"
        )

        reasons.append(
            "LOW_OUTLIER_PERCENT"
        )

    elif (
        outlier_percent_observed
        <= OUTLIER_HIGH_THRESHOLD_PERCENT
    ):

        priority = (
            "MEDIUM"
        )

        reasons.append(
            "MODERATE_OUTLIER_PERCENT"
        )

    else:

        priority = (
            "HIGH"
        )

        reasons.append(
            "HIGH_OUTLIER_PERCENT"
        )

    review_reason = (
        " | ".join(
            reasons
        )
        if reasons
        else ""
    )

    return (
        priority,
        review_reason,
    )


# ---------------------------------------------------------------------
# Feature analysis
# ---------------------------------------------------------------------

def analyze_numeric_outliers(
    dataframe: pd.DataFrame,
    feature: str,
    feature_metadata: dict[
        str,
        dict[str, str],
    ],
) -> dict[str, object]:
    """
    Analyze IQR-based outliers for one numeric feature.
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
            "review_priority": "NO_REVIEW",
            "review_reason": "NO_OBSERVED_VALUES",
            "dictionary_data_type": "",
            "feature_description": "",
            "missing_percent": missing_percent,
            "unique_values": unique_values,
            "q1": float("nan"),
            "q3": float("nan"),
            "iqr": float("nan"),
            "lower_bound": float("nan"),
            "upper_bound": float("nan"),
            "outlier_count": 0,
            "outlier_percent_observed": float("nan"),
        }

    (
        q1,
        q3,
        iqr,
        lower_bound,
        upper_bound,
    ) = calculate_iqr_bounds(
        observed
    )

    outlier_mask = (
        (observed < lower_bound)
        |
        (observed > upper_bound)
    )

    outlier_count = int(
        outlier_mask.sum()
    )

    outlier_percent_observed = (
        outlier_count
        / len(observed)
        * 100
    )

    (
        review_priority,
        review_reason,
    ) = assign_review_information(
        unique_values=unique_values,
        outlier_percent_observed=(
            outlier_percent_observed
        ),
    )

    dictionary_data_type = ""
    feature_description = ""

    # Only load descriptive metadata into the table
    # for features worth manual investigation.
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
        "unique_values": (
            unique_values
        ),
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_bound": (
            lower_bound
        ),
        "upper_bound": (
            upper_bound
        ),
        "outlier_count": (
            outlier_count
        ),
        "outlier_percent_observed": (
            outlier_percent_observed
        ),
    }


# ---------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------

def build_outlier_summary(
    dataframe: pd.DataFrame,
    feature_metadata: dict[
        str,
        dict[str, str],
    ],
) -> pd.DataFrame:
    """
    Build A.2 outlier summary for all numeric features.

    Results are sorted by outlier percentage from highest to lowest.
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
            analyze_numeric_outliers(
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
            by="outlier_percent_observed",
            ascending=False,
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )


# ---------------------------------------------------------------------
# Box plots
# ---------------------------------------------------------------------

def save_boxplot(
    dataframe: pd.DataFrame,
    feature: str,
) -> None:
    """
    Save a box plot for one numeric feature.
    """

    observed = (
        dataframe[
            feature
        ]
        .dropna()
    )

    if observed.empty:
        return

    REPORTS_FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(
            8,
            4,
        )
    )

    axis.boxplot(
        observed,
        vert=False,
    )

    axis.set_title(
        f"Box Plot of {feature}"
    )

    axis.set_xlabel(
        feature
    )

    axis.set_yticks(
        []
    )

    figure.tight_layout()

    output_path = (
        REPORTS_FIGURES_DIR
        / f"{feature}.png"
    )

    figure.savefig(
        output_path,
        dpi=150,
    )

    plt.close(
        figure
    )


def save_all_boxplots(
    dataframe: pd.DataFrame,
) -> None:
    """
    Save box plots for all numeric predictor features.
    """

    features = (
        get_numeric_features(
            dataframe
        )
    )

    for feature in features:

        save_boxplot(
            dataframe=dataframe,
            feature=feature,
        )


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

def save_summary(
    results: pd.DataFrame,
) -> None:
    """
    Save A.2 outlier summary.
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
    Print A.2 numeric outlier results.
    """

    print("=" * 180)

    print(
        "EDA LEVEL 1 — A.2 NUMERIC FEATURE OUTLIERS"
    )

    print("=" * 180)

    print(
        f"Numeric features investigated: "
        f"{len(results)}"
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

    print("\n" + "=" * 180)

    print(
        "OUTLIER SUMMARY — SORTED BY OUTLIER PERCENT"
    )

    print("=" * 180)

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

    print("\n" + "=" * 180)

    print(
        "TOP 20 FEATURES FOR OUTLIER REVIEW"
    )

    print("=" * 180)

    if not results.empty:

        review_features = (
            results[
                results[
                    "review_priority"
                ].isin(
                    [
                        "HIGH",
                        "MEDIUM",
                    ]
                )
            ]
        )

        if review_features.empty:

            print(
                "No medium/high priority features."
            )

        else:

            print(
                review_features[
                    [
                        "feature",
                        "review_priority",
                        "review_reason",
                        "dictionary_data_type",
                        "unique_values",
                        "outlier_count",
                        "outlier_percent_observed",
                        "lower_bound",
                        "upper_bound",
                        "feature_description",
                    ]
                ]
                .head(
                    20
                )
                .to_string(
                    index=False,
                    float_format=lambda value: (
                        f"{value:.2f}"
                    ),
                )
            )

    print("\n" + "=" * 180)

    print(
        "ARTIFACTS SAVED"
    )

    print("=" * 180)

    print(
        SUMMARY_OUTPUT_PATH
    )

    print(
        REPORTS_FIGURES_DIR
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def run_a2_numeric_outliers() -> pd.DataFrame:
    """
    Run EDA Level 1 A.2.
    """

    dataframe = (
        load_training_data()
    )

    feature_metadata = (
        load_feature_dictionary()
    )

    results = (
        build_outlier_summary(
            dataframe=dataframe,
            feature_metadata=(
                feature_metadata
            ),
        )
    )

    save_summary(
        results
    )

    save_all_boxplots(
        dataframe
    )

    print_results(
        results
    )

    return results


if __name__ == "__main__":
    run_a2_numeric_outliers()