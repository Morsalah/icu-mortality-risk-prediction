"""EDA Level 1 - A.1: Numeric feature distribution analysis."""

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
    / "numeric_distribution"
)

SUMMARY_OUTPUT_PATH = (
    REPORTS_TABLES_DIR
    / "eda_level1_a1_numeric_distribution_summary.csv"
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

RELATIVE_DIFFERENCE_NOTICEABLE = 10.0
RELATIVE_DIFFERENCE_HIGH = 20.0

SKEWNESS_NOTICEABLE = 0.5
SKEWNESS_HIGH = 1.0
SKEWNESS_VERY_HIGH = 2.0


# ---------------------------------------------------------------------
# Feature dictionary
# ---------------------------------------------------------------------

def load_feature_dictionary() -> dict[str, dict[str, str]]:
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

        feature = row["Variable Name"]

        if pd.isna(feature):
            continue

        description = (
            ""
            if pd.isna(row["Description"])
            else str(row["Description"])
        )

        data_type = (
            ""
            if pd.isna(row["Data Type"])
            else str(row["Data Type"])
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
    Return numeric predictor features.

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

        features.append(feature)

    return features


# ---------------------------------------------------------------------
# Relative difference
# ---------------------------------------------------------------------

def calculate_relative_difference(
    mean: float,
    median: float,
) -> float:
    """
    Calculate relative difference between mean and median.

        |mean - median|
        ------------------------- * 100
        (|mean| + |median|) / 2
    """

    if pd.isna(mean) or pd.isna(median):
        return float("nan")

    denominator = (
        abs(mean)
        + abs(median)
    ) / 2

    if denominator == 0:
        return 0.0

    return (
        abs(mean - median)
        / denominator
        * 100
    )


# ---------------------------------------------------------------------
# Review / triage
# ---------------------------------------------------------------------

def calculate_review_information(
    unique_values: int,
    relative_difference_percent: float,
    skewness: float,
) -> tuple[int, str, str]:
    """
    Calculate A.1 review score, priority and reason.

    Low-cardinality numeric features are treated separately because
    skewness and mean/median behavior may not have the same
    interpretation as for continuous numeric variables.
    """

    score = 0
    reasons: list[str] = []

    # --------------------------------------------------------------
    # Low-cardinality numeric feature
    # --------------------------------------------------------------

    if unique_values < LOW_UNIQUE_VALUES_THRESHOLD:

        reasons.append(
            "LOW_UNIQUE_VALUES"
        )

        # We want to review its real meaning / data type,
        # rather than rank it using continuous-distribution metrics.
        score = 1

    else:

        # ----------------------------------------------------------
        # Relative difference
        # ----------------------------------------------------------

        if not pd.isna(
            relative_difference_percent
        ):

            if (
                relative_difference_percent
                > RELATIVE_DIFFERENCE_HIGH
            ):

                score += 2

                reasons.append(
                    "LARGE_MEAN_MEDIAN_DIFFERENCE"
                )

            elif (
                relative_difference_percent
                >= RELATIVE_DIFFERENCE_NOTICEABLE
            ):

                score += 1

                reasons.append(
                    "NOTICEABLE_MEAN_MEDIAN_DIFFERENCE"
                )

        # ----------------------------------------------------------
        # Skewness
        # ----------------------------------------------------------

        if not pd.isna(skewness):

            absolute_skewness = abs(
                skewness
            )

            if (
                absolute_skewness
                > SKEWNESS_VERY_HIGH
            ):

                score += 3

                reasons.append(
                    "VERY_HIGH_SKEWNESS"
                )

            elif (
                absolute_skewness
                > SKEWNESS_HIGH
            ):

                score += 2

                reasons.append(
                    "HIGH_SKEWNESS"
                )

            elif (
                absolute_skewness
                >= SKEWNESS_NOTICEABLE
            ):

                score += 1

                reasons.append(
                    "NOTICEABLE_SKEWNESS"
                )

    # --------------------------------------------------------------
    # Priority
    # --------------------------------------------------------------

    if score == 0:

        priority = "NO_REVIEW"

    elif score == 1:

        priority = "LOW"

    elif score <= 3:

        priority = "MEDIUM"

    else:

        priority = "HIGH"

    review_reason = (
        " | ".join(reasons)
        if reasons
        else ""
    )

    return (
        score,
        priority,
        review_reason,
    )


# ---------------------------------------------------------------------
# Feature analysis
# ---------------------------------------------------------------------

def analyze_numeric_distribution(
    dataframe: pd.DataFrame,
    feature: str,
    feature_metadata: dict[
        str,
        dict[str, str],
    ],
) -> dict[str, object]:

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
            "review_score": 0,
            "review_priority": "NO_REVIEW",
            "review_reason": "",
            "dictionary_data_type": "",
            "feature_description": "",
            "missing_percent": missing_percent,
            "unique_values": unique_values,
            "mean": float("nan"),
            "median": float("nan"),
            "relative_difference_percent": float("nan"),
            "skewness": float("nan"),
        }

    mean = float(
        observed.mean()
    )

    median = float(
        observed.median()
    )

    skewness = float(
        observed.skew()
    )

    relative_difference = (
        calculate_relative_difference(
            mean=mean,
            median=median,
        )
    )

    (
        review_score,
        review_priority,
        review_reason,
    ) = calculate_review_information(
        unique_values=unique_values,
        relative_difference_percent=(
            relative_difference
        ),
        skewness=skewness,
    )

    feature_description = ""
    dictionary_data_type = ""

    if review_score > 0:

        metadata = (
            feature_metadata.get(feature)
        )

        if metadata is None:

            feature_description = (
                "Description not found in dictionary"
            )

            dictionary_data_type = (
                "Data type not found in dictionary"
            )

        else:

            feature_description = (
                metadata["description"]
            )

            dictionary_data_type = (
                metadata["data_type"]
            )

    return {
        "feature": feature,
        "review_score": review_score,
        "review_priority": review_priority,
        "review_reason": review_reason,
        "dictionary_data_type": (
            dictionary_data_type
        ),
        "feature_description": (
            feature_description
        ),
        "missing_percent": missing_percent,
        "unique_values": unique_values,
        "mean": mean,
        "median": median,
        "relative_difference_percent": (
            relative_difference
        ),
        "skewness": skewness,
    }


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def build_distribution_summary(
    dataframe: pd.DataFrame,
    feature_metadata: dict[
        str,
        dict[str, str],
    ],
) -> pd.DataFrame:

    features = (
        get_numeric_features(
            dataframe
        )
    )

    rows = []

    for feature in features:

        rows.append(
            analyze_numeric_distribution(
                dataframe=dataframe,
                feature=feature,
                feature_metadata=feature_metadata,
            )
        )

    results = pd.DataFrame(
        rows
    )

    if results.empty:
        return results

    # Temporary value used only for secondary sorting.
    results["_abs_skewness"] = (
        results[
            "skewness"
        ].abs()
    )

    results = (
        results
        .sort_values(
            by=[
                "review_score",
                "_abs_skewness",
            ],
            ascending=[
                False,
                False,
            ],
            na_position="last",
        )
        .drop(
            columns=[
                "_abs_skewness"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return results


# ---------------------------------------------------------------------
# Histograms
# ---------------------------------------------------------------------

def save_histogram(
    dataframe: pd.DataFrame,
    feature: str,
) -> None:

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
        figsize=(8, 5)
    )

    axis.hist(
        observed,
        bins=30,
    )

    axis.set_title(
        f"Distribution of {feature}"
    )

    axis.set_xlabel(
        feature
    )

    axis.set_ylabel(
        "Frequency"
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


def save_all_histograms(
    dataframe: pd.DataFrame,
) -> None:

    features = (
        get_numeric_features(
            dataframe
        )
    )

    for feature in features:

        save_histogram(
            dataframe=dataframe,
            feature=feature,
        )


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

def save_summary(
    results: pd.DataFrame,
) -> None:

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

    print("=" * 190)

    print(
        "EDA LEVEL 1 — A.1 NUMERIC FEATURE DISTRIBUTION"
    )

    print("=" * 190)

    print(
        f"Numeric features investigated: "
        f"{len(results)}"
    )

    if not results.empty:

        print("\nREVIEW PRIORITY COUNTS")

        print(
            results[
                "review_priority"
            ]
            .value_counts()
            .to_string()
        )

        print(
            "\nTOP FEATURES FOR MANUAL REVIEW"
        )

        review_features = (
            results[
                results[
                    "review_score"
                ] > 0
            ]
        )

        print(
            review_features[
                [
                    "feature",
                    "review_score",
                    "review_priority",
                    "review_reason",
                    "dictionary_data_type",
                    "unique_values",
                    "relative_difference_percent",
                    "skewness",
                    "feature_description",
                ]
            ]
            .head(20)
            .to_string(
                index=False,
                float_format=lambda value: (
                    f"{value:.2f}"
                ),
            )
        )

    print("\nARTIFACTS SAVED")

    print(
        SUMMARY_OUTPUT_PATH
    )

    print(
        REPORTS_FIGURES_DIR
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def run_a1_numeric_distribution() -> pd.DataFrame:

    dataframe = (
        load_training_data()
    )

    feature_metadata = (
        load_feature_dictionary()
    )

    results = (
        build_distribution_summary(
            dataframe=dataframe,
            feature_metadata=feature_metadata,
        )
    )

    save_summary(
        results
    )

    save_all_histograms(
        dataframe
    )

    print_results(
        results
    )

    return results


if __name__ == "__main__":
    run_a1_numeric_distribution()