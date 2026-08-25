"""EDA Level 1 - B.1: Categorical feature cardinality analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype

from icu_mortality.data import (
    TARGET_COLUMN,
    load_training_data,
)


REPORTS_TABLES_DIR = Path("reports") / "tables"

SUMMARY_OUTPUT_PATH = (
    REPORTS_TABLES_DIR
    / "eda_level1_b1_categorical_cardinality_summary.csv"
)


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

    These thresholds are working EDA thresholds rather than
    automatic preprocessing rules.
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
) -> dict[str, object]:
    """Analyze cardinality of one categorical feature."""

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

    observed_count = len(
        observed
    )

    unique_categories = int(
        observed.nunique()
    )

    cardinality_ratio_percent = (
        unique_categories
        / observed_count
        * 100
        if observed_count > 0
        else float("nan")
    )

    cardinality_level = (
        classify_cardinality(
            unique_categories
        )
    )

    return {
        "feature": feature,
        "missing_percent": (
            missing_percent
        ),
        "observed_count": (
            observed_count
        ),
        "unique_categories": (
            unique_categories
        ),
        "cardinality_ratio_percent": (
            cardinality_ratio_percent
        ),
        "cardinality_level": (
            cardinality_level
        ),
    }


# ---------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------

def build_cardinality_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Build B.1 cardinality summary for categorical features."""

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
    """Save B.1 categorical cardinality summary."""

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
    """Print B.1 categorical cardinality results."""

    print("=" * 130)
    print(
        "EDA LEVEL 1 — B.1 CATEGORICAL FEATURE CARDINALITY"
    )
    print("=" * 130)

    print(
        f"Categorical features investigated: "
        f"{len(results)}"
    )

    print("\n" + "=" * 130)
    print(
        "CARDINALITY SUMMARY"
    )
    print("=" * 130)

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

    print("\n" + "=" * 130)
    print(
        "CARDINALITY LEVEL COUNTS"
    )
    print("=" * 130)

    if not results.empty:

        print(
            results[
                "cardinality_level"
            ]
            .value_counts()
            .to_string()
        )

    print("\n" + "=" * 130)
    print(
        "HIGH-CARDINALITY FEATURES"
    )
    print("=" * 130)

    if not results.empty:

        high_cardinality = (
            results[
                results[
                    "cardinality_level"
                ] == "high"
            ]
        )

        if high_cardinality.empty:

            print(
                "No high-cardinality categorical features detected."
            )

        else:

            print(
                high_cardinality[
                    [
                        "feature",
                        "unique_categories",
                        "cardinality_ratio_percent",
                        "missing_percent",
                    ]
                ]
                .to_string(
                    index=False,
                    float_format=lambda value: (
                        f"{value:.2f}"
                    ),
                )
            )

    print("\n" + "=" * 130)
    print(
        "ARTIFACT SAVED"
    )
    print("=" * 130)

    print(
        SUMMARY_OUTPUT_PATH
    )


def run_b1_categorical_cardinality() -> pd.DataFrame:
    """Run EDA Level 1 B.1."""

    dataframe = (
        load_training_data()
    )

    results = (
        build_cardinality_summary(
            dataframe
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