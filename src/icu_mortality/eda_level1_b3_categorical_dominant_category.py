"""EDA Level 1 - B.3: Categorical dominant-category analysis."""

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
    / "eda_level1_b3_categorical_dominant_category_summary.csv"
)


# Working EDA threshold.
# This is used only for flagging, not for automatic feature removal.
DOMINANT_CATEGORY_THRESHOLD_PERCENT = 90.0


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
# Dominant-category analysis
# ---------------------------------------------------------------------

def analyze_dominant_category(
    dataframe: pd.DataFrame,
    feature: str,
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

    total_count = len(
        series
    )

    observed_count = len(
        observed
    )

    missing_count = int(
        series.isna().sum()
    )

    missing_percent = (
        missing_count
        / total_count
        * 100
        if total_count > 0
        else float("nan")
    )

    unique_categories = int(
        observed.nunique()
    )

    if observed.empty:

        return {
            "feature": feature,
            "observed_count": observed_count,
            "missing_percent": missing_percent,
            "unique_categories": unique_categories,
            "dominant_category": None,
            "dominant_category_count": 0,
            "dominant_category_percent": float("nan"),
            "dominant_category_flag": False,
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
        / observed_count
        * 100
    )

    dominant_category_flag = bool(
        dominant_category_percent
        >= DOMINANT_CATEGORY_THRESHOLD_PERCENT
    )

    return {
        "feature": feature,
        "observed_count": observed_count,
        "missing_percent": missing_percent,
        "unique_categories": unique_categories,
        "dominant_category": dominant_category,
        "dominant_category_count": (
            dominant_category_count
        ),
        "dominant_category_percent": (
            dominant_category_percent
        ),
        "dominant_category_flag": (
            dominant_category_flag
        ),
    }


# ---------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------

def build_dominant_category_summary(
    dataframe: pd.DataFrame,
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
    """Save B.3 dominant-category summary."""

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
    """Print B.3 categorical dominant-category results."""

    print("=" * 140)
    print(
        "EDA LEVEL 1 — B.3 CATEGORICAL DOMINANT CATEGORY"
    )
    print("=" * 140)

    print(
        f"Categorical features investigated: "
        f"{len(results)}"
    )

    print(
        f"Dominant-category flag threshold: "
        f"{DOMINANT_CATEGORY_THRESHOLD_PERCENT:.2f}%"
    )

    print("\n" + "=" * 140)
    print(
        "DOMINANT CATEGORY SUMMARY"
    )
    print("=" * 140)

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

    print("\n" + "=" * 140)
    print(
        "HIGHLY DOMINATED CATEGORICAL FEATURES"
    )
    print("=" * 140)

    if not results.empty:

        dominated_features = (
            results[
                results[
                    "dominant_category_flag"
                ]
            ]
        )

        if dominated_features.empty:

            print(
                "No categorical features exceed "
                "the dominant-category threshold."
            )

        else:

            print(
                dominated_features[
                    [
                        "feature",
                        "dominant_category",
                        "dominant_category_count",
                        "dominant_category_percent",
                    ]
                ]
                .to_string(
                    index=False,
                    float_format=lambda value: (
                        f"{value:.2f}"
                    ),
                )
            )

    print("\n" + "=" * 140)
    print(
        "ARTIFACT SAVED"
    )
    print("=" * 140)

    print(
        SUMMARY_OUTPUT_PATH
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def run_b3_categorical_dominant_category() -> pd.DataFrame:
    """Run EDA Level 1 B.3."""

    dataframe = (
        load_training_data()
    )

    results = (
        build_dominant_category_summary(
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
    run_b3_categorical_dominant_category()