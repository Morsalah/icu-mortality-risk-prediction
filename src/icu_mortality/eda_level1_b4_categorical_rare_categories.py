"""EDA Level 1 - B.4: Categorical rare-category analysis."""

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
    / "eda_level1_b4_categorical_rare_categories_summary.csv"
)

DETAIL_OUTPUT_PATH = (
    REPORTS_TABLES_DIR
    / "eda_level1_b4_categorical_rare_categories_detail.csv"
)


RARE_CATEGORY_THRESHOLD_PERCENT = 1.0


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
) -> tuple[
    dict[str, object],
    pd.DataFrame,
]:
    """
    Analyze rare categories for one categorical feature.

    Missing values are excluded from the rare-category calculation.

    A category is considered rare when its frequency among observed
    values is below RARE_CATEGORY_THRESHOLD_PERCENT.
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
        series.isna()
        .sum()
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

        summary = {
            "feature": feature,
            "observed_count": observed_count,
            "missing_percent": missing_percent,
            "unique_categories": unique_categories,
            "rare_categories_count": 0,
            "rare_observations_count": 0,
            "rare_observations_percent": float("nan"),
            "has_rare_categories": False,
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

    category_counts = (
        observed
        .value_counts()
    )

    category_percentages = (
        category_counts
        / observed_count
        * 100
    )

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

    rare_observations_count = int(
        rare_counts.sum()
    )

    rare_observations_percent = (
        rare_observations_count
        / observed_count
        * 100
    )

    has_rare_categories = bool(
        rare_categories_count > 0
    )

    summary = {
        "feature": feature,
        "observed_count": observed_count,
        "missing_percent": missing_percent,
        "unique_categories": unique_categories,
        "rare_categories_count": (
            rare_categories_count
        ),
        "rare_observations_count": (
            rare_observations_count
        ),
        "rare_observations_percent": (
            rare_observations_percent
        ),
        "has_rare_categories": (
            has_rare_categories
        ),
    }

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
        )

        summary_rows.append(
            summary
        )

        if not detail.empty:
            detail_tables.append(
                detail
            )

    summary_results = pd.DataFrame(
        summary_rows
    )

    if not summary_results.empty:

        summary_results = (
            summary_results
            .sort_values(
                by="rare_observations_percent",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

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
    """Save B.4 summary and detail tables."""

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
    """Print B.4 rare-category results."""

    print("=" * 140)
    print(
        "EDA LEVEL 1 — B.4 CATEGORICAL RARE CATEGORIES"
    )
    print("=" * 140)

    print(
        f"Categorical features investigated: "
        f"{len(summary_results)}"
    )

    print(
        f"Rare-category threshold: "
        f"< {RARE_CATEGORY_THRESHOLD_PERCENT:.2f}%"
    )

    print("\n" + "=" * 140)
    print(
        "RARE CATEGORY SUMMARY"
    )
    print("=" * 140)

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

    print("\n" + "=" * 140)
    print(
        "RARE CATEGORY DETAILS"
    )
    print("=" * 140)

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

    print("\n" + "=" * 140)
    print(
        "SUMMARY"
    )
    print("=" * 140)

    if not summary_results.empty:

        features_with_rare = int(
            summary_results[
                "has_rare_categories"
            ].sum()
        )

        total_rare_categories = int(
            summary_results[
                "rare_categories_count"
            ].sum()
        )

        print(
            f"Features containing rare categories: "
            f"{features_with_rare}"
        )

        print(
            f"Total rare categories detected: "
            f"{total_rare_categories}"
        )

    print("\n" + "=" * 140)
    print(
        "ARTIFACTS SAVED"
    )
    print("=" * 140)

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
    """Run EDA Level 1 B.4."""

    dataframe = (
        load_training_data()
    )

    (
        summary_results,
        detail_results,
    ) = build_rare_category_tables(
        dataframe
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