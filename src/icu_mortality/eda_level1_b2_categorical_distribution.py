"""EDA Level 1 - B.2: Categorical feature distribution analysis."""

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
    / "categorical_distribution"
)

SUMMARY_OUTPUT_PATH = (
    REPORTS_TABLES_DIR
    / "eda_level1_b2_categorical_distribution_summary.csv"
)


# ---------------------------------------------------------------------
# Categorical feature selection
# ---------------------------------------------------------------------

def get_categorical_features(
    dataframe: pd.DataFrame,
) -> list[str]:
    """
    Return categorical predictor features for B.2 analysis.

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
# Category distribution analysis
# ---------------------------------------------------------------------

def analyze_category_distribution(
    dataframe: pd.DataFrame,
    feature: str,
) -> pd.DataFrame:
    """
    Calculate category counts and percentages for one feature.

    Missing values are represented explicitly as <MISSING>.

    Percentages are calculated relative to all rows so the
    visualization represents the raw dataset distribution,
    including missing values.
    """

    series = (
        dataframe[
            feature
        ]
        .astype(
            "object"
        )
        .where(
            dataframe[
                feature
            ].notna(),
            "<MISSING>",
        )
    )

    counts = (
        series
        .value_counts(
            dropna=False
        )
    )

    percentages = (
        counts
        / len(
            dataframe
        )
        * 100
    )

    result = pd.DataFrame(
        {
            "feature": feature,
            "category_value": (
                counts.index
            ),
            "count": (
                counts.values
            ),
            "percent": (
                percentages.values
            ),
        }
    )

    return result


# ---------------------------------------------------------------------
# Build summary table
# ---------------------------------------------------------------------

def build_distribution_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build category-distribution table for all categorical features.

    The output uses long format:
    one row per category per feature.
    """

    features = (
        get_categorical_features(
            dataframe
        )
    )

    tables: list[
        pd.DataFrame
    ] = []

    for feature in features:

        distribution = (
            analyze_category_distribution(
                dataframe=dataframe,
                feature=feature,
            )
        )

        tables.append(
            distribution
        )

    if not tables:

        return pd.DataFrame(
            columns=[
                "feature",
                "category_value",
                "count",
                "percent",
            ]
        )

    results = (
        pd.concat(
            tables,
            ignore_index=True,
        )
    )

    return results


# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------

def save_category_barplot(
    dataframe: pd.DataFrame,
    feature: str,
) -> None:
    """
    Save category distribution bar plot for one categorical feature.

    Categories are displayed using a horizontal bar plot and sorted
    from the least common category at the bottom to the most common
    category at the top.

    Percentage labels are displayed next to each bar.
    """

    distribution = (
        analyze_category_distribution(
            dataframe=dataframe,
            feature=feature,
        )
    )

    if distribution.empty:
        return

    REPORTS_FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Sort ascending for horizontal plotting so the most frequent
    # category appears at the top of the graph.
    plot_data = (
        distribution
        .sort_values(
            by="percent",
            ascending=True,
        )
        .reset_index(
            drop=True
        )
    )

    # Dynamic height:
    # features with many categories receive more vertical space.
    figure_height = max(
        4,
        0.45
        * len(
            plot_data
        ),
    )

    figure, axis = (
        plt.subplots(
            figsize=(
                9,
                figure_height,
            )
        )
    )

    bars = axis.barh(
        plot_data[
            "category_value"
        ].astype(
            str
        ),
        plot_data[
            "percent"
        ],
    )

    axis.set_title(
        f"Category Distribution — {feature}"
    )

    axis.set_xlabel(
        "Observations (%)"
    )

    axis.set_ylabel(
        "Category"
    )

    # -------------------------------------------------------------
    # Percentage labels
    # -------------------------------------------------------------

    for bar, percent in zip(
        bars,
        plot_data[
            "percent"
        ],
    ):

        axis.text(
            bar.get_width(),
            bar.get_y()
            + bar.get_height()
            / 2,
            f"  {percent:.2f}%",
            va="center",
            ha="left",
        )

    # -------------------------------------------------------------
    # Extra horizontal space for percentage labels
    # -------------------------------------------------------------

    max_percent = (
        plot_data[
            "percent"
        ].max()
    )

    if max_percent > 0:

        axis.set_xlim(
            0,
            max_percent
            * 1.15,
        )

    else:

        axis.set_xlim(
            0,
            1,
        )

    figure.tight_layout()

    output_path = (
        REPORTS_FIGURES_DIR
        / f"{feature}.png"
    )

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def save_all_category_barplots(
    dataframe: pd.DataFrame,
) -> None:
    """
    Save category-distribution plots for all categorical features.
    """

    features = (
        get_categorical_features(
            dataframe
        )
    )

    for feature in features:

        save_category_barplot(
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
    Save B.2 category-distribution summary.
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
    Print B.2 categorical-distribution results.
    """

    print(
        "="
        * 140
    )

    print(
        "EDA LEVEL 1 — "
        "B.2 CATEGORICAL FEATURE DISTRIBUTION"
    )

    print(
        "="
        * 140
    )

    if results.empty:

        print(
            "No categorical features were found."
        )

    else:

        feature_count = (
            results[
                "feature"
            ]
            .nunique()
        )

        print(
            f"Categorical features investigated: "
            f"{feature_count}"
        )

        print(
            f"Category rows generated: "
            f"{len(results)}"
        )

    print(
        "\n"
        + "="
        * 140
    )

    print(
        "CATEGORY DISTRIBUTIONS"
    )

    print(
        "="
        * 140
    )

    if not results.empty:

        print(
            results.to_string(
                index=False,
                float_format=lambda value: (
                    f"{value:.2f}"
                ),
            )
        )

    print(
        "\n"
        + "="
        * 140
    )

    print(
        "ARTIFACTS SAVED"
    )

    print(
        "="
        * 140
    )

    print(
        SUMMARY_OUTPUT_PATH
    )

    print(
        REPORTS_FIGURES_DIR
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def run_b2_categorical_distribution() -> pd.DataFrame:
    """
    Run EDA Level 1 B.2.
    """

    dataframe = (
        load_training_data()
    )

    results = (
        build_distribution_summary(
            dataframe
        )
    )

    save_summary(
        results
    )

    save_all_category_barplots(
        dataframe
    )

    print_results(
        results
    )

    return results


if __name__ == "__main__":
    run_b2_categorical_distribution()