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


KNOWN_IDENTIFIER_COLUMNS = {
    "encounter_id",
    "patient_id",
}


def get_numeric_features(
    dataframe: pd.DataFrame,
) -> list[str]:
    """
    Return numeric predictor features for A.1 distribution analysis.

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


def analyze_numeric_distribution(
    dataframe: pd.DataFrame,
    feature: str,
) -> dict[str, object]:
    """Calculate distribution statistics for one numeric feature."""

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
            "missing_percent": missing_percent,
            "unique_values": unique_values,
            "mean": float("nan"),
            "median": float("nan"),
            "skewness": float("nan"),
        }

    return {
        "feature": feature,
        "missing_percent": missing_percent,
        "unique_values": unique_values,
        "mean": observed.mean(),
        "median": observed.median(),
        "skewness": observed.skew(),
    }


def build_distribution_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Build A.1 distribution summary for all numeric features."""

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
            analyze_numeric_distribution(
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
            by="feature"
        )
        .reset_index(
            drop=True
        )
    )


def save_histogram(
    dataframe: pd.DataFrame,
    feature: str,
) -> None:
    """Save histogram for one numeric feature."""

    observed = (
        dataframe[feature]
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
        edgecolor="black",
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
    """Save histograms for all numeric predictor features."""

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


def save_summary(
    results: pd.DataFrame,
) -> None:
    """Save A.1 distribution summary."""

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
    """Print A.1 numeric distribution results."""

    print("=" * 120)
    print(
        "EDA LEVEL 1 — A.1 NUMERIC FEATURE DISTRIBUTION"
    )
    print("=" * 120)

    print(
        f"Numeric features investigated: "
        f"{len(results)}"
    )

    print("\n" + "=" * 120)
    print(
        "DISTRIBUTION SUMMARY"
    )
    print("=" * 120)

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

    print("\n" + "=" * 120)
    print(
        "ARTIFACTS SAVED"
    )
    print("=" * 120)

    print(
        SUMMARY_OUTPUT_PATH
    )

    print(
        REPORTS_FIGURES_DIR
    )


def run_a1_numeric_distribution() -> pd.DataFrame:
    """Run EDA Level 1 A.1."""

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

    save_all_histograms(
        dataframe
    )

    print_results(
        results
    )

    return results


if __name__ == "__main__":
    run_a1_numeric_distribution()