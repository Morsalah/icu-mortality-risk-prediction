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


KNOWN_IDENTIFIER_COLUMNS = {
    "encounter_id",
    "patient_id",
}


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


def analyze_numeric_outliers(
    dataframe: pd.DataFrame,
    feature: str,
) -> dict[str, object]:
    """Analyze IQR-based outliers for one numeric feature."""

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

    if observed.empty:
        return {
            "feature": feature,
            "missing_percent": missing_percent,
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

    return {
        "feature": feature,
        "missing_percent": missing_percent,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outlier_count": outlier_count,
        "outlier_percent_observed": (
            outlier_percent_observed
        ),
    }


def build_outlier_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Build A.2 outlier summary for all numeric features."""

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
        )
        .reset_index(
            drop=True
        )
    )


def save_boxplot(
    dataframe: pd.DataFrame,
    feature: str,
) -> None:
    """Save a box plot for one numeric feature."""

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
        figsize=(8, 4)
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
    """Save box plots for all numeric predictor features."""

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


def save_summary(
    results: pd.DataFrame,
) -> None:
    """Save A.2 outlier summary."""

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
    """Print A.2 numeric outlier results."""

    print("=" * 130)
    print(
        "EDA LEVEL 1 — A.2 NUMERIC FEATURE OUTLIERS"
    )
    print("=" * 130)

    print(
        f"Numeric features investigated: "
        f"{len(results)}"
    )

    print("\n" + "=" * 130)
    print(
        "OUTLIER SUMMARY"
    )
    print("=" * 130)

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

    print("\n" + "=" * 130)
    print(
        "TOP 20 FEATURES BY OUTLIER PERCENT"
    )
    print("=" * 130)

    if results.empty:
        print(
            "No results."
        )
    else:
        print(
            results.head(
                20
            )[
                [
                    "feature",
                    "outlier_count",
                    "outlier_percent_observed",
                    "lower_bound",
                    "upper_bound",
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
        "ARTIFACTS SAVED"
    )
    print("=" * 130)

    print(
        SUMMARY_OUTPUT_PATH
    )

    print(
        REPORTS_FIGURES_DIR
    )


def run_a2_numeric_outliers() -> pd.DataFrame:
    """Run EDA Level 1 A.2."""

    dataframe = (
        load_training_data()
    )

    results = (
        build_outlier_summary(
            dataframe
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