"""EDA Level 1 - A.4: Numeric near-constant feature analysis."""

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
    / "eda_level1_a4_numeric_near_constant_summary.csv"
)


KNOWN_IDENTIFIER_COLUMNS = {
    "encounter_id",
    "patient_id",
}


NEAR_CONSTANT_THRESHOLD_PERCENT = 95.0


# ---------------------------------------------------------------------
# Numeric feature selection
# ---------------------------------------------------------------------

def get_numeric_features(
    dataframe: pd.DataFrame,
) -> list[str]:
    """
    Return numeric predictor features for A.4 analysis.

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
# Near-constant analysis
# ---------------------------------------------------------------------

def analyze_near_constant_feature(
    dataframe: pd.DataFrame,
    feature: str,
) -> dict[str, object]:
    """
    Analyze whether one observed value dominates a numeric feature.

    Missing values are excluded when calculating the dominant-value
    percentage because missingness has already been investigated
    separately.
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

    observed_count = len(
        observed
    )

    unique_values = int(
        observed.nunique()
    )

    if observed.empty:

        return {
            "feature": feature,
            "missing_percent": missing_percent,
            "observed_count": observed_count,
            "unique_values": unique_values,
            "dominant_value": float("nan"),
            "dominant_value_count": 0,
            "dominant_value_percent": float("nan"),
            "near_constant": False,
        }

    value_counts = (
        observed
        .value_counts(
            dropna=False
        )
    )

    dominant_value = (
        value_counts.index[0]
    )

    dominant_value_count = int(
        value_counts.iloc[0]
    )

    dominant_value_percent = (
        dominant_value_count
        / observed_count
        * 100
    )

    near_constant = bool(
        dominant_value_percent
        >= NEAR_CONSTANT_THRESHOLD_PERCENT
    )

    return {
        "feature": feature,
        "missing_percent": missing_percent,
        "observed_count": observed_count,
        "unique_values": unique_values,
        "dominant_value": dominant_value,
        "dominant_value_count": dominant_value_count,
        "dominant_value_percent": dominant_value_percent,
        "near_constant": near_constant,
    }


# ---------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------

def build_near_constant_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Build A.4 summary for all numeric predictor features."""

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
            analyze_near_constant_feature(
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
            by="dominant_value_percent",
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
    """Save A.4 near-constant summary."""

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
    """Print A.4 numeric near-constant results."""

    print("=" * 140)
    print(
        "EDA LEVEL 1 — A.4 NUMERIC NEAR-CONSTANT FEATURES"
    )
    print("=" * 140)

    print(
        f"Numeric features investigated: "
        f"{len(results)}"
    )

    print(
        f"Near-constant threshold: "
        f"{NEAR_CONSTANT_THRESHOLD_PERCENT:.2f}%"
    )

    print("\n" + "=" * 140)
    print(
        "NEAR-CONSTANT FEATURES"
    )
    print("=" * 140)

    if results.empty:

        print(
            "No numeric features were found."
        )

    else:

        near_constant_features = (
            results[
                results[
                    "near_constant"
                ]
            ]
        )

        if near_constant_features.empty:

            print(
                "No near-constant numeric features were detected."
            )

        else:

            print(
                near_constant_features[
                    [
                        "feature",
                        "missing_percent",
                        "unique_values",
                        "dominant_value",
                        "dominant_value_count",
                        "dominant_value_percent",
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
        "TOP 20 MOST DOMINATED FEATURES"
    )
    print("=" * 140)

    if not results.empty:

        print(
            results.head(
                20
            )[
                [
                    "feature",
                    "unique_values",
                    "dominant_value",
                    "dominant_value_percent",
                    "near_constant",
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
        "SUMMARY"
    )
    print("=" * 140)

    if not results.empty:

        near_constant_count = int(
            results[
                "near_constant"
            ].sum()
        )

        print(
            f"Near-constant features: "
            f"{near_constant_count}"
        )

        print(
            f"Other numeric features: "
            f"{len(results) - near_constant_count}"
        )

    print("\n" + "=" * 140)
    print(
        "ARTIFACT SAVED"
    )
    print("=" * 140)

    print(
        SUMMARY_OUTPUT_PATH
    )


def run_a4_numeric_near_constant() -> pd.DataFrame:
    """Run EDA Level 1 A.4."""

    dataframe = (
        load_training_data()
    )

    results = (
        build_near_constant_summary(
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
    run_a4_numeric_near_constant()