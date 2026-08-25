"""EDA Level 1 - A.3: Numeric suspicious values / range analysis."""

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
    / "eda_level1_a3_numeric_suspicious_values_summary.csv"
)


KNOWN_IDENTIFIER_COLUMNS = {
    "encounter_id",
    "patient_id",
}


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
# Suspicious-value analysis
# ---------------------------------------------------------------------

def analyze_numeric_range(
    dataframe: pd.DataFrame,
    feature: str,
) -> dict[str, object]:
    """
    Analyze the observed range of one numeric feature.

    This function describes potentially suspicious values but does not
    automatically decide whether a value is medically invalid.
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
            "min_value": float("nan"),
            "max_value": float("nan"),
            "range": float("nan"),
            "negative_values_count": 0,
            "negative_values_percent": float("nan"),
            "zero_values_count": 0,
            "zero_values_percent": float("nan"),
        }

    min_value = (
        observed.min()
    )

    max_value = (
        observed.max()
    )

    value_range = (
        max_value
        - min_value
    )

    negative_values_count = int(
        (
            observed < 0
        ).sum()
    )

    negative_values_percent = (
        negative_values_count
        / observed_count
        * 100
    )

    zero_values_count = int(
        (
            observed == 0
        ).sum()
    )

    zero_values_percent = (
        zero_values_count
        / observed_count
        * 100
    )

    return {
        "feature": feature,
        "missing_percent": missing_percent,
        "observed_count": observed_count,
        "unique_values": unique_values,
        "min_value": min_value,
        "max_value": max_value,
        "range": value_range,
        "negative_values_count": (
            negative_values_count
        ),
        "negative_values_percent": (
            negative_values_percent
        ),
        "zero_values_count": (
            zero_values_count
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
) -> pd.DataFrame:
    """Build A.3 range summary for all numeric predictor features."""

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


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

def save_summary(
    results: pd.DataFrame,
) -> None:
    """Save A.3 suspicious-value / range summary."""

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
    """Print A.3 suspicious-value / range results."""

    print("=" * 150)
    print(
        "EDA LEVEL 1 — A.3 NUMERIC SUSPICIOUS VALUES / RANGE"
    )
    print("=" * 150)

    print(
        f"Numeric features investigated: "
        f"{len(results)}"
    )

    print("\n" + "=" * 150)
    print(
        "RANGE SUMMARY"
    )
    print("=" * 150)

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

    print("\n" + "=" * 150)
    print(
        "FEATURES CONTAINING NEGATIVE VALUES"
    )
    print("=" * 150)

    if not results.empty:

        negative_features = (
            results[
                results[
                    "negative_values_count"
                ] > 0
            ]
        )

        if negative_features.empty:

            print(
                "No numeric features contain negative values."
            )

        else:

            print(
                negative_features[
                    [
                        "feature",
                        "min_value",
                        "max_value",
                        "negative_values_count",
                        "negative_values_percent",
                    ]
                ]
                .to_string(
                    index=False,
                    float_format=lambda value: (
                        f"{value:.2f}"
                    ),
                )
            )

    print("\n" + "=" * 150)
    print(
        "FEATURES CONTAINING ZERO VALUES"
    )
    print("=" * 150)

    if not results.empty:

        zero_features = (
            results[
                results[
                    "zero_values_count"
                ] > 0
            ]
        )

        if zero_features.empty:

            print(
                "No numeric features contain zero values."
            )

        else:

            print(
                zero_features[
                    [
                        "feature",
                        "min_value",
                        "max_value",
                        "zero_values_count",
                        "zero_values_percent",
                    ]
                ]
                .to_string(
                    index=False,
                    float_format=lambda value: (
                        f"{value:.2f}"
                    ),
                )
            )

    print("\n" + "=" * 150)
    print(
        "ARTIFACT SAVED"
    )
    print("=" * 150)

    print(
        SUMMARY_OUTPUT_PATH
    )


def run_a3_numeric_suspicious_values() -> pd.DataFrame:
    """Run EDA Level 1 A.3."""

    dataframe = (
        load_training_data()
    )

    results = (
        build_range_summary(
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
    run_a3_numeric_suspicious_values()