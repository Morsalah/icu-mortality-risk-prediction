"""Research Q5: investigate feature cardinality."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype

from icu_mortality.data import (
    TARGET_COLUMN,
    load_training_data,
)


REPORTS_TABLES_DIR = Path("reports") / "tables"

OUTPUT_PATH = (
    REPORTS_TABLES_DIR
    / "q5_cardinality_analysis.csv"
)


def classify_feature_type(
    dataframe: pd.DataFrame,
    feature: str,
) -> str:
    """Classify a feature as numeric or categorical."""

    if is_numeric_dtype(
        dataframe[feature]
    ):
        return "numeric"

    return "categorical"


def classify_cardinality(
    unique_values: int,
    total_rows: int,
    feature_type: str,
) -> str:
    """Assign a descriptive cardinality classification."""

    uniqueness_ratio = (
        unique_values
        / total_rows
    )

    if uniqueness_ratio >= 0.95:
        return "identifier_like"

    if feature_type == "numeric":
        return "numeric_cardinality_not_flagged"

    if unique_values <= 10:
        return "low_cardinality"

    if unique_values <= 30:
        return "moderate_cardinality"

    return "high_cardinality"


def assign_cardinality_review(
    feature: str,
    feature_type: str,
    cardinality_level: str,
) -> tuple[str, str]:
    """Assign a high-level Q5 review result."""

    if feature == TARGET_COLUMN:
        return (
            "TARGET",
            "Target variable; cardinality treatment is not applicable.",
        )

    if cardinality_level == "identifier_like":
        return (
            "INVESTIGATE",
            (
                "Feature is unique or nearly unique across rows "
                "and may behave as an identifier."
            ),
        )

    if (
        feature_type == "categorical"
        and cardinality_level == "high_cardinality"
    ):
        return (
            "INVESTIGATE",
            (
                "Categorical feature has high cardinality; "
                "encoding strategy requires investigation."
            ),
        )

    if (
        feature_type == "categorical"
        and cardinality_level == "moderate_cardinality"
    ):
        return (
            "REVIEW_FOR_ENCODING",
            (
                "Categorical feature has moderate cardinality; "
                "encoding method should be selected before modeling."
            ),
        )

    if feature_type == "categorical":
        return (
            "STANDARD_CATEGORICAL",
            (
                "Categorical feature has relatively low cardinality."
            ),
        )

    return (
        "NO_CARDINALITY_CONCERN",
        (
            "High uniqueness in a numeric feature is not treated "
            "as a cardinality problem by itself."
        ),
    )


def build_cardinality_analysis(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Build cardinality statistics for all features."""

    total_rows = len(
        dataframe
    )

    rows: list[
        dict[str, object]
    ] = []

    for feature in dataframe.columns:

        feature_type = (
            classify_feature_type(
                dataframe=dataframe,
                feature=feature,
            )
        )

        unique_values = int(
            dataframe[feature]
            .nunique(
                dropna=True
            )
        )

        uniqueness_ratio = (
            unique_values
            / total_rows
        )

        cardinality_level = (
            classify_cardinality(
                unique_values=unique_values,
                total_rows=total_rows,
                feature_type=feature_type,
            )
        )

        review, reason = (
            assign_cardinality_review(
                feature=feature,
                feature_type=feature_type,
                cardinality_level=cardinality_level,
            )
        )

        rows.append(
            {
                "feature": feature,
                "feature_type": feature_type,
                "unique_values": unique_values,
                "uniqueness_ratio_percent": (
                    uniqueness_ratio
                    * 100
                ),
                "cardinality_level": (
                    cardinality_level
                ),
                "q5_review": review,
                "q5_reason": reason,
            }
        )

    results = pd.DataFrame(
        rows
    )

    return (
        results
        .sort_values(
            by="uniqueness_ratio_percent",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


def save_results(
    results: pd.DataFrame,
) -> None:
    """Save Q5 analysis."""

    REPORTS_TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )


def print_results(
    results: pd.DataFrame,
) -> None:
    """Print Q5 results."""

    print("=" * 100)
    print(
        "Q5 — CARDINALITY INVESTIGATION"
    )
    print("=" * 100)

    print(
        f"Features investigated: "
        f"{len(results)}"
    )

    print("\n" + "=" * 100)
    print(
        "FEATURES REQUIRING CARDINALITY REVIEW"
    )
    print("=" * 100)

    review = results[
        results["q5_review"].isin(
            [
                "INVESTIGATE",
                "REVIEW_FOR_ENCODING",
            ]
        )
    ]

    if review.empty:
        print(
            "None"
        )
    else:
        print(
            review.to_string(
                index=False,
                float_format=lambda value: (
                    f"{value:.2f}"
                ),
            )
        )

    print("\n" + "=" * 100)
    print(
        "Q5 REVIEW COUNTS"
    )
    print("=" * 100)

    print(
        results[
            "q5_review"
        ]
        .value_counts()
        .to_string()
    )

    print("\nSaved to:")
    print(
        OUTPUT_PATH
    )


def run_q5_analysis() -> pd.DataFrame:
    """Run Q5 cardinality investigation."""

    dataframe = (
        load_training_data()
    )

    results = (
        build_cardinality_analysis(
            dataframe
        )
    )

    save_results(
        results
    )

    print_results(
        results
    )

    return results


if __name__ == "__main__":
    run_q5_analysis()