"""Build baseline preprocessing decisions for categorical features."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype

from icu_mortality.data import load_training_data


REPORTS_TABLES_DIR = Path("reports") / "tables"

OUTPUT_PATH = (
    REPORTS_TABLES_DIR
    / "baseline_categorical_feature_decisions.csv"
)

RARE_CATEGORY_THRESHOLD_PERCENT = 1.0


# ---------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------

def get_categorical_features(
    dataframe: pd.DataFrame,
) -> list[str]:
    """Return all non-numeric features."""

    return [
        feature
        for feature in dataframe.columns
        if not is_numeric_dtype(
            dataframe[feature]
        )
    ]


# ---------------------------------------------------------------------
# Missing-value policy
# ---------------------------------------------------------------------

def assign_missing_value_action(
    missing_percent: float,
) -> tuple[str, str]:
    """Assign baseline missing-value treatment."""

    if missing_percent == 0:
        return (
            "NO IMPUTATION",
            "Feature contains no missing values.",
        )

    if missing_percent < 5:
        return (
            "MODE IMPUTATION",
            (
                "Missingness is below 5%; "
                "mode imputation is used for the baseline."
            ),
        )

    if missing_percent < 80:
        return (
            "ADD UNKNOWN CATEGORY",
            (
                "Missingness is at least 5% but below 80%; "
                "missing values are preserved explicitly "
                "through an UNKNOWN category."
            ),
        )

    return (
        "DROP FROM BASELINE",
        (
            "Feature has at least 80% missing values and is "
            "removed according to the baseline policy."
        ),
    )


# ---------------------------------------------------------------------
# Encoding policy
# ---------------------------------------------------------------------

def assign_encoding_action(
    unique_categories: int,
) -> tuple[str, str]:
    """Assign baseline categorical encoding."""

    if unique_categories <= 1:
        return (
            "NO USEFUL ENCODING",
            (
                "Feature has one or fewer observed categories "
                "and does not provide meaningful categorical variation."
            ),
        )

    if unique_categories == 2:
        return (
            "BINARY ENCODING",
            (
                "Feature contains two observed categories."
            ),
        )

    if unique_categories <= 20:
        return (
            "ONE-HOT ENCODING",
            (
                "Feature has low-to-moderate categorical cardinality "
                "and is suitable for baseline one-hot encoding."
            ),
        )

    return (
        "FURTHER INVESTIGATION",
        (
            "Feature has more than 20 categories; "
            "encoding strategy requires additional investigation."
        ),
    )


# ---------------------------------------------------------------------
# Rare-category analysis
# ---------------------------------------------------------------------

def count_rare_categories(
    series: pd.Series,
) -> int:
    """
    Count observed categories occurring in less than the configured
    percentage of rows.

    Missing values are excluded from rare-category counting.
    """

    observed = (
        series.dropna()
    )

    if observed.empty:
        return 0

    percentages = (
        observed
        .value_counts()
        / len(series)
        * 100
    )

    return int(
        (
            percentages
            < RARE_CATEGORY_THRESHOLD_PERCENT
        ).sum()
    )


# ---------------------------------------------------------------------
# Baseline decision
# ---------------------------------------------------------------------

def assign_baseline_action(
    missing_value_action: str,
    encoding_action: str,
) -> tuple[str, str]:
    """Combine missing-value and encoding decisions."""

    if missing_value_action == "DROP FROM BASELINE":
        return (
            "DROP FROM BASELINE",
            (
                "Feature is removed because categorical "
                "missingness exceeds the baseline threshold."
            ),
        )

    if encoding_action == "NO USEFUL ENCODING":
        return (
            "DROP FROM BASELINE",
            (
                "Feature has insufficient categorical variation "
                "for baseline modeling."
            ),
        )

    if encoding_action == "FURTHER INVESTIGATION":
        return (
            "KEEP — FURTHER INVESTIGATION",
            (
                "Feature can be retained, but its encoding strategy "
                "is not resolved by the baseline rules."
            ),
        )

    return (
        "KEEP",
        (
            "Feature satisfies the baseline categorical "
            "missing-value and encoding policies."
        ),
    )


# ---------------------------------------------------------------------
# Feature analysis
# ---------------------------------------------------------------------

def analyze_categorical_feature(
    dataframe: pd.DataFrame,
    feature: str,
) -> dict[str, object]:
    """Analyze one categorical feature."""

    series = dataframe[
        feature
    ]

    missing_percent = (
        series.isna()
        .mean()
        * 100
    )

    unique_categories = int(
        series.nunique(
            dropna=True
        )
    )

    rare_categories_count = (
        count_rare_categories(
            series
        )
    )

    (
        missing_value_action,
        missing_value_reason,
    ) = assign_missing_value_action(
        missing_percent
    )

    (
        encoding_action,
        encoding_reason,
    ) = assign_encoding_action(
        unique_categories
    )

    (
        baseline_action,
        baseline_reason,
    ) = assign_baseline_action(
        missing_value_action=missing_value_action,
        encoding_action=encoding_action,
    )

    return {
        "feature": feature,
        "feature_data_type": "categorical",
        "unique_categories": (
            unique_categories
        ),
        "missing_percent": (
            missing_percent
        ),
        "rare_categories_count": (
            rare_categories_count
        ),
        "missing_value_action": (
            missing_value_action
        ),
        "missing_value_reason": (
            missing_value_reason
        ),
        "encoding_action": (
            encoding_action
        ),
        "encoding_reason": (
            encoding_reason
        ),
        "baseline_action": (
            baseline_action
        ),
        "baseline_reason": (
            baseline_reason
        ),
    }


def build_categorical_decision_table(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Build baseline decisions for all categorical features."""

    categorical_features = (
        get_categorical_features(
            dataframe
        )
    )

    rows: list[
        dict[str, object]
    ] = []

    for feature in categorical_features:

        rows.append(
            analyze_categorical_feature(
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
            by=[
                "missing_percent",
                "unique_categories",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

def save_results(
    results: pd.DataFrame,
) -> None:
    """Save categorical baseline decisions."""

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
    """Print categorical baseline decisions."""

    print("=" * 140)
    print(
        "BASELINE CATEGORICAL FEATURE DECISIONS"
    )
    print("=" * 140)

    print(
        f"Categorical features investigated: "
        f"{len(results)}"
    )

    print("\n" + "=" * 140)
    print(
        "MISSING-VALUE ACTION COUNTS"
    )
    print("=" * 140)

    print(
        results[
            "missing_value_action"
        ]
        .value_counts()
        .to_string()
    )

    print("\n" + "=" * 140)
    print(
        "ENCODING ACTION COUNTS"
    )
    print("=" * 140)

    print(
        results[
            "encoding_action"
        ]
        .value_counts()
        .to_string()
    )

    print("\n" + "=" * 140)
    print(
        "BASELINE ACTION COUNTS"
    )
    print("=" * 140)

    print(
        results[
            "baseline_action"
        ]
        .value_counts()
        .to_string()
    )

    print("\n" + "=" * 140)
    print(
        "FEATURE DECISION TABLE"
    )
    print("=" * 140)

    display_columns = [
        "feature",
        "unique_categories",
        "missing_percent",
        "rare_categories_count",
        "missing_value_action",
        "encoding_action",
        "baseline_action",
    ]

    print(
        results[
            display_columns
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
        OUTPUT_PATH
    )


def run_baseline_categorical_decisions() -> pd.DataFrame:
    """Build and save categorical baseline preprocessing decisions."""

    dataframe = (
        load_training_data()
    )

    results = (
        build_categorical_decision_table(
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
    run_baseline_categorical_decisions()