"""Research Q6: investigate categorical features before encoding."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype

from icu_mortality.data import load_training_data


REPORTS_TABLES_DIR = Path("reports") / "tables"

SUMMARY_OUTPUT_PATH = (
    REPORTS_TABLES_DIR
    / "q6_categorical_feature_summary.csv"
)

CATEGORY_OUTPUT_PATH = (
    REPORTS_TABLES_DIR
    / "q6_categorical_value_distribution.csv"
)

RARE_CATEGORY_THRESHOLD_PERCENT = 1.0


def get_categorical_features(
    dataframe: pd.DataFrame,
) -> list[str]:
    """Return non-numeric features."""

    return [
        feature
        for feature in dataframe.columns
        if not is_numeric_dtype(
            dataframe[feature]
        )
    ]


def assign_encoding_candidate(
    unique_values: int,
) -> tuple[str, str]:
    """Assign an initial encoding candidate from cardinality."""

    if unique_values <= 2:
        return (
            "BINARY / ONE-HOT CANDIDATE",
            (
                "Feature contains two categories; "
                "simple binary or one-hot encoding is appropriate "
                "for later validation."
            ),
        )

    if unique_values <= 15:
        return (
            "ONE-HOT ENCODING CANDIDATE",
            (
                "Feature has relatively low categorical cardinality."
            ),
        )

    return (
        "FURTHER INVESTIGATION",
        (
            "Feature has relatively high categorical cardinality; "
            "one-hot encoding may create many columns."
        ),
    )


def analyze_categorical_feature(
    dataframe: pd.DataFrame,
    feature: str,
) -> tuple[
    dict[str, object],
    pd.DataFrame,
]:
    """Analyze one categorical feature."""

    series = dataframe[
        feature
    ]

    unique_values = int(
        series.nunique(
            dropna=True
        )
    )

    missing_percent = (
        series.isna()
        .mean()
        * 100
    )

    encoding_candidate, reason = (
        assign_encoding_candidate(
            unique_values
        )
    )

    value_counts = (
        series
        .fillna("<MISSING>")
        .value_counts(
            dropna=False
        )
        .rename_axis(
            "category_value"
        )
        .reset_index(
            name="count"
        )
    )

    value_counts[
        "percent"
    ] = (
        value_counts[
            "count"
        ]
        / len(dataframe)
        * 100
    )

    value_counts[
        "is_rare_category"
    ] = (
        value_counts[
            "percent"
        ]
        < RARE_CATEGORY_THRESHOLD_PERCENT
    )

    value_counts.insert(
        0,
        "feature",
        feature,
    )

    rare_categories = int(
        value_counts[
            "is_rare_category"
        ]
        .sum()
    )

    summary = {
        "feature": feature,
        "unique_categories": (
            unique_values
        ),
        "missing_percent": (
            missing_percent
        ),
        "rare_categories_count": (
            rare_categories
        ),
        "encoding_candidate": (
            encoding_candidate
        ),
        "encoding_reason": (
            reason
        ),
    }

    return (
        summary,
        value_counts,
    )


def build_q6_analysis(
    dataframe: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Analyze all categorical features."""

    categorical_features = (
        get_categorical_features(
            dataframe
        )
    )

    summaries: list[
        dict[str, object]
    ] = []

    distributions: list[
        pd.DataFrame
    ] = []

    for feature in categorical_features:

        summary, distribution = (
            analyze_categorical_feature(
                dataframe=dataframe,
                feature=feature,
            )
        )

        summaries.append(
            summary
        )

        distributions.append(
            distribution
        )

    summary_df = pd.DataFrame(
        summaries
    )

    if distributions:
        distribution_df = pd.concat(
            distributions,
            ignore_index=True,
        )
    else:
        distribution_df = (
            pd.DataFrame()
        )

    if not summary_df.empty:
        summary_df = (
            summary_df
            .sort_values(
                by="unique_categories",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

    return (
        summary_df,
        distribution_df,
    )


def save_results(
    summary: pd.DataFrame,
    distribution: pd.DataFrame,
) -> None:
    """Save Q6 analysis artifacts."""

    REPORTS_TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        SUMMARY_OUTPUT_PATH,
        index=False,
    )

    distribution.to_csv(
        CATEGORY_OUTPUT_PATH,
        index=False,
    )


def print_results(
    summary: pd.DataFrame,
    distribution: pd.DataFrame,
) -> None:
    """Print Q6 analysis."""

    print("=" * 110)
    print(
        "Q6 — CATEGORICAL FEATURE & ENCODING INVESTIGATION"
    )
    print("=" * 110)

    print(
        f"Categorical features investigated: "
        f"{len(summary)}"
    )

    print("\n" + "=" * 110)
    print(
        "CATEGORICAL FEATURE SUMMARY"
    )
    print("=" * 110)

    if summary.empty:
        print(
            "No categorical features were found."
        )
    else:
        print(
            summary.to_string(
                index=False,
                float_format=lambda value: (
                    f"{value:.2f}"
                ),
            )
        )

    print("\n" + "=" * 110)
    print(
        "ENCODING CANDIDATE COUNTS"
    )
    print("=" * 110)

    if not summary.empty:
        print(
            summary[
                "encoding_candidate"
            ]
            .value_counts()
            .to_string()
        )

    print("\n" + "=" * 110)
    print(
        "CATEGORY VALUE DISTRIBUTIONS"
    )
    print("=" * 110)

    if distribution.empty:
        print(
            "None"
        )
    else:
        print(
            distribution.to_string(
                index=False,
                float_format=lambda value: (
                    f"{value:.2f}"
                ),
            )
        )

    print("\nArtifacts saved:")
    print(
        SUMMARY_OUTPUT_PATH
    )
    print(
        CATEGORY_OUTPUT_PATH
    )


def run_q6_analysis() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Run Q6 categorical-feature investigation."""

    dataframe = (
        load_training_data()
    )

    summary, distribution = (
        build_q6_analysis(
            dataframe
        )
    )

    save_results(
        summary=summary,
        distribution=distribution,
    )

    print_results(
        summary=summary,
        distribution=distribution,
    )

    return (
        summary,
        distribution,
    )


if __name__ == "__main__":
    run_q6_analysis()