"""Build baseline preprocessing decisions for numeric features."""

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
    / "baseline_numeric_feature_decisions.csv"
)


KNOWN_IDENTIFIER_COLUMNS = {
    "encounter_id",
    "patient_id",
}


# ---------------------------------------------------------------------
# Feature type classification
# ---------------------------------------------------------------------

def classify_numeric_feature_type(
    dataframe: pd.DataFrame,
    feature: str,
) -> str:
    """Classify a numeric feature for baseline preprocessing."""

    if feature == TARGET_COLUMN:
        return "target"

    if feature in KNOWN_IDENTIFIER_COLUMNS:
        return "identifier"

    observed_values = (
        dataframe[feature]
        .dropna()
    )

    unique_values = set(
        observed_values.unique()
    )

    if unique_values and unique_values.issubset(
        {0, 1}
    ):
        return "binary_numeric"

    return "continuous_numeric"


def statistical_analysis_is_applicable(
    feature_data_type: str,
) -> bool:
    """Return whether mean/median distribution analysis is applicable."""

    return (
        feature_data_type
        == "continuous_numeric"
    )


# ---------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------

def calculate_relative_difference(
    mean: float,
    median: float,
) -> float:
    """Calculate relative difference between mean and median."""

    if pd.isna(mean) or pd.isna(median):
        return float("nan")

    if median == 0:
        return float("nan")

    return (
        abs(mean - median)
        / abs(median)
        * 100
    )


def classify_relative_difference(
    relative_difference: float,
) -> str:
    """Classify relative difference between mean and median."""

    if pd.isna(relative_difference):
        return "not_applicable"

    if relative_difference < 5:
        return "very_small"

    if relative_difference < 10:
        return "small"

    if relative_difference < 20:
        return "noticeable"

    if relative_difference <= 50:
        return "significant"

    return "very_large"


def classify_skewness(
    skewness: float,
) -> str:
    """Classify distribution skewness."""

    if pd.isna(skewness):
        return "not_applicable"

    absolute_skewness = abs(
        skewness
    )

    if absolute_skewness < 0.5:
        return "approximately_symmetric"

    if absolute_skewness < 1:
        return "moderately_skewed"

    if absolute_skewness < 2:
        return "strongly_skewed"

    return "very_strongly_skewed"


def assign_statistical_candidate(
    relative_difference: float,
    skewness: float,
) -> tuple[str, str]:
    """Assign mean/median candidate for continuous numeric features."""

    if (
        pd.isna(relative_difference)
        or pd.isna(skewness)
    ):
        return (
            "FURTHER INVESTIGATION",
            (
                "Statistical indicators are unavailable or "
                "insufficient for selecting mean or median."
            ),
        )

    absolute_skewness = abs(
        skewness
    )

    if (
        absolute_skewness < 0.5
        and relative_difference < 5
    ):
        return (
            "MEAN CANDIDATE",
            (
                "Distribution is approximately symmetric and "
                "mean and median are highly consistent."
            ),
        )

    if absolute_skewness >= 2:
        return (
            "MEDIAN CANDIDATE",
            (
                "Distribution is very strongly skewed; "
                "median is more robust to extreme values."
            ),
        )

    if (
        absolute_skewness >= 1
        and relative_difference >= 10
    ):
        return (
            "MEDIAN CANDIDATE",
            (
                "Distribution is strongly skewed and "
                "mean and median differ meaningfully."
            ),
        )

    if (
        absolute_skewness < 0.5
        and relative_difference >= 10
    ):
        return (
            "FURTHER INVESTIGATION",
            (
                "Distribution appears approximately symmetric, "
                "but mean and median differ noticeably."
            ),
        )

    if (
        absolute_skewness >= 1
        and relative_difference < 10
    ):
        return (
            "FURTHER INVESTIGATION",
            (
                "Distribution is skewed, but mean and median "
                "remain relatively similar."
            ),
        )

    if 0.5 <= absolute_skewness < 1:
        return (
            "FURTHER INVESTIGATION",
            (
                "Distribution has moderate skewness and does "
                "not clearly favor mean or median."
            ),
        )

    return (
        "FURTHER INVESTIGATION",
        (
            "Current distribution indicators do not clearly "
            "favor mean or median."
        ),
    )


# ---------------------------------------------------------------------
# Baseline policy
# ---------------------------------------------------------------------

def assign_continuous_baseline_action(
    missing_percent: float,
    statistical_candidate: str,
) -> tuple[str, str]:
    """Apply baseline missing-value policy to continuous numeric features."""

    if missing_percent == 0:
        return (
            "KEEP — NO IMPUTATION",
            "Feature contains no missing values.",
        )

    # Less than 50% missing
    if missing_percent < 50:

        if statistical_candidate == "MEAN CANDIDATE":
            return (
                "KEEP — MEAN IMPUTATION",
                (
                    "Missingness is below 50% and the observed "
                    "distribution supports mean imputation."
                ),
            )

        if statistical_candidate == "MEDIAN CANDIDATE":
            return (
                "KEEP — MEDIAN IMPUTATION",
                (
                    "Missingness is below 50% and the observed "
                    "distribution supports median imputation."
                ),
            )

        return (
            "KEEP — MEDIAN IMPUTATION",
            (
                "Missingness is below 50%, but statistical "
                "behavior is inconclusive. Median is used as "
                "the conservative baseline default."
            ),
        )

    # 50% <= missing < 80%
    if missing_percent < 80:

        if statistical_candidate == "MEAN CANDIDATE":
            return (
                "KEEP — MEAN IMPUTATION",
                (
                    "Missingness is between 50% and 80% and "
                    "mean is a clear statistical candidate."
                ),
            )

        if statistical_candidate == "MEDIAN CANDIDATE":
            return (
                "KEEP — MEDIAN IMPUTATION",
                (
                    "Missingness is between 50% and 80% and "
                    "median is a clear statistical candidate."
                ),
            )

        return (
            "DROP FROM BASELINE",
            (
                "Missingness is between 50% and 80%, but "
                "no clear simple-imputation candidate was found."
            ),
        )

    # 80% or more missing
    return (
        "DROP FROM BASELINE",
        (
            "Feature has at least 80% missing values and is "
            "removed according to the initial baseline policy."
        ),
    )


def assign_binary_baseline_action(
    missing_percent: float,
) -> tuple[str, str]:
    """Assign baseline action for binary numeric features."""

    if missing_percent == 0:
        return (
            "KEEP — BINARY NO IMPUTATION",
            "Binary feature contains no missing values.",
        )

    if missing_percent < 50:
        return (
            "KEEP — BINARY MODE IMPUTATION",
            (
                "Binary feature has less than 50% missingness. "
                "Mode imputation is used for the baseline."
            ),
        )

    return (
        "DROP FROM BASELINE",
        (
            "Binary feature has at least 50% missingness and is "
            "removed from the initial baseline."
        ),
    )


# ---------------------------------------------------------------------
# Feature analysis
# ---------------------------------------------------------------------

def analyze_numeric_feature(
    dataframe: pd.DataFrame,
    feature: str,
) -> dict[str, object]:
    """Analyze one numeric feature and assign a baseline action."""

    series = dataframe[
        feature
    ]

    missing_percent = (
        series.isna().mean()
        * 100
    )

    feature_data_type = (
        classify_numeric_feature_type(
            dataframe=dataframe,
            feature=feature,
        )
    )

    analysis_applicable = (
        statistical_analysis_is_applicable(
            feature_data_type
        )
    )

    # -------------------------------------------------------------
    # Target
    # -------------------------------------------------------------

    if feature_data_type == "target":
        return {
            "feature": feature,
            "feature_data_type": feature_data_type,
            "statistical_analysis_applicable": False,
            "missing_percent": missing_percent,
            "mean": float("nan"),
            "median": float("nan"),
            "relative_difference_percent": float("nan"),
            "relative_difference_category": "not_applicable",
            "skewness": float("nan"),
            "skewness_category": "not_applicable",
            "statistical_candidate": "not_applicable",
            "statistical_reason": (
                "Target column is excluded from feature analysis."
            ),
            "baseline_action": (
                "TARGET — NO FEATURE PREPROCESSING"
            ),
            "baseline_reason": (
                "Target column is not treated as a predictor."
            ),
        }

    # -------------------------------------------------------------
    # Identifier
    # -------------------------------------------------------------

    if feature_data_type == "identifier":
        return {
            "feature": feature,
            "feature_data_type": feature_data_type,
            "statistical_analysis_applicable": False,
            "missing_percent": missing_percent,
            "mean": float("nan"),
            "median": float("nan"),
            "relative_difference_percent": float("nan"),
            "relative_difference_category": "not_applicable",
            "skewness": float("nan"),
            "skewness_category": "not_applicable",
            "statistical_candidate": "not_applicable",
            "statistical_reason": (
                "Identifier features are not evaluated "
                "using distribution-based imputation rules."
            ),
            "baseline_action": (
                "DROP FROM BASELINE"
            ),
            "baseline_reason": (
                "Feature is an identifier and should not be "
                "used as a model predictor."
            ),
        }

    # -------------------------------------------------------------
    # Binary numeric
    # -------------------------------------------------------------

    if feature_data_type == "binary_numeric":

        (
            baseline_action,
            baseline_reason,
        ) = assign_binary_baseline_action(
            missing_percent=missing_percent
        )

        return {
            "feature": feature,
            "feature_data_type": feature_data_type,
            "statistical_analysis_applicable": False,
            "missing_percent": missing_percent,
            "mean": float("nan"),
            "median": float("nan"),
            "relative_difference_percent": float("nan"),
            "relative_difference_category": "not_applicable",
            "skewness": float("nan"),
            "skewness_category": "not_applicable",
            "statistical_candidate": (
                "BINARY MODE CANDIDATE"
                if missing_percent > 0
                else "not_applicable"
            ),
            "statistical_reason": (
                "Binary feature is handled using categorical/binary "
                "missing-value logic rather than mean/median analysis."
            ),
            "baseline_action": baseline_action,
            "baseline_reason": baseline_reason,
        }

    # -------------------------------------------------------------
    # Continuous numeric
    # -------------------------------------------------------------

    observed_values = (
        series.dropna()
    )

    if observed_values.empty:

        mean = float("nan")
        median = float("nan")
        skewness = float("nan")
        relative_difference = float("nan")

        statistical_candidate = (
            "FURTHER INVESTIGATION"
        )

        statistical_reason = (
            "No observed values are available."
        )

    else:

        mean = observed_values.mean()
        median = observed_values.median()
        skewness = observed_values.skew()

        relative_difference = (
            calculate_relative_difference(
                mean=mean,
                median=median,
            )
        )

        (
            statistical_candidate,
            statistical_reason,
        ) = assign_statistical_candidate(
            relative_difference=relative_difference,
            skewness=skewness,
        )

    (
        baseline_action,
        baseline_reason,
    ) = assign_continuous_baseline_action(
        missing_percent=missing_percent,
        statistical_candidate=statistical_candidate,
    )

    return {
        "feature": feature,
        "feature_data_type": feature_data_type,
        "statistical_analysis_applicable": (
            analysis_applicable
        ),
        "missing_percent": missing_percent,
        "mean": mean,
        "median": median,
        "relative_difference_percent": (
            relative_difference
        ),
        "relative_difference_category": (
            classify_relative_difference(
                relative_difference
            )
        ),
        "skewness": skewness,
        "skewness_category": (
            classify_skewness(
                skewness
            )
        ),
        "statistical_candidate": (
            statistical_candidate
        ),
        "statistical_reason": (
            statistical_reason
        ),
        "baseline_action": (
            baseline_action
        ),
        "baseline_reason": (
            baseline_reason
        ),
    }


def build_numeric_decision_table(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Build baseline decisions for all numeric features."""

    numeric_features = [
        feature
        for feature in dataframe.columns
        if is_numeric_dtype(
            dataframe[feature]
        )
    ]

    rows: list[
        dict[str, object]
    ] = []

    for feature in numeric_features:

        rows.append(
            analyze_numeric_feature(
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
                "feature_data_type",
                "missing_percent",
            ],
            ascending=[
                True,
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
    """Save baseline numeric feature decisions."""

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
    """Print baseline numeric preprocessing decisions."""

    print("=" * 150)
    print(
        "BASELINE NUMERIC FEATURE DECISIONS"
    )
    print("=" * 150)

    print(
        f"Numeric features investigated: "
        f"{len(results)}"
    )

    print("\n" + "=" * 150)
    print(
        "FEATURE TYPE COUNTS"
    )
    print("=" * 150)

    print(
        results[
            "feature_data_type"
        ]
        .value_counts()
        .to_string()
    )

    print("\n" + "=" * 150)
    print(
        "BASELINE ACTION COUNTS"
    )
    print("=" * 150)

    print(
        results[
            "baseline_action"
        ]
        .value_counts()
        .to_string()
    )

    print("\n" + "=" * 150)
    print(
        "FEATURE DECISION TABLE"
    )
    print("=" * 150)

    display_columns = [
        "feature",
        "feature_data_type",
        "statistical_analysis_applicable",
        "missing_percent",
        "relative_difference_percent",
        "skewness",
        "statistical_candidate",
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

    print("\n" + "=" * 150)
    print(
        "ARTIFACT SAVED"
    )
    print("=" * 150)

    print(
        OUTPUT_PATH
    )


def run_baseline_numeric_decisions() -> pd.DataFrame:
    """Build and save numeric baseline preprocessing decisions."""

    dataframe = (
        load_training_data()
    )

    results = (
        build_numeric_decision_table(
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
    run_baseline_numeric_decisions()