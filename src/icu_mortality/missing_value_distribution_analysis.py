"""Statistical distribution analysis for high-missing numeric features."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import is_numeric_dtype

from icu_mortality.data import load_training_data


REPORTS_TABLES_DIR = Path("reports") / "tables"

REPORTS_FIGURES_DIR = (
    Path("reports")
    / "figures"
    / "missing_feature_distributions"
)

FEATURE_CATALOG_PATH = (
    REPORTS_TABLES_DIR
    / "feature_catalog.csv"
)

OUTPUT_PATH = (
    REPORTS_TABLES_DIR
    / "high_missing_numeric_distributions.csv"
)

CROSSTAB_OUTPUT_PATH = (
    REPORTS_TABLES_DIR
    / "missingness_level_treatment_crosstab.csv"
)

HIGH_MISSING_THRESHOLD = 50.0


def load_feature_catalog(
    path: Path = FEATURE_CATALOG_PATH,
) -> pd.DataFrame:
    """Load the feature catalog produced during Phase 3.1."""

    if not path.exists():
        raise FileNotFoundError(
            "Feature catalog was not found: "
            f"{path.resolve()}"
        )

    catalog = pd.read_csv(path)

    if catalog.empty:
        raise ValueError(
            "Feature catalog is empty."
        )

    return catalog


def get_high_missing_numeric_features(
    dataframe: pd.DataFrame,
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    """Return numeric features with at least 50% missing values."""

    required_columns = {
        "feature",
        "missing_percent",
    }

    missing_columns = (
        required_columns
        - set(catalog.columns)
    )

    if missing_columns:
        raise ValueError(
            "Feature catalog is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    high_missing = catalog.loc[
        catalog["missing_percent"]
        >= HIGH_MISSING_THRESHOLD,
        [
            "feature",
            "missing_percent",
        ],
    ].copy()

    high_missing = high_missing[
        high_missing["feature"].isin(
            dataframe.columns
        )
    ]

    numeric_mask = (
        high_missing["feature"]
        .map(
            lambda feature: is_numeric_dtype(
                dataframe[feature]
            )
        )
    )

    return (
        high_missing.loc[numeric_mask]
        .reset_index(drop=True)
    )


def classify_missingness_level(
    missing_percent: float,
) -> str:
    """Classify missingness severity for investigation."""

    if pd.isna(missing_percent):
        return "not_available"

    if missing_percent < 70:
        return "high"

    if missing_percent < 80:
        return "very_high"

    if missing_percent < 90:
        return "extreme"

    return "very_extreme"


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
        return "not_available"

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
        return "not_available"

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


def assign_treatment_candidate(
    relative_difference: float,
    skewness: float,
) -> tuple[str, str]:
    """Assign a preliminary imputation candidate."""

    if (
        pd.isna(relative_difference)
        or pd.isna(skewness)
    ):
        return (
            "KEEP — FURTHER INVESTIGATION",
            (
                "A distribution indicator is unavailable; "
                "the imputation candidate cannot be selected reliably."
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
            "KEEP — MEAN CANDIDATE",
            (
                "Distribution is approximately symmetric and "
                "mean and median are highly consistent."
            ),
        )

    if absolute_skewness >= 2:
        return (
            "KEEP — MEDIAN CANDIDATE",
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
            "KEEP — MEDIAN CANDIDATE",
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
            "KEEP — FURTHER INVESTIGATION",
            (
                "Distribution appears approximately symmetric, "
                "but the mean-median difference is noticeable."
            ),
        )

    if (
        absolute_skewness >= 1
        and relative_difference < 10
    ):
        return (
            "KEEP — FURTHER INVESTIGATION",
            (
                "Distribution is strongly skewed, but mean and "
                "median remain relatively similar."
            ),
        )

    if 0.5 <= absolute_skewness < 1:
        return (
            "KEEP — FURTHER INVESTIGATION",
            (
                "Distribution shows moderate skewness; "
                "current indicators do not clearly favor "
                "mean or median imputation."
            ),
        )

    return (
        "KEEP — FURTHER INVESTIGATION",
        (
            "Current distribution indicators do not provide "
            "a sufficiently clear imputation candidate."
        ),
    )


def analyze_feature_distribution(
    dataframe: pd.DataFrame,
    feature: str,
    missing_percent: float,
) -> dict[str, object]:
    """Calculate statistics and a treatment candidate."""

    values = (
        dataframe[feature]
        .dropna()
    )

    missingness_level = (
        classify_missingness_level(
            missing_percent
        )
    )

    if values.empty:
        return {
            "feature": feature,
            "missing_percent": missing_percent,
            "missingness_level": missingness_level,
            "mean": float("nan"),
            "median": float("nan"),
            "relative_difference_percent": float("nan"),
            "relative_difference_category": "not_available",
            "skewness": float("nan"),
            "skewness_category": "not_available",
            "missing_value_treatment_candidate": (
                "KEEP — FURTHER INVESTIGATION"
            ),
            "treatment_reason": (
                "No observed values are available "
                "for distribution analysis."
            ),
        }

    mean = values.mean()
    median = values.median()
    skewness = values.skew()

    relative_difference = (
        calculate_relative_difference(
            mean=mean,
            median=median,
        )
    )

    candidate, reason = (
        assign_treatment_candidate(
            relative_difference=relative_difference,
            skewness=skewness,
        )
    )

    return {
        "feature": feature,
        "missing_percent": missing_percent,
        "missingness_level": missingness_level,
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
        "missing_value_treatment_candidate": (
            candidate
        ),
        "treatment_reason": (
            reason
        ),
    }


def build_distribution_summary(
    dataframe: pd.DataFrame,
    high_missing_features: pd.DataFrame,
) -> pd.DataFrame:
    """Build summaries for all high-missing numeric features."""

    rows: list[dict[str, object]] = []

    for _, row in high_missing_features.iterrows():
        rows.append(
            analyze_feature_distribution(
                dataframe=dataframe,
                feature=row["feature"],
                missing_percent=row[
                    "missing_percent"
                ],
            )
        )

    results = pd.DataFrame(rows)

    if results.empty:
        return results

    return (
        results
        .sort_values(
            by="missing_percent",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


def build_missingness_treatment_crosstab(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Cross-tabulate missingness level and treatment candidate."""

    if results.empty:
        return pd.DataFrame()

    return pd.crosstab(
        results[
            "missing_value_treatment_candidate"
        ],
        results[
            "missingness_level"
        ],
    )


def save_histogram(
    dataframe: pd.DataFrame,
    feature: str,
) -> None:
    """Save a histogram for one feature."""

    values = (
        dataframe[feature]
        .dropna()
    )

    if values.empty:
        return

    REPORTS_FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(8, 5)
    )

    axis.hist(
        values,
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
    features: list[str],
) -> None:
    """Save histograms for all investigated features."""

    for feature in features:
        save_histogram(
            dataframe=dataframe,
            feature=feature,
        )


def save_results(
    results: pd.DataFrame,
    crosstab: pd.DataFrame,
) -> None:
    """Save analysis results."""

    REPORTS_TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    crosstab.to_csv(
        CROSSTAB_OUTPUT_PATH,
    )


def print_results(
    results: pd.DataFrame,
    crosstab: pd.DataFrame,
) -> None:
    """Print analysis results."""

    print("=" * 140)
    print(
        "HIGH-MISSING NUMERIC FEATURE DISTRIBUTION ANALYSIS"
    )
    print("=" * 140)

    print(
        f"Numeric high-missing features investigated: "
        f"{len(results)}"
    )

    print("\n" + "=" * 140)
    print(
        "DISTRIBUTION AND TREATMENT-CANDIDATE SUMMARY"
    )
    print("=" * 140)

    if results.empty:
        print(
            "No high-missing numeric features were found."
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

    if not results.empty:

        print("\n" + "=" * 140)
        print(
            "MISSINGNESS LEVEL COUNTS"
        )
        print("=" * 140)

        print(
            results[
                "missingness_level"
            ]
            .value_counts()
            .to_string()
        )

        print("\n" + "=" * 140)
        print(
            "TREATMENT CANDIDATE COUNTS"
        )
        print("=" * 140)

        print(
            results[
                "missing_value_treatment_candidate"
            ]
            .value_counts()
            .to_string()
        )

        print("\n" + "=" * 140)
        print(
            "MISSINGNESS LEVEL × TREATMENT CANDIDATE"
        )
        print("=" * 140)

        print(
            crosstab.to_string()
        )

    print("\n" + "=" * 140)
    print(
        "ARTIFACTS SAVED"
    )
    print("=" * 140)

    print(
        OUTPUT_PATH
    )

    print(
        CROSSTAB_OUTPUT_PATH
    )

    print(
        REPORTS_FIGURES_DIR
    )


def run_distribution_analysis() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Run the high-missing numeric distribution analysis."""

    dataframe = (
        load_training_data()
    )

    catalog = (
        load_feature_catalog()
    )

    high_missing_features = (
        get_high_missing_numeric_features(
            dataframe=dataframe,
            catalog=catalog,
        )
    )

    results = (
        build_distribution_summary(
            dataframe=dataframe,
            high_missing_features=high_missing_features,
        )
    )

    crosstab = (
        build_missingness_treatment_crosstab(
            results
        )
    )

    save_results(
        results=results,
        crosstab=crosstab,
    )

    save_all_histograms(
        dataframe=dataframe,
        features=high_missing_features[
            "feature"
        ].tolist(),
    )

    print_results(
        results=results,
        crosstab=crosstab,
    )

    return (
        results,
        crosstab,
    )


if __name__ == "__main__":
    run_distribution_analysis()