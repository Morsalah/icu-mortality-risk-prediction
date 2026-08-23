"""Missingness investigation for ICU mortality features."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from icu_mortality.data import (
    TARGET_COLUMN,
    load_training_data,
)


REPORTS_TABLES_DIR = Path("reports") / "tables"

FEATURE_CATALOG_PATH = (
    REPORTS_TABLES_DIR
    / "feature_catalog.csv"
)

FEATURE_MISSINGNESS_PATH = (
    REPORTS_TABLES_DIR
    / "missingness_investigation.csv"
)

PATIENT_MISSINGNESS_PATH = (
    REPORTS_TABLES_DIR
    / "patient_missingness_summary.csv"
)


def load_feature_catalog(
    path: Path = FEATURE_CATALOG_PATH,
) -> pd.DataFrame:
    """Load the feature catalog created during the high-level audit."""

    if not path.exists():
        raise FileNotFoundError(
            "Feature catalog was not found. "
            "Run feature_audit first: "
            f"{path.resolve()}"
        )

    catalog = pd.read_csv(path)

    if catalog.empty:
        raise ValueError(
            "Feature catalog is empty."
        )

    return catalog


def get_investigate_features(
    catalog: pd.DataFrame,
) -> list[str]:
    """Return features marked as INVESTIGATE."""

    required_columns = {
        "feature",
        "decision",
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

    return (
        catalog.loc[
            catalog["decision"] == "INVESTIGATE",
            "feature",
        ]
        .tolist()
    )


def analyze_single_feature_missingness(
    dataframe: pd.DataFrame,
    feature: str,
) -> dict[str, object]:
    """Compare mortality when a feature is missing versus present."""

    missing_mask = dataframe[feature].isna()
    present_mask = ~missing_mask

    missing_count = int(
        missing_mask.sum()
    )

    present_count = int(
        present_mask.sum()
    )

    missing_percent = (
        missing_count
        / len(dataframe)
        * 100
    )

    mortality_when_missing = (
        dataframe.loc[
            missing_mask,
            TARGET_COLUMN,
        ]
        .mean()
    )

    mortality_when_present = (
        dataframe.loc[
            present_mask,
            TARGET_COLUMN,
        ]
        .mean()
    )

    mortality_difference = (
        mortality_when_present
        - mortality_when_missing
    )

    return {
        "feature": feature,
        "missing_count": missing_count,
        "present_count": present_count,
        "missing_percent": missing_percent,
        "mortality_when_missing": mortality_when_missing,
        "mortality_when_present": mortality_when_present,
        "mortality_difference": mortality_difference,
        "absolute_mortality_difference": abs(
            mortality_difference
        ),
    }


def build_feature_missingness_analysis(
    dataframe: pd.DataFrame,
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    """Analyze missingness for all INVESTIGATE features."""

    investigate_features = (
        get_investigate_features(
            catalog
        )
    )

    rows: list[dict[str, object]] = []

    for feature in investigate_features:
        if feature not in dataframe.columns:
            continue

        row = analyze_single_feature_missingness(
            dataframe=dataframe,
            feature=feature,
        )

        catalog_row = (
            catalog.loc[
                catalog["feature"] == feature
            ]
            .iloc[0]
        )

        row["dictionary_category"] = (
            catalog_row.get(
                "dictionary_category"
            )
        )

        rows.append(row)

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    return (
        result
        .sort_values(
            by="absolute_mortality_difference",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


def build_patient_missingness_summary(
    dataframe: pd.DataFrame,
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize mortality by patient-level missingness burden."""

    investigate_features = (
        get_investigate_features(
            catalog
        )
    )

    available_features = [
        feature
        for feature in investigate_features
        if feature in dataframe.columns
    ]

    if not available_features:
        raise ValueError(
            "No INVESTIGATE features were found in the dataset."
        )

    missing_count_per_patient = (
        dataframe[
            available_features
        ]
        .isna()
        .sum(
            axis=1
        )
    )

    missing_percent_per_patient = (
        missing_count_per_patient
        / len(available_features)
        * 100
    )

    patient_analysis = pd.DataFrame(
        {
            "missing_feature_count": (
                missing_count_per_patient
            ),
            "missing_feature_percent": (
                missing_percent_per_patient
            ),
            TARGET_COLUMN: (
                dataframe[TARGET_COLUMN]
                .values
            ),
        }
    )

    bins = [
        -0.1,
        25,
        50,
        75,
        100,
    ]

    labels = [
        "0-25%",
        "25-50%",
        "50-75%",
        "75-100%",
    ]

    patient_analysis[
        "missingness_group"
    ] = pd.cut(
        patient_analysis[
            "missing_feature_percent"
        ],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    summary = (
        patient_analysis
        .groupby(
            "missingness_group",
            observed=False,
        )
        .agg(
            patients=(
                TARGET_COLUMN,
                "size",
            ),
            mean_missing_feature_count=(
                "missing_feature_count",
                "mean",
            ),
            mean_missing_percent=(
                "missing_feature_percent",
                "mean",
            ),
            mortality_rate=(
                TARGET_COLUMN,
                "mean",
            ),
        )
        .reset_index()
    )

    return summary


def save_analysis_results(
    feature_missingness: pd.DataFrame,
    patient_missingness: pd.DataFrame,
) -> None:
    """Save missingness investigation artifacts."""

    REPORTS_TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_missingness.to_csv(
        FEATURE_MISSINGNESS_PATH,
        index=False,
    )

    patient_missingness.to_csv(
        PATIENT_MISSINGNESS_PATH,
        index=False,
    )


def print_analysis_summary(
    feature_missingness: pd.DataFrame,
    patient_missingness: pd.DataFrame,
) -> None:
    """Print a concise missingness-analysis summary."""

    print("=" * 70)
    print("MISSINGNESS INVESTIGATION")
    print("=" * 70)

    print(
        f"Features investigated: "
        f"{len(feature_missingness)}"
    )

    print("\n" + "=" * 70)
    print(
        "TOP 20 FEATURES BY ABSOLUTE "
        "MORTALITY DIFFERENCE"
    )
    print("=" * 70)

    if feature_missingness.empty:
        print("None")
    else:
        print(
            feature_missingness[
                [
                    "feature",
                    "dictionary_category",
                    "missing_percent",
                    "mortality_when_missing",
                    "mortality_when_present",
                    "absolute_mortality_difference",
                ]
            ]
            .head(20)
            .to_string(
                index=False
            )
        )

    print("\n" + "=" * 70)
    print("PATIENT-LEVEL MISSINGNESS")
    print("=" * 70)

    print(
        patient_missingness
        .to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("ARTIFACTS SAVED")
    print("=" * 70)

    print(
        FEATURE_MISSINGNESS_PATH
    )

    print(
        PATIENT_MISSINGNESS_PATH
    )


def run_missingness_analysis() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Run and persist the missingness investigation."""

    dataframe = load_training_data()

    catalog = load_feature_catalog()

    feature_missingness = (
        build_feature_missingness_analysis(
            dataframe=dataframe,
            catalog=catalog,
        )
    )

    patient_missingness = (
        build_patient_missingness_summary(
            dataframe=dataframe,
            catalog=catalog,
        )
    )

    save_analysis_results(
        feature_missingness=feature_missingness,
        patient_missingness=patient_missingness,
    )

    print_analysis_summary(
        feature_missingness=feature_missingness,
        patient_missingness=patient_missingness,
    )

    return (
        feature_missingness,
        patient_missingness,
    )


if __name__ == "__main__":
    run_missingness_analysis()