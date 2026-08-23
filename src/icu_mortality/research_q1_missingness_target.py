# Q1 — Is missingness related to the target? For each feature with a high rate of
# missing values: Is there a difference in the mortality rate (hospital_death) between patients
# for whom the feature is missing and patients for whom the feature is present?

"""Research Q1: association between feature missingness and hospital mortality."""

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

OUTPUT_PATH = (
    REPORTS_TABLES_DIR
    / "q1_missingness_vs_target.csv"
)


def load_feature_catalog(
    path: Path = FEATURE_CATALOG_PATH,
) -> pd.DataFrame:
    """Load the feature catalog produced during Phase 3.1."""

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


def analyze_feature(
    dataframe: pd.DataFrame,
    feature: str,
) -> dict[str, object]:
    """Compare mortality between missing and present feature values."""

    missing_mask = (
        dataframe[feature]
        .isna()
    )

    present_mask = (
        ~missing_mask
    )

    missing_percent = (
        missing_mask.mean()
        * 100
    )

    mortality_when_missing_percent = (
        dataframe.loc[
            missing_mask,
            TARGET_COLUMN,
        ]
        .mean()
        * 100
    )

    mortality_when_present_percent = (
        dataframe.loc[
            present_mask,
            TARGET_COLUMN,
        ]
        .mean()
        * 100
    )

    mortality_difference_pp = (
        mortality_when_present_percent
        - mortality_when_missing_percent
    )

    return {
        "feature": feature,
        "missing_percent": missing_percent,
        "mortality_when_missing_percent": (
            mortality_when_missing_percent
        ),
        "mortality_when_present_percent": (
            mortality_when_present_percent
        ),
        "mortality_difference_pp": (
            mortality_difference_pp
        ),
    }


def build_q1_results(
    dataframe: pd.DataFrame,
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    """Build Q1 results for all INVESTIGATE features."""

    investigate_features = (
        get_investigate_features(
            catalog
        )
    )

    rows: list[dict[str, object]] = []

    for feature in investigate_features:
        if feature not in dataframe.columns:
            continue

        row = analyze_feature(
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

        rows.append(
            row
        )

    results = pd.DataFrame(
        rows
    )

    if results.empty:
        return results

    return (
        results
        .sort_values(
            by="mortality_difference_pp",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


def save_results(
    results: pd.DataFrame,
) -> None:
    """Save Q1 research results."""

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
    """Print the complete Q1 result table."""

    print("=" * 90)
    print(
        "Q1 — MISSINGNESS VS HOSPITAL MORTALITY"
    )
    print("=" * 90)

    print(
        f"Features investigated: "
        f"{len(results)}"
    )

    print("\n" + "=" * 90)
    print(
        "ALL INVESTIGATED FEATURES"
    )
    print("=" * 90)

    if results.empty:
        print(
            "None"
        )
    else:
        print(
            results[
                [
                    "feature",
                    "dictionary_category",
                    "missing_percent",
                    "mortality_when_missing_percent",
                    "mortality_when_present_percent",
                    "mortality_difference_pp",
                ]
            ]
            .to_string(
                index=False,
                float_format=lambda value: (
                    f"{value:.2f}"
                ),
            )
        )

    print("\n" + "=" * 90)
    print(
        "RESULT SAVED"
    )
    print("=" * 90)

    print(
        OUTPUT_PATH
    )


def run_q1_analysis() -> pd.DataFrame:
    """Run Research Question 1 analysis."""

    dataframe = (
        load_training_data()
    )

    catalog = (
        load_feature_catalog()
    )

    results = (
        build_q1_results(
            dataframe=dataframe,
            catalog=catalog,
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
    run_q1_analysis()