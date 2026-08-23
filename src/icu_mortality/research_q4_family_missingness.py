"""Research Q4: investigate missingness patterns within feature families."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from icu_mortality.data import load_training_data


REPORTS_TABLES_DIR = Path("reports") / "tables"

FEATURE_CATALOG_PATH = (
    REPORTS_TABLES_DIR
    / "feature_catalog.csv"
)

SUMMARY_OUTPUT_PATH = (
    REPORTS_TABLES_DIR
    / "q4_family_missingness_summary.csv"
)

DISTRIBUTION_OUTPUT_PATH = (
    REPORTS_TABLES_DIR
    / "q4_family_missingness_distribution.csv"
)


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


def get_investigate_catalog(
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    """Return INVESTIGATE features with valid categories."""

    required_columns = {
        "feature",
        "decision",
        "dictionary_category",
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

    investigate = catalog.loc[
        catalog["decision"] == "INVESTIGATE",
        [
            "feature",
            "dictionary_category",
        ],
    ].copy()

    investigate = investigate.dropna(
        subset=["dictionary_category"]
    )

    return investigate


def get_feature_families(
    catalog: pd.DataFrame,
) -> dict[str, list[str]]:
    """Group INVESTIGATE features by dictionary category."""

    investigate = get_investigate_catalog(
        catalog
    )

    families: dict[str, list[str]] = {}

    grouped = investigate.groupby(
        "dictionary_category"
    )

    for category, group in grouped:
        features = (
            group["feature"]
            .tolist()
        )

        if len(features) >= 2:
            families[str(category)] = features

    return families


def analyze_family_summary(
    dataframe: pd.DataFrame,
    category: str,
    features: list[str],
) -> dict[str, object]:
    """Summarize missingness patterns for one feature family."""

    available_features = [
        feature
        for feature in features
        if feature in dataframe.columns
    ]

    missing_matrix = (
        dataframe[
            available_features
        ]
        .isna()
    )

    all_missing = (
        missing_matrix.all(
            axis=1
        )
    )

    all_present = (
        ~missing_matrix.any(
            axis=1
        )
    )

    partial_missing = (
        ~(all_missing | all_present)
    )

    return {
        "dictionary_category": category,
        "features_count": len(
            available_features
        ),
        "all_missing_percent": (
            all_missing.mean()
            * 100
        ),
        "all_present_percent": (
            all_present.mean()
            * 100
        ),
        "partial_missing_percent": (
            partial_missing.mean()
            * 100
        ),
    }


def build_family_summary(
    dataframe: pd.DataFrame,
    families: dict[str, list[str]],
) -> pd.DataFrame:
    """Build one summary row per feature family."""

    rows: list[
        dict[str, object]
    ] = []

    for category, features in families.items():
        row = analyze_family_summary(
            dataframe=dataframe,
            category=category,
            features=features,
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
            by="partial_missing_percent",
            ascending=True,
        )
        .reset_index(
            drop=True
        )
    )


def analyze_family_distribution(
    dataframe: pd.DataFrame,
    category: str,
    features: list[str],
) -> pd.DataFrame:
    """Build the patient-level missingness distribution for one family."""

    available_features = [
        feature
        for feature in features
        if feature in dataframe.columns
    ]

    missing_count = (
        dataframe[
            available_features
        ]
        .isna()
        .sum(
            axis=1
        )
    )

    distribution = (
        missing_count
        .value_counts()
        .sort_index()
        .rename_axis(
            "missing_features"
        )
        .reset_index(
            name="patients"
        )
    )

    distribution[
        "missing_features_percent"
    ] = (
        distribution[
            "missing_features"
        ]
        / len(available_features)
        * 100
    )

    distribution[
        "patients_percent"
    ] = (
        distribution[
            "patients"
        ]
        / len(dataframe)
        * 100
    )

    distribution.insert(
        0,
        "dictionary_category",
        category,
    )

    distribution.insert(
        1,
        "features_count",
        len(available_features),
    )

    return distribution


def build_family_distribution(
    dataframe: pd.DataFrame,
    families: dict[str, list[str]],
) -> pd.DataFrame:
    """Build missingness distributions for all feature families."""

    frames: list[
        pd.DataFrame
    ] = []

    for category, features in families.items():
        family_distribution = (
            analyze_family_distribution(
                dataframe=dataframe,
                category=category,
                features=features,
            )
        )

        frames.append(
            family_distribution
        )

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
    )


def save_results(
    summary: pd.DataFrame,
    distribution: pd.DataFrame,
) -> None:
    """Save Q4 research artifacts."""

    REPORTS_TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        SUMMARY_OUTPUT_PATH,
        index=False,
    )

    distribution.to_csv(
        DISTRIBUTION_OUTPUT_PATH,
        index=False,
    )


def print_results(
    summary: pd.DataFrame,
    distribution: pd.DataFrame,
) -> None:
    """Print the Q4 result tables."""

    print("=" * 100)
    print(
        "Q4 — FEATURE FAMILY MISSINGNESS"
    )
    print("=" * 100)

    print(
        f"Feature families investigated: "
        f"{len(summary)}"
    )

    print("\n" + "=" * 100)
    print(
        "FAMILY SUMMARY"
    )
    print("=" * 100)

    if summary.empty:
        print(
            "No feature families were found."
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

    print("\n" + "=" * 100)
    print(
        "FAMILY MISSINGNESS DISTRIBUTION"
    )
    print("=" * 100)

    if distribution.empty:
        print(
            "No distribution results were produced."
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

    print("\n" + "=" * 100)
    print(
        "RESULTS SAVED"
    )
    print("=" * 100)

    print(
        SUMMARY_OUTPUT_PATH
    )

    print(
        DISTRIBUTION_OUTPUT_PATH
    )


def run_q4_analysis() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Run Research Question 4 analysis."""

    dataframe = (
        load_training_data()
    )

    catalog = (
        load_feature_catalog()
    )

    families = (
        get_feature_families(
            catalog
        )
    )

    summary = (
        build_family_summary(
            dataframe=dataframe,
            families=families,
        )
    )

    distribution = (
        build_family_distribution(
            dataframe=dataframe,
            families=families,
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
    run_q4_analysis()