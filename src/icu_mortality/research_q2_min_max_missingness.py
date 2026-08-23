"""Research Q2: investigate shared missingness patterns in MIN/MAX feature pairs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REPORTS_TABLES_DIR = Path("reports") / "tables"

FEATURE_CATALOG_PATH = (
    REPORTS_TABLES_DIR
    / "feature_catalog.csv"
)

OUTPUT_PATH = (
    REPORTS_TABLES_DIR
    / "q2_min_max_missingness.csv"
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


def find_min_max_pairs(
    features: list[str],
) -> list[tuple[str, str, str]]:
    """Find matching MIN/MAX feature pairs."""

    feature_set = set(features)

    pairs: list[
        tuple[str, str, str]
    ] = []

    for feature in features:
        if not feature.endswith("_min"):
            continue

        base_feature = feature[:-4]

        max_feature = (
            f"{base_feature}_max"
        )

        if max_feature in feature_set:
            pairs.append(
                (
                    base_feature,
                    feature,
                    max_feature,
                )
            )

    return pairs


def analyze_min_max_pair(
    dataframe: pd.DataFrame,
    base_feature: str,
    min_feature: str,
    max_feature: str,
) -> dict[str, object]:
    """Analyze missingness combinations for one MIN/MAX pair."""

    min_missing = (
        dataframe[min_feature]
        .isna()
    )

    max_missing = (
        dataframe[max_feature]
        .isna()
    )

    both_missing = (
        min_missing
        & max_missing
    )

    both_present = (
        ~min_missing
        & ~max_missing
    )

    min_only_missing = (
        min_missing
        & ~max_missing
    )

    max_only_missing = (
        ~min_missing
        & max_missing
    )

    both_missing_percent = (
        both_missing.mean()
        * 100
    )

    both_present_percent = (
        both_present.mean()
        * 100
    )

    min_only_missing_percent = (
        min_only_missing.mean()
        * 100
    )

    max_only_missing_percent = (
        max_only_missing.mean()
        * 100
    )

    same_missingness_pattern = (
        not min_only_missing.any()
        and not max_only_missing.any()
    )

    return {
        "base_feature": base_feature,
        "min_feature": min_feature,
        "max_feature": max_feature,
        "both_missing_percent": (
            both_missing_percent
        ),
        "both_present_percent": (
            both_present_percent
        ),
        "min_only_missing_percent": (
            min_only_missing_percent
        ),
        "max_only_missing_percent": (
            max_only_missing_percent
        ),
        "same_missingness_pattern": (
            same_missingness_pattern
        ),
    }


def build_q2_results(
    dataframe: pd.DataFrame,
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    """Build Q2 results for all MIN/MAX pairs."""

    investigate_features = (
        get_investigate_features(
            catalog
        )
    )

    pairs = find_min_max_pairs(
        investigate_features
    )

    rows: list[
        dict[str, object]
    ] = []

    for (
        base_feature,
        min_feature,
        max_feature,
    ) in pairs:

        if (
            min_feature not in dataframe.columns
            or max_feature not in dataframe.columns
        ):
            continue

        row = analyze_min_max_pair(
            dataframe=dataframe,
            base_feature=base_feature,
            min_feature=min_feature,
            max_feature=max_feature,
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
            by=[
                "same_missingness_pattern",
                "both_missing_percent",
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


def save_results(
    results: pd.DataFrame,
) -> None:
    """Save Q2 results."""

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
    """Print the complete Q2 result table."""

    print("=" * 100)
    print(
        "Q2 — MIN/MAX MISSINGNESS PATTERNS"
    )
    print("=" * 100)

    print(
        f"MIN/MAX pairs investigated: "
        f"{len(results)}"
    )

    print("\n" + "=" * 100)
    print(
        "ALL MIN/MAX PAIRS"
    )
    print("=" * 100)

    if results.empty:
        print(
            "No MIN/MAX pairs were found."
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

        same_pattern_count = int(
            results[
                "same_missingness_pattern"
            ]
            .sum()
        )

        print("\n" + "=" * 100)
        print(
            "SUMMARY"
        )
        print("=" * 100)

        print(
            f"Pairs with exactly the same "
            f"missingness pattern: "
            f"{same_pattern_count} / "
            f"{len(results)}"
        )

    print("\n" + "=" * 100)
    print(
        "RESULT SAVED"
    )
    print("=" * 100)

    print(
        OUTPUT_PATH
    )


def run_q2_analysis(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Run Research Question 2 analysis."""

    catalog = (
        load_feature_catalog()
    )

    results = (
        build_q2_results(
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
    from icu_mortality.data import (
        load_training_data,
    )

    training_data = (
        load_training_data()
    )

    run_q2_analysis(
        dataframe=training_data
    )