"""Research Q3: compare missingness patterns between h1 and d1 features."""

from __future__ import annotations
from pathlib import Path
import pandas as pd
from icu_mortality.data import load_training_data

REPORTS_TABLES_DIR = Path("reports") / "tables"

FEATURE_CATALOG_PATH = (
    REPORTS_TABLES_DIR
    / "feature_catalog.csv"
)

OUTPUT_PATH = (
    REPORTS_TABLES_DIR
    / "q3_h1_vs_d1_missingness.csv"
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


def find_h1_d1_pairs(
    features: list[str],
) -> list[tuple[str, str, str]]:
    """Find matching h1/d1 feature pairs."""

    feature_set = set(
        features
    )

    pairs: list[
        tuple[str, str, str]
    ] = []

    for feature in features:
        if not feature.startswith(
            "h1_"
        ):
            continue

        base_feature = feature[
            len("h1_"):
        ]

        d1_feature = (
            f"d1_{base_feature}"
        )

        if d1_feature in feature_set:
            pairs.append(
                (
                    base_feature,
                    feature,
                    d1_feature,
                )
            )

    return pairs


def analyze_h1_d1_pair(
    dataframe: pd.DataFrame,
    base_feature: str,
    h1_feature: str,
    d1_feature: str,
) -> dict[str, object]:
    """Analyze missingness combinations for one h1/d1 feature pair."""

    h1_missing = (
        dataframe[h1_feature]
        .isna()
    )

    d1_missing = (
        dataframe[d1_feature]
        .isna()
    )

    both_missing = (
        h1_missing
        & d1_missing
    )

    h1_missing_d1_present = (
        h1_missing
        & ~d1_missing
    )

    both_present = (
        ~h1_missing
        & ~d1_missing
    )

    h1_present_d1_missing = (
        ~h1_missing
        & d1_missing
    )

    h1_missing_percent = (
        h1_missing.mean()
        * 100
    )

    d1_missing_percent = (
        d1_missing.mean()
        * 100
    )

    both_missing_percent = (
        both_missing.mean()
        * 100
    )

    h1_missing_d1_present_percent = (
        h1_missing_d1_present.mean()
        * 100
    )

    both_present_percent = (
        both_present.mean()
        * 100
    )

    h1_present_d1_missing_percent = (
        h1_present_d1_missing.mean()
        * 100
    )

    missingness_reduction_pp = (
        h1_missing_percent
        - d1_missing_percent
    )

    return {
        "base_feature": base_feature,
        "h1_feature": h1_feature,
        "d1_feature": d1_feature,
        "h1_missing_percent": (
            h1_missing_percent
        ),
        "d1_missing_percent": (
            d1_missing_percent
        ),
        "missingness_reduction_pp": (
            missingness_reduction_pp
        ),
        "both_missing_percent": (
            both_missing_percent
        ),
        "h1_missing_d1_present_percent": (
            h1_missing_d1_present_percent
        ),
        "both_present_percent": (
            both_present_percent
        ),
        "h1_present_d1_missing_percent": (
            h1_present_d1_missing_percent
        ),
    }


def build_q3_results(
    dataframe: pd.DataFrame,
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    """Build Q3 results for all matching h1/d1 pairs."""

    investigate_features = (
        get_investigate_features(
            catalog
        )
    )

    pairs = find_h1_d1_pairs(
        investigate_features
    )

    rows: list[
        dict[str, object]
    ] = []

    for (
        base_feature,
        h1_feature,
        d1_feature,
    ) in pairs:

        if (
            h1_feature not in dataframe.columns
            or d1_feature not in dataframe.columns
        ):
            continue

        row = analyze_h1_d1_pair(
            dataframe=dataframe,
            base_feature=base_feature,
            h1_feature=h1_feature,
            d1_feature=d1_feature,
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
            by="missingness_reduction_pp",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


def save_results(
    results: pd.DataFrame,
) -> None:
    """Save Q3 results."""

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
    """Print the complete Q3 result table."""

    print("=" * 110)
    print(
        "Q3 — H1 VS D1 MISSINGNESS PATTERNS"
    )
    print("=" * 110)

    print(
        f"h1/d1 pairs investigated: "
        f"{len(results)}"
    )

    print("\n" + "=" * 110)
    print(
        "ALL H1/D1 PAIRS"
    )
    print("=" * 110)

    if results.empty:
        print(
            "No matching h1/d1 pairs were found."
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

    print("\n" + "=" * 110)
    print(
        "RESULT SAVED"
    )
    print("=" * 110)

    print(
        OUTPUT_PATH
    )


def run_q3_analysis() -> pd.DataFrame:
    """Run Research Question 3 analysis."""

    dataframe = (
        load_training_data()
    )

    catalog = (
        load_feature_catalog()
    )

    results = (
        build_q3_results(
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
    run_q3_analysis()