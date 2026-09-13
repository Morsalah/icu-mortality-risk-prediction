from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = Path(
    "data/raw/training_v2.csv"
)

REGISTRY_PATH = Path(
    "reports/tables/eda_level1_feature_decision_registry.csv"
)

C1_CORRELATION_PATH = Path(
    "reports/tables/eda_level2_c1_numeric_feature_correlation.csv"
)

OUTPUT_EXACT_DUPLICATES_PATH = Path(
    "reports/tables/eda_level2_c2_exact_duplicate_features.csv"
)

OUTPUT_NEAR_DUPLICATES_PATH = Path(
    "reports/tables/eda_level2_c2_near_duplicate_features.csv"
)

TARGET = "hospital_death"

EXCLUDED_STATUSES = {
    "DROP",
    "DROP_BASELINE",
    "TARGET",
}

# A pair must match in at least this percentage
# of comparable rows to be reported as near-duplicate.
NEAR_DUPLICATE_THRESHOLD = 99.0

# Numerical tolerance used when comparing floating-point values.
RTOL = 1e-05
ATOL = 1e-08


# ============================================================
# LOAD DATA
# ============================================================

def load_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Load:
        - raw training data
        - Level 1 feature registry
        - C.1 high-correlation candidate pairs
    """

    print("Loading training dataset...")
    df = pd.read_csv(DATA_PATH)

    print(f"Dataset shape: {df.shape}")

    print("Loading Level 1 feature registry...")
    registry = pd.read_csv(REGISTRY_PATH)

    print(f"Registry shape: {registry.shape}")

    print("Loading C.1 correlation pairs...")

    if C1_CORRELATION_PATH.exists():
        correlation_pairs = pd.read_csv(
            C1_CORRELATION_PATH
        )
    else:
        correlation_pairs = pd.DataFrame(
            columns=[
                "feature_1",
                "feature_2",
                "correlation",
                "absolute_correlation",
            ]
        )

    print(
        "C.1 candidate pairs: "
        f"{len(correlation_pairs)}"
    )

    return (
        df,
        registry,
        correlation_pairs,
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_inputs(
    df: pd.DataFrame,
    registry: pd.DataFrame,
) -> None:
    """
    Validate required columns.
    """

    required_registry_columns = {
        "feature",
        "semantic_type",
        "level1_status",
    }

    missing_registry_columns = (
        required_registry_columns
        - set(registry.columns)
    )

    if missing_registry_columns:
        raise ValueError(
            "Registry is missing required columns: "
            f"{sorted(missing_registry_columns)}"
        )

    if TARGET not in df.columns:
        raise ValueError(
            f"Target '{TARGET}' was not found."
        )


# ============================================================
# SELECT FEATURES FOR EXACT DUPLICATE CHECK
# ============================================================

def select_relevant_features(
    df: pd.DataFrame,
    registry: pd.DataFrame,
) -> list[str]:
    """
    Select all features that remain relevant after Level 1.

    Exact duplicates may occur in both numerical and
    categorical features, therefore this step is not limited
    to numeric semantic types.

    Excluded:
        DROP
        DROP_BASELINE
        TARGET
    """

    selected = registry[
        ~registry["level1_status"].isin(
            EXCLUDED_STATUSES
        )
    ].copy()

    selected = selected[
        selected["feature"].isin(
            df.columns
        )
    ]

    features = (
        selected["feature"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    return features


# ============================================================
# EXACT DUPLICATE HELPERS
# ============================================================

def build_column_fingerprints(
    df: pd.DataFrame,
    features: list[str],
) -> dict[int, list[str]]:
    """
    Generate a lightweight fingerprint for every column.

    Instead of directly comparing every possible pair
    row-by-row, columns are first grouped by a hash.

    Columns with different fingerprints cannot be exact
    duplicates.

    Candidate columns with identical fingerprints are later
    verified using an exact comparison.
    """

    fingerprint_groups = {}

    for feature in features:

        series = df[feature]

        # Hash values including the index.
        hashed = pd.util.hash_pandas_object(
            series,
            index=True,
        )

        fingerprint = int(
            hashed.sum()
        )

        fingerprint_groups.setdefault(
            fingerprint,
            [],
        ).append(feature)

    return fingerprint_groups


def series_exactly_equal(
    series_1: pd.Series,
    series_2: pd.Series,
) -> bool:
    """
    Check whether two columns contain exactly the same
    information.

    Missing values in the same row are considered equal.
    """

    if len(series_1) != len(series_2):
        return False

    both_missing = (
        series_1.isna()
        & series_2.isna()
    )

    both_present = (
        series_1.notna()
        & series_2.notna()
    )

    missing_pattern_equal = (
        (
            series_1.isna()
            == series_2.isna()
        )
        .all()
    )

    if not missing_pattern_equal:
        return False

    if both_present.sum() == 0:
        return True

    values_equal = (
        series_1[both_present]
        .astype(str)
        .to_numpy()
        ==
        series_2[both_present]
        .astype(str)
        .to_numpy()
    )

    return bool(
        values_equal.all()
    )


# ============================================================
# FIND EXACT DUPLICATES
# ============================================================

def find_exact_duplicates(
    df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """
    Find columns containing exactly the same information.

    Hashing is used first for efficiency.
    Candidate hash collisions are then verified explicitly.
    """

    fingerprint_groups = (
        build_column_fingerprints(
            df=df,
            features=features,
        )
    )

    rows = []

    for feature_group in fingerprint_groups.values():

        if len(feature_group) < 2:
            continue

        for i in range(len(feature_group)):
            for j in range(i + 1, len(feature_group)):

                feature_1 = feature_group[i]
                feature_2 = feature_group[j]

                if series_exactly_equal(
                    df[feature_1],
                    df[feature_2],
                ):
                    rows.append(
                        {
                            "feature_1": feature_1,
                            "feature_2": feature_2,
                            "match_percent": 100.0,
                        }
                    )

    if not rows:
        return pd.DataFrame(
            columns=[
                "feature_1",
                "feature_2",
                "match_percent",
            ]
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "feature_1",
                "feature_2",
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# COMPARE NUMERIC PAIR
# ============================================================

def compare_numeric_pair(
    df: pd.DataFrame,
    feature_1: str,
    feature_2: str,
) -> dict:
    """
    Compare two numerical features row-by-row.

    Only rows where both features have observed values are
    used for value agreement.

    np.isclose() is used because floating-point values should
    not be expected to have perfect bit-level equality.

    Missing-pattern agreement is calculated separately.
    """

    series_1 = pd.to_numeric(
        df[feature_1],
        errors="coerce",
    )

    series_2 = pd.to_numeric(
        df[feature_2],
        errors="coerce",
    )

    comparable_mask = (
        series_1.notna()
        & series_2.notna()
    )

    comparable_rows = int(
        comparable_mask.sum()
    )

    if comparable_rows == 0:
        matching_rows = 0
        match_percent = np.nan

    else:

        values_1 = series_1[
            comparable_mask
        ].to_numpy()

        values_2 = series_2[
            comparable_mask
        ].to_numpy()

        matches = np.isclose(
            values_1,
            values_2,
            rtol=RTOL,
            atol=ATOL,
            equal_nan=False,
        )

        matching_rows = int(
            matches.sum()
        )

        match_percent = (
            matching_rows
            / comparable_rows
            * 100
        )

    missing_pattern_matches = (
        series_1.isna()
        == series_2.isna()
    )

    missing_pattern_match_percent = (
        missing_pattern_matches.mean()
        * 100
    )

    return {
        "comparable_rows": comparable_rows,
        "matching_rows": matching_rows,
        "match_percent": match_percent,
        "missing_pattern_match_percent":
            missing_pattern_match_percent,
    }


# ============================================================
# FIND NEAR DUPLICATES
# ============================================================

def find_near_duplicates(
    df: pd.DataFrame,
    correlation_pairs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Investigate C.1 high-correlation pairs for actual
    row-level agreement.

    Correlation tells us whether two variables move together.

    C.2 asks whether they are actually storing nearly the
    same values.
    """

    if correlation_pairs.empty:
        return pd.DataFrame(
            columns=[
                "feature_1",
                "feature_2",
                "correlation",
                "comparable_rows",
                "matching_rows",
                "match_percent",
                "missing_pattern_match_percent",
            ]
        )

    rows = []

    for _, pair in correlation_pairs.iterrows():

        feature_1 = pair["feature_1"]
        feature_2 = pair["feature_2"]

        if (
            feature_1 not in df.columns
            or feature_2 not in df.columns
        ):
            continue

        comparison = compare_numeric_pair(
            df=df,
            feature_1=feature_1,
            feature_2=feature_2,
        )

        match_percent = comparison[
            "match_percent"
        ]

        if (
            pd.notna(match_percent)
            and match_percent
            >= NEAR_DUPLICATE_THRESHOLD
        ):

            rows.append(
                {
                    "feature_1": feature_1,
                    "feature_2": feature_2,
                    "correlation": pair[
                        "correlation"
                    ],
                    "comparable_rows":
                        comparison[
                            "comparable_rows"
                        ],
                    "matching_rows":
                        comparison[
                            "matching_rows"
                        ],
                    "match_percent":
                        match_percent,
                    "missing_pattern_match_percent":
                        comparison[
                            "missing_pattern_match_percent"
                        ],
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "feature_1",
                "feature_2",
                "correlation",
                "comparable_rows",
                "matching_rows",
                "match_percent",
                "missing_pattern_match_percent",
            ]
        )

    near_duplicates = pd.DataFrame(rows)

    near_duplicates = (
        near_duplicates
        .sort_values(
            [
                "match_percent",
                "correlation",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    return near_duplicates


# ============================================================
# ROUND OUTPUTS
# ============================================================

def round_outputs(
    exact_duplicates: pd.DataFrame,
    near_duplicates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Round percentages and correlations for readability.
    """

    exact_duplicates = (
        exact_duplicates.copy()
    )

    near_duplicates = (
        near_duplicates.copy()
    )

    columns_to_round = [
        "correlation",
        "match_percent",
        "missing_pattern_match_percent",
    ]

    for column in columns_to_round:

        if column in near_duplicates.columns:
            near_duplicates[column] = (
                near_duplicates[column]
                .round(4)
            )

    return (
        exact_duplicates,
        near_duplicates,
    )


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
    features: list[str],
    correlation_pairs: pd.DataFrame,
    exact_duplicates: pd.DataFrame,
    near_duplicates: pd.DataFrame,
) -> None:
    """
    Print compact C.2 summary.
    """

    print(
        "\n" + "=" * 72
    )

    print(
        "EDA LEVEL 2 — C.2 "
        "DUPLICATE / NEAR-DUPLICATE FEATURES"
    )

    print(
        "=" * 72
    )

    print(
        "\nFeatures checked for exact duplicates: "
        f"{len(features)}"
    )

    print(
        "C.1 candidate pairs checked for "
        "near duplicates: "
        f"{len(correlation_pairs)}"
    )

    print(
        "Exact duplicate pairs found: "
        f"{len(exact_duplicates)}"
    )

    print(
        "Near-duplicate threshold: "
        f"{NEAR_DUPLICATE_THRESHOLD:.2f}%"
    )

    print(
        "Near-duplicate pairs found: "
        f"{len(near_duplicates)}"
    )

    if not exact_duplicates.empty:

        print(
            "\nExact duplicates:"
        )

        print(
            exact_duplicates.to_string(
                index=False
            )
        )

    if not near_duplicates.empty:

        print(
            "\nNear duplicates:"
        )

        print(
            near_duplicates.head(20).to_string(
                index=False
            )
        )

    print(
        "\nInterpretation reminder:"
        "\n- Exact duplicate = same information row-by-row"
        "\n- Near duplicate = values agree in almost all"
        "\n  comparable rows"
        "\n- High correlation alone does NOT mean duplicate"
        "\n- comparable_rows includes only rows where both"
        "\n  numeric values are observed"
        "\n- missing_pattern_match_percent is reported"
        "\n  separately"
        "\n- Duplicate candidate does NOT automatically mean DROP"
        "\n- Review semantic meaning before removing features"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    (
        df,
        registry,
        correlation_pairs,
    ) = load_data()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_inputs(
        df=df,
        registry=registry,
    )

    # --------------------------------------------------------
    # Select features for exact duplicate check
    # --------------------------------------------------------

    features = select_relevant_features(
        df=df,
        registry=registry,
    )

    print(
        "\nRelevant features selected: "
        f"{len(features)}"
    )

    # --------------------------------------------------------
    # Exact duplicates
    # --------------------------------------------------------

    print(
        "\nSearching for exact duplicate features..."
    )

    exact_duplicates = find_exact_duplicates(
        df=df,
        features=features,
    )

    # --------------------------------------------------------
    # Near duplicates
    # --------------------------------------------------------

    print(
        "\nChecking C.1 high-correlation pairs "
        "for near duplicates..."
    )

    near_duplicates = find_near_duplicates(
        df=df,
        correlation_pairs=correlation_pairs,
    )

    # --------------------------------------------------------
    # Round outputs
    # --------------------------------------------------------

    (
        exact_duplicates,
        near_duplicates,
    ) = round_outputs(
        exact_duplicates=exact_duplicates,
        near_duplicates=near_duplicates,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_EXACT_DUPLICATES_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    exact_duplicates.to_csv(
        OUTPUT_EXACT_DUPLICATES_PATH,
        index=False,
    )

    near_duplicates.to_csv(
        OUTPUT_NEAR_DUPLICATES_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_summary(
        features=features,
        correlation_pairs=correlation_pairs,
        exact_duplicates=exact_duplicates,
        near_duplicates=near_duplicates,
    )

    print(
        "\nSaved exact duplicates:"
        f"\n{OUTPUT_EXACT_DUPLICATES_PATH}"
    )

    print(
        "\nSaved near duplicates:"
        f"\n{OUTPUT_NEAR_DUPLICATES_PATH}"
    )


if __name__ == "__main__":
    main()