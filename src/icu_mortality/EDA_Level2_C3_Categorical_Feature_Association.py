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

OUTPUT_TABLE_PATH = Path(
    "reports/tables/eda_level2_c3_categorical_feature_association.csv"
)

TARGET = "hospital_death"

INCLUDED_SEMANTIC_TYPES = {
    "CATEGORICAL",
    "CATEGORICAL_CODE",
    "BINARY",
}

EXCLUDED_STATUSES = {
    "DROP",
    "DROP_BASELINE",
    "TARGET",
}

MISSING_LABEL = "<MISSING>"

# Only report relatively strong associations.
CRAMERS_V_THRESHOLD = 0.70


# ============================================================
# LOAD DATA
# ============================================================

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw training data and Level 1 feature registry.
    """

    print("Loading training dataset...")
    df = pd.read_csv(DATA_PATH)

    print(f"Dataset shape: {df.shape}")

    print("Loading Level 1 feature registry...")
    registry = pd.read_csv(REGISTRY_PATH)

    print(f"Registry shape: {registry.shape}")

    return df, registry


# ============================================================
# VALIDATION
# ============================================================

def validate_inputs(
    df: pd.DataFrame,
    registry: pd.DataFrame,
) -> None:
    """
    Validate required registry columns.
    """

    required_registry_columns = {
        "feature",
        "semantic_type",
        "level1_status",
    }

    missing_columns = (
        required_registry_columns
        - set(registry.columns)
    )

    if missing_columns:
        raise ValueError(
            "Registry is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if TARGET not in df.columns:
        raise ValueError(
            f"Target '{TARGET}' was not found."
        )


# ============================================================
# SELECT FEATURES
# ============================================================

def select_categorical_features(
    df: pd.DataFrame,
    registry: pd.DataFrame,
) -> list[str]:
    """
    Select categorical-like features for association analysis.

    Included:
        CATEGORICAL
        CATEGORICAL_CODE
        BINARY

    Excluded:
        DROP
        DROP_BASELINE
        TARGET
    """

    selected = registry[
        registry["semantic_type"].isin(
            INCLUDED_SEMANTIC_TYPES
        )
        & ~registry["level1_status"].isin(
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
# PREPARE CATEGORY SERIES
# ============================================================

def prepare_category_series(
    series: pd.Series,
) -> pd.Series:
    """
    Prepare values for categorical comparison.

    Missing values are represented explicitly.

    This is for EDA only and does not define final
    preprocessing behavior.
    """

    prepared = series.astype("object").copy()

    prepared = prepared.where(
        prepared.notna(),
        MISSING_LABEL,
    )

    prepared = (
        prepared
        .astype(str)
        .str.strip()
    )

    return prepared


# ============================================================
# CRAMER'S V
# ============================================================

def cramers_v(
    series_1: pd.Series,
    series_2: pd.Series,
) -> float:
    """
    Calculate bias-corrected Cramer's V.

    Cramer's V ranges approximately from:

        0 -> little/no association
        1 -> very strong association

    This implementation uses the bias correction proposed
    for finite samples.
    """

    confusion_matrix = pd.crosstab(
        series_1,
        series_2,
    )

    observed = confusion_matrix.to_numpy(
        dtype=float
    )

    n = observed.sum()

    if n == 0:
        return np.nan

    row_totals = observed.sum(
        axis=1,
        keepdims=True,
    )

    column_totals = observed.sum(
        axis=0,
        keepdims=True,
    )

    expected = (
        row_totals
        @ column_totals
        / n
    )

    valid_mask = expected > 0

    chi2 = (
        (
            (observed - expected) ** 2
            / expected
        )[valid_mask]
        .sum()
    )

    phi2 = chi2 / n

    r, k = observed.shape

    if n <= 1:
        return np.nan

    phi2_corrected = max(
        0,
        phi2
        - (
            (k - 1)
            * (r - 1)
            / (n - 1)
        ),
    )

    r_corrected = (
        r
        - (
            (r - 1) ** 2
            / (n - 1)
        )
    )

    k_corrected = (
        k
        - (
            (k - 1) ** 2
            / (n - 1)
        )
    )

    denominator = min(
        k_corrected - 1,
        r_corrected - 1,
    )

    if denominator <= 0:
        return np.nan

    return float(
        np.sqrt(
            phi2_corrected
            / denominator
        )
    )


# ============================================================
# COMPUTE ASSOCIATIONS
# ============================================================

def compute_categorical_associations(
    df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """
    Compute pairwise Cramer's V for categorical-like features.

    Each feature pair appears only once.
    """

    prepared_data = {}

    print(
        "\nPreparing categorical features..."
    )

    for feature in features:
        prepared_data[feature] = (
            prepare_category_series(
                df[feature]
            )
        )

    rows = []

    total_pairs = (
        len(features)
        * (len(features) - 1)
        // 2
    )

    pair_counter = 0

    print(
        "Total feature pairs to evaluate: "
        f"{total_pairs}\n"
    )

    for i in range(len(features)):
        for j in range(i + 1, len(features)):

            feature_1 = features[i]
            feature_2 = features[j]

            pair_counter += 1

            value = cramers_v(
                prepared_data[feature_1],
                prepared_data[feature_2],
            )

            if pd.isna(value):
                continue

            if value >= CRAMERS_V_THRESHOLD:
                rows.append(
                    {
                        "feature_1": feature_1,
                        "feature_2": feature_2,
                        "cramers_v": value,
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=[
                "feature_1",
                "feature_2",
                "cramers_v",
            ]
        )

    result = pd.DataFrame(rows)

    result = result.sort_values(
        "cramers_v",
        ascending=False,
    ).reset_index(drop=True)

    return result


# ============================================================
# ROUND OUTPUT
# ============================================================

def round_output(
    association_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Round Cramer's V values for readability.
    """

    result = association_df.copy()

    if "cramers_v" in result.columns:
        result["cramers_v"] = (
            result["cramers_v"]
            .round(4)
        )

    return result


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
    features: list[str],
    association_df: pd.DataFrame,
) -> None:
    """
    Print compact C.3 summary.
    """

    total_pairs = (
        len(features)
        * (len(features) - 1)
        // 2
    )

    print(
        "\n" + "=" * 72
    )

    print(
        "EDA LEVEL 2 — C.3 "
        "CATEGORICAL FEATURE ASSOCIATION"
    )

    print(
        "=" * 72
    )

    print(
        "\nCategorical-like features analyzed: "
        f"{len(features)}"
    )

    print(
        "Total pairs evaluated: "
        f"{total_pairs}"
    )

    print(
        "Reporting threshold: "
        f"Cramer's V >= {CRAMERS_V_THRESHOLD}"
    )

    print(
        "Strong association pairs found: "
        f"{len(association_df)}"
    )

    if not association_df.empty:

        print(
            "\nStrongest associations:"
        )

        print(
            association_df.head(20).to_string(
                index=False
            )
        )

    print(
        "\nInterpretation reminder:"
        "\n- Cramer's V measures association between"
        "\n  categorical variables"
        "\n- Values closer to 1 indicate stronger association"
        "\n- Strong association may indicate redundancy"
        "\n  or related clinical information"
        "\n- Strong association does NOT automatically mean"
        "\n  one feature should be removed"
        "\n- Missing values are treated as an explicit"
        "\n  EDA category"
        "\n- Review feature semantics before making"
        "\n  preprocessing decisions"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df, registry = load_data()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_inputs(
        df=df,
        registry=registry,
    )

    # --------------------------------------------------------
    # Select categorical-like features
    # --------------------------------------------------------

    features = select_categorical_features(
        df=df,
        registry=registry,
    )

    print(
        "\nSelected categorical-like features: "
        f"{len(features)}"
    )

    if len(features) < 2:
        raise ValueError(
            "At least two categorical-like features "
            "are required."
        )

    # --------------------------------------------------------
    # Calculate associations
    # --------------------------------------------------------

    association_df = (
        compute_categorical_associations(
            df=df,
            features=features,
        )
    )

    # --------------------------------------------------------
    # Round
    # --------------------------------------------------------

    association_df = round_output(
        association_df
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_TABLE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    association_df.to_csv(
        OUTPUT_TABLE_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print_summary(
        features=features,
        association_df=association_df,
    )

    print(
        "\nSaved categorical association table:"
        f"\n{OUTPUT_TABLE_PATH}"
    )


if __name__ == "__main__":
    main()