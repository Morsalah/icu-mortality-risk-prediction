from pathlib import Path

import matplotlib.pyplot as plt
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
    "reports/tables/eda_level2_c1_numeric_feature_correlation.csv"
)

OUTPUT_HEATMAP_PATH = Path(
    "reports/figures/eda_level2/c1_numeric_feature_correlation/"
    "high_correlation_heatmap.png"
)

TARGET = "hospital_death"

INCLUDED_SEMANTIC_TYPES = {
    "CONTINUOUS",
    "DISCRETE",
    "ORDINAL",
}

EXCLUDED_STATUSES = {
    "DROP",
    "DROP_BASELINE",
    "TARGET",
}

CORRELATION_THRESHOLD = 0.85


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
    Validate required input columns.
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
            f"Target column '{TARGET}' was not found."
        )


# ============================================================
# SELECT NUMERIC FEATURES
# ============================================================

def select_numeric_features(
    df: pd.DataFrame,
    registry: pd.DataFrame,
) -> list[str]:
    """
    Select numeric/ordered features eligible for correlation.

    Included semantic types:
        CONTINUOUS
        DISCRETE
        ORDINAL

    Excluded statuses:
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
    ].copy()

    features = (
        selected["feature"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    return features


# ============================================================
# COMPUTE CORRELATION MATRIX
# ============================================================

def compute_correlation_matrix(
    df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """
    Compute Pearson correlation matrix.

    Pandas correlation uses pairwise complete observations,
    meaning that for each feature pair, rows with missing
    values in either feature are ignored.
    """

    numeric_df = df[
        features
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    correlation_matrix = numeric_df.corr(
        method="pearson"
    )

    return correlation_matrix


# ============================================================
# EXTRACT HIGH-CORRELATION PAIRS
# ============================================================

def extract_high_correlation_pairs(
    correlation_matrix: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """
    Convert the correlation matrix into a compact table
    containing each feature pair only once.

    Keep only pairs where:

        abs(correlation) >= threshold
    """

    features = correlation_matrix.columns.tolist()

    rows = []

    for i in range(len(features)):
        for j in range(i + 1, len(features)):

            feature_1 = features[i]
            feature_2 = features[j]

            correlation = correlation_matrix.loc[
                feature_1,
                feature_2,
            ]

            if pd.isna(correlation):
                continue

            if abs(correlation) >= threshold:
                rows.append(
                    {
                        "feature_1": feature_1,
                        "feature_2": feature_2,
                        "correlation": correlation,
                        "absolute_correlation": abs(
                            correlation
                        ),
                    }
                )

    pairs_df = pd.DataFrame(rows)

    if pairs_df.empty:
        return pd.DataFrame(
            columns=[
                "feature_1",
                "feature_2",
                "correlation",
                "absolute_correlation",
            ]
        )

    pairs_df = pairs_df.sort_values(
        "absolute_correlation",
        ascending=False,
    ).reset_index(drop=True)

    return pairs_df


# ============================================================
# CREATE HEATMAP
# ============================================================

def create_high_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
    pairs_df: pd.DataFrame,
) -> None:
    """
    Create a heatmap containing only features that appear
    in at least one high-correlation pair.

    This avoids generating a huge unreadable heatmap
    containing every numeric feature.
    """

    if pairs_df.empty:
        print(
            "No high-correlation pairs found. "
            "Heatmap will not be created."
        )
        return

    high_corr_features = sorted(
        set(
            pairs_df["feature_1"].tolist()
            + pairs_df["feature_2"].tolist()
        )
    )

    heatmap_matrix = correlation_matrix.loc[
        high_corr_features,
        high_corr_features,
    ]

    number_of_features = len(
        high_corr_features
    )

    figure_size = max(
        8,
        number_of_features * 0.55,
    )

    OUTPUT_HEATMAP_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(
            figure_size,
            figure_size,
        )
    )

    image = plt.imshow(
        heatmap_matrix,
        aspect="auto",
        vmin=-1,
        vmax=1,
    )

    plt.colorbar(
        image,
        label="Pearson Correlation",
    )

    plt.xticks(
        range(number_of_features),
        high_corr_features,
        rotation=90,
    )

    plt.yticks(
        range(number_of_features),
        high_corr_features,
    )

    plt.title(
        "High-Correlation Numeric Features"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_HEATMAP_PATH,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# ROUND OUTPUT
# ============================================================

def round_output(
    pairs_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Round correlation values for easier inspection.
    """

    rounded_df = pairs_df.copy()

    for column in [
        "correlation",
        "absolute_correlation",
    ]:
        if column in rounded_df.columns:
            rounded_df[column] = (
                rounded_df[column]
                .round(4)
            )

    return rounded_df


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
    features: list[str],
    pairs_df: pd.DataFrame,
) -> None:
    """
    Print compact execution summary.
    """

    print(
        "\n" + "=" * 72
    )

    print(
        "EDA LEVEL 2 — C.1 "
        "NUMERIC FEATURE CORRELATION"
    )

    print(
        "=" * 72
    )

    print(
        "\nNumeric features analyzed: "
        f"{len(features)}"
    )

    print(
        "Correlation threshold: "
        f"|r| >= {CORRELATION_THRESHOLD}"
    )

    print(
        "High-correlation pairs found: "
        f"{len(pairs_df)}"
    )

    if not pairs_df.empty:
        print(
            "\nTop high-correlation pairs:"
        )

        print(
            pairs_df.head(20).to_string(
                index=False
            )
        )

    print(
        "\nInterpretation reminder:"
        "\n- High correlation suggests possible redundancy"
        "\n- High correlation does NOT automatically mean"
        "\n  one feature should be dropped"
        "\n- Correlation is calculated pairwise using"
        "\n  available non-missing observations"
        "\n- Pearson correlation mainly captures linear"
        "\n  relationships"
        "\n- Strong correlations may be expected for related"
        "\n  measurements such as min/max or different"
        "\n  time windows"
        "\n- Final feature-removal decisions should consider"
        "\n  semantics, missingness, model behavior, and"
        "\n  generalization"
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
    # Select numeric features
    # --------------------------------------------------------

    features = select_numeric_features(
        df=df,
        registry=registry,
    )

    print(
        "\nSelected numeric features: "
        f"{len(features)}"
    )

    if len(features) < 2:
        raise ValueError(
            "At least two numeric features are required "
            "for correlation analysis."
        )

    # --------------------------------------------------------
    # Correlation matrix
    # --------------------------------------------------------

    correlation_matrix = (
        compute_correlation_matrix(
            df=df,
            features=features,
        )
    )

    # --------------------------------------------------------
    # Extract strong pairs
    # --------------------------------------------------------

    pairs_df = (
        extract_high_correlation_pairs(
            correlation_matrix=correlation_matrix,
            threshold=CORRELATION_THRESHOLD,
        )
    )

    # --------------------------------------------------------
    # Round table output
    # --------------------------------------------------------

    pairs_df = round_output(
        pairs_df
    )

    # --------------------------------------------------------
    # Save compact pair table
    # --------------------------------------------------------

    OUTPUT_TABLE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pairs_df.to_csv(
        OUTPUT_TABLE_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Create focused heatmap
    # --------------------------------------------------------

    create_high_correlation_heatmap(
        correlation_matrix=correlation_matrix,
        pairs_df=pairs_df,
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print_summary(
        features=features,
        pairs_df=pairs_df,
    )

    print(
        "\nSaved pair table:"
        f"\n{OUTPUT_TABLE_PATH}"
    )

    if not pairs_df.empty:
        print(
            "\nSaved heatmap:"
            f"\n{OUTPUT_HEATMAP_PATH}"
        )


if __name__ == "__main__":
    main()