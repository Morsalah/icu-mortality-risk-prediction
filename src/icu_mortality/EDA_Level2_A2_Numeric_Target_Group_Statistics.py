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
    "reports/tables/eda_level2_a2_numeric_target_group_statistics.csv"
)

TARGET = "hospital_death"

NUMERIC_SEMANTIC_TYPES = {
    "CONTINUOUS",
    "DISCRETE",
    "ORDINAL",
}

EXCLUDED_STATUSES = {
    "DROP",
    "DROP_BASELINE",
    "TARGET",
}


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
    Validate required columns and target structure.
    """

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' was not found."
        )

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

    target_values = set(
        df[TARGET]
        .dropna()
        .unique()
        .tolist()
    )

    if not target_values.issubset({0, 1}):
        raise ValueError(
            f"Unexpected target values: {target_values}"
        )


# ============================================================
# SELECT FEATURES
# ============================================================

def select_numeric_features(
    df: pd.DataFrame,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select features eligible for A.2.

    Included:
        CONTINUOUS
        DISCRETE
        ORDINAL

    Excluded:
        DROP
        DROP_BASELINE
        TARGET

    INVESTIGATE features remain included.
    """

    selected = registry[
        registry["semantic_type"].isin(
            NUMERIC_SEMANTIC_TYPES
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

    selected = selected.sort_values(
        [
            "semantic_type",
            "feature",
        ]
    ).reset_index(drop=True)

    return selected


# ============================================================
# GROUP STATISTICS
# ============================================================

def calculate_group_statistics(
    series: pd.Series,
) -> dict:
    """
    Calculate robust descriptive statistics
    for one target group.

    A.2 keeps only:
        Q1
        Median
        Q3
    """

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return {
            "q1": np.nan,
            "median": np.nan,
            "q3": np.nan,
        }

    return {
        "q1": values.quantile(0.25),
        "median": values.median(),
        "q3": values.quantile(0.75),
    }


# ============================================================
# CREATE SUMMARY ROW
# ============================================================

def create_summary_row(
    df: pd.DataFrame,
    registry_row: pd.Series,
) -> dict:
    """
    Create one A.2 summary row.

    Differences are always defined as:

        Died - Survived

    Therefore:
        Positive value -> Died is higher
        Negative value -> Died is lower
        Zero / near-zero -> similar values

    A.2 is descriptive only.
    """

    feature = registry_row["feature"]
    semantic_type = registry_row["semantic_type"]

    survived_series = df.loc[
        df[TARGET] == 0,
        feature,
    ]

    died_series = df.loc[
        df[TARGET] == 1,
        feature,
    ]

    survived_stats = calculate_group_statistics(
        survived_series
    )

    died_stats = calculate_group_statistics(
        died_series
    )

    q1_difference = (
        died_stats["q1"]
        - survived_stats["q1"]
    )

    median_difference = (
        died_stats["median"]
        - survived_stats["median"]
    )

    q3_difference = (
        died_stats["q3"]
        - survived_stats["q3"]
    )

    return {
        "feature": feature,
        "semantic_type": semantic_type,

        "survived_q1":
            survived_stats["q1"],

        "survived_median":
            survived_stats["median"],

        "survived_q3":
            survived_stats["q3"],

        "died_q1":
            died_stats["q1"],

        "died_median":
            died_stats["median"],

        "died_q3":
            died_stats["q3"],

        "q1_difference":
            q1_difference,

        "median_difference":
            median_difference,

        "q3_difference":
            q3_difference,
    }


# ============================================================
# RUN ANALYSIS
# ============================================================

def run_analysis(
    df: pd.DataFrame,
    selected_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run A.2 descriptive group statistics.
    """

    summary_rows = []

    total_features = len(
        selected_features
    )

    print(
        "\nNumeric / ordinal features selected: "
        f"{total_features}\n"
    )

    for index, registry_row in (
        selected_features.iterrows()
    ):

        feature = registry_row["feature"]

        print(
            f"[{index + 1}/{total_features}] "
            f"Analyzing {feature}"
        )

        try:
            summary_row = create_summary_row(
                df=df,
                registry_row=registry_row,
            )

            summary_rows.append(
                summary_row
            )

        except Exception as exc:
            print(
                "WARNING: Failed to analyze "
                f"{feature}: {exc}"
            )

    return pd.DataFrame(
        summary_rows
    )


# ============================================================
# ROUND OUTPUT
# ============================================================

def round_numeric_columns(
    summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Round numeric statistics for easier inspection.
    """

    rounded_df = summary_df.copy()

    numeric_columns = (
        rounded_df.select_dtypes(
            include="number"
        ).columns
    )

    rounded_df[numeric_columns] = (
        rounded_df[numeric_columns]
        .round(4)
    )

    return rounded_df


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
    summary_df: pd.DataFrame,
) -> None:
    """
    Print compact execution summary.
    """

    print(
        "\n" + "=" * 72
    )

    print(
        "EDA LEVEL 2 — A.2 "
        "NUMERIC FEATURE ↔ TARGET GROUP STATISTICS"
    )

    print(
        "=" * 72
    )

    print(
        "\nFeatures successfully analyzed: "
        f"{len(summary_df)}"
    )

    if summary_df.empty:
        print(
            "\nNo features were analyzed."
        )
        return

    print(
        "\nSemantic type distribution:"
    )

    print(
        summary_df[
            "semantic_type"
        ].value_counts()
    )

    print(
        "\nInterpretation reminder:"
        "\n- All differences are calculated as Died - Survived"
        "\n- q1_difference compares the lower quartile"
        "\n- median_difference compares the center"
        "\n- q3_difference compares the upper quartile"
        "\n- Positive difference -> Died is higher"
        "\n- Negative difference -> Died is lower"
        "\n- Similar signs across Q1 / Median / Q3 may indicate"
        "\n  a consistent directional shift"
        "\n- Mixed signs may indicate differences in spread or shape"
        "\n- A.2 is descriptive only"
        "\n- No p-values"
        "\n- No effect size"
        "\n- No feature ranking"
        "\n- No KEEP / DROP decision"
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
        df,
        registry,
    )

    # --------------------------------------------------------
    # Select eligible features
    # --------------------------------------------------------

    selected_features = (
        select_numeric_features(
            df,
            registry,
        )
    )

    # --------------------------------------------------------
    # Run A.2
    # --------------------------------------------------------

    summary_df = run_analysis(
        df,
        selected_features,
    )

    # --------------------------------------------------------
    # Round
    # --------------------------------------------------------

    summary_df = round_numeric_columns(
        summary_df
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_TABLE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_df.to_csv(
        OUTPUT_TABLE_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print_summary(
        summary_df
    )

    print(
        "\nSaved CSV:"
        f"\n{OUTPUT_TABLE_PATH}"
    )


if __name__ == "__main__":
    main()