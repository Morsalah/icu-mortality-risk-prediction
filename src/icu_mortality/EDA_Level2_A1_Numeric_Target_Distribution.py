from pathlib import Path

import matplotlib.pyplot as plt
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
    "reports/tables/eda_level2_a1_numeric_target_distribution_summary.csv"
)

OUTPUT_PLOT_DIR = Path(
    "reports/figures/eda_level2/a1_numeric_target_distribution"
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

TARGET_LABELS = {
    0: "Survived",
    1: "Died",
}


# ============================================================
# LOAD DATA
# ============================================================

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the raw training dataset and the
    Level 1 Feature Decision Registry.
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
    Validate that:
    1. Target exists.
    2. Required registry columns exist.
    3. Target contains only 0 / 1.
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

    missing_columns = (
        required_registry_columns
        - set(registry.columns)
    )

    if missing_columns:
        raise ValueError(
            "Registry is missing required columns: "
            f"{sorted(missing_columns)}"
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
# SELECT NUMERIC FEATURES
# ============================================================

def select_numeric_features(
    df: pd.DataFrame,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select numeric / ordered features for A.1.

    Included semantic types:
        CONTINUOUS
        DISCRETE
        ORDINAL

    Excluded statuses:
        DROP
        DROP_BASELINE
        TARGET

    INVESTIGATE features are intentionally kept.
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
        selected["feature"].isin(df.columns)
    ].copy()

    selected = selected.sort_values(
        ["semantic_type", "feature"]
    ).reset_index(drop=True)

    return selected


# ============================================================
# HISTOGRAM BINS
# ============================================================

def get_histogram_bins(
    values: np.ndarray,
    semantic_type: str,
):
    """
    Choose histogram bins.

    ORDINAL features:
        Use discrete-style bins when possible.

    CONTINUOUS / DISCRETE:
        Use 30 bins.
    """

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return 30

    if semantic_type == "ORDINAL":

        unique_values = np.sort(
            np.unique(values)
        )

        # Single-value feature
        if len(unique_values) == 1:

            value = unique_values[0]

            return np.array([
                value - 0.5,
                value + 0.5,
            ])

        # Small ordinal scale
        if len(unique_values) <= 20:

            differences = np.diff(
                unique_values
            )

            positive_differences = (
                differences[
                    differences > 0
                ]
            )

            if len(positive_differences) > 0:
                step = (
                    positive_differences.min()
                )
            else:
                step = 1

            return np.arange(
                unique_values.min()
                - step / 2,

                unique_values.max()
                + step,

                step,
            )

    return 30


# ============================================================
# PLOT FEATURE
# ============================================================

def plot_feature_distribution(
    df: pd.DataFrame,
    feature: str,
    semantic_type: str,
    output_path: Path,
) -> None:
    """
    Plot the full feature distribution for:

        hospital_death = 0 → Survived
        hospital_death = 1 → Died

    Important:
        - Full observed range is displayed.
        - No observations are removed.
        - No clipping is performed.
        - No transformation is applied.
    """

    survived = (
        df.loc[
            df[TARGET] == 0,
            feature,
        ]
        .dropna()
        .astype(float)
        .to_numpy()
    )

    died = (
        df.loc[
            df[TARGET] == 1,
            feature,
        ]
        .dropna()
        .astype(float)
        .to_numpy()
    )

    combined = np.concatenate(
        [
            survived,
            died,
        ]
    )

    combined = combined[
        np.isfinite(combined)
    ]

    if len(combined) == 0:

        print(
            f"Skipping {feature}: "
            "no valid numeric observations."
        )

        return

    bins = get_histogram_bins(
        combined,
        semantic_type,
    )

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    # Survived
    ax.hist(
        survived,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2,
        label=TARGET_LABELS[0],
    )

    # Died
    ax.hist(
        died,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2,
        label=TARGET_LABELS[1],
    )

    # --------------------------------------------------------
    # Titles and labels
    # --------------------------------------------------------

    ax.set_title(
        (
            f"{feature}\n"
            "Distribution by Hospital Mortality"
        )
    )

    ax.set_xlabel(
        feature
    )

    ax.set_ylabel(
        "Density"
    )

    ax.legend()

    ax.grid(
        alpha=0.2
    )

    # --------------------------------------------------------
    # Feature information
    # --------------------------------------------------------

    ax.text(
        0.01,
        0.98,
        (
            f"Semantic type: {semantic_type}\n"
            f"Survived observations: "
            f"{len(survived):,}\n"
            f"Died observations: "
            f"{len(died):,}"
        ),
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=9,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# CREATE SUMMARY ROW
# ============================================================

def create_summary_row(
    df: pd.DataFrame,
    registry_row: pd.Series,
    plot_filename: str,
) -> dict:
    """
    Create a compact summary row for the feature.

    A.1 focuses on visual distribution comparison.

    We do not calculate here:
        - group means
        - group medians
        - statistical significance
        - effect size
        - feature importance
    """

    feature = registry_row[
        "feature"
    ]

    survived_mask = (
        df[TARGET] == 0
    )

    died_mask = (
        df[TARGET] == 1
    )

    survived_total = int(
        survived_mask.sum()
    )

    died_total = int(
        died_mask.sum()
    )

    survived_nonmissing = int(
        df.loc[
            survived_mask,
            feature,
        ]
        .notna()
        .sum()
    )

    died_nonmissing = int(
        df.loc[
            died_mask,
            feature,
        ]
        .notna()
        .sum()
    )

    survived_missing_percent = (
        100
        * (
            survived_total
            - survived_nonmissing
        )
        / survived_total
        if survived_total > 0
        else np.nan
    )

    died_missing_percent = (
        100
        * (
            died_total
            - died_nonmissing
        )
        / died_total
        if died_total > 0
        else np.nan
    )

    return {
        "feature": feature,

        "semantic_type":
            registry_row[
                "semantic_type"
            ],

        "level1_status":
            registry_row[
                "level1_status"
            ],

        "survived_nonmissing":
            survived_nonmissing,

        "died_nonmissing":
            died_nonmissing,

        "survived_missing_percent":
            round(
                survived_missing_percent,
                2,
            ),

        "died_missing_percent":
            round(
                died_missing_percent,
                2,
            ),

        "plot_file":
            plot_filename,
    }


# ============================================================
# RUN ANALYSIS
# ============================================================

def run_analysis(
    df: pd.DataFrame,
    selected_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run A.1 analysis for all selected features.
    """

    OUTPUT_PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

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

        feature = registry_row[
            "feature"
        ]

        semantic_type = registry_row[
            "semantic_type"
        ]

        print(
            f"[{index + 1}/{total_features}] "
            f"Analyzing {feature}"
        )

        plot_filename = (
            f"{feature}"
            "_target_distribution.png"
        )

        plot_path = (
            OUTPUT_PLOT_DIR
            / plot_filename
        )

        try:

            # ----------------------------------------------
            # Create plot
            # ----------------------------------------------

            plot_feature_distribution(
                df=df,
                feature=feature,
                semantic_type=semantic_type,
                output_path=plot_path,
            )

            # ----------------------------------------------
            # Create summary row
            # ----------------------------------------------

            summary_row = (
                create_summary_row(
                    df=df,
                    registry_row=registry_row,
                    plot_filename=plot_filename,
                )
            )

            summary_rows.append(
                summary_row
            )

        except Exception as exc:

            print(
                "WARNING: Failed to analyze "
                f"{feature}: {exc}"
            )

    summary_df = pd.DataFrame(
        summary_rows
    )

    return summary_df


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
    summary_df: pd.DataFrame,
) -> None:
    """
    Print a compact execution summary.
    """

    print(
        "\n" + "=" * 70
    )

    print(
        "EDA LEVEL 2 — A.1 "
        "NUMERIC FEATURE ↔ TARGET DISTRIBUTION"
    )

    print(
        "=" * 70
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

    # --------------------------------------------------------
    # Semantic type summary
    # --------------------------------------------------------

    print(
        "\nSemantic type distribution:"
    )

    print(
        summary_df[
            "semantic_type"
        ].value_counts()
    )

    # --------------------------------------------------------
    # Level 1 status summary
    # --------------------------------------------------------

    print(
        "\nLevel 1 status distribution:"
    )

    print(
        summary_df[
            "level1_status"
        ].value_counts()
    )

    print(
        "\nImportant:"
        "\n- One plot is created per feature."
        "\n- Full observed range is displayed."
        "\n- No observations are removed."
        "\n- No clipping is performed."
        "\n- No transformations are performed."
        "\n- No feature decisions are made here."
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
    # Select numeric features
    # --------------------------------------------------------

    selected_features = (
        select_numeric_features(
            df,
            registry,
        )
    )

    # --------------------------------------------------------
    # Run analysis
    # --------------------------------------------------------

    summary_df = run_analysis(
        df,
        selected_features,
    )

    # --------------------------------------------------------
    # Save CSV
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

    print(
        "\nSaved plots:"
        f"\n{OUTPUT_PLOT_DIR}"
    )


if __name__ == "__main__":
    main()