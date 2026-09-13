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
    "reports/tables/eda_level2_b2_binary_target_mortality.csv"
)

OUTPUT_PLOT_DIR = Path(
    "reports/figures/eda_level2/b2_binary_target_mortality"
)

TARGET = "hospital_death"

INCLUDED_SEMANTIC_TYPE = "BINARY"

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
    Validate required columns and target values.
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

def select_binary_features(
    df: pd.DataFrame,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select features eligible for B.2.

    Included:
        BINARY

    Excluded:
        DROP
        DROP_BASELINE
        TARGET

    INVESTIGATE and KEEP_FOR_NOW features remain included.
    """

    selected = registry[
        (registry["semantic_type"] == INCLUDED_SEMANTIC_TYPE)
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
        "feature"
    ).reset_index(drop=True)

    return selected


# ============================================================
# ANALYZE SINGLE FEATURE
# ============================================================

def analyze_feature(
    df: pd.DataFrame,
    feature: str,
) -> pd.DataFrame:
    """
    Calculate mortality statistics for one binary feature.

    Expected values:
        0
        1

    Missing values are excluded from the 0/1 comparison.

    mortality_rate:
        death_count / patient_count * 100
    """

    temp_df = df[
        [
            feature,
            TARGET,
        ]
    ].copy()

    temp_df = temp_df.dropna(
        subset=[feature]
    )

    summary = (
        temp_df
        .groupby(feature)[TARGET]
        .agg(
            patient_count="count",
            death_count="sum",
        )
        .reset_index()
        .rename(
            columns={
                feature: "value"
            }
        )
    )

    summary["mortality_rate"] = (
        summary["death_count"]
        / summary["patient_count"]
        * 100
    )

    summary.insert(
        0,
        "feature",
        feature,
    )

    summary = summary.sort_values(
        "value"
    ).reset_index(drop=True)

    return summary


# ============================================================
# PLOT SINGLE FEATURE
# ============================================================

def plot_feature_mortality(
    feature_summary: pd.DataFrame,
    feature: str,
    overall_mortality_rate: float,
) -> Path:
    """
    Create a bar plot comparing mortality rate
    between binary values 0 and 1.
    """

    plot_df = feature_summary.copy()

    plot_df["value"] = (
        plot_df["value"]
        .astype(str)
    )

    plt.figure(
        figsize=(6, 5)
    )

    plt.bar(
        plot_df["value"],
        plot_df["mortality_rate"],
    )

    plt.axhline(
        y=overall_mortality_rate,
        linestyle="--",
        linewidth=1.5,
        label=(
            "Overall mortality "
            f"({overall_mortality_rate:.2f}%)"
        ),
    )

    plt.title(
        f"{feature} — Mortality Rate by Binary Value"
    )

    plt.xlabel(
        "Binary Value"
    )

    plt.ylabel(
        "Mortality Rate (%)"
    )

    plt.legend()

    plt.tight_layout()

    safe_feature_name = (
        feature
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )

    plot_path = (
        OUTPUT_PLOT_DIR
        / f"{safe_feature_name}_mortality_rate.png"
    )

    plt.savefig(
        plot_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    return plot_path


# ============================================================
# RUN ANALYSIS
# ============================================================

def run_analysis(
    df: pd.DataFrame,
    selected_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run B.2 binary-target analysis.
    """

    all_feature_summaries = []

    overall_mortality_rate = (
        df[TARGET].mean()
        * 100
    )

    total_features = len(
        selected_features
    )

    print(
        "\nBinary features selected: "
        f"{total_features}"
    )

    print(
        "Overall mortality rate: "
        f"{overall_mortality_rate:.2f}%\n"
    )

    OUTPUT_PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
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
            feature_summary = analyze_feature(
                df=df,
                feature=feature,
            )

            all_feature_summaries.append(
                feature_summary
            )

            plot_feature_mortality(
                feature_summary=feature_summary,
                feature=feature,
                overall_mortality_rate=overall_mortality_rate,
            )

        except Exception as exc:
            print(
                "WARNING: Failed to analyze "
                f"{feature}: {exc}"
            )

    if not all_feature_summaries:
        return pd.DataFrame(
            columns=[
                "feature",
                "value",
                "patient_count",
                "death_count",
                "mortality_rate",
            ]
        )

    summary_df = pd.concat(
        all_feature_summaries,
        ignore_index=True,
    )

    return summary_df


# ============================================================
# ROUND OUTPUT
# ============================================================

def round_output(
    summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Round mortality rate for easier inspection.
    """

    rounded_df = summary_df.copy()

    if "mortality_rate" in rounded_df.columns:
        rounded_df["mortality_rate"] = (
            rounded_df["mortality_rate"]
            .round(4)
        )

    return rounded_df


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
    summary_df: pd.DataFrame,
    selected_features: pd.DataFrame,
    df: pd.DataFrame,
) -> None:
    """
    Print compact B.2 execution summary.
    """

    overall_mortality_rate = (
        df[TARGET].mean()
        * 100
    )

    print(
        "\n" + "=" * 72
    )

    print(
        "EDA LEVEL 2 — B.2 "
        "BINARY FEATURE ↔ TARGET"
    )

    print(
        "=" * 72
    )

    print(
        "\nFeatures selected: "
        f"{len(selected_features)}"
    )

    print(
        "Rows in final table: "
        f"{len(summary_df)}"
    )

    print(
        "Overall mortality rate: "
        f"{overall_mortality_rate:.2f}%"
    )

    print(
        "\nInterpretation reminder:"
        "\n- Compare value 0 vs value 1"
        "\n- mortality_rate = death_count / patient_count"
        "\n- Always inspect patient_count together with mortality_rate"
        "\n- Rare value=1 groups may produce unstable mortality rates"
        "\n- High mortality rate does not automatically mean"
        "\n  strong predictive value"
        "\n- Association does not imply causation"
        "\n- No KEEP / DROP decision from B.2 alone"
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
    # Select binary features
    # --------------------------------------------------------

    selected_features = (
        select_binary_features(
            df,
            registry,
        )
    )

    # --------------------------------------------------------
    # Run B.2
    # --------------------------------------------------------

    summary_df = run_analysis(
        df,
        selected_features,
    )

    # --------------------------------------------------------
    # Round
    # --------------------------------------------------------

    summary_df = round_output(
        summary_df
    )

    # --------------------------------------------------------
    # Save table
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
        summary_df=summary_df,
        selected_features=selected_features,
        df=df,
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