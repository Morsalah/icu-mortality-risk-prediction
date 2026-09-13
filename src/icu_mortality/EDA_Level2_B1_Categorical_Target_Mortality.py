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
    "reports/tables/eda_level2_b1_categorical_target_mortality.csv"
)

OUTPUT_PLOT_DIR = Path(
    "reports/figures/eda_level2/b1_categorical_target_mortality"
)

TARGET = "hospital_death"

INCLUDED_SEMANTIC_TYPE = "CATEGORICAL"

EXCLUDED_STATUSES = {
    "DROP",
    "DROP_BASELINE",
    "TARGET",
}

MISSING_LABEL = "<MISSING>"


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
    Validate required input columns and target values.
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

def select_categorical_features(
    df: pd.DataFrame,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select features eligible for B.1.

    Included:
        CATEGORICAL

    Excluded:
        DROP
        DROP_BASELINE
        TARGET

    Binary and categorical-code features are intentionally
    handled later in B.2 / B.3.
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
# PREPARE CATEGORICAL SERIES
# ============================================================

def prepare_category_series(
    series: pd.Series,
) -> pd.Series:
    """
    Convert category values to strings and preserve
    missing values as an explicit category.
    """

    category_series = series.astype("object").copy()

    category_series = category_series.where(
        category_series.notna(),
        MISSING_LABEL,
    )

    category_series = (
        category_series
        .astype(str)
        .str.strip()
    )

    return category_series


# ============================================================
# ANALYZE SINGLE FEATURE
# ============================================================

def analyze_feature(
    df: pd.DataFrame,
    feature: str,
) -> pd.DataFrame:
    """
    Calculate mortality statistics for every category
    of one categorical feature.

    mortality_rate is calculated as:

        death_count / patient_count * 100
    """

    temp_df = pd.DataFrame(
        {
            "category": prepare_category_series(
                df[feature]
            ),
            TARGET: df[TARGET],
        }
    )

    summary = (
        temp_df
        .groupby(
            "category",
            dropna=False,
        )[TARGET]
        .agg(
            patient_count="count",
            death_count="sum",
        )
        .reset_index()
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
        [
            "mortality_rate",
            "patient_count",
        ],
        ascending=[
            False,
            False,
        ],
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
    Create a mortality-rate bar plot for one feature.

    X-axis:
        Category

    Y-axis:
        Mortality rate (%)

    Horizontal line:
        Overall mortality rate in the dataset
    """

    plot_df = feature_summary.copy()

    fig_width = max(
        8,
        len(plot_df) * 0.9,
    )

    plt.figure(
        figsize=(
            fig_width,
            6,
        )
    )

    plt.bar(
        plot_df["category"],
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
        f"{feature} — Mortality Rate by Category"
    )

    plt.xlabel(
        "Category"
    )

    plt.ylabel(
        "Mortality Rate (%)"
    )

    plt.xticks(
        rotation=45,
        ha="right",
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
    Run B.1 categorical-target analysis.
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
        "\nCategorical features selected: "
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
                "category",
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
    Print compact B.1 execution summary.
    """

    overall_mortality_rate = (
        df[TARGET].mean()
        * 100
    )

    print(
        "\n" + "=" * 72
    )

    print(
        "EDA LEVEL 2 — B.1 "
        "CATEGORICAL FEATURE ↔ TARGET"
    )

    print(
        "=" * 72
    )

    print(
        "\nFeatures successfully selected: "
        f"{len(selected_features)}"
    )

    print(
        "Rows in final category table: "
        f"{len(summary_df)}"
    )

    print(
        "Overall mortality rate: "
        f"{overall_mortality_rate:.2f}%"
    )

    print(
        "\nInterpretation reminder:"
        "\n- mortality_rate = death_count / patient_count"
        "\n- Compare mortality rates between categories"
        "\n- Always inspect patient_count before interpreting"
        "\n  a high or low mortality rate"
        "\n- <MISSING> is treated as an explicit EDA category"
        "\n- Different mortality rates show association,"
        "\n  not causation"
        "\n- Do not make KEEP / DROP decisions from B.1 alone"
        "\n- Binary features are handled separately in B.2"
        "\n- Categorical codes are handled separately in B.3"
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
    # Select eligible categorical features
    # --------------------------------------------------------

    selected_features = (
        select_categorical_features(
            df,
            registry,
        )
    )

    # --------------------------------------------------------
    # Run B.1
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



    # MortalityRate = (Death_Count/Patient_Count)×100