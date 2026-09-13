from pathlib import Path

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
    "reports/tables/eda_level2_b3_categorical_code_target_mortality.csv"
)

OUTPUT_SUPPORT_SUMMARY_PATH = Path(
    "reports/tables/eda_level2_b3_categorical_code_support_summary.csv"
)

TARGET = "hospital_death"

INCLUDED_SEMANTIC_TYPE = "CATEGORICAL_CODE"

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
    Validate required input columns.
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
# SELECT CATEGORICAL-CODE FEATURES
# ============================================================

def select_categorical_code_features(
    df: pd.DataFrame,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select features classified as CATEGORICAL_CODE.

    Numeric storage does not imply that these features
    are continuous numerical variables.

    Example:
        apache_2_diagnosis
        apache_3j_diagnosis

    The numeric values represent category codes.
    """

    selected = registry[
        (registry["semantic_type"] == INCLUDED_SEMANTIC_TYPE)
        & ~registry["level1_status"].isin(
            EXCLUDED_STATUSES
        )
    ].copy()

    selected = selected[
        selected["feature"].isin(df.columns)
    ].copy()

    selected = selected.sort_values(
        "feature"
    ).reset_index(drop=True)

    return selected


# ============================================================
# PREPARE CODE SERIES
# ============================================================

def prepare_code_series(
    series: pd.Series,
) -> pd.Series:
    """
    Prepare diagnosis/category codes for grouping.

    Missing values are kept explicitly as <MISSING>.

    Integer-like numeric codes are displayed without '.0'
    when possible.

    No rare-category grouping is performed here.
    """

    def format_code(value):
        if pd.isna(value):
            return MISSING_LABEL

        if isinstance(value, (int, float)):
            numeric_value = float(value)

            if numeric_value.is_integer():
                return str(int(numeric_value))

        return str(value).strip()

    return series.map(format_code)


# ============================================================
# ANALYZE SINGLE FEATURE
# ============================================================

def analyze_feature(
    df: pd.DataFrame,
    feature: str,
) -> pd.DataFrame:
    """
    Calculate support and mortality rate for every code.

    mortality_rate =
        death_count / patient_count * 100

    Important:
    No minimum-support threshold is applied at this stage.
    """

    temp_df = pd.DataFrame(
        {
            "diagnosis_code": prepare_code_series(
                df[feature]
            ),
            TARGET: df[TARGET],
        }
    )

    summary = (
        temp_df
        .groupby(
            "diagnosis_code",
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

    # Most supported codes first.
    #
    # This is intentional:
    # sorting only by mortality rate would put tiny,
    # unstable categories at the top of the table.
    summary = summary.sort_values(
        [
            "patient_count",
            "mortality_rate",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(drop=True)

    return summary


# ============================================================
# SUPPORT SUMMARY
# ============================================================

def build_support_summary(
    detailed_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize the support distribution of codes for each
    categorical-code feature.

    This table helps us choose a sensible support threshold
    later instead of choosing one arbitrarily now.
    """

    rows = []

    for feature, feature_df in detailed_summary.groupby(
        "feature"
    ):

        # The missing category is useful in the detailed table,
        # but it should not influence the distribution of
        # actual diagnosis-code support.
        actual_codes = feature_df[
            feature_df["diagnosis_code"] != MISSING_LABEL
        ].copy()

        if actual_codes.empty:
            continue

        support = actual_codes["patient_count"]

        rows.append(
            {
                "feature": feature,
                "number_of_codes": len(actual_codes),
                "min_support": support.min(),
                "q1_support": support.quantile(0.25),
                "median_support": support.median(),
                "q3_support": support.quantile(0.75),
                "max_support": support.max(),
                "codes_support_lt_10": (
                    support < 10
                ).sum(),
                "codes_support_lt_50": (
                    support < 50
                ).sum(),
                "codes_support_lt_100": (
                    support < 100
                ).sum(),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# RUN ANALYSIS
# ============================================================

def run_analysis(
    df: pd.DataFrame,
    selected_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run B.3 categorical-code ↔ target analysis.
    """

    all_feature_summaries = []

    total_features = len(selected_features)

    overall_mortality_rate = (
        df[TARGET].mean()
        * 100
    )

    print(
        "\nCategorical-code features selected: "
        f"{total_features}"
    )

    print(
        "Overall mortality rate: "
        f"{overall_mortality_rate:.2f}%\n"
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

            actual_codes = feature_summary[
                feature_summary[
                    "diagnosis_code"
                ] != MISSING_LABEL
            ]

            print(
                "    Number of observed codes: "
                f"{len(actual_codes)}"
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
                "diagnosis_code",
                "patient_count",
                "death_count",
                "mortality_rate",
            ]
        )

    return pd.concat(
        all_feature_summaries,
        ignore_index=True,
    )


# ============================================================
# ROUND OUTPUTS
# ============================================================

def round_outputs(
    detailed_summary: pd.DataFrame,
    support_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Round percentage and quartile values for easier inspection.
    """

    detailed_summary = detailed_summary.copy()
    support_summary = support_summary.copy()

    if "mortality_rate" in detailed_summary.columns:
        detailed_summary["mortality_rate"] = (
            detailed_summary["mortality_rate"]
            .round(4)
        )

    support_columns = [
        "q1_support",
        "median_support",
        "q3_support",
    ]

    for column in support_columns:
        if column in support_summary.columns:
            support_summary[column] = (
                support_summary[column]
                .round(2)
            )

    return detailed_summary, support_summary


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
    detailed_summary: pd.DataFrame,
    support_summary: pd.DataFrame,
    selected_features: pd.DataFrame,
    df: pd.DataFrame,
) -> None:
    """
    Print compact execution and interpretation summary.
    """

    overall_mortality_rate = (
        df[TARGET].mean()
        * 100
    )

    print("\n" + "=" * 72)

    print(
        "EDA LEVEL 2 — B.3 "
        "CATEGORICAL CODE ↔ TARGET"
    )

    print("=" * 72)

    print(
        "\nFeatures selected: "
        f"{len(selected_features)}"
    )

    print(
        "Rows in detailed table: "
        f"{len(detailed_summary)}"
    )

    print(
        "Overall mortality rate: "
        f"{overall_mortality_rate:.2f}%"
    )

    if not support_summary.empty:
        print(
            "\nSupport summary:"
        )

        print(
            support_summary.to_string(
                index=False
            )
        )

    print(
        "\nInterpretation reminder:"
        "\n- Diagnosis codes are categories, not continuous numbers"
        "\n- patient_count is the support of each code"
        "\n- mortality_rate must always be interpreted together"
        "\n  with patient_count"
        "\n- A very high mortality rate with tiny support may"
        "\n  be unstable"
        "\n- <MISSING> is shown explicitly for EDA"
        "\n- No minimum-support threshold is applied yet"
        "\n- No codes are grouped into OTHER yet"
        "\n- No KEEP / DROP decision is made from B.3 alone"
        "\n- Association does not imply causation"
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
    # Select categorical-code features
    # --------------------------------------------------------

    selected_features = (
        select_categorical_code_features(
            df=df,
            registry=registry,
        )
    )

    # --------------------------------------------------------
    # Run detailed analysis
    # --------------------------------------------------------

    detailed_summary = run_analysis(
        df=df,
        selected_features=selected_features,
    )

    # --------------------------------------------------------
    # Analyze support distribution
    # --------------------------------------------------------

    support_summary = build_support_summary(
        detailed_summary=detailed_summary,
    )

    # --------------------------------------------------------
    # Round outputs
    # --------------------------------------------------------

    (
        detailed_summary,
        support_summary,
    ) = round_outputs(
        detailed_summary=detailed_summary,
        support_summary=support_summary,
    )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    OUTPUT_TABLE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    detailed_summary.to_csv(
        OUTPUT_TABLE_PATH,
        index=False,
    )

    support_summary.to_csv(
        OUTPUT_SUPPORT_SUMMARY_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print_summary(
        detailed_summary=detailed_summary,
        support_summary=support_summary,
        selected_features=selected_features,
        df=df,
    )

    print(
        "\nSaved detailed table:"
        f"\n{OUTPUT_TABLE_PATH}"
    )

    print(
        "\nSaved support summary:"
        f"\n{OUTPUT_SUPPORT_SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()