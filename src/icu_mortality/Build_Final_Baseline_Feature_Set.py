from pathlib import Path
import json

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

REGISTRY_PATH = Path(
    "reports/tables/final_feature_decision_registry.csv"
)

OUTPUT_SUMMARY_PATH = Path(
    "reports/tables/final_baseline_feature_set.csv"
)

OUTPUT_JSON_PATH = Path(
    "reports/tables/final_baseline_feature_set.json"
)

TARGET = "hospital_death"


# ============================================================
# LOAD
# ============================================================

def load_registry() -> pd.DataFrame:
    """
    Load the final decision registry created in Step 3.1.
    """

    print("Loading final feature decision registry...")

    registry = pd.read_csv(
        REGISTRY_PATH
    )

    print(
        f"Registry shape: {registry.shape}"
    )

    return registry


# ============================================================
# VALIDATE
# ============================================================

def validate_registry(
    registry: pd.DataFrame,
) -> None:
    """
    Validate required columns.
    """

    required_columns = {
        "feature",
        "semantic_type",
        "final_data_status",
        "baseline_status",
        "missing_strategy",
        "encoding_strategy",
    }

    missing_columns = (
        required_columns
        - set(registry.columns)
    )

    if missing_columns:
        raise ValueError(
            "Registry is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    target_rows = registry[
        registry["feature"] == TARGET
    ]

    if len(target_rows) != 1:
        raise ValueError(
            f"Target '{TARGET}' must appear exactly once."
        )


# ============================================================
# SELECT BASELINE FEATURES
# ============================================================

def get_baseline_registry(
    registry: pd.DataFrame,
) -> pd.DataFrame:
    """
    Keep only features entering the baseline model.
    """

    baseline = registry[
        registry["baseline_status"] == "KEEP"
    ].copy()

    return baseline


# ============================================================
# BUILD FEATURE GROUPS
# ============================================================

def build_feature_groups(
    baseline: pd.DataFrame,
) -> dict:
    """
    Group baseline features by semantic type.

    These groups will later drive the preprocessing pipeline.
    """

    groups = {
        "numeric_features": [],
        "ordinal_features": [],
        "binary_features": [],
        "categorical_features": [],
        "categorical_code_features": [],
    }

    # --------------------------------------------------------
    # Numeric
    # --------------------------------------------------------

    groups["numeric_features"] = (
        baseline[
            baseline["semantic_type"].isin(
                {
                    "CONTINUOUS",
                    "DISCRETE",
                }
            )
        ]["feature"]
        .sort_values()
        .tolist()
    )

    # --------------------------------------------------------
    # Ordinal
    # --------------------------------------------------------

    groups["ordinal_features"] = (
        baseline[
            baseline["semantic_type"]
            == "ORDINAL"
        ]["feature"]
        .sort_values()
        .tolist()
    )

    # --------------------------------------------------------
    # Binary
    # --------------------------------------------------------

    groups["binary_features"] = (
        baseline[
            baseline["semantic_type"]
            == "BINARY"
        ]["feature"]
        .sort_values()
        .tolist()
    )

    # --------------------------------------------------------
    # Categorical
    # --------------------------------------------------------

    groups["categorical_features"] = (
        baseline[
            baseline["semantic_type"]
            == "CATEGORICAL"
        ]["feature"]
        .sort_values()
        .tolist()
    )

    # --------------------------------------------------------
    # Categorical codes
    # --------------------------------------------------------

    groups["categorical_code_features"] = (
        baseline[
            baseline["semantic_type"]
            == "CATEGORICAL_CODE"
        ]["feature"]
        .sort_values()
        .tolist()
    )

    # --------------------------------------------------------
    # All baseline features
    # --------------------------------------------------------

    groups["all_baseline_features"] = (
        baseline["feature"]
        .sort_values()
        .tolist()
    )

    return groups


# ============================================================
# BUILD NON-BASELINE GROUPS
# ============================================================

def build_excluded_groups(
    registry: pd.DataFrame,
) -> dict:
    """
    Record features excluded either permanently or only from
    the baseline model.
    """

    dropped_features = (
        registry[
            registry["final_data_status"]
            == "DROP"
        ]["feature"]
        .sort_values()
        .tolist()
    )

    baseline_dropped_features = (
        registry[
            registry["baseline_status"]
            == "DROP_BASELINE"
        ]["feature"]
        .sort_values()
        .tolist()
    )

    target_features = (
        registry[
            registry["baseline_status"]
            == "TARGET"
        ]["feature"]
        .sort_values()
        .tolist()
    )

    return {
        "dropped_features":
            dropped_features,

        "baseline_dropped_features":
            baseline_dropped_features,

        "target_features":
            target_features,
    }


# ============================================================
# BUILD SUMMARY TABLE
# ============================================================

def build_summary_table(
    groups: dict,
    excluded_groups: dict,
) -> pd.DataFrame:
    """
    Create a compact summary table.

    Each row represents one feature group.
    """

    rows = []

    combined = {
        **groups,
        **excluded_groups,
    }

    for group_name, features in combined.items():

        rows.append(
            {
                "feature_group":
                    group_name,

                "feature_count":
                    len(features),

                "features":
                    "; ".join(features),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# SANITY CHECKS
# ============================================================

def run_sanity_checks(
    baseline: pd.DataFrame,
    groups: dict,
) -> None:
    """
    Confirm that every baseline feature belongs to exactly one
    semantic preprocessing group.
    """

    grouped_features = (
        groups["numeric_features"]
        + groups["ordinal_features"]
        + groups["binary_features"]
        + groups["categorical_features"]
        + groups["categorical_code_features"]
    )

    grouped_set = set(
        grouped_features
    )

    baseline_set = set(
        baseline["feature"]
    )

    # --------------------------------------------------------
    # Missing group assignment
    # --------------------------------------------------------

    missing_from_groups = (
        baseline_set
        - grouped_set
    )

    if missing_from_groups:
        raise ValueError(
            "Baseline features missing from preprocessing "
            f"groups: {sorted(missing_from_groups)}"
        )

    # --------------------------------------------------------
    # Unexpected grouped feature
    # --------------------------------------------------------

    extra_features = (
        grouped_set
        - baseline_set
    )

    if extra_features:
        raise ValueError(
            "Grouped features not found in baseline set: "
            f"{sorted(extra_features)}"
        )

    # --------------------------------------------------------
    # Duplicate group membership
    # --------------------------------------------------------

    if (
        len(grouped_features)
        != len(grouped_set)
    ):
        raise ValueError(
            "At least one feature appears in more than one "
            "preprocessing group."
        )


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
    groups: dict,
    excluded_groups: dict,
) -> None:
    """
    Print compact baseline feature-set summary.
    """

    print(
        "\n" + "=" * 72
    )

    print(
        "FINAL BASELINE FEATURE SET"
    )

    print(
        "=" * 72
    )

    print(
        "\nBaseline preprocessing groups:"
    )

    print(
        f"Numeric: "
        f"{len(groups['numeric_features'])}"
    )

    print(
        f"Ordinal: "
        f"{len(groups['ordinal_features'])}"
    )

    print(
        f"Binary: "
        f"{len(groups['binary_features'])}"
    )

    print(
        f"Categorical: "
        f"{len(groups['categorical_features'])}"
    )

    print(
        f"Categorical code: "
        f"{len(groups['categorical_code_features'])}"
    )

    print(
        "\nTotal baseline features: "
        f"{len(groups['all_baseline_features'])}"
    )

    print(
        "\nPermanently dropped features: "
        f"{len(excluded_groups['dropped_features'])}"
    )

    print(
        "Baseline-only dropped features: "
        f"{len(excluded_groups['baseline_dropped_features'])}"
    )

    print(
        "Target features: "
        f"{len(excluded_groups['target_features'])}"
    )


# ============================================================
# SAVE JSON
# ============================================================

def save_json(
    groups: dict,
    excluded_groups: dict,
) -> None:
    """
    Save feature lists in machine-readable form.

    This JSON will be convenient when building the pipeline.
    """

    payload = {
        "target": TARGET,
        **groups,
        **excluded_groups,
    }

    OUTPUT_JSON_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_JSON_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            indent=4,
            ensure_ascii=False,
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    registry = load_registry()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_registry(
        registry
    )

    # --------------------------------------------------------
    # Select baseline
    # --------------------------------------------------------

    baseline = get_baseline_registry(
        registry
    )

    # --------------------------------------------------------
    # Build groups
    # --------------------------------------------------------

    groups = build_feature_groups(
        baseline
    )

    excluded_groups = (
        build_excluded_groups(
            registry
        )
    )

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    run_sanity_checks(
        baseline=baseline,
        groups=groups,
    )

    # --------------------------------------------------------
    # Summary table
    # --------------------------------------------------------

    summary_table = (
        build_summary_table(
            groups=groups,
            excluded_groups=excluded_groups,
        )
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    OUTPUT_SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_table.to_csv(
        OUTPUT_SUMMARY_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    save_json(
        groups=groups,
        excluded_groups=excluded_groups,
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print_summary(
        groups=groups,
        excluded_groups=excluded_groups,
    )

    print(
        "\nSaved summary table:"
        f"\n{OUTPUT_SUMMARY_PATH}"
    )

    print(
        "\nSaved machine-readable feature set:"
        f"\n{OUTPUT_JSON_PATH}"
    )


if __name__ == "__main__":
    main()