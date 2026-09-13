from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = Path(
    "data/raw/training_v2.csv"
)

LEVEL1_REGISTRY_PATH = Path(
    "reports/tables/eda_level1_feature_decision_registry.csv"
)

C1_CORRELATION_PATH = Path(
    "reports/tables/eda_level2_c1_numeric_feature_correlation.csv"
)

C2_EXACT_DUPLICATES_PATH = Path(
    "reports/tables/eda_level2_c2_exact_duplicate_features.csv"
)

C2_NEAR_DUPLICATES_PATH = Path(
    "reports/tables/eda_level2_c2_near_duplicate_features.csv"
)

C3_CATEGORICAL_ASSOCIATION_PATH = Path(
    "reports/tables/eda_level2_c3_categorical_feature_association.csv"
)

OUTPUT_REGISTRY_PATH = Path(
    "reports/tables/final_feature_decision_registry.csv"
)

OUTPUT_REVIEW_QUEUE_PATH = Path(
    "reports/tables/final_feature_review_queue.csv"
)

TARGET = "hospital_death"


# ============================================================
# GLOBAL POLICIES
# ============================================================

EXTREME_MISSINGNESS_THRESHOLD = 80.0

HIGH_MISSINGNESS_THRESHOLD = 50.0

HIGH_CORRELATION_THRESHOLD = 0.85

HIGH_CATEGORICAL_ASSOCIATION_THRESHOLD = 0.70

NEAR_DUPLICATE_THRESHOLD = 99.0


# ============================================================
# FEATURE GROUPS / MANUAL SEMANTIC POLICIES
# ============================================================

PURE_IDENTIFIERS = {
    "encounter_id",
    "patient_id",
}

SITE_IDENTIFIERS = {
    "hospital_id",
    "icu_id",
}

DERIVED_MORTALITY_PREDICTIONS = {
    "apache_4a_hospital_death_prob",
    "apache_4a_icu_death_prob",
}

PHYSIOLOGICAL_ZERO_FEATURES = {
    "d1_heartrate_max",
    "d1_heartrate_min",
    "d1_resprate_max",
    "d1_resprate_min",
    "d1_spo2_max",
    "d1_spo2_min",
    "h1_heartrate_max",
    "h1_heartrate_min",
    "h1_resprate_max",
    "h1_resprate_min",
    "h1_spo2_max",
    "h1_spo2_min",
}

SPECIAL_SEMANTIC_FEATURES = {
    "pre_icu_los_days",
}


# ============================================================
# EXPLICIT CLEANING RULES
# ============================================================

CATEGORY_CLEANING_FEATURES = {
    "apache_2_bodysystem": "STANDARDIZE_CATEGORY_LABELS",
}


# ============================================================
# CATEGORICAL MISSING VALUE POLICY
# ============================================================

UNKNOWN_CATEGORY_FEATURES = {
    "hospital_admit_source",
}


# ============================================================
# OPTIONAL MANUAL OVERRIDES
# ============================================================
#
# These dictionaries allow us to override an automatic rule
# without rewriting the decision engine.
#
# Leave them empty unless a manual review gives us a reason
# to change a feature.
#
# Example:
#
# MANUAL_BASELINE_STATUS_OVERRIDES = {
#     "some_feature": "DROP_BASELINE"
# }
#

MANUAL_FINAL_STATUS_OVERRIDES = {}

MANUAL_BASELINE_STATUS_OVERRIDES = {}

MANUAL_MISSING_STRATEGY_OVERRIDES = {}

MANUAL_ENCODING_STRATEGY_OVERRIDES = {}

MANUAL_SPECIAL_VALUE_OVERRIDES = {}

MANUAL_CLEANING_OVERRIDES = {}


# ============================================================
# SAFE CSV LOADER
# ============================================================

def load_optional_csv(path: Path) -> pd.DataFrame:
    """
    Load a CSV if it exists.

    If it does not exist, return an empty DataFrame.
    """

    if not path.exists():
        print(
            f"WARNING: Optional file not found: {path}"
        )
        return pd.DataFrame()

    return pd.read_csv(path)


# ============================================================
# LOAD INPUTS
# ============================================================

def load_inputs():
    """
    Load raw data, Level 1 registry, and Level 2
    feature-feature analysis outputs.
    """

    print("Loading raw training data...")
    df = pd.read_csv(DATA_PATH)

    print(
        f"Training shape: {df.shape}"
    )

    print(
        "\nLoading Level 1 feature registry..."
    )

    level1_registry = pd.read_csv(
        LEVEL1_REGISTRY_PATH
    )

    print(
        f"Level 1 registry shape: "
        f"{level1_registry.shape}"
    )

    print(
        "\nLoading Level 2 C-analysis outputs..."
    )

    c1 = load_optional_csv(
        C1_CORRELATION_PATH
    )

    c2_exact = load_optional_csv(
        C2_EXACT_DUPLICATES_PATH
    )

    c2_near = load_optional_csv(
        C2_NEAR_DUPLICATES_PATH
    )

    c3 = load_optional_csv(
        C3_CATEGORICAL_ASSOCIATION_PATH
    )

    return (
        df,
        level1_registry,
        c1,
        c2_exact,
        c2_near,
        c3,
    )


# ============================================================
# VALIDATE INPUTS
# ============================================================

def validate_inputs(
    df: pd.DataFrame,
    registry: pd.DataFrame,
) -> None:
    """
    Validate required inputs.
    """

    required_registry_columns = {
        "feature",
        "semantic_type",
        "missing_percent",
        "level1_status",
    }

    missing_columns = (
        required_registry_columns
        - set(registry.columns)
    )

    if missing_columns:
        raise ValueError(
            "Level 1 registry is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if TARGET not in df.columns:
        raise ValueError(
            f"Target '{TARGET}' not found in dataset."
        )


# ============================================================
# REDUNDANCY LOOKUP HELPERS
# ============================================================

def build_pair_lookup(
    table: pd.DataFrame,
    value_column: str,
) -> dict:
    """
    Convert a pairwise feature table into an easy lookup.

    Example:

        feature A -> [(feature B, value)]
        feature B -> [(feature A, value)]
    """

    lookup = {}

    if table.empty:
        return lookup

    required = {
        "feature_1",
        "feature_2",
        value_column,
    }

    if not required.issubset(
        table.columns
    ):
        return lookup

    for _, row in table.iterrows():

        feature_1 = row["feature_1"]
        feature_2 = row["feature_2"]
        value = row[value_column]

        lookup.setdefault(
            feature_1,
            [],
        ).append(
            (feature_2, value)
        )

        lookup.setdefault(
            feature_2,
            [],
        ).append(
            (feature_1, value)
        )

    return lookup


# ============================================================
# EXACT DUPLICATE POLICY
# ============================================================

def build_exact_duplicate_drop_map(
    exact_duplicates: pd.DataFrame,
) -> dict:
    """
    Decide which exact duplicate feature to drop.

    For each exact duplicate pair we need one retained feature
    and one removed feature.

    Current deterministic baseline policy:
        keep alphabetically first feature
        drop alphabetically second feature

    This avoids random behavior.

    IMPORTANT:
    The choice of which exact duplicate to keep is not based
    on predictive superiority because both columns contain the
    same row-level information.

    Manual overrides can later replace this policy if feature
    semantics justify preferring one name over another.
    """

    drop_map = {}

    if exact_duplicates.empty:
        return drop_map

    required = {
        "feature_1",
        "feature_2",
    }

    if not required.issubset(
        exact_duplicates.columns
    ):
        return drop_map

    for _, row in exact_duplicates.iterrows():

        feature_1 = str(
            row["feature_1"]
        )

        feature_2 = str(
            row["feature_2"]
        )

        keep_feature = min(
            feature_1,
            feature_2,
        )

        drop_feature = max(
            feature_1,
            feature_2,
        )

        drop_map[
            drop_feature
        ] = keep_feature

    return drop_map


# ============================================================
# REDUNDANCY INFORMATION
# ============================================================

def collect_redundancy_information(
    feature: str,
    exact_drop_map: dict,
    correlation_lookup: dict,
    near_duplicate_lookup: dict,
    categorical_lookup: dict,
) -> tuple[str, str, list[str]]:
    """
    Collect redundancy evidence for one feature.

    Priority:
        EXACT_DUPLICATE
        NEAR_DUPLICATE
        HIGH_CORRELATION
        HIGH_CATEGORICAL_ASSOCIATION
        NONE
    """

    evidence = []

    # --------------------------------------------------------
    # Exact duplicate
    # --------------------------------------------------------

    if feature in exact_drop_map:

        kept_feature = (
            exact_drop_map[feature]
        )

        evidence.append(
            f"EXACT_DUPLICATE_OF={kept_feature}"
        )

        return (
            "EXACT_DUPLICATE",
            kept_feature,
            evidence,
        )

    # --------------------------------------------------------
    # Near duplicate
    # --------------------------------------------------------

    near_pairs = (
        near_duplicate_lookup.get(
            feature,
            [],
        )
    )

    if near_pairs:

        strongest = max(
            near_pairs,
            key=lambda item: item[1],
        )

        other_feature = strongest[0]
        match_percent = strongest[1]

        evidence.append(
            f"NEAR_DUPLICATE_WITH="
            f"{other_feature}:"
            f"{match_percent:.4f}%"
        )

        return (
            "NEAR_DUPLICATE",
            other_feature,
            evidence,
        )

    # --------------------------------------------------------
    # High numeric correlation
    # --------------------------------------------------------

    correlation_pairs = (
        correlation_lookup.get(
            feature,
            [],
        )
    )

    if correlation_pairs:

        strongest = max(
            correlation_pairs,
            key=lambda item: abs(
                item[1]
            ),
        )

        other_feature = strongest[0]
        correlation = strongest[1]

        evidence.append(
            f"HIGH_CORRELATION_WITH="
            f"{other_feature}:"
            f"{correlation:.4f}"
        )

        return (
            "HIGH_CORRELATION",
            other_feature,
            evidence,
        )

    # --------------------------------------------------------
    # High categorical association
    # --------------------------------------------------------

    categorical_pairs = (
        categorical_lookup.get(
            feature,
            [],
        )
    )

    if categorical_pairs:

        strongest = max(
            categorical_pairs,
            key=lambda item: item[1],
        )

        other_feature = strongest[0]
        cramers_v = strongest[1]

        evidence.append(
            f"HIGH_CATEGORICAL_ASSOCIATION_WITH="
            f"{other_feature}:"
            f"{cramers_v:.4f}"
        )

        return (
            "HIGH_CATEGORICAL_ASSOCIATION",
            other_feature,
            evidence,
        )

    return (
        "NONE",
        "",
        evidence,
    )


# ============================================================
# FINAL DATA STATUS
# ============================================================

def determine_final_data_status(
    feature: str,
    level1_status: str,
    exact_duplicate_of: str | None,
) -> tuple[str, str]:
    """
    Determine whether a feature exists in the final modeling
    feature universe.

    KEEP:
        feature remains available for current/future modeling

    DROP:
        feature is intentionally removed

    TARGET:
        prediction target
    """

    if feature == TARGET:
        return (
            "TARGET",
            "TARGET_VARIABLE",
        )

    if feature in PURE_IDENTIFIERS:
        return (
            "DROP",
            "PURE_IDENTIFIER",
        )

    if feature in DERIVED_MORTALITY_PREDICTIONS:
        return (
            "DROP",
            "EXISTING_MORTALITY_PREDICTION_EXCLUDED",
        )

    if exact_duplicate_of:
        return (
            "DROP",
            f"EXACT_DUPLICATE_OF_{exact_duplicate_of}",
        )

    if level1_status == "DROP":
        return (
            "DROP",
            "LEVEL1_CONFIRMED_DROP",
        )

    return (
        "KEEP",
        "FEATURE_RETAINED",
    )


# ============================================================
# BASELINE STATUS
# ============================================================

def determine_baseline_status(
    feature: str,
    semantic_type: str,
    missing_percent: float,
    final_status: str,
) -> tuple[str, str]:
    """
    Decide whether feature enters the first baseline model.
    """

    if final_status == "TARGET":
        return (
            "TARGET",
            "TARGET_VARIABLE",
        )

    if final_status == "DROP":
        return (
            "DROP",
            "FINAL_DATA_DROP",
        )

    # Site identifiers are retained in the overall data
    # universe but excluded from baseline to avoid relying
    # on hospital/unit-specific patterns immediately.

    if feature in SITE_IDENTIFIERS:
        return (
            "DROP_BASELINE",
            "SITE_GENERALIZATION_RISK",
        )

    # Conservative baseline missingness policy.

    if (
        pd.notna(missing_percent)
        and missing_percent
        >= EXTREME_MISSINGNESS_THRESHOLD
    ):
        return (
            "DROP_BASELINE",
            "EXTREME_MISSINGNESS",
        )

    return (
        "KEEP",
        "BASELINE_FEATURE",
    )


# ============================================================
# MISSING VALUE STRATEGY
# ============================================================

def determine_missing_strategy(
    feature: str,
    semantic_type: str,
    missing_percent: float,
    baseline_status: str,
) -> str:
    """
    Define missing-value policy.

    IMPORTANT:
    This function only defines the strategy.

    It does NOT calculate median/mode values.

    Those values will be learned from the TRAINING SET only
    after train/validation/test splitting.
    """

    if baseline_status != "KEEP":
        return "NONE"

    if (
        pd.isna(missing_percent)
        or missing_percent == 0
    ):
        return "NONE"

    if feature in UNKNOWN_CATEGORY_FEATURES:
        return "UNKNOWN_CATEGORY"

    if semantic_type in {
        "CONTINUOUS",
        "DISCRETE",
        "ORDINAL",
    }:
        return "MEDIAN"

    if semantic_type in {
        "CATEGORICAL",
        "CATEGORICAL_CODE",
        "BINARY",
    }:
        return "MODE"

    return "NONE"


# ============================================================
# ENCODING STRATEGY
# ============================================================

def determine_encoding_strategy(
    semantic_type: str,
    baseline_status: str,
) -> str:
    """
    Define how a feature will eventually be represented
    in the preprocessing pipeline.
    """

    if baseline_status != "KEEP":
        return "NONE"

    if semantic_type in {
        "CONTINUOUS",
        "DISCRETE",
    }:
        return "NONE"

    if semantic_type == "ORDINAL":
        # Numeric clinical ordinal values such as GCS retain
        # their meaningful order.
        return "ORDINAL"

    if semantic_type == "BINARY":
        return "BINARY"

    if semantic_type in {
        "CATEGORICAL",
        "CATEGORICAL_CODE",
    }:
        return "ONE_HOT"

    return "NONE"


# ============================================================
# CLEANING ACTION
# ============================================================

def determine_cleaning_action(
    feature: str,
) -> str:
    """
    Determine deterministic data-cleaning operations.
    """

    return CATEGORY_CLEANING_FEATURES.get(
        feature,
        "NONE",
    )


# ============================================================
# SPECIAL VALUE ACTION
# ============================================================

def determine_special_value_action(
    feature: str,
) -> str:
    """
    Handle known special-value concerns.

    We intentionally avoid silently converting physiological
    zero values to NaN without semantic confirmation.
    """

    if feature in PHYSIOLOGICAL_ZERO_FEATURES:
        return "REVIEW_ZERO_VALIDITY"

    if feature == "pre_icu_los_days":
        return "REVIEW_NEGATIVE_VALUES"

    return "NONE"


# ============================================================
# LEAKAGE STATUS
# ============================================================

def determine_leakage_status(
    feature: str,
) -> str:
    """
    Mark potential leakage or derived prediction features.
    """

    if feature in DERIVED_MORTALITY_PREDICTIONS:
        return "EXCLUDE"

    return "NONE"


# ============================================================
# FUTURE EXPERIMENT
# ============================================================

def determine_future_experiment(
    feature: str,
    baseline_status: str,
    redundancy_status: str,
    special_value_action: str,
) -> str:
    """
    Record possible post-baseline experiments.
    """

    experiments = []

    if (
        baseline_status == "DROP_BASELINE"
        and feature in SITE_IDENTIFIERS
    ):
        experiments.append(
            "SITE_ID_ABLATION"
        )

    if (
        baseline_status == "DROP_BASELINE"
        and feature not in SITE_IDENTIFIERS
    ):
        experiments.append(
            "REINTRODUCE_HIGH_MISSINGNESS_FEATURE"
        )

    if redundancy_status in {
        "HIGH_CORRELATION",
        "HIGH_CATEGORICAL_ASSOCIATION",
        "NEAR_DUPLICATE",
    }:
        experiments.append(
            "REDUNDANCY_ABLATION"
        )

    if special_value_action != "NONE":
        experiments.append(
            "SPECIAL_VALUE_REVIEW"
        )

    if not experiments:
        return "NONE"

    return "; ".join(experiments)


# ============================================================
# APPLY MANUAL OVERRIDES
# ============================================================

def apply_manual_overrides(
    row: dict,
) -> dict:
    """
    Apply explicit human-reviewed overrides.

    This keeps manual decisions visible in code rather than
    silently editing the output CSV.
    """

    feature = row["feature"]

    if feature in MANUAL_FINAL_STATUS_OVERRIDES:
        row["final_data_status"] = (
            MANUAL_FINAL_STATUS_OVERRIDES[
                feature
            ]
        )

    if feature in MANUAL_BASELINE_STATUS_OVERRIDES:
        row["baseline_status"] = (
            MANUAL_BASELINE_STATUS_OVERRIDES[
                feature
            ]
        )

    if feature in MANUAL_MISSING_STRATEGY_OVERRIDES:
        row["missing_strategy"] = (
            MANUAL_MISSING_STRATEGY_OVERRIDES[
                feature
            ]
        )

    if feature in MANUAL_ENCODING_STRATEGY_OVERRIDES:
        row["encoding_strategy"] = (
            MANUAL_ENCODING_STRATEGY_OVERRIDES[
                feature
            ]
        )

    if feature in MANUAL_SPECIAL_VALUE_OVERRIDES:
        row["special_value_action"] = (
            MANUAL_SPECIAL_VALUE_OVERRIDES[
                feature
            ]
        )

    if feature in MANUAL_CLEANING_OVERRIDES:
        row["cleaning_action"] = (
            MANUAL_CLEANING_OVERRIDES[
                feature
            ]
        )

    return row


# ============================================================
# REVIEW QUEUE LOGIC
# ============================================================

def determine_review_requirement(
    feature: str,
    redundancy_status: str,
    special_value_action: str,
) -> tuple[bool, str]:
    """
    Determine whether the feature requires human review.

    The queue should remain small and actionable.
    """

    reasons = []

    if feature in SITE_IDENTIFIERS:
        reasons.append(
            "SITE_GENERALIZATION"
        )

    if redundancy_status == "NEAR_DUPLICATE":
        reasons.append(
            "NEAR_DUPLICATE"
        )

    if redundancy_status == "HIGH_CORRELATION":
        reasons.append(
            "HIGH_CORRELATION"
        )

    if redundancy_status == (
        "HIGH_CATEGORICAL_ASSOCIATION"
    ):
        reasons.append(
            "HIGH_CATEGORICAL_ASSOCIATION"
        )

    if special_value_action != "NONE":
        reasons.append(
            special_value_action
        )

    if not reasons:
        return (
            False,
            "",
        )

    return (
        True,
        "; ".join(reasons),
    )


# ============================================================
# BUILD FINAL REGISTRY
# ============================================================

def build_final_registry(
    level1_registry: pd.DataFrame,
    exact_drop_map: dict,
    correlation_lookup: dict,
    near_duplicate_lookup: dict,
    categorical_lookup: dict,
) -> pd.DataFrame:
    """
    Build the final feature decision registry.
    """

    rows = []

    for _, source_row in (
        level1_registry.iterrows()
    ):

        feature = source_row["feature"]

        semantic_type = source_row[
            "semantic_type"
        ]

        level1_status = source_row[
            "level1_status"
        ]

        missing_percent = source_row.get(
            "missing_percent",
            np.nan,
        )

        # ----------------------------------------------------
        # Redundancy evidence
        # ----------------------------------------------------

        (
            redundancy_status,
            redundancy_with,
            redundancy_evidence,
        ) = collect_redundancy_information(
            feature=feature,
            exact_drop_map=exact_drop_map,
            correlation_lookup=correlation_lookup,
            near_duplicate_lookup=near_duplicate_lookup,
            categorical_lookup=categorical_lookup,
        )

        exact_duplicate_of = None

        if (
            redundancy_status
            == "EXACT_DUPLICATE"
        ):
            exact_duplicate_of = (
                redundancy_with
            )

        # ----------------------------------------------------
        # Final status
        # ----------------------------------------------------

        (
            final_status,
            decision_reason,
        ) = determine_final_data_status(
            feature=feature,
            level1_status=level1_status,
            exact_duplicate_of=exact_duplicate_of,
        )

        # ----------------------------------------------------
        # Baseline status
        # ----------------------------------------------------

        (
            baseline_status,
            baseline_reason,
        ) = determine_baseline_status(
            feature=feature,
            semantic_type=semantic_type,
            missing_percent=missing_percent,
            final_status=final_status,
        )

        # ----------------------------------------------------
        # Missing strategy
        # ----------------------------------------------------

        missing_strategy = (
            determine_missing_strategy(
                feature=feature,
                semantic_type=semantic_type,
                missing_percent=missing_percent,
                baseline_status=baseline_status,
            )
        )

        # ----------------------------------------------------
        # Encoding
        # ----------------------------------------------------

        encoding_strategy = (
            determine_encoding_strategy(
                semantic_type=semantic_type,
                baseline_status=baseline_status,
            )
        )

        # ----------------------------------------------------
        # Cleaning
        # ----------------------------------------------------

        cleaning_action = (
            determine_cleaning_action(
                feature
            )
        )

        # ----------------------------------------------------
        # Special values
        # ----------------------------------------------------

        special_value_action = (
            determine_special_value_action(
                feature
            )
        )

        # ----------------------------------------------------
        # Leakage
        # ----------------------------------------------------

        leakage_status = (
            determine_leakage_status(
                feature
            )
        )

        # ----------------------------------------------------
        # Future experiments
        # ----------------------------------------------------

        future_experiment = (
            determine_future_experiment(
                feature=feature,
                baseline_status=baseline_status,
                redundancy_status=redundancy_status,
                special_value_action=special_value_action,
            )
        )

        # ----------------------------------------------------
        # Review queue
        # ----------------------------------------------------

        (
            requires_review,
            review_reason,
        ) = determine_review_requirement(
            feature=feature,
            redundancy_status=redundancy_status,
            special_value_action=special_value_action,
        )

        # ----------------------------------------------------
        # Level 2 evidence
        # ----------------------------------------------------

        if redundancy_evidence:
            level2_evidence = "; ".join(
                redundancy_evidence
            )
        else:
            level2_evidence = "NONE"

        # ----------------------------------------------------
        # Build row
        # ----------------------------------------------------

        row = {
            "feature": feature,
            "semantic_type": semantic_type,
            "missing_percent": missing_percent,

            "final_data_status":
                final_status,

            "baseline_status":
                baseline_status,

            "missing_strategy":
                missing_strategy,

            "encoding_strategy":
                encoding_strategy,

            "cleaning_action":
                cleaning_action,

            "special_value_action":
                special_value_action,

            "redundancy_status":
                redundancy_status,

            "redundancy_with":
                redundancy_with,

            "leakage_status":
                leakage_status,

            "decision_reason":
                decision_reason,

            "baseline_reason":
                baseline_reason,

            "level2_evidence":
                level2_evidence,

            "future_experiment":
                future_experiment,

            "requires_review":
                requires_review,

            "review_reason":
                review_reason,
        }

        row = apply_manual_overrides(
            row
        )

        rows.append(row)

    final_registry = pd.DataFrame(
        rows
    )

    return final_registry


# ============================================================
# BUILD REVIEW QUEUE
# ============================================================

def build_review_queue(
    registry: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a compact table containing only features that
    still deserve explicit human review.
    """

    review_columns = [
        "feature",
        "semantic_type",
        "missing_percent",
        "final_data_status",
        "baseline_status",
        "special_value_action",
        "redundancy_status",
        "redundancy_with",
        "leakage_status",
        "level2_evidence",
        "review_reason",
        "future_experiment",
    ]

    review_queue = registry[
        registry["requires_review"]
        == True
    ][
        review_columns
    ].copy()

    return review_queue.reset_index(
        drop=True
    )


# ============================================================
# SANITY CHECKS
# ============================================================

def run_sanity_checks(
    registry: pd.DataFrame,
) -> None:
    """
    Validate important registry invariants.
    """

    # Target must exist exactly once.

    target_rows = registry[
        registry["feature"] == TARGET
    ]

    if len(target_rows) != 1:
        raise ValueError(
            "Target must appear exactly once."
        )

    # Final DROP cannot enter baseline.

    invalid_drop = registry[
        (registry["final_data_status"] == "DROP")
        &
        (registry["baseline_status"] == "KEEP")
    ]

    if not invalid_drop.empty:
        raise ValueError(
            "A final DROP feature cannot be "
            "kept in baseline."
        )

    # Target cannot receive imputation.

    invalid_target = registry[
        (registry["final_data_status"] == "TARGET")
        &
        (registry["missing_strategy"] != "NONE")
    ]

    if not invalid_target.empty:
        raise ValueError(
            "Target must not receive a missing "
            "value strategy."
        )

    # Dropped features should not receive encoding.

    invalid_encoding = registry[
        registry["baseline_status"].isin(
            {
                "DROP",
                "DROP_BASELINE",
                "TARGET",
            }
        )
        &
        (
            registry["encoding_strategy"]
            != "NONE"
        )
    ]

    if not invalid_encoding.empty:
        raise ValueError(
            "Dropped/non-baseline features should "
            "not receive baseline encoding."
        )


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
    registry: pd.DataFrame,
    review_queue: pd.DataFrame,
) -> None:
    """
    Print compact registry summary.
    """

    print(
        "\n" + "=" * 72
    )

    print(
        "FINAL FEATURE DECISION REGISTRY"
    )

    print(
        "=" * 72
    )

    print(
        f"\nTotal features: "
        f"{len(registry)}"
    )

    print(
        "\nFinal data status:"
    )

    print(
        registry[
            "final_data_status"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nBaseline status:"
    )

    print(
        registry[
            "baseline_status"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nMissing strategies:"
    )

    print(
        registry[
            "missing_strategy"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nEncoding strategies:"
    )

    print(
        registry[
            "encoding_strategy"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nRedundancy findings:"
    )

    print(
        registry[
            "redundancy_status"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nLeakage status:"
    )

    print(
        registry[
            "leakage_status"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nFeatures requiring review: "
        f"{len(review_queue)}"
    )

    if not review_queue.empty:

        print(
            "\nReview Queue:"
        )

        print(
            review_queue[
                [
                    "feature",
                    "baseline_status",
                    "review_reason",
                ]
            ]
            .to_string(
                index=False
            )
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    # --------------------------------------------------------
    # Load inputs
    # --------------------------------------------------------

    (
        df,
        level1_registry,
        c1,
        c2_exact,
        c2_near,
        c3,
    ) = load_inputs()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_inputs(
        df=df,
        registry=level1_registry,
    )

    # --------------------------------------------------------
    # Exact duplicate map
    # --------------------------------------------------------

    exact_drop_map = (
        build_exact_duplicate_drop_map(
            c2_exact
        )
    )

    # --------------------------------------------------------
    # Level 2 pair lookups
    # --------------------------------------------------------

    correlation_lookup = (
        build_pair_lookup(
            table=c1,
            value_column="correlation",
        )
    )

    near_duplicate_lookup = (
        build_pair_lookup(
            table=c2_near,
            value_column="match_percent",
        )
    )

    categorical_lookup = (
        build_pair_lookup(
            table=c3,
            value_column="cramers_v",
        )
    )

    # --------------------------------------------------------
    # Build registry
    # --------------------------------------------------------

    final_registry = (
        build_final_registry(
            level1_registry=level1_registry,
            exact_drop_map=exact_drop_map,
            correlation_lookup=correlation_lookup,
            near_duplicate_lookup=near_duplicate_lookup,
            categorical_lookup=categorical_lookup,
        )
    )

    # --------------------------------------------------------
    # Round
    # --------------------------------------------------------

    if (
        "missing_percent"
        in final_registry.columns
    ):
        final_registry[
            "missing_percent"
        ] = (
            final_registry[
                "missing_percent"
            ]
            .round(4)
        )

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    run_sanity_checks(
        final_registry
    )

    # --------------------------------------------------------
    # Review queue
    # --------------------------------------------------------

    review_queue = (
        build_review_queue(
            final_registry
        )
    )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    OUTPUT_REGISTRY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_registry.to_csv(
        OUTPUT_REGISTRY_PATH,
        index=False,
    )

    review_queue.to_csv(
        OUTPUT_REVIEW_QUEUE_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print_summary(
        registry=final_registry,
        review_queue=review_queue,
    )

    print(
        "\nSaved final registry:"
        f"\n{OUTPUT_REGISTRY_PATH}"
    )

    print(
        "\nSaved review queue:"
        f"\n{OUTPUT_REVIEW_QUEUE_PATH}"
    )


if __name__ == "__main__":
    main()