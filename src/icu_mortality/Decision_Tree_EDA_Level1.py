from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path("data/raw/training_v2.csv")

DICTIONARY_PATH = Path(
    "data/reference/WiDS Datathon 2020 Dictionary.csv"
)

REPORTS_PATH = Path("reports/tables")


# ------------------------------------------------------------
# EDA LEVEL 1 RESULTS
# ------------------------------------------------------------

A1_PATH = REPORTS_PATH / "eda_level1_a1_numeric_distribution_summary.csv"
A2_PATH = REPORTS_PATH / "eda_level1_a2_numeric_outliers_summary.csv"
A3_PATH = REPORTS_PATH / "eda_level1_a3_numeric_range_summary.csv"
A4_PATH = REPORTS_PATH / "eda_level1_a4_numeric_near_constant_summary.csv"

B1_PATH = REPORTS_PATH / "eda_level1_b1_categorical_cardinality_summary.csv"
B3_PATH = REPORTS_PATH / "eda_level1_b3_categorical_dominance_summary.csv"
B4_PATH = REPORTS_PATH / "eda_level1_b4_categorical_rare_categories_summary.csv"


# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

OUTPUT_PATH = (
    REPORTS_PATH
    / "eda_level1_feature_decision_registry.csv"
)

TARGET = "hospital_death"


# ============================================================
# DECISION POLICY
# ============================================================

EXTREME_MISSINGNESS_THRESHOLD = 80.0
HIGH_MISSINGNESS_THRESHOLD = 50.0


# ============================================================
# KNOWN SEMANTIC GROUPS
# ============================================================

# Pure identifiers should not be model features.
PURE_IDENTIFIERS = {
    "encounter_id",
    "patient_id",
}


# Site/unit identifiers may contain useful site effects,
# but can also damage generalization.
SITE_IDENTIFIERS = {
    "hospital_id",
    "icu_id",
}


# Existing mortality predictions / scores.
DERIVED_PREDICTION_FEATURES = {
    "apache_4a_hospital_death_prob",
    "apache_4a_icu_death_prob",
}


# Numeric storage, but semantically diagnosis/category codes.
CATEGORICAL_CODE_FEATURES = {
    "apache_2_diagnosis",
    "apache_3j_diagnosis",
}


# Ordered clinical scores.
ORDINAL_FEATURES = {
    "gcs_eyes_apache",
    "gcs_motor_apache",
    "gcs_verbal_apache",
}


# Dictionary datatype is not always analytically correct.
CONTINUOUS_OVERRIDES = {
    "bmi",
}


# Features already identified during Level 1
# as requiring explicit semantic investigation.
SPECIAL_INVESTIGATION_FEATURES = {
    "pre_icu_los_days",
}


# Categorical features for which UNKNOWN was already
# selected as the baseline missing-value strategy.
UNKNOWN_CATEGORY_FEATURES = {
    "hospital_admit_source",
}


# Physiological variables where zero may represent
# clinical signal rather than an automatic invalid value.
PHYSIOLOGICAL_ZERO_KEYWORDS = (
    "heartrate",
    "resprate",
    "spo2",
)


# ============================================================
# LOAD FUNCTIONS
# ============================================================

def load_optional_csv(path: Path) -> pd.DataFrame:
    """
    Load an EDA result table.

    If the file does not exist, return an empty DataFrame
    so the registry can still be built and inspected.
    """

    if not path.exists():
        print(f"[WARNING] EDA file not found: {path}")
        return pd.DataFrame()

    return pd.read_csv(path)


def normalize_dictionary(
    dictionary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize dictionary column names.
    """

    dictionary = dictionary.copy()

    dictionary.columns = (
        dictionary.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    return dictionary


# ============================================================
# GENERAL HELPERS
# ============================================================

def get_feature_row(
    dataframe: pd.DataFrame,
    feature: str,
):
    """
    Return the row belonging to a feature from an EDA table.
    """

    if dataframe.empty:
        return None

    if "feature" not in dataframe.columns:
        return None

    rows = dataframe[
        dataframe["feature"] == feature
    ]

    if rows.empty:
        return None

    return rows.iloc[0]


def get_value(
    row,
    column: str,
    default=None,
):
    """
    Safely read a value from an EDA row.
    """

    if row is None:
        return default

    if column not in row.index:
        return default

    return row[column]


def add_finding(
    findings: list[str],
    finding: str,
) -> None:
    """
    Add finding without duplicates.
    """

    if finding not in findings:
        findings.append(finding)


# ============================================================
# SEMANTIC TYPE
# ============================================================

def infer_semantic_type(
    feature: str,
    series: pd.Series,
    dictionary_type,
) -> str:
    """
    Determine the analytical semantic type.

    Priority:
        explicit semantic knowledge
        -> observed data
        -> dictionary datatype
        -> pandas dtype

    pandas dtype alone is NOT considered sufficient.
    """

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    if feature == TARGET:
        return "TARGET"

    # --------------------------------------------------------
    # IDENTIFIERS
    # --------------------------------------------------------

    if (
        feature in PURE_IDENTIFIERS
        or feature in SITE_IDENTIFIERS
    ):
        return "IDENTIFIER"

    # --------------------------------------------------------
    # DERIVED PREDICTIONS
    # --------------------------------------------------------

    if feature in DERIVED_PREDICTION_FEATURES:
        return "DERIVED_PREDICTION"

    # --------------------------------------------------------
    # EXPLICIT SEMANTIC OVERRIDES
    # --------------------------------------------------------

    if feature in CATEGORICAL_CODE_FEATURES:
        return "CATEGORICAL_CODE"

    if feature in ORDINAL_FEATURES:
        return "ORDINAL"

    if feature in CONTINUOUS_OVERRIDES:
        return "CONTINUOUS"

    # --------------------------------------------------------
    # OBSERVED VALUES
    # --------------------------------------------------------

    observed = series.dropna()

    unique_values = observed.nunique()

    # --------------------------------------------------------
    # STRING / CATEGORY
    # --------------------------------------------------------

    if (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(
            series.dtype,
            pd.CategoricalDtype,
        )
    ):
        return "CATEGORICAL"

    # --------------------------------------------------------
    # BINARY
    # --------------------------------------------------------

    if unique_values <= 2:

        unique_set = set(
            observed.unique().tolist()
        )

        if unique_set.issubset({0, 1}):
            return "BINARY"

    # --------------------------------------------------------
    # DICTIONARY
    # --------------------------------------------------------

    dictionary_type_normalized = (
        str(dictionary_type)
        .strip()
        .lower()
    )

    # --------------------------------------------------------
    # NUMERIC
    # --------------------------------------------------------

    if pd.api.types.is_numeric_dtype(series):

        # Dictionary explicitly defines a numeric measurement.
        if dictionary_type_normalized == "numeric":
            return "CONTINUOUS"

        # Detect small integer-like discrete variables.
        if len(observed) > 0:

            values = observed.to_numpy(
                dtype=float
            )

            integer_like = np.all(
                np.isclose(
                    values,
                    np.round(values),
                )
            )

            if (
                integer_like
                and unique_values <= 20
            ):
                return "DISCRETE"

        return "CONTINUOUS"

    return "CATEGORICAL"


# ============================================================
# CATEGORY CONSISTENCY
# ============================================================

def find_case_inconsistent_categories(
    series: pd.Series,
) -> list[list[str]]:
    """
    Detect categories differing only by case / whitespace.

    Example:
        Undefined diagnoses
        Undefined Diagnoses
    """

    values = (
        series
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )

    normalized_groups = {}

    for value in values:

        key = value.casefold()

        normalized_groups.setdefault(
            key,
            [],
        ).append(value)

    inconsistencies = []

    for categories in normalized_groups.values():

        unique_categories = list(
            dict.fromkeys(categories)
        )

        if len(unique_categories) > 1:
            inconsistencies.append(
                unique_categories
            )

    return inconsistencies


# ============================================================
# A1 - DISTRIBUTION FINDINGS
# ============================================================

def collect_a1_findings(
    row,
    findings: list[str],
) -> None:

    if row is None:
        return

    review_reason = get_value(
        row,
        "review_reason",
    )

    if (
        review_reason is not None
        and pd.notna(review_reason)
    ):

        reasons = str(
            review_reason
        ).split("|")

        for reason in reasons:

            reason = reason.strip()

            if (
                reason
                and reason != "NO_REVIEW"
            ):
                add_finding(
                    findings,
                    f"A1_{reason}",
                )


# ============================================================
# A2 - OUTLIER FINDINGS
# ============================================================

def collect_a2_findings(
    row,
    findings: list[str],
) -> None:

    if row is None:
        return

    outlier_percent = get_value(
        row,
        "outlier_percent_observed",
    )

    if (
        outlier_percent is None
        or pd.isna(outlier_percent)
    ):
        return

    if outlier_percent >= 10:

        add_finding(
            findings,
            "A2_HIGH_OUTLIER_PREVALENCE",
        )

    elif outlier_percent >= 5:

        add_finding(
            findings,
            "A2_NOTICEABLE_OUTLIERS",
        )


# ============================================================
# A3 - RANGE / SUSPICIOUS VALUES
# ============================================================

def collect_a3_findings(
    feature: str,
    semantic_type: str,
    row,
    findings: list[str],
) -> None:

    if row is None:
        return

    negative_percent = get_value(
        row,
        "negative_values_percent",
    )

    zero_percent = get_value(
        row,
        "zero_values_percent",
    )

    if (
        negative_percent is not None
        and pd.notna(negative_percent)
        and negative_percent > 0
    ):
        add_finding(
            findings,
            "A3_HAS_NEGATIVE_VALUES",
        )

    # Zero is useful mainly for quantitative measurements.
    if (
        semantic_type
        in {"CONTINUOUS", "DISCRETE"}
        and zero_percent is not None
        and pd.notna(zero_percent)
        and zero_percent > 0
    ):
        add_finding(
            findings,
            "A3_HAS_ZERO_VALUES",
        )

    # Explicit probability-domain issue.
    if feature in DERIVED_PREDICTION_FEATURES:

        min_value = get_value(
            row,
            "min_value",
        )

        if (
            min_value is not None
            and pd.notna(min_value)
            and min_value < 0
        ):
            add_finding(
                findings,
                "A3_VALUE_OUTSIDE_PROBABILITY_DOMAIN",
            )


# ============================================================
# A4 - NEAR CONSTANT
# ============================================================

def collect_a4_findings(
    row,
    findings: list[str],
) -> None:

    if row is None:
        return

    dominant_percent = get_value(
        row,
        "dominant_value_percent",
    )

    if (
        dominant_percent is None
        or pd.isna(dominant_percent)
    ):
        return

    if dominant_percent > 99:

        add_finding(
            findings,
            "A4_EXTREME_VALUE_DOMINANCE",
        )

    elif dominant_percent > 98:

        add_finding(
            findings,
            "A4_VERY_HIGH_VALUE_DOMINANCE",
        )

    elif dominant_percent >= 95:

        add_finding(
            findings,
            "A4_HIGH_VALUE_DOMINANCE",
        )


# ============================================================
# B1 - CARDINALITY
# ============================================================

def collect_b1_findings(
    row,
    findings: list[str],
) -> None:

    if row is None:
        return

    cardinality_level = get_value(
        row,
        "cardinality_level",
    )

    if (
        cardinality_level is not None
        and pd.notna(cardinality_level)
    ):

        level = str(
            cardinality_level
        ).strip().upper()

        if level == "HIGH":
            add_finding(
                findings,
                "B1_HIGH_CARDINALITY",
            )

        elif level == "MODERATE":
            add_finding(
                findings,
                "B1_MODERATE_CARDINALITY",
            )


# ============================================================
# B3 - DOMINANT CATEGORY
# ============================================================

def collect_b3_findings(
    row,
    findings: list[str],
) -> None:

    if row is None:
        return

    dominant_percent = get_value(
        row,
        "dominant_category_percent",
    )

    if (
        dominant_percent is None
        or pd.isna(dominant_percent)
    ):
        return

    if dominant_percent > 95:

        add_finding(
            findings,
            "B3_EXTREME_CATEGORY_DOMINANCE",
        )

    elif dominant_percent > 90:

        add_finding(
            findings,
            "B3_VERY_HIGH_CATEGORY_DOMINANCE",
        )

    elif dominant_percent >= 80:

        add_finding(
            findings,
            "B3_HIGH_CATEGORY_DOMINANCE",
        )


# ============================================================
# B4 - RARE CATEGORIES
# ============================================================

def collect_b4_findings(
    row,
    findings: list[str],
) -> None:

    if row is None:
        return

    rare_count = get_value(
        row,
        "rare_categories_count",
    )

    if (
        rare_count is not None
        and pd.notna(rare_count)
        and rare_count > 0
    ):
        add_finding(
            findings,
            "B4_HAS_RARE_CATEGORIES",
        )


# ============================================================
# BUILD ALL LEVEL 1 FINDINGS
# ============================================================

def build_level1_findings(
    feature: str,
    semantic_type: str,
    missing_percent: float,
    unique_values: int,
    a1_row,
    a2_row,
    a3_row,
    a4_row,
    b1_row,
    b3_row,
    b4_row,
    category_inconsistencies,
) -> list[str]:

    findings = []

    # --------------------------------------------------------
    # BASIC DATA QUALITY
    # --------------------------------------------------------

    if missing_percent >= EXTREME_MISSINGNESS_THRESHOLD:

        add_finding(
            findings,
            "EXTREME_MISSINGNESS",
        )

    elif missing_percent > HIGH_MISSINGNESS_THRESHOLD:

        add_finding(
            findings,
            "HIGH_MISSINGNESS",
        )

    if unique_values <= 1:

        add_finding(
            findings,
            "CONSTANT_FEATURE",
        )

    # --------------------------------------------------------
    # EDA ANALYSES
    # --------------------------------------------------------

    collect_a1_findings(
        a1_row,
        findings,
    )

    collect_a2_findings(
        a2_row,
        findings,
    )

    collect_a3_findings(
        feature,
        semantic_type,
        a3_row,
        findings,
    )

    collect_a4_findings(
        a4_row,
        findings,
    )

    collect_b1_findings(
        b1_row,
        findings,
    )

    collect_b3_findings(
        b3_row,
        findings,
    )

    collect_b4_findings(
        b4_row,
        findings,
    )

    # --------------------------------------------------------
    # CATEGORY CONSISTENCY
    # --------------------------------------------------------

    if category_inconsistencies:

        add_finding(
            findings,
            "CATEGORY_LABEL_INCONSISTENCY",
        )

    # --------------------------------------------------------
    # SEMANTIC FINDINGS
    # --------------------------------------------------------

    if feature in DERIVED_PREDICTION_FEATURES:

        add_finding(
            findings,
            "DERIVED_PREDICTION_REVIEW",
        )

    if feature in CATEGORICAL_CODE_FEATURES:

        add_finding(
            findings,
            "NUMERIC_CATEGORICAL_CODE",
        )

    if feature in CONTINUOUS_OVERRIDES:

        add_finding(
            findings,
            "DICTIONARY_ANALYTICAL_TYPE_MISMATCH",
        )

    if feature == "pre_icu_los_days":

        add_finding(
            findings,
            "REPEATED_A1_A2_A3_FOLLOWUP",
        )

    return findings


# ============================================================
# BASELINE IMPUTATION POLICY
# ============================================================

def has_baseline_imputation_strategy(
    semantic_type: str,
) -> bool:
    """
    For the 50-80% missingness branch.

    Baseline retention is currently allowed for
    quantitative / ordered numeric features because
    a robust baseline imputation strategy exists.

    This remains a baseline policy, not a final rule.
    """

    return semantic_type in {
        "CONTINUOUS",
        "DISCRETE",
        "ORDINAL",
        "BINARY",
    }


# ============================================================
# DEFAULT BASELINE ACTION
# ============================================================

def get_default_baseline_action(
    feature: str,
    semantic_type: str,
    missing_percent: float,
) -> str:

    if semantic_type == "CATEGORICAL":

        if feature in UNKNOWN_CATEGORY_FEATURES:

            return (
                "ADD_UNKNOWN_CATEGORY_THEN_ENCODE"
            )

        if missing_percent > 0:

            return (
                "MODE_IMPUTATION_THEN_ENCODE"
            )

        return "ENCODE_LATER"

    if semantic_type == "CATEGORICAL_CODE":

        return (
            "KEEP_RAW_UNTIL_ENCODING_STRATEGY_DEFINED"
        )

    if semantic_type == "ORDINAL":

        if missing_percent > 0:
            return "NUMERIC_IMPUTATION_KEEP_ORDER"

        return "KEEP_ORDINAL"

    if semantic_type in {
        "CONTINUOUS",
        "DISCRETE",
        "BINARY",
    }:

        if missing_percent > 0:

            return "NUMERIC_BASELINE_IMPUTATION"

        return "KEEP_RAW"

    return "KEEP_RAW"


# ============================================================
# DECISION TREE
# ============================================================

def decide_feature(
    feature: str,
    semantic_type: str,
    missing_percent: float,
    unique_values: int,
    findings: list[str],
) -> dict:
    """
    Apply the official EDA Level 1 decision tree.

    The order below is important.

    A stronger rule prevents weaker statistical findings
    from overriding the decision.
    """

    # ========================================================
    # NODE 1 - TARGET
    # ========================================================

    if feature == TARGET:

        return {
            "level1_status": "TARGET",
            "decision_rule": "01_TARGET",
            "decision_reason": "TARGET_VARIABLE",
            "baseline_action": "TARGET_ONLY",
            "level2_followup":
                "IMBALANCE_AWARE_MODEL_EVALUATION",
        }

    # ========================================================
    # NODE 2 - PURE IDENTIFIER
    # ========================================================

    if feature in PURE_IDENTIFIERS:

        return {
            "level1_status": "DROP",
            "decision_rule": "02_PURE_IDENTIFIER",
            "decision_reason": "PURE_IDENTIFIER",
            "baseline_action": "REMOVE",
            "level2_followup": "NONE",
        }

    # ========================================================
    # NODE 3 - SITE / UNIT IDENTIFIER
    # ========================================================

    if feature in SITE_IDENTIFIERS:

        return {
            "level1_status": "INVESTIGATE",
            "decision_rule": "03_SITE_IDENTIFIER",
            "decision_reason":
                "SITE_OR_UNIT_IDENTIFIER",
            "baseline_action":
                "KEEP_OUT_OF_BASELINE_UNTIL_REVIEWED",
            "level2_followup":
                "SITE_EFFECT_AND_GENERALIZATION",
        }

    # ========================================================
    # NODE 4 - CONSTANT
    # ========================================================

    if unique_values <= 1:

        return {
            "level1_status": "DROP",
            "decision_rule": "04_CONSTANT",
            "decision_reason":
                "CONSTANT_FEATURE",
            "baseline_action": "REMOVE",
            "level2_followup": "NONE",
        }

    # ========================================================
    # NODE 5 - DERIVED PREDICTION / LEAKAGE
    # ========================================================

    if feature in DERIVED_PREDICTION_FEATURES:

        return {
            "level1_status": "INVESTIGATE",
            "decision_rule":
                "05_DERIVED_PREDICTION",
            "decision_reason":
                "DERIVED_PREDICTION_OR_LEAKAGE_RISK",
            "baseline_action":
                "KEEP_OUT_OF_MODEL_UNTIL_RESOLVED",
            "level2_followup":
                (
                    "VERIFY_SENTINEL_TIMING_"
                    "AVAILABILITY_AND_LEAKAGE"
                ),
        }

    # ========================================================
    # NODE 6 - SPECIAL SEMANTIC INVESTIGATION
    # ========================================================

    if feature in SPECIAL_INVESTIGATION_FEATURES:

        return {
            "level1_status": "INVESTIGATE",
            "decision_rule":
                "06_SPECIAL_SEMANTIC_INVESTIGATION",
            "decision_reason":
                (
                    "REPEATED_UNUSUAL_BEHAVIOR_"
                    "ACROSS_LEVEL1"
                ),
            "baseline_action":
                "KEEP_RAW_UNTIL_RESOLVED",
            "level2_followup":
                (
                    "VERIFY_FEATURE_DEFINITION_"
                    "AND_TARGET_RELATIONSHIP"
                ),
        }

    # ========================================================
    # NODE 7 - EXTREME MISSINGNESS
    # ========================================================

    if (
        missing_percent
        >= EXTREME_MISSINGNESS_THRESHOLD
    ):

        return {
            "level1_status": "DROP_BASELINE",
            "decision_rule":
                "07_EXTREME_MISSINGNESS",
            "decision_reason":
                "MISSINGNESS_GE_80_PERCENT",
            "baseline_action":
                "REMOVE_FROM_BASELINE",
            "level2_followup":
                "OPTIONAL_REVISIT_AFTER_BASELINE",
        }

    # ========================================================
    # NODE 8 - DATA CONSISTENCY
    # ========================================================

    if (
        "CATEGORY_LABEL_INCONSISTENCY"
        in findings
    ):

        return {
            "level1_status": "CLEAN_KEEP",
            "decision_rule":
                "08_CATEGORY_CONSISTENCY",
            "decision_reason":
                "CATEGORY_LABEL_INCONSISTENCY",
            "baseline_action":
                (
                    "VERIFY_AND_STANDARDIZE_"
                    "CATEGORY_LABELS"
                ),
            "level2_followup":
                "TARGET_RELATIONSHIP_AFTER_CLEANING",
        }

    # ========================================================
    # NODE 9 - HIGH MISSINGNESS 50-80%
    # ========================================================

    if (
        missing_percent
        > HIGH_MISSINGNESS_THRESHOLD
    ):

        if has_baseline_imputation_strategy(
            semantic_type
        ):

            return {
                "level1_status":
                    "KEEP_FOR_NOW",
                "decision_rule":
                    "09_HIGH_MISSINGNESS_IMPUTABLE",
                "decision_reason":
                    (
                        "MISSINGNESS_50_TO_80_"
                        "WITH_BASELINE_IMPUTATION"
                    ),
                "baseline_action":
                    "NUMERIC_BASELINE_IMPUTATION",
                "level2_followup":
                    (
                        "TARGET_RELATIONSHIP_AND_"
                        "REASSESS_MISSINGNESS"
                    ),
            }

        return {
            "level1_status": "DROP_BASELINE",
            "decision_rule":
                "09_HIGH_MISSINGNESS_NO_STRATEGY",
            "decision_reason":
                (
                    "MISSINGNESS_50_TO_80_"
                    "WITHOUT_RETENTION_RULE"
                ),
            "baseline_action":
                "REMOVE_FROM_BASELINE",
            "level2_followup":
                "OPTIONAL_REVISIT_AFTER_BASELINE",
        }

    # ========================================================
    # NODE 10 - PHYSIOLOGICAL ZERO
    # ========================================================

    feature_lower = feature.lower()

    physiological_zero = (
        "A3_HAS_ZERO_VALUES" in findings
        and any(
            keyword in feature_lower
            for keyword
            in PHYSIOLOGICAL_ZERO_KEYWORDS
        )
    )

    if physiological_zero:

        return {
            "level1_status": "KEEP_FOR_NOW",
            "decision_rule":
                "10_PHYSIOLOGICAL_ZERO",
            "decision_reason":
                (
                    "ZERO_MAY_REPRESENT_"
                    "CLINICAL_SIGNAL"
                ),
            "baseline_action":
                get_default_baseline_action(
                    feature,
                    semantic_type,
                    missing_percent,
                ),
            "level2_followup":
                "ZERO_VALUE_VS_TARGET_ANALYSIS",
        }

    # ========================================================
    # NODE 11 - CATEGORICAL CODE
    # ========================================================

    if semantic_type == "CATEGORICAL_CODE":

        return {
            "level1_status": "KEEP_FOR_NOW",
            "decision_rule":
                "11_CATEGORICAL_CODE",
            "decision_reason":
                (
                    "NUMERIC_STORAGE_WITH_"
                    "CATEGORICAL_SEMANTICS"
                ),
            "baseline_action":
                (
                    "KEEP_RAW_UNTIL_ENCODING_"
                    "STRATEGY_DEFINED"
                ),
            "level2_followup":
                (
                    "TARGET_RELATIONSHIP_AND_"
                    "ENCODING_REVIEW"
                ),
        }

    # ========================================================
    # NODE 12 - ORDINAL
    # ========================================================

    if semantic_type == "ORDINAL":

        return {
            "level1_status": "KEEP_FOR_NOW",
            "decision_rule": "12_ORDINAL",
            "decision_reason":
                "ORDERED_CLINICAL_FEATURE",
            "baseline_action":
                get_default_baseline_action(
                    feature,
                    semantic_type,
                    missing_percent,
                ),
            "level2_followup":
                "TARGET_RELATIONSHIP",
        }

    # ========================================================
    # NODE 13 - BINARY
    # ========================================================

    if semantic_type == "BINARY":

        if (
            "A4_EXTREME_VALUE_DOMINANCE"
            in findings
            or
            "A4_VERY_HIGH_VALUE_DOMINANCE"
            in findings
        ):

            return {
                "level1_status":
                    "KEEP_FOR_NOW",
                "decision_rule":
                    "13_RARE_BINARY",
                "decision_reason":
                    (
                        "RARE_BINARY_CLINICAL_"
                        "INDICATOR"
                    ),
                "baseline_action":
                    get_default_baseline_action(
                        feature,
                        semantic_type,
                        missing_percent,
                    ),
                "level2_followup":
                    "POSITIVE_CLASS_VS_TARGET",
            }

        return {
            "level1_status": "KEEP_FOR_NOW",
            "decision_rule":
                "13_BINARY_DEFAULT",
            "decision_reason":
                "VALID_BINARY_FEATURE",
            "baseline_action":
                get_default_baseline_action(
                    feature,
                    semantic_type,
                    missing_percent,
                ),
            "level2_followup":
                "TARGET_RELATIONSHIP",
        }

    # ========================================================
    # NODE 14 - CATEGORICAL
    # ========================================================

    if semantic_type == "CATEGORICAL":

        return {
            "level1_status": "KEEP_FOR_NOW",
            "decision_rule":
                "14_CATEGORICAL_DEFAULT",
            "decision_reason":
                "NO_LEVEL1_REASON_TO_REMOVE",
            "baseline_action":
                get_default_baseline_action(
                    feature,
                    semantic_type,
                    missing_percent,
                ),
            "level2_followup":
                "TARGET_RELATIONSHIP",
        }

    # ========================================================
    # NODE 15 - CONTINUOUS / DISCRETE
    # ========================================================

    if semantic_type in {
        "CONTINUOUS",
        "DISCRETE",
    }:

        # Skewness / outliers are deliberately not
        # automatic reasons for removal or transformation.

        return {
            "level1_status": "KEEP_FOR_NOW",
            "decision_rule":
                "15_NUMERIC_DEFAULT",
            "decision_reason":
                "NO_LEVEL1_REASON_TO_REMOVE",
            "baseline_action":
                get_default_baseline_action(
                    feature,
                    semantic_type,
                    missing_percent,
                ),
            "level2_followup":
                "TARGET_RELATIONSHIP",
        }

    # ========================================================
    # FALLBACK
    # ========================================================

    return {
        "level1_status": "KEEP_FOR_NOW",
        "decision_rule": "99_FALLBACK",
        "decision_reason":
            "NO_LEVEL1_REASON_TO_REMOVE",
        "baseline_action": "KEEP_RAW",
        "level2_followup":
            "TARGET_RELATIONSHIP",
    }


# ============================================================
# BUILD FEATURE REGISTRY
# ============================================================

def build_feature_registry() -> pd.DataFrame:

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    print("Loading training dataset...")

    df = pd.read_csv(
        DATA_PATH
    )

    print(
        f"Dataset shape: {df.shape}"
    )

    # --------------------------------------------------------
    # DICTIONARY
    # --------------------------------------------------------

    print("Loading dictionary...")

    dictionary = pd.read_csv(
        DICTIONARY_PATH
    )

    dictionary = normalize_dictionary(
        dictionary
    )

    dictionary_lookup = (
        dictionary
        .drop_duplicates(
            subset="variable_name"
        )
        .set_index("variable_name")
    )

    # --------------------------------------------------------
    # EDA TABLES
    # --------------------------------------------------------

    print("Loading EDA Level 1 tables...")

    a1 = load_optional_csv(A1_PATH)
    a2 = load_optional_csv(A2_PATH)
    a3 = load_optional_csv(A3_PATH)
    a4 = load_optional_csv(A4_PATH)

    b1 = load_optional_csv(B1_PATH)
    b3 = load_optional_csv(B3_PATH)
    b4 = load_optional_csv(B4_PATH)

    records = []

    total_rows = len(df)

    # ========================================================
    # FEATURE LOOP
    # ========================================================

    for feature in df.columns:

        series = df[feature]

        missing_percent = (
            series.isna().mean()
            * 100
        )

        unique_values = (
            series.nunique(
                dropna=True
            )
        )

        # ----------------------------------------------------
        # DICTIONARY METADATA
        # ----------------------------------------------------

        dictionary_type = np.nan
        feature_description = np.nan

        if feature in dictionary_lookup.index:

            dictionary_row = (
                dictionary_lookup.loc[
                    feature
                ]
            )

            dictionary_type = (
                dictionary_row.get(
                    "data_type",
                    np.nan,
                )
            )

            feature_description = (
                dictionary_row.get(
                    "description",
                    np.nan,
                )
            )

        # ----------------------------------------------------
        # SEMANTIC TYPE
        # ----------------------------------------------------

        semantic_type = (
            infer_semantic_type(
                feature=feature,
                series=series,
                dictionary_type=
                    dictionary_type,
            )
        )

        # ----------------------------------------------------
        # GET EDA RESULTS
        # ----------------------------------------------------

        a1_row = get_feature_row(
            a1,
            feature,
        )

        a2_row = get_feature_row(
            a2,
            feature,
        )

        a3_row = get_feature_row(
            a3,
            feature,
        )

        a4_row = get_feature_row(
            a4,
            feature,
        )

        b1_row = get_feature_row(
            b1,
            feature,
        )

        b3_row = get_feature_row(
            b3,
            feature,
        )

        b4_row = get_feature_row(
            b4,
            feature,
        )

        # ----------------------------------------------------
        # CATEGORY CONSISTENCY
        # ----------------------------------------------------

        category_inconsistencies = []

        if semantic_type == "CATEGORICAL":

            category_inconsistencies = (
                find_case_inconsistent_categories(
                    series
                )
            )

        # ----------------------------------------------------
        # LEVEL 1 FINDINGS
        # ----------------------------------------------------

        findings = build_level1_findings(
            feature=feature,
            semantic_type=semantic_type,
            missing_percent=
                missing_percent,
            unique_values=
                unique_values,
            a1_row=a1_row,
            a2_row=a2_row,
            a3_row=a3_row,
            a4_row=a4_row,
            b1_row=b1_row,
            b3_row=b3_row,
            b4_row=b4_row,
            category_inconsistencies=
                category_inconsistencies,
        )

        # ----------------------------------------------------
        # DECISION TREE
        # ----------------------------------------------------

        decision = decide_feature(
            feature=feature,
            semantic_type=semantic_type,
            missing_percent=
                missing_percent,
            unique_values=
                unique_values,
            findings=findings,
        )

        # ----------------------------------------------------
        # CATEGORY INCONSISTENCY TEXT
        # ----------------------------------------------------

        if category_inconsistencies:

            inconsistency_text = "; ".join(
                [
                    " / ".join(group)
                    for group
                    in category_inconsistencies
                ]
            )

        else:

            inconsistency_text = ""

        # ----------------------------------------------------
        # RECORD
        # ----------------------------------------------------

        records.append(
            {
                "feature":
                    feature,

                "semantic_type":
                    semantic_type,

                "dictionary_data_type":
                    dictionary_type,

                "feature_description":
                    feature_description,

                "missing_percent":
                    round(
                        missing_percent,
                        2,
                    ),

                "unique_values":
                    unique_values,

                "level1_findings":
                    (
                        " | ".join(findings)
                        if findings
                        else
                        "NO_SPECIAL_FINDING"
                    ),

                "category_inconsistency":
                    inconsistency_text,

                "level1_status":
                    decision[
                        "level1_status"
                    ],

                "decision_rule":
                    decision[
                        "decision_rule"
                    ],

                "decision_reason":
                    decision[
                        "decision_reason"
                    ],

                "baseline_action":
                    decision[
                        "baseline_action"
                    ],

                "level2_followup":
                    decision[
                        "level2_followup"
                    ],
            }
        )

    registry = pd.DataFrame(
        records
    )

    return registry


# ============================================================
# SANITY CHECKS
# ============================================================

def run_sanity_checks(
    registry: pd.DataFrame,
) -> None:

    print("\n")
    print("=" * 80)
    print("SANITY CHECKS")
    print("=" * 80)

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    target_rows = registry[
        registry["level1_status"]
        == "TARGET"
    ]

    if len(target_rows) != 1:

        print(
            "[WARNING] Expected exactly "
            "one TARGET feature."
        )

    # --------------------------------------------------------
    # CONSTANT
    # --------------------------------------------------------

    constant_not_dropped = registry[
        (registry["unique_values"] <= 1)
        &
        (~registry["level1_status"].isin(
            ["DROP", "TARGET"]
        ))
    ]

    if not constant_not_dropped.empty:

        print(
            "\n[WARNING] Constant features "
            "not dropped:"
        )

        print(
            constant_not_dropped[
                [
                    "feature",
                    "level1_status",
                ]
            ].to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # EXTREME MISSINGNESS
    # --------------------------------------------------------

    extreme_missing_kept = registry[
        (
            registry["missing_percent"]
            >= EXTREME_MISSINGNESS_THRESHOLD
        )
        &
        (
            registry["level1_status"]
            == "KEEP_FOR_NOW"
        )
    ]

    if not extreme_missing_kept.empty:

        print(
            "\n[WARNING] >=80% missing "
            "features still marked KEEP:"
        )

        print(
            extreme_missing_kept[
                [
                    "feature",
                    "missing_percent",
                    "level1_status",
                ]
            ].to_string(
                index=False
            )
        )

    print("\nSanity checks completed.")


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_registry_summary(
    registry: pd.DataFrame,
) -> None:

    print("\n")
    print("=" * 80)
    print(
        "EDA LEVEL 1 FEATURE DECISION REGISTRY"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    print("\nSTATUS DISTRIBUTION")
    print("-" * 80)

    print(
        registry[
            "level1_status"
        ]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # SEMANTIC TYPES
    # --------------------------------------------------------

    print("\nSEMANTIC TYPE DISTRIBUTION")
    print("-" * 80)

    print(
        registry[
            "semantic_type"
        ]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # DROP
    # --------------------------------------------------------

    removed = registry[
        registry[
            "level1_status"
        ].isin(
            [
                "DROP",
                "DROP_BASELINE",
            ]
        )
    ]

    print("\nDROP / DROP_BASELINE")
    print("-" * 80)

    if removed.empty:

        print("None")

    else:

        print(
            removed[
                [
                    "feature",
                    "missing_percent",
                    "level1_status",
                    "decision_reason",
                ]
            ].to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # INVESTIGATE
    # --------------------------------------------------------

    investigate = registry[
        registry["level1_status"]
        == "INVESTIGATE"
    ]

    print("\nINVESTIGATE")
    print("-" * 80)

    if investigate.empty:

        print("None")

    else:

        print(
            investigate[
                [
                    "feature",
                    "decision_reason",
                    "level2_followup",
                ]
            ].to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # CLEAN
    # --------------------------------------------------------

    clean = registry[
        registry["level1_status"]
        == "CLEAN_KEEP"
    ]

    print("\nCLEAN_KEEP")
    print("-" * 80)

    if clean.empty:

        print("None")

    else:

        print(
            clean[
                [
                    "feature",
                    "category_inconsistency",
                    "baseline_action",
                ]
            ].to_string(
                index=False
            )
        )


# ============================================================
# SEMANTIC TYPE REVIEW TABLE
# ============================================================

def save_semantic_review_table(
    registry: pd.DataFrame,
) -> None:

    output_path = (
        REPORTS_PATH
        / "eda_level1_semantic_type_review.csv"
    )

    review = registry[
        [
            "feature",
            "semantic_type",
            "dictionary_data_type",
            "unique_values",
            "missing_percent",
            "level1_status",
        ]
    ].copy()

    review.to_csv(
        output_path,
        index=False,
    )

    print(
        "\nSemantic review table saved to:"
    )

    print(output_path)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    registry = build_feature_registry()

    # --------------------------------------------------------
    # SAVE MAIN REGISTRY
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    registry.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # CHECK / REPORT
    # --------------------------------------------------------

    run_sanity_checks(
        registry
    )

    print_registry_summary(
        registry
    )

    save_semantic_review_table(
        registry
    )

    print("\n")
    print("=" * 80)

    print(
        "Main registry saved to:"
    )

    print(
        OUTPUT_PATH
    )

    print("=" * 80)


if __name__ == "__main__":
    main()