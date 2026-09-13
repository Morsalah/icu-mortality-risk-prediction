from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = Path("data/raw/training_v2.csv")

OUTPUT_DIR = Path("data/processed")

REPORT_PATH = Path(
    "reports/tables/data_split_summary.csv"
)

TARGET = "hospital_death"

RANDOM_STATE = 42

TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15


# ============================================================
# LOAD DATA
# ============================================================

def load_data() -> pd.DataFrame:
    """
    Load the complete labeled dataset.

    Important:
    No feature selection or preprocessing is performed here.
    The split is intentionally independent of any experiment.
    """

    print("Loading labeled dataset...")

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset shape: {df.shape}")

    return df


# ============================================================
# VALIDATE DATA
# ============================================================

def validate_data(df: pd.DataFrame) -> None:
    """
    Validate requirements before creating the fixed split.
    """

    if TARGET not in df.columns:
        raise ValueError(
            f"Target '{TARGET}' was not found."
        )

    if df[TARGET].isna().any():
        raise ValueError(
            "Target contains missing values."
        )

    target_values = set(
        df[TARGET].unique()
    )

    if not target_values.issubset({0, 1}):
        raise ValueError(
            f"Unexpected target values: {target_values}"
        )

    if df.columns.duplicated().any():
        duplicates = (
            df.columns[
                df.columns.duplicated()
            ]
            .tolist()
        )

        raise ValueError(
            f"Duplicate columns found: {duplicates}"
        )

    split_total = (
        TRAIN_SIZE
        + VALIDATION_SIZE
        + TEST_SIZE
    )

    if abs(split_total - 1.0) > 1e-10:
        raise ValueError(
            "Split proportions must sum to 1."
        )


# ============================================================
# CREATE FIXED SPLIT
# ============================================================

def create_fixed_split(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Create one permanent stratified split:

        Train          70%
        Validation     15%
        Internal Test  15%

    The complete raw rows are preserved.

    Feature selection and preprocessing are intentionally
    deferred to later experiment stages.
    """

    # --------------------------------------------------------
    # First split:
    # 70% Train / 30% Temporary
    # --------------------------------------------------------

    train_df, temp_df = train_test_split(
        df,
        test_size=(
            VALIDATION_SIZE
            + TEST_SIZE
        ),
        stratify=df[TARGET],
        random_state=RANDOM_STATE,
    )

    # --------------------------------------------------------
    # Second split:
    # Temporary -> 15% Validation / 15% Internal Test
    # --------------------------------------------------------

    relative_test_size = (
        TEST_SIZE
        / (
            VALIDATION_SIZE
            + TEST_SIZE
        )
    )

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        stratify=temp_df[TARGET],
        random_state=RANDOM_STATE,
    )

    return (
        train_df,
        validation_df,
        test_df,
    )


# ============================================================
# SANITY CHECKS
# ============================================================

def run_sanity_checks(
    original_df: pd.DataFrame,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """
    Verify split integrity before saving.
    """

    # --------------------------------------------------------
    # Row count
    # --------------------------------------------------------

    total_rows = (
        len(train_df)
        + len(validation_df)
        + len(test_df)
    )

    if total_rows != len(original_df):
        raise ValueError(
            "Split row counts do not match original dataset."
        )

    # --------------------------------------------------------
    # Same columns
    # --------------------------------------------------------

    expected_columns = list(
        original_df.columns
    )

    for name, split_df in [
        ("train", train_df),
        ("validation", validation_df),
        ("internal_test", test_df),
    ]:

        if list(split_df.columns) != expected_columns:
            raise ValueError(
                f"Column mismatch in {name}."
            )

    # --------------------------------------------------------
    # No row overlap
    # --------------------------------------------------------

    train_indices = set(train_df.index)
    validation_indices = set(validation_df.index)
    test_indices = set(test_df.index)

    if train_indices & validation_indices:
        raise ValueError(
            "Train / Validation overlap detected."
        )

    if train_indices & test_indices:
        raise ValueError(
            "Train / Internal Test overlap detected."
        )

    if validation_indices & test_indices:
        raise ValueError(
            "Validation / Internal Test overlap detected."
        )

    # --------------------------------------------------------
    # All original rows represented exactly once
    # --------------------------------------------------------

    all_split_indices = (
        train_indices
        | validation_indices
        | test_indices
    )

    if all_split_indices != set(original_df.index):
        raise ValueError(
            "Not all original rows are represented "
            "exactly once."
        )

    print("\nAll split sanity checks passed.")


# ============================================================
# SPLIT SUMMARY
# ============================================================

def summarize_split(
    name: str,
    split_df: pd.DataFrame,
) -> dict:

    rows = len(split_df)

    death_count = int(
        (split_df[TARGET] == 1).sum()
    )

    survival_count = int(
        (split_df[TARGET] == 0).sum()
    )

    mortality_rate = (
        death_count
        / rows
        * 100
    )

    return {
        "split": name,
        "rows": rows,
        "survival_count": survival_count,
        "death_count": death_count,
        "mortality_rate_percent": mortality_rate,
    }


def build_summary(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:

    summary = pd.DataFrame(
        [
            summarize_split(
                "train",
                train_df,
            ),
            summarize_split(
                "validation",
                validation_df,
            ),
            summarize_split(
                "internal_test",
                test_df,
            ),
        ]
    )

    summary[
        "mortality_rate_percent"
    ] = (
        summary[
            "mortality_rate_percent"
        ]
        .round(4)
    )

    return summary


# ============================================================
# SAVE
# ============================================================

def save_outputs(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Reset index only when saving.
    # Original indices were kept until now for overlap checks.

    train_df.reset_index(
        drop=True
    ).to_csv(
        OUTPUT_DIR / "train.csv",
        index=False,
    )

    validation_df.reset_index(
        drop=True
    ).to_csv(
        OUTPUT_DIR / "validation.csv",
        index=False,
    )

    test_df.reset_index(
        drop=True
    ).to_csv(
        OUTPUT_DIR / "internal_test.csv",
        index=False,
    )

    summary.to_csv(
        REPORT_PATH,
        index=False,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "\n"
        + "=" * 72
    )

    print(
        "STEP 3.3 — FIXED TRAIN / VALIDATION / INTERNAL TEST SPLIT"
    )

    print(
        "=" * 72
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_data(df)

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    (
        train_df,
        validation_df,
        test_df,
    ) = create_fixed_split(df)

    # --------------------------------------------------------
    # Integrity checks
    # --------------------------------------------------------

    run_sanity_checks(
        original_df=df,
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = build_summary(
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_outputs(
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        summary=summary,
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        "\nSplit summary:\n"
    )

    print(
        summary.to_string(index=False)
    )

    print(
        "\nSaved:"
    )

    print(
        "  data/processed/train.csv"
    )

    print(
        "  data/processed/validation.csv"
    )

    print(
        "  data/processed/internal_test.csv"
    )

    print(
        "  reports/tables/data_split_summary.csv"
    )

    print(
        "\nNo feature selection or preprocessing "
        "was performed."
    )

    print(
        "This split should remain fixed across "
        "all future experiments."
    )


if __name__ == "__main__":
    main()