"""EDA Level 1 - C.1: Target class distribution analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from icu_mortality.data import (
    TARGET_COLUMN,
    load_training_data,
)


REPORTS_TABLES_DIR = Path("reports") / "tables"

REPORTS_FIGURES_DIR = (
    Path("reports")
    / "figures"
    / "eda_level1"
    / "target"
)

SUMMARY_OUTPUT_PATH = (
    REPORTS_TABLES_DIR
    / "eda_level1_c1_target_distribution_summary.csv"
)

FIGURE_OUTPUT_PATH = (
    REPORTS_FIGURES_DIR
    / "hospital_death_distribution.png"
)


# ---------------------------------------------------------------------
# Target analysis
# ---------------------------------------------------------------------

def analyze_target_distribution(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate target class counts and percentages."""

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' "
            "was not found in the dataframe."
        )

    target = dataframe[
        TARGET_COLUMN
    ]

    if target.isna().any():
        raise ValueError(
            "Target column contains missing values."
        )

    counts = (
        target
        .value_counts()
        .sort_index()
    )

    percentages = (
        counts
        / len(target)
        * 100
    )

    results = pd.DataFrame(
        {
            "target_value": counts.index,
            "count": counts.values,
            "percent": percentages.values,
        }
    )

    return results


# ---------------------------------------------------------------------
# Class-imbalance summary
# ---------------------------------------------------------------------

def calculate_class_balance_summary(
    results: pd.DataFrame,
) -> dict[str, object]:
    """Calculate simple class-balance statistics."""

    if results.empty:
        return {
            "majority_class": None,
            "majority_percent": float("nan"),
            "minority_class": None,
            "minority_percent": float("nan"),
            "majority_to_minority_ratio": float("nan"),
        }

    sorted_results = (
        results
        .sort_values(
            by="count",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    majority_class = (
        sorted_results.loc[
            0,
            "target_value",
        ]
    )

    majority_percent = (
        sorted_results.loc[
            0,
            "percent",
        ]
    )

    minority_class = (
        sorted_results.loc[
            len(sorted_results) - 1,
            "target_value",
        ]
    )

    minority_percent = (
        sorted_results.loc[
            len(sorted_results) - 1,
            "percent",
        ]
    )

    minority_count = (
        sorted_results.loc[
            len(sorted_results) - 1,
            "count",
        ]
    )

    majority_count = (
        sorted_results.loc[
            0,
            "count",
        ]
    )

    if minority_count == 0:
        ratio = float("inf")
    else:
        ratio = (
            majority_count
            / minority_count
        )

    return {
        "majority_class": majority_class,
        "majority_percent": majority_percent,
        "minority_class": minority_class,
        "minority_percent": minority_percent,
        "majority_to_minority_ratio": ratio,
    }


# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------

def save_target_barplot(
    results: pd.DataFrame,
) -> None:
    """Save target class-distribution bar plot."""

    if results.empty:
        return

    REPORTS_FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(7, 5)
    )

    axis.bar(
        results[
            "target_value"
        ].astype(str),
        results[
            "percent"
        ],
    )

    axis.set_title(
        f"Class Distribution of {TARGET_COLUMN}"
    )

    axis.set_xlabel(
        TARGET_COLUMN
    )

    axis.set_ylabel(
        "Percentage of Rows (%)"
    )

    for index, row in results.iterrows():

        axis.text(
            index,
            row["percent"],
            f"{row['percent']:.2f}%",
            ha="center",
            va="bottom",
        )

    figure.tight_layout()

    figure.savefig(
        FIGURE_OUTPUT_PATH,
        dpi=150,
    )

    plt.close(
        figure
    )


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

def save_summary(
    results: pd.DataFrame,
) -> None:
    """Save target class-distribution table."""

    REPORTS_TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        SUMMARY_OUTPUT_PATH,
        index=False,
    )


def print_results(
    results: pd.DataFrame,
    balance_summary: dict[str, object],
) -> None:
    """Print C.1 target-distribution results."""

    print("=" * 120)
    print(
        "EDA LEVEL 1 — C.1 TARGET CLASS DISTRIBUTION"
    )
    print("=" * 120)

    print(
        f"Target column: "
        f"{TARGET_COLUMN}"
    )

    print(
        f"Rows analyzed: "
        f"{results['count'].sum()}"
    )

    print("\n" + "=" * 120)
    print(
        "CLASS DISTRIBUTION"
    )
    print("=" * 120)

    if results.empty:

        print(
            "No target values were found."
        )

    else:

        print(
            results.to_string(
                index=False,
                float_format=lambda value: (
                    f"{value:.2f}"
                ),
            )
        )

    print("\n" + "=" * 120)
    print(
        "CLASS BALANCE SUMMARY"
    )
    print("=" * 120)

    print(
        f"Majority class: "
        f"{balance_summary['majority_class']}"
    )

    print(
        f"Majority class percent: "
        f"{balance_summary['majority_percent']:.2f}%"
    )

    print(
        f"Minority class: "
        f"{balance_summary['minority_class']}"
    )

    print(
        f"Minority class percent: "
        f"{balance_summary['minority_percent']:.2f}%"
    )

    print(
        f"Majority/minority ratio: "
        f"{balance_summary['majority_to_minority_ratio']:.2f}"
    )

    print("\n" + "=" * 120)
    print(
        "ARTIFACTS SAVED"
    )
    print("=" * 120)

    print(
        SUMMARY_OUTPUT_PATH
    )

    print(
        FIGURE_OUTPUT_PATH
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def run_c1_target_distribution() -> pd.DataFrame:
    """Run EDA Level 1 C.1."""

    dataframe = (
        load_training_data()
    )

    results = (
        analyze_target_distribution(
            dataframe
        )
    )

    balance_summary = (
        calculate_class_balance_summary(
            results
        )
    )

    save_summary(
        results
    )

    save_target_barplot(
        results
    )

    print_results(
        results=results,
        balance_summary=balance_summary,
    )

    return results


if __name__ == "__main__":
    run_c1_target_distribution()