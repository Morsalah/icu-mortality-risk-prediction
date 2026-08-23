"""Data-loading utilities for the ICU mortality project."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"

TRAINING_DATA_PATH = RAW_DATA_DIR / "training_v2.csv"

TARGET_COLUMN = "hospital_death"


def load_training_data(
    path: Path = TRAINING_DATA_PATH,
) -> pd.DataFrame:
    """Load and validate the raw WiDS training dataset.

    Parameters
    ----------
    path:
        Path to the raw WiDS training CSV file.

    Returns
    -------
    pd.DataFrame
        The raw training dataset.

    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist.

    ValueError
        If the dataset is empty or the target column is missing.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Training dataset was not found: {path.resolve()}"
        )

    try:
        dataframe = pd.read_csv(path)
    except pd.errors.EmptyDataError as error:
        raise ValueError(
            "Training dataset is empty."
        ) from error

    if dataframe.empty:
        raise ValueError(
            "Training dataset is empty."
        )

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' "
            "is missing from the training dataset."
        )

    return dataframe


if __name__ == "__main__":
    dataframe = load_training_data()

    print("=" * 60)
    print("RAW TRAINING DATA")
    print("=" * 60)

    print(
        f"Shape: {dataframe.shape}"
    )

    print(
        f"Rows: {len(dataframe)}"
    )

    print(
        f"Columns: {len(dataframe.columns)}"
    )

    print(
        f"Target column: {TARGET_COLUMN}"
    )

    print(
        "\nTarget distribution:"
    )

    print(
        dataframe[TARGET_COLUMN]
        .value_counts(
            dropna=False
        )
        .sort_index()
    )