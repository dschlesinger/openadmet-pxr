"""Load local PXR challenge CSV splits into Polars DataFrames."""

from pathlib import Path

import polars as pl

_DEFAULT_DATA_DIR = Path("data")
_TRAIN_FILE = "train.csv"
_TEST_FILE = "test.csv"


def load_train(path: Path = _DEFAULT_DATA_DIR / _TRAIN_FILE) -> pl.DataFrame:
    """Return the training split as a Polars DataFrame."""
    return pl.read_csv(path)


def load_test(path: Path = _DEFAULT_DATA_DIR / _TEST_FILE) -> pl.DataFrame:
    """Return the test split as a Polars DataFrame."""
    return pl.read_csv(path)


def load_splits(
    train_path: Path = _DEFAULT_DATA_DIR / _TRAIN_FILE,
    test_path: Path = _DEFAULT_DATA_DIR / _TEST_FILE,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (train, test) DataFrames."""
    return load_train(train_path), load_test(test_path)
