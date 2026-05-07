"""Load local PXR challenge CSV splits into Polars DataFrames."""

from pathlib import Path

import polars as pl

_DEFAULT_DATA_DIR = Path("data")
_TRAIN_FILE = "train.csv"
_TEST_FILE = "test.csv"
_COUNTER_TRAIN_FILE = "counter_train.csv"
_TRAIN_SPLIT_FILE = "train_split.csv"
_VAL_SPLIT_FILE = "val_split.csv"


def load_train(path: Path = _DEFAULT_DATA_DIR / _TRAIN_FILE) -> pl.DataFrame:
    """Return the training split as a Polars DataFrame."""
    return pl.read_csv(path)


def load_test(path: Path = _DEFAULT_DATA_DIR / _TEST_FILE) -> pl.DataFrame:
    """Return the test split as a Polars DataFrame."""
    return pl.read_csv(path)


def load_counter_train(path: Path = _DEFAULT_DATA_DIR / _COUNTER_TRAIN_FILE) -> pl.DataFrame:
    """Return the counter-assay training split as a Polars DataFrame."""
    return pl.read_csv(path)


def load_train_split(path: Path = _DEFAULT_DATA_DIR / _TRAIN_SPLIT_FILE) -> pl.DataFrame:
    """Return the canonical training portion (from download-data) as a Polars DataFrame."""
    return pl.read_csv(path)


def load_val_split(path: Path = _DEFAULT_DATA_DIR / _VAL_SPLIT_FILE) -> pl.DataFrame:
    """Return the canonical validation portion (from download-data) as a Polars DataFrame."""
    return pl.read_csv(path)


def load_splits(
    train_path: Path = _DEFAULT_DATA_DIR / _TRAIN_FILE,
    test_path: Path = _DEFAULT_DATA_DIR / _TEST_FILE,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (train, test) DataFrames."""
    return load_train(train_path), load_test(test_path)
