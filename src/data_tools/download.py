"""Download PXR challenge dataset splits from HuggingFace to a local data folder."""

import argparse
from pathlib import Path

import polars as pl

_HF_BASE = "hf://datasets/openadmet/pxr-challenge-train-test/"
_HF_TRAIN = "pxr-challenge_TRAIN.csv"
_HF_TEST = "pxr-challenge_TEST_BLINDED.csv"
_HF_COUNTER_TRAIN = "pxr-challenge_counter-assay_TRAIN.csv"
_TRAIN_FILE = "train.csv"
_TEST_FILE = "test.csv"
_COUNTER_TRAIN_FILE = "counter_train.csv"
_TRAIN_SPLIT_FILE = "train_split.csv"
_VAL_SPLIT_FILE = "val_split.csv"
_DEFAULT_DATA_DIR = Path("data")
_DEFAULT_VAL_SPLIT = 0.1
_DEFAULT_SEED = 42


def _download_split(hf_path: str, local_path: Path) -> None:
    if local_path.exists():
        print(f"Skipping (already exists): {local_path}")
        return
    local_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {hf_path} ...")
    df = pl.read_csv(hf_path)
    df.write_csv(local_path)
    print(f"Saved {len(df)} rows -> {local_path}")


def _write_train_val_split(
    train_path: Path,
    train_split_path: Path,
    val_split_path: Path,
    val_fraction: float,
    seed: int,
) -> None:
    if train_split_path.exists() and val_split_path.exists():
        print(f"Skipping split (already exist): {train_split_path}, {val_split_path}")
        return
    if not train_path.exists():
        print(f"Skipping split: source {train_path} does not exist.")
        return
    df = pl.read_csv(train_path)
    shuffled = df.sample(fraction=1.0, shuffle=True, seed=seed)
    n_val = max(1, int(len(shuffled) * val_fraction))
    val_df = shuffled[:n_val]
    train_df = shuffled[n_val:]
    train_split_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.write_csv(train_split_path)
    val_df.write_csv(val_split_path)
    print(f"Train split: {len(train_df)} rows -> {train_split_path}")
    print(f"Val split:   {len(val_df)} rows -> {val_split_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PXR challenge dataset splits.")
    parser.add_argument("--no-train", action="store_true", help="Skip the train split.")
    parser.add_argument("--no-test", action="store_true", help="Skip the test split.")
    parser.add_argument("--no-counter-train", action="store_true", help="Skip the counter-assay train split.")
    parser.add_argument(
        "--train-location",
        type=Path,
        default=_DEFAULT_DATA_DIR / _TRAIN_FILE,
        help="Destination path for the train CSV.",
    )
    parser.add_argument(
        "--test-location",
        type=Path,
        default=_DEFAULT_DATA_DIR / _TEST_FILE,
        help="Destination path for the test CSV.",
    )
    parser.add_argument(
        "--counter-train-location",
        type=Path,
        default=_DEFAULT_DATA_DIR / _COUNTER_TRAIN_FILE,
        help="Destination path for the counter-assay train CSV.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=_DEFAULT_VAL_SPLIT,
        help="Fraction of train.csv to hold out as validation (default: 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_DEFAULT_SEED,
        help="Random seed for the train/val shuffle (default: 42).",
    )
    parser.add_argument(
        "--train-split-location",
        type=Path,
        default=_DEFAULT_DATA_DIR / _TRAIN_SPLIT_FILE,
        help="Destination path for the training-portion CSV.",
    )
    parser.add_argument(
        "--val-split-location",
        type=Path,
        default=_DEFAULT_DATA_DIR / _VAL_SPLIT_FILE,
        help="Destination path for the validation-portion CSV.",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Skip writing train_split.csv / val_split.csv.",
    )
    args = parser.parse_args()

    if not args.no_train:
        _download_split(_HF_BASE + _HF_TRAIN, args.train_location)

    if not args.no_test:
        _download_split(_HF_BASE + _HF_TEST, args.test_location)

    if not args.no_counter_train:
        _download_split(_HF_BASE + _HF_COUNTER_TRAIN, args.counter_train_location)

    if not args.no_train and not args.no_split:
        _write_train_val_split(
            args.train_location,
            args.train_split_location,
            args.val_split_location,
            args.val_split,
            args.seed,
        )


if __name__ == "__main__":
    main()
