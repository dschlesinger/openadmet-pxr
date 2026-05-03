"""Download PXR challenge dataset splits from HuggingFace to a local data folder."""

import argparse
from pathlib import Path

import polars as pl

_HF_BASE = "hf://datasets/openadmet/pxr-challenge-train-test/"
_HF_TRAIN = "pxr-challenge_TRAIN.csv"
_HF_TEST = "pxr-challenge_TEST_BLINDED.csv"
_TRAIN_FILE = "train.csv"
_TEST_FILE = "test.csv"
_DEFAULT_DATA_DIR = Path("data")


def _download_split(hf_path: str, local_path: Path) -> None:
    if local_path.exists():
        print(f"Skipping (already exists): {local_path}")
        return
    local_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {hf_path} ...")
    df = pl.read_csv(hf_path)
    df.write_csv(local_path)
    print(f"Saved {len(df)} rows -> {local_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PXR challenge dataset splits.")
    parser.add_argument("--no-train", action="store_true", help="Skip the train split.")
    parser.add_argument("--no-test", action="store_true", help="Skip the test split.")
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
    args = parser.parse_args()

    if not args.no_train:
        _download_split(_HF_BASE + _HF_TRAIN, args.train_location)

    if not args.no_test:
        _download_split(_HF_BASE + _HF_TEST, args.test_location)


if __name__ == "__main__":
    main()
