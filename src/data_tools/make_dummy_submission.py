"""Generate a dummy submission file from test.csv for upload testing."""

from pathlib import Path

import polars as pl

from data_tools.load import load_test

_RESULTS_DIR = Path("results")
_DEFAULT_OUTPUT = _RESULTS_DIR / "dummy_submission.parquet"


def make_dummy_submission(
    output_path: Path = _DEFAULT_OUTPUT,
    pec50_value: float = 0.0,
) -> pl.DataFrame:
    """Load test split, add a constant pEC50 column, and write to output_path."""
    df = load_test()
    df = df.with_columns(pl.lit(pec50_value).alias("pEC50"))
    df = df.select(["SMILES", "Molecule Name", "pEC50"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)
    print(f"Wrote {len(df)} rows -> {output_path}")
    return df


if __name__ == "__main__":
    make_dummy_submission()
