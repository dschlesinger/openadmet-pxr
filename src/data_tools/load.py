"""Load local PXR challenge CSV splits into Polars DataFrames."""

import random
from pathlib import Path
from typing import Literal

import polars as pl

_DEFAULT_DATA_DIR = Path("data")
_TRAIN_FILE = "train.csv"
_TEST_FILE = "test.csv"
_TEST_UNBLINDED_FILE = "test_unblinded.csv"
_COUNTER_TRAIN_FILE = "counter_train.csv"
_SINGLE_CONC_TRAIN_FILE = "single_concentration_train.csv"
# test_unblinded.csv omits unit annotations in column names; map to train.csv naming
_UNBLINDED_COL_MAP: dict[str, str] = {
    "Emax_estimate": "Emax_estimate (log2FC vs. baseline)",
    "Emax.vs.pos.ctrl_estimate": "Emax.vs.pos.ctrl_estimate (dimensionless)",
    "pEC50_std.error": "pEC50_std.error (-log10(molarity))",
    "Emax_std.error": "Emax_std.error (log2FC vs. baseline)",
    "Emax.vs.pos.ctrl_std.error": "Emax.vs.pos.ctrl_std.error (dimensionless)",
    "pEC50_ci.lower": "pEC50_ci.lower (-log10(molarity))",
    "pEC50_ci.upper": "pEC50_ci.upper (-log10(molarity))",
    "Emax_ci.lower": "Emax_ci.lower (log2FC vs. baseline)",
    "Emax_ci.upper": "Emax_ci.upper (log2FC vs. baseline)",
    "Emax.vs.pos.ctrl_ci.lower": "Emax.vs.pos.ctrl_ci.lower (dimensionless)",
    "Emax.vs.pos.ctrl_ci.upper": "Emax.vs.pos.ctrl_ci.upper (dimensionless)",
}



def load_train(path: Path = _DEFAULT_DATA_DIR / _TRAIN_FILE) -> pl.DataFrame:
    """Return the training split as a Polars DataFrame."""
    return pl.read_csv(path)


def load_test(path: Path = _DEFAULT_DATA_DIR / _TEST_FILE) -> pl.DataFrame:
    """Return the blinded test split as a Polars DataFrame."""
    return pl.read_csv(path)


def load_test_unblinded(path: Path = _DEFAULT_DATA_DIR / _TEST_UNBLINDED_FILE) -> pl.DataFrame:
    """Return the phase-1 unblinded test split (has pEC50 labels) as a Polars DataFrame."""
    return pl.read_csv(path)


def load_counter_train(path: Path = _DEFAULT_DATA_DIR / _COUNTER_TRAIN_FILE) -> pl.DataFrame:
    """Return the counter-assay training split as a Polars DataFrame."""
    return pl.read_csv(path)


def load_single_concentration_train(path: Path = _DEFAULT_DATA_DIR / _SINGLE_CONC_TRAIN_FILE) -> pl.DataFrame:
    """Return the single-concentration training split as a Polars DataFrame."""
    return pl.read_csv(path)


def load_splits(
    train_path: Path = _DEFAULT_DATA_DIR / _TRAIN_FILE,
    test_path: Path = _DEFAULT_DATA_DIR / _TEST_FILE,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (train, test) DataFrames."""
    return load_train(train_path), load_test(test_path)


def load_test_holdout(
    test_path: Path = _DEFAULT_DATA_DIR / _TEST_FILE,
    unblinded_path: Path = _DEFAULT_DATA_DIR / _TEST_UNBLINDED_FILE,
) -> pl.DataFrame:
    """Return the 260 test molecules NOT present in the phase-1 unblinded set.

    Use this for unbiased local evaluation when training with include_unblinded=True.
    The full 513-molecule test.csv is still required for submission.
    """
    test = load_test(test_path)
    unblinded_smiles = set(load_test_unblinded(unblinded_path)["SMILES"].to_list())
    return test.filter(~pl.col("SMILES").is_in(unblinded_smiles))


def _scaffold_split(
    df: pl.DataFrame,
    val_fraction: float,
    seed: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split df by Bemis-Murcko scaffold so no scaffold family spans both splits."""
    try:
        from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("rdkit is required for scaffold splitting") from exc

    scaffold_groups: dict[str, list[int]] = {}
    for idx, smi in enumerate(df["SMILES"].to_list()):
        try:
            scaffold = MurckoScaffoldSmiles(smi, includeChirality=False)
        except Exception:  # pylint: disable=broad-except
            scaffold = "__invalid__"
        scaffold_groups.setdefault(scaffold, []).append(idx)

    # Stable sort by group size, then shuffle with seeded RNG for reproducibility
    sorted_groups = sorted(scaffold_groups.items(), key=lambda x: len(x[1]), reverse=True)
    random.Random(seed).shuffle(sorted_groups)

    n_total = len(df)
    val_indices: list[int] = []
    train_indices: list[int] = []
    for scaffold, indices in sorted_groups:
        if scaffold != "__invalid__" and (n_total == 0 or len(val_indices) / n_total < val_fraction):
            val_indices.extend(indices)
        else:
            train_indices.extend(indices)

    return df[train_indices], df[val_indices]


def _assemble_pool(  # pylint: disable=too-many-arguments
    *,
    include_unblinded: bool = False,
    include_counter_assay: bool = False,
    include_single_concentration: bool = False,
    data_dir: Path = Path("data"),
) -> pl.DataFrame:
    """Assemble the full training pool from multiple sources without splitting."""
    base = load_train(data_dir / _TRAIN_FILE)

    if include_unblinded:
        unblinded = load_test_unblinded(data_dir / _TEST_UNBLINDED_FILE)
        rename_map = {k: v for k, v in _UNBLINDED_COL_MAP.items() if k in unblinded.columns}
        unblinded = unblinded.rename(rename_map).with_columns(
            pl.col("OCNT Batch").str.extract(r"^(OCNT-\d+)", 1).alias("OCNT_ID"),
            pl.lit("test_unblinded").alias("source"),
        )
        # Pad with typed-null columns so schemas match for concat
        for col in base.columns:
            if col not in unblinded.columns:
                unblinded = unblinded.with_columns(pl.lit(None).cast(base[col].dtype).alias(col))
        base = pl.concat([base, unblinded.select(base.columns)])

    if include_counter_assay:
        counter = (
            load_counter_train(data_dir / _COUNTER_TRAIN_FILE)
            .select(["OCNT_ID", "pEC50"])
            .rename({"pEC50": "pEC50_counter"})
        )
        base = base.join(counter, on="OCNT_ID", how="left")

    if include_single_concentration:
        single = load_single_concentration_train(data_dir / _SINGLE_CONC_TRAIN_FILE)
        agg = single.group_by("OCNT_ID").agg(
            pl.col("log2_fc_estimate").median().alias("log2_fc_single")
        )
        base = base.join(agg, on="OCNT_ID", how="left")

    return base


def _build_unblinded_val(
    include_counter_assay: bool,
    include_single_concentration: bool,
    data_dir: Path,
    base_schema: pl.DataFrame,
) -> pl.DataFrame:
    """Load test_unblinded.csv aligned to the train.csv column schema."""
    val_raw = load_test_unblinded(data_dir / _TEST_UNBLINDED_FILE)
    rename_map = {k: v for k, v in _UNBLINDED_COL_MAP.items() if k in val_raw.columns}
    val_df = val_raw.rename(rename_map).with_columns(
        pl.col("OCNT Batch").str.extract(r"^(OCNT-\d+)", 1).alias("OCNT_ID"),
        pl.lit("test_unblinded").alias("source"),
    )
    for col in base_schema.columns:
        if col not in val_df.columns:
            val_df = val_df.with_columns(pl.lit(None).cast(base_schema[col].dtype).alias(col))
    val_df = val_df.select(base_schema.columns)

    if include_counter_assay:
        counter = (
            load_counter_train(data_dir / _COUNTER_TRAIN_FILE)
            .select(["OCNT_ID", "pEC50"])
            .rename({"pEC50": "pEC50_counter"})
        )
        val_df = val_df.join(counter, on="OCNT_ID", how="left")
    if include_single_concentration:
        single = load_single_concentration_train(data_dir / _SINGLE_CONC_TRAIN_FILE)
        agg = single.group_by("OCNT_ID").agg(
            pl.col("log2_fc_estimate").median().alias("log2_fc_single")
        )
        val_df = val_df.join(agg, on="OCNT_ID", how="left")

    return val_df


def load_data(  # pylint: disable=too-many-arguments
    *,
    include_unblinded: bool = False,
    include_counter_assay: bool = False,
    include_single_concentration: bool = False,
    split_type: Literal["random", "scaffold", "unblinded"] = "unblinded",
    val_fraction: float = 0.1,
    seed: int = 42,
    data_dir: Path = Path("data"),
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Assemble training pool and return (train_df, val_df).

    split_type options:
      'scaffold'  — Bemis-Murcko scaffold split; no scaffold leakage (default)
      'random'    — seeded random split
      'unblinded' — train=train.csv, val=test_unblinded.csv (253 phase-1 labeled molecules);
                    best proxy for the prospective test distribution. include_unblinded is
                    ignored in this mode.
    """
    if split_type == "unblinded":
        base_schema = load_train(data_dir / _TRAIN_FILE)
        train_df = _assemble_pool(
            include_unblinded=False,
            include_counter_assay=include_counter_assay,
            include_single_concentration=include_single_concentration,
            data_dir=data_dir,
        )
        val_df = _build_unblinded_val(include_counter_assay, include_single_concentration, data_dir, base_schema)
        return train_df, val_df

    base = _assemble_pool(
        include_unblinded=include_unblinded,
        include_counter_assay=include_counter_assay,
        include_single_concentration=include_single_concentration,
        data_dir=data_dir,
    )

    if split_type == "scaffold":
        return _scaffold_split(base, val_fraction, seed)

    shuffled = base.sample(fraction=1.0, shuffle=True, seed=seed)
    n_val = int(len(shuffled) * val_fraction)
    return shuffled[n_val:], shuffled[:n_val]
