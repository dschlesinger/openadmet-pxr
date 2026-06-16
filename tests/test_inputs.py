"""Tests for the data_tools input registry and featurize()."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from data_tools.inputs import INPUT_REGISTRY, featurize

_DF = pl.DataFrame({"SMILES": ["c1ccccc1", "CCO", "CC(=O)O"]})


# --- INPUT_REGISTRY ---


def test_registry_keys() -> None:
    assert "morgan" in INPUT_REGISTRY
    assert "rdkit" in INPUT_REGISTRY
    assert "jazzy" in INPUT_REGISTRY
    assert "xtb" in INPUT_REGISTRY


def test_morgan_shape() -> None:
    out = INPUT_REGISTRY["morgan"](_DF)
    assert out.shape == (3, 2048)
    assert out.dtype == np.float64


def test_rdkit_shape() -> None:
    out = INPUT_REGISTRY["rdkit"](_DF)
    assert out.shape[0] == 3
    assert out.dtype == np.float64


def test_jazzy_in_registry() -> None:
    assert "jazzy" in INPUT_REGISTRY


def test_xtb_in_registry() -> None:
    assert "xtb" in INPUT_REGISTRY


# --- featurize() caching ---


def test_featurize_computes_and_saves(tmp_path: Path) -> None:
    out = featurize(_DF, "morgan", cache_dir=tmp_path)
    assert out.shape == (3, 2048)
    assert (tmp_path / "morgan.npy").exists()
    assert (tmp_path / "morgan.idx").exists()
    assert (tmp_path / "morgan.idx").read_text().splitlines() == _DF["SMILES"].to_list()


def test_featurize_loads_from_cache(tmp_path: Path) -> None:
    first = featurize(_DF, "morgan", cache_dir=tmp_path)
    # Poison the registry function so a second compute would be detectable
    original = INPUT_REGISTRY["morgan"]
    INPUT_REGISTRY["morgan"] = lambda df: (_ for _ in ()).throw(AssertionError("should not recompute"))  # type: ignore[assignment]
    try:
        second = featurize(_DF, "morgan", cache_dir=tmp_path)
    finally:
        INPUT_REGISTRY["morgan"] = original
    np.testing.assert_array_equal(first, second)


def test_featurize_different_inputs_separate_files(tmp_path: Path) -> None:
    featurize(_DF, "morgan", cache_dir=tmp_path)
    featurize(_DF, "rdkit", cache_dir=tmp_path)
    assert (tmp_path / "morgan.npy").exists()
    assert (tmp_path / "rdkit.npy").exists()


def test_featurize_subset_slices_without_recompute(tmp_path: Path) -> None:
    full = featurize(_DF, "morgan", cache_dir=tmp_path)

    subset_df = pl.DataFrame({"SMILES": ["CCO", "c1ccccc1"]})
    original = INPUT_REGISTRY["morgan"]
    calls: list[int] = []
    INPUT_REGISTRY["morgan"] = lambda df: (calls.append(1), original(df))[1]  # type: ignore[assignment]
    try:
        sub = featurize(subset_df, "morgan", cache_dir=tmp_path)
    finally:
        INPUT_REGISTRY["morgan"] = original

    assert calls == [], "featurizer should not be called for a known subset"
    # Rows should match the correct rows from the full array
    ethanol_row = full[_DF["SMILES"].to_list().index("CCO")]
    benzene_row = full[_DF["SMILES"].to_list().index("c1ccccc1")]
    np.testing.assert_array_equal(sub[0], ethanol_row)
    np.testing.assert_array_equal(sub[1], benzene_row)


def test_featurize_appends_new_molecules(tmp_path: Path) -> None:
    featurize(_DF, "morgan", cache_dir=tmp_path)
    npy_rows_before = np.load(str(tmp_path / "morgan.npy")).shape[0]

    extended_df = pl.DataFrame({"SMILES": ["c1ccccc1", "CCO", "CC(=O)O", "CCCC"]})
    out = featurize(extended_df, "morgan", cache_dir=tmp_path)

    npy_rows_after = np.load(str(tmp_path / "morgan.npy")).shape[0]
    assert npy_rows_after == npy_rows_before + 1  # only butane was new
    assert out.shape == (4, 2048)
    idx = (tmp_path / "morgan.idx").read_text().splitlines()
    assert "CCCC" in idx


def test_featurize_unknown_input_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown input"):
        featurize(_DF, "no_such_input", cache_dir=tmp_path)
