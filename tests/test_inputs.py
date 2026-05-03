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
    assert set(INPUT_REGISTRY.keys()) == {"morgan", "rdkit", "rdkit+morgan"}


def test_morgan_shape() -> None:
    out = INPUT_REGISTRY["morgan"](_DF)
    assert out.shape == (3, 2048)
    assert out.dtype == np.float64


def test_rdkit_shape() -> None:
    out = INPUT_REGISTRY["rdkit"](_DF)
    assert out.shape[0] == 3
    assert out.dtype == np.float64


def test_rdkit_morgan_shape() -> None:
    rdkit_cols = INPUT_REGISTRY["rdkit"](_DF).shape[1]
    morgan_cols = INPUT_REGISTRY["morgan"](_DF).shape[1]
    out = INPUT_REGISTRY["rdkit+morgan"](_DF)
    assert out.shape == (3, rdkit_cols + morgan_cols)


# --- featurize() caching ---


def test_featurize_computes_and_saves(tmp_path: Path) -> None:
    out = featurize(_DF, "morgan", cache_dir=tmp_path)
    assert out.shape == (3, 2048)
    cache_files = list(tmp_path.glob("morgan_*.npy"))
    assert len(cache_files) == 1


def test_featurize_loads_from_cache(tmp_path: Path) -> None:
    featurize(_DF, "morgan", cache_dir=tmp_path)
    # second call should hit the cache (file already exists)
    out = featurize(_DF, "morgan", cache_dir=tmp_path)
    assert out.shape == (3, 2048)
    assert len(list(tmp_path.glob("morgan_*.npy"))) == 1


def test_featurize_different_inputs_separate_files(tmp_path: Path) -> None:
    featurize(_DF, "morgan", cache_dir=tmp_path)
    featurize(_DF, "rdkit", cache_dir=tmp_path)
    assert len(list(tmp_path.glob("*.npy"))) == 2


def test_featurize_unknown_input_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown input"):
        featurize(_DF, "no_such_input", cache_dir=tmp_path)
