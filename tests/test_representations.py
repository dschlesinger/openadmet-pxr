"""Tests for the representations package."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from representations import REGISTRY, Representation
from representations.fingerprints import MorganFingerprint
from representations.rdkit_descriptors import RDKitDescriptors

VALID_SMILES = pl.Series(["c1ccccc1", "CCO", "CC(=O)O"])
INVALID_SMILES = pl.Series(["not_a_smiles", "c1ccccc1"])


# --- registry ---


def test_registry_contains_both() -> None:
    assert "morgan_fingerprint" in REGISTRY
    assert "rdkit_descriptors" in REGISTRY


# --- MorganFingerprint ---


def test_morgan_shape() -> None:
    fp = MorganFingerprint()
    out = fp.transform(VALID_SMILES)
    assert out.shape == (3, 2048)
    assert out.dtype == np.uint8


def test_morgan_custom_params() -> None:
    fp = MorganFingerprint(radius=3, n_bits=1024)
    out = fp.transform(VALID_SMILES)
    assert out.shape == (3, 1024)


def test_morgan_invalid_smiles_is_zeros() -> None:
    fp = MorganFingerprint()
    out = fp.transform(INVALID_SMILES)
    assert np.all(out[0] == 0)
    assert out[1].sum() > 0


def test_morgan_valid_not_all_zeros() -> None:
    fp = MorganFingerprint()
    out = fp.transform(VALID_SMILES)
    assert out.sum() > 0


# --- RDKitDescriptors ---


def test_rdkit_shape() -> None:
    rd = RDKitDescriptors()
    out = rd.transform(VALID_SMILES)
    assert out.shape[0] == 3
    assert out.shape[1] == len(rd.feature_names)
    assert out.dtype == np.float64


def test_rdkit_invalid_smiles_is_nan() -> None:
    rd = RDKitDescriptors()
    out = rd.transform(INVALID_SMILES)
    assert np.all(np.isnan(out[0]))
    assert not np.all(np.isnan(out[1]))


def test_rdkit_feature_names() -> None:
    rd = RDKitDescriptors()
    names = rd.feature_names
    assert isinstance(names, list)
    assert len(names) > 0
    assert all(isinstance(n, str) for n in names)
