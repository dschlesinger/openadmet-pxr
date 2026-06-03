"""Tests for the representations package."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from representations import REGISTRY, Representation
from representations.fingerprints import MorganFingerprint
from representations.jazzy_descriptors import JazzyDescriptors
from representations.rdkit_descriptors import RDKitDescriptors
from representations.xtb_descriptors import N_XTB_FEATURES, XTBDescriptors

VALID_SMILES = pl.Series(["c1ccccc1", "CCO", "CC(=O)O"])
INVALID_SMILES = pl.Series(["not_a_smiles", "c1ccccc1"])


# --- registry ---


def test_registry_contains_both() -> None:
    assert "morgan" in REGISTRY
    assert "rdkit" in REGISTRY
    assert "jazzy" in REGISTRY
    assert "xtb" in REGISTRY


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


# --- JazzyDescriptors ---


def test_jazzy_shape() -> None:
    pytest.importorskip("jazzy")
    jd = JazzyDescriptors()
    out = jd.transform(VALID_SMILES)
    assert out.shape[0] == 3
    assert out.shape[1] == len(jd.feature_names)
    assert out.dtype == np.float64


def test_jazzy_invalid_smiles_is_nan() -> None:
    pytest.importorskip("jazzy")
    jd = JazzyDescriptors()
    out = jd.transform(INVALID_SMILES)
    assert np.all(np.isnan(out[0]))
    assert not np.all(np.isnan(out[1]))


def test_jazzy_feature_names() -> None:
    pytest.importorskip("jazzy")
    jd = JazzyDescriptors()
    names = jd.feature_names
    assert isinstance(names, list)
    assert len(names) > 0
    assert all(isinstance(n, str) for n in names)


# --- XTBDescriptors ---


@pytest.mark.slow
def test_xtb_shape() -> None:
    pytest.importorskip("xtb")
    xd = XTBDescriptors()
    out = xd.transform(VALID_SMILES)
    assert out.shape == (3, N_XTB_FEATURES)
    assert out.dtype == np.float64


@pytest.mark.slow
def test_xtb_invalid_smiles_is_nan() -> None:
    pytest.importorskip("xtb")
    xd = XTBDescriptors()
    out = xd.transform(INVALID_SMILES)
    assert np.all(np.isnan(out[0]))
    assert not np.all(np.isnan(out[1]))


@pytest.mark.slow
def test_xtb_features_finite_for_valid() -> None:
    pytest.importorskip("xtb")
    xd = XTBDescriptors()
    out = xd.transform(pl.Series(["CCO"]))
    assert np.all(np.isfinite(out[0]))
