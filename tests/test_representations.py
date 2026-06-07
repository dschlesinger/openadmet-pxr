"""Tests for the representations package."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from pathlib import Path

from representations import REGISTRY
from representations.crystal_similarity import (
    RECEPTOR_ORDER,
    STAT_ORDER,
    CrystalLigandSimilarity,
)
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


# --- CrystalLigandSimilarity ---


def _write_ligands_csv(path: Path) -> None:
    """Write a tiny per-receptor reference ligand CSV for testing."""
    rows = {
        "pxr": ["c1ccccc1", "CCO"],
        "fxr": ["CC(=O)O"],
        "rxra": ["c1ccccc1C"],
        "vdr": ["C1CCCCC1"],
        "car": ["CCN"],
    }
    df = pl.DataFrame(
        {
            "receptor": [r for r, smis in rows.items() for _ in smis],
            "het_code": [f"L{i}" for r, smis in rows.items() for i, _ in enumerate(smis)],
            "smiles": [s for smis in rows.values() for s in smis],
            "name": [""] * sum(len(s) for s in rows.values()),
        }
    )
    df.write_csv(path)


def test_crystal_similarity_registered() -> None:
    assert "crystal_similarity" in REGISTRY


def test_crystal_similarity_shape(tmp_path: Path) -> None:
    csv = tmp_path / "crystal_ligands.csv"
    _write_ligands_csv(csv)
    rep = CrystalLigandSimilarity(ligands_path=csv)
    out = rep.transform(VALID_SMILES)
    expected_cols = len(RECEPTOR_ORDER) * len(STAT_ORDER)
    assert out.shape == (3, expected_cols)
    assert out.dtype == np.float64
    assert len(rep.feature_names) == expected_cols


def test_crystal_similarity_values_in_unit_interval(tmp_path: Path) -> None:
    csv = tmp_path / "crystal_ligands.csv"
    _write_ligands_csv(csv)
    rep = CrystalLigandSimilarity(ligands_path=csv)
    out = rep.transform(VALID_SMILES)
    assert np.all(out >= 0.0)
    assert np.all(out <= 1.0)


def test_crystal_similarity_self_match_is_one(tmp_path: Path) -> None:
    csv = tmp_path / "crystal_ligands.csv"
    _write_ligands_csv(csv)
    rep = CrystalLigandSimilarity(ligands_path=csv)
    # Benzene is a PXR reference ligand, so its pxr_max_sim must be 1.0.
    out = rep.transform(pl.Series(["c1ccccc1"]))
    pxr_max_idx = RECEPTOR_ORDER.index("pxr") * len(STAT_ORDER) + STAT_ORDER.index("max")
    assert out[0, pxr_max_idx] == pytest.approx(1.0)


def test_crystal_similarity_invalid_smiles_is_zeros(tmp_path: Path) -> None:
    csv = tmp_path / "crystal_ligands.csv"
    _write_ligands_csv(csv)
    rep = CrystalLigandSimilarity(ligands_path=csv)
    out = rep.transform(INVALID_SMILES)
    assert np.all(out[0] == 0)
    assert np.any(out[1] != 0)


def test_crystal_similarity_missing_csv_raises(tmp_path: Path) -> None:
    rep = CrystalLigandSimilarity(ligands_path=tmp_path / "does_not_exist.csv")
    with pytest.raises(FileNotFoundError):
        rep.transform(VALID_SMILES)
