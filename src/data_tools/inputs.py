"""Registry of molecular input featurizations for the PXR challenge."""

from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl

from representations.chemeleon import ChemeleonFingerprint, FinetunedChemeleonFingerprint
from representations.chemprop_rep import ChempropFingerprint
from representations.crystal_similarity import CrystalLigandSimilarity
from representations.fingerprints import MACCSKeysFingerprint, MorganFingerprint
from representations.jazzy_descriptors import JazzyDescriptors
from representations.mole import MolERepresentation
from representations.rdkit_descriptors import RDKitDescriptors
from representations.unimol import UniMolRepresentation
from representations.unimol_finetuned import FinetunedUniMolRepresentation
from representations.xtb_descriptors import XTBDescriptors

_morgan = MorganFingerprint()
_maccs = MACCSKeysFingerprint()
_rdkit = RDKitDescriptors()
_chemeleon = ChemeleonFingerprint()
_chemprop = ChempropFingerprint()
_finetuned_chemeleon = FinetunedChemeleonFingerprint()
_unimol = UniMolRepresentation()
_finetuned_unimol = FinetunedUniMolRepresentation()
_mole = MolERepresentation()
_jazzy = JazzyDescriptors()
_xtb = XTBDescriptors()
_crystal_similarity = CrystalLigandSimilarity()


def _morgan_features(df: pl.DataFrame) -> np.ndarray:
    return _morgan.transform(df["SMILES"]).astype(np.float64)


def _maccs_features(df: pl.DataFrame) -> np.ndarray:
    return _maccs.transform(df["SMILES"]).astype(np.float64)


def _rdkit_features(df: pl.DataFrame) -> np.ndarray:
    return _rdkit.transform(df["SMILES"])


def _chemeleon_features(df: pl.DataFrame) -> np.ndarray:
    return _chemeleon.transform(df["SMILES"]).astype(np.float64)


def _chemprop_features(df: pl.DataFrame) -> np.ndarray:
    return _chemprop.transform(df["SMILES"]).astype(np.float64)


def _finetuned_chemeleon_features(df: pl.DataFrame) -> np.ndarray:
    return _finetuned_chemeleon.transform(df["SMILES"]).astype(np.float64)


def _unimol_features(df: pl.DataFrame) -> np.ndarray:
    return _unimol.transform(df["SMILES"]).astype(np.float64)


def _finetuned_unimol_features(df: pl.DataFrame) -> np.ndarray:
    return _finetuned_unimol.transform(df["SMILES"]).astype(np.float64)


def _mole_features(df: pl.DataFrame) -> np.ndarray:
    return _mole.transform(df["SMILES"]).astype(np.float64)


def _jazzy_features(df: pl.DataFrame) -> np.ndarray:
    return _jazzy.transform(df["SMILES"])


def _xtb_features(df: pl.DataFrame) -> np.ndarray:
    return _xtb.transform(df["SMILES"])


def _crystal_similarity_features(df: pl.DataFrame) -> np.ndarray:
    return _crystal_similarity.transform(df["SMILES"])


INPUT_REGISTRY: dict[str, Callable[[pl.DataFrame], np.ndarray]] = {
    _morgan.name: _morgan_features,
    _maccs.name: _maccs_features,
    _rdkit.name: _rdkit_features,
    _chemeleon.name: _chemeleon_features,
    _chemprop.name: _chemprop_features,
    _finetuned_chemeleon.name: _finetuned_chemeleon_features,
    _unimol.name: _unimol_features,
    _finetuned_unimol.name: _finetuned_unimol_features,
    _mole.name: _mole_features,
    _jazzy.name: _jazzy_features,
    _xtb.name: _xtb_features,
    _crystal_similarity.name: _crystal_similarity_features,
}


def _featurize_one(df: pl.DataFrame, input_name: str, cache_dir: Path) -> np.ndarray:
    """Return features for df, computing only molecules not already in the cache.

    Cache layout (one pair of files per representation):
      {input_name}.npy  — float64 array of shape (N_cached, D); grows as new molecules arrive
      {input_name}.idx  — text file, one canonical SMILES per line; line i = row i of .npy
    """
    smiles_req = df["SMILES"].to_list()
    cache_dir.mkdir(parents=True, exist_ok=True)
    npy_path = cache_dir / f"{input_name}.npy"
    idx_path = cache_dir / f"{input_name}.idx"

    # Load existing index
    if idx_path.exists():
        indexed = idx_path.read_text().splitlines()
        pos: dict[str, int] = {s: i for i, s in enumerate(indexed)}
    else:
        indexed, pos = [], {}

    # Identify molecules not yet cached, preserving order and deduplicating
    missing = list(dict.fromkeys(s for s in smiles_req if s not in pos))
    if missing:
        new_features = INPUT_REGISTRY[input_name](pl.DataFrame({"SMILES": missing}))
        full = np.vstack([np.load(str(npy_path)), new_features]) if npy_path.exists() else new_features
        np.save(str(npy_path), full)
        for s in missing:
            pos[s] = len(indexed)
            indexed.append(s)
        idx_path.write_text("\n".join(indexed))
        print(f"[{input_name}] appended {len(missing)} molecules (cache now {len(indexed)} total)")
    else:
        print(f"[{input_name}] loaded from cache ({len(indexed)} molecules indexed)")

    return np.load(str(npy_path))[[pos[s] for s in smiles_req]]


def featurize(
    df: pl.DataFrame,
    input_names: str | list[str],
    cache_dir: Path = Path("data/features"),
) -> np.ndarray:
    """Return feature matrix for df, computing only uncached molecules.

    Pass a single name or a list; multiple inputs are hstacked after loading
    each from its own cache file.
    """
    names = [input_names] if isinstance(input_names, str) else input_names
    unknown = [n for n in names if n not in INPUT_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown input(s): {unknown}. Available: {list(INPUT_REGISTRY.keys())}")
    arrays = [_featurize_one(df, name, cache_dir) for name in names]
    return arrays[0] if len(arrays) == 1 else np.hstack(arrays)
