"""Registry of molecular input featurizations for the PXR challenge."""

import hashlib
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl

from representations.fingerprints import MorganFingerprint
from representations.rdkit_descriptors import RDKitDescriptors

_morgan = MorganFingerprint()
_rdkit = RDKitDescriptors()


def _morgan_features(df: pl.DataFrame) -> np.ndarray:
    return _morgan.transform(df["SMILES"]).astype(np.float64)


def _rdkit_features(df: pl.DataFrame) -> np.ndarray:
    return _rdkit.transform(df["SMILES"])


def _rdkit_morgan_features(df: pl.DataFrame) -> np.ndarray:
    return np.hstack([_rdkit_features(df), _morgan_features(df)])


INPUT_REGISTRY: dict[str, Callable[[pl.DataFrame], np.ndarray]] = {
    "morgan": _morgan_features,
    "rdkit": _rdkit_features,
    "rdkit+morgan": _rdkit_morgan_features,
}


def _cache_key(smiles: pl.Series) -> str:
    """Return a short SHA-256 hex digest of the SMILES list."""
    content = "|".join(smiles.to_list())
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def featurize(
    df: pl.DataFrame,
    input_name: str,
    cache_dir: Path = Path("data/features"),
) -> np.ndarray:
    """Return feature matrix for df, loading from cache when available.

    On a cache miss the features are computed (with a tqdm bar) and saved so
    that subsequent calls with the same SMILES and input_name are instant.
    """
    if input_name not in INPUT_REGISTRY:
        raise ValueError(f"Unknown input: {input_name!r}. Available: {list(INPUT_REGISTRY.keys())}")

    cache_path = cache_dir / f"{input_name}_{_cache_key(df['SMILES'])}.npy"

    if cache_path.exists():
        print(f"Loading cached {input_name} features from {cache_path}")
        return np.load(str(cache_path))

    features = INPUT_REGISTRY[input_name](df)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(cache_path), features)
    print(f"Saved {input_name} features to {cache_path}")
    return features
