"""Jazzy solvation and hydrogen-bonding descriptor representation."""

from typing import ClassVar

import numpy as np
import polars as pl
from tqdm import tqdm

from representations.base import Representation

_REFERENCE_SMILES = "c1ccccc1"
_FEATURE_NAMES: list[str] | None = None


def _get_feature_names() -> list[str]:
    global _FEATURE_NAMES
    if _FEATURE_NAMES is None:
        from jazzy.api import molecular_vector_from_smiles

        vec = molecular_vector_from_smiles(_REFERENCE_SMILES)
        _FEATURE_NAMES = sorted(vec.keys())
    return _FEATURE_NAMES


class JazzyDescriptors(Representation):
    """Jazzy solvation and hydrogen-bonding descriptors.

    Invalid SMILES produce an all-NaN row.
    """

    name: ClassVar[str] = "jazzy"

    @property
    def feature_names(self) -> list[str]:
        return _get_feature_names()

    def transform(self, smiles: pl.Series) -> np.ndarray:
        """Return a (n_molecules, n_features) float64 array of Jazzy descriptors."""
        from jazzy.api import molecular_vector_from_smiles

        names = self.feature_names
        n_feat = len(names)
        out = np.full((len(smiles), n_feat), np.nan, dtype=np.float64)
        for i, smi in enumerate(tqdm(smiles.to_list(), desc="Jazzy descriptors", unit="mol", leave=True)):
            try:
                vec = molecular_vector_from_smiles(smi)
                if vec is not None:
                    out[i] = [vec[k] for k in names]
            except Exception:
                pass
        return out
