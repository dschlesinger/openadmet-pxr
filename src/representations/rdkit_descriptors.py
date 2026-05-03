"""RDKit 2D physicochemical descriptor representation."""

from typing import ClassVar

import numpy as np
import polars as pl
from rdkit import Chem
from rdkit.Chem import Descriptors

from representations.base import Representation

_DESC_FUNCS: list[tuple[str, object]] = Descriptors.descList


class RDKitDescriptors(Representation):
    """All 217 RDKit 2D physicochemical descriptors.

    Invalid SMILES produce an all-NaN row.
    """

    name: ClassVar[str] = "rdkit_descriptors"

    @property
    def feature_names(self) -> list[str]:
        """Return the ordered list of descriptor names."""
        return [name for name, _ in _DESC_FUNCS]

    def transform(self, smiles: pl.Series) -> np.ndarray:
        """Return a (n_molecules, 217) float64 array of RDKit descriptors."""
        n_desc = len(_DESC_FUNCS)
        out = np.full((len(smiles), n_desc), np.nan, dtype=np.float64)
        for i, smi in enumerate(smiles.to_list()):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                out[i] = [func(mol) for _, func in _DESC_FUNCS]
        return out
