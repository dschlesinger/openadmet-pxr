"""Morgan (ECFP) fingerprint representation."""

from typing import ClassVar

import numpy as np
import polars as pl
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from tqdm import tqdm

from representations.base import Representation


class MorganFingerprint(Representation):
    """Binary Morgan (ECFP) fingerprints.

    Invalid SMILES produce an all-zero vector.
    """

    name: ClassVar[str] = "morgan_fingerprint"

    def __init__(self, radius: int = 2, n_bits: int = 2048) -> None:
        self.radius = radius
        self.n_bits = n_bits
        self._generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

    def transform(self, smiles: pl.Series) -> np.ndarray:
        """Return a (n_molecules, n_bits) uint8 array of Morgan fingerprints."""
        out = np.zeros((len(smiles), self.n_bits), dtype=np.uint8)
        for i, smi in enumerate(tqdm(smiles.to_list(), desc="Morgan fingerprints", unit="mol", leave=True)):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                fp = self._generator.GetFingerprintAsNumPy(mol)
                out[i] = fp
        return out
