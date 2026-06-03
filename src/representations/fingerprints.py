"""Morgan (ECFP) and MACCS keys fingerprint representations."""

from typing import ClassVar

import numpy as np
import polars as pl
from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
from tqdm import tqdm

from representations.base import Representation


class MACCSKeysFingerprint(Representation):
    """167-bit MACCS structural keys fingerprints.

    Invalid SMILES produce an all-zero vector.
    """

    name: ClassVar[str] = "maccs"

    def transform(self, smiles: pl.Series) -> np.ndarray:
        """Return a (n_molecules, 167) uint8 array of MACCS keys fingerprints."""
        out = np.zeros((len(smiles), 167), dtype=np.uint8)
        for i, smi in enumerate(tqdm(smiles.to_list(), desc="MACCS keys", unit="mol", leave=True)):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                fp = MACCSkeys.GenMACCSKeys(mol)
                out[i] = np.frombuffer(fp.ToBitString().encode(), dtype=np.uint8) - ord("0")
        return out


class MorganFingerprint(Representation):
    """Binary Morgan (ECFP) fingerprints.

    Invalid SMILES produce an all-zero vector.
    """

    name: ClassVar[str] = "morgan"

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
