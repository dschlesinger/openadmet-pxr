"""Co-crystal ligand Tanimoto similarity features.

For each query molecule, computes ECFP4 Tanimoto similarity against the set of
ligands co-crystallized with each nuclear-receptor family member (PXR, FXR,
RXRA, VDR, CAR) and aggregates into per-receptor statistics. This encodes
"how much does this compound resemble a confirmed binder of receptor X" in a
way orthogonal to within-training pairwise similarity.

Reference ligand SMILES come from ``data/crystal_ligands.csv``, produced by
``scripts/extract_crystal_ligands.py``. See ldbc1999 (rank 65) and
``os_reports/GAP.md`` for the motivation.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import polars as pl
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from tqdm import tqdm

from representations.base import Representation

# Receptor column order is fixed so feature columns are stable across runs.
RECEPTOR_ORDER: tuple[str, ...] = ("pxr", "fxr", "rxra", "vdr", "car")
# Aggregation statistics produced per receptor, in output order.
STAT_ORDER: tuple[str, ...] = ("max", "mean", "top3", "std")


class CrystalLigandSimilarity(Representation):
    """Per-receptor ECFP4 Tanimoto similarity to co-crystal ligands.

    Output is (n_molecules, len(RECEPTOR_ORDER) * len(STAT_ORDER)) = (n, 20),
    column-ordered as (pxr_max, pxr_mean, pxr_top3, pxr_std, fxr_max, ...).
    Invalid query SMILES produce an all-zero row.
    """

    name: ClassVar[str] = "crystal_similarity"

    def __init__(
        self,
        ligands_path: Path = Path("data/crystal_ligands.csv"),
        radius: int = 2,
        n_bits: int = 2048,
        top_k: int = 3,
    ) -> None:
        self.ligands_path = ligands_path
        self.top_k = top_k
        self._generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        self._reference_fps: dict[str, list] = {}

    @property
    def feature_names(self) -> list[str]:
        """Stable column labels matching the output array order."""
        return [f"{receptor}_{stat}_sim" for receptor in RECEPTOR_ORDER for stat in STAT_ORDER]

    def _load_reference_fps(self) -> dict[str, list]:
        """Lazily build {receptor: [fingerprints]} from the cached ligand CSV."""
        if self._reference_fps:
            return self._reference_fps
        if not self.ligands_path.exists():
            raise FileNotFoundError(
                f"{self.ligands_path} not found. Run " "`python scripts/extract_crystal_ligands.py` first."
            )
        df = pl.read_csv(self.ligands_path)
        for receptor in RECEPTOR_ORDER:
            smiles_list = df.filter(pl.col("receptor") == receptor)["smiles"].to_list()
            fps = []
            for smi in smiles_list:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    fps.append(self._generator.GetFingerprint(mol))
            self._reference_fps[receptor] = fps
        return self._reference_fps

    def _aggregate(self, sims: np.ndarray) -> list[float]:
        """Reduce a similarity vector to [max, mean, top-k mean, std]."""
        if sims.size == 0:
            return [0.0] * len(STAT_ORDER)
        top_k = np.sort(sims)[::-1][: self.top_k]
        return [float(sims.max()), float(sims.mean()), float(top_k.mean()), float(sims.std())]

    def transform(self, smiles: pl.Series) -> np.ndarray:
        """Return a (n_molecules, 20) float64 similarity-statistics array."""
        reference_fps = self._load_reference_fps()
        n_features = len(RECEPTOR_ORDER) * len(STAT_ORDER)
        out = np.zeros((len(smiles), n_features), dtype=np.float64)
        for i, smi in enumerate(tqdm(smiles.to_list(), desc="Crystal similarity", unit="mol", leave=True)):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            query_fp = self._generator.GetFingerprint(mol)
            row: list[float] = []
            for receptor in RECEPTOR_ORDER:
                fps = reference_fps[receptor]
                sims = np.asarray(DataStructs.BulkTanimotoSimilarity(query_fp, fps)) if fps else np.empty(0)
                row.extend(self._aggregate(sims))
            out[i] = row
        return out
