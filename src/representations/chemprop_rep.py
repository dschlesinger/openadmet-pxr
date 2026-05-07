"""Chemprop MPNN fingerprints from a trained checkpoint.

Loads a chemprop Lightning checkpoint (e.g. from scripts/train_chemprop.py)
and extracts message-passing fingerprints as fixed-size float32 vectors.
Invalid SMILES produce an all-NaN row (cleaned by drop_nan_rows downstream).
"""

from pathlib import Path
from typing import ClassVar

import numpy as np
import polars as pl
import torch
from chemprop import data, featurizers
from chemprop.data import BatchMolGraph
from chemprop.models import MPNN
from rdkit.Chem import MolFromSmiles
from tqdm import tqdm

from representations.base import Representation

_DEFAULT_CKPT_DIR = Path("checkpoints/chemprop")


def _find_best_checkpoint(ckpt_dir: Path) -> Path:
    candidates = sorted(ckpt_dir.glob("best-epoch*.ckpt"))
    if not candidates:
        raise FileNotFoundError(
            f"No best-epoch*.ckpt found in {ckpt_dir}. "
            "Train one with: python scripts/train_chemprop.py"
        )
    return candidates[-1]


class ChempropFingerprint(Representation):
    """Chemprop MPNN fingerprints from a trained Lightning checkpoint.

    The checkpoint is produced by scripts/train_chemprop.py. Invalid SMILES
    produce an all-NaN row (cleaned by drop_nan_rows downstream).
    """

    name: ClassVar[str] = "chemprop_finetuned"

    def __init__(
        self,
        ckpt_dir: Path | str = _DEFAULT_CKPT_DIR,
        device: str | torch.device | None = None,
        batch_size: int = 64,
        num_workers: int = 0,
    ) -> None:
        self._ckpt_dir = Path(ckpt_dir)
        self._device = device
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._model: MPNN | None = None
        self._featurizer: featurizers.SimpleMoleculeMolGraphFeaturizer | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        ckpt = _find_best_checkpoint(self._ckpt_dir)
        print('Found checkpoint:', ckpt)
        self._model = MPNN.load_from_checkpoint(str(ckpt))
        self._model.eval()
        if self._device is not None:
            self._model.to(device=self._device)
        self._featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    def transform(self, smiles: pl.Series) -> np.ndarray:
        """Return a (n_molecules, embedding_dim) float32 array of chemprop fingerprints."""
        self._ensure_loaded()
        assert self._model is not None and self._featurizer is not None

        smiles_list = smiles.to_list()
        valid_idx = [i for i, s in enumerate(smiles_list) if MolFromSmiles(s) is not None]
        valid_smiles = [smiles_list[i] for i in valid_idx]

        datapoints = [data.MoleculeDatapoint.from_smi(s) for s in valid_smiles]
        dset = data.MoleculeDataset(datapoints, self._featurizer)
        loader = data.build_dataloader(dset, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=False)

        fps: list[np.ndarray] = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Chemprop fingerprints", unit="batch", leave=True):
                bmg, *_ = batch
                bmg.to(device=self._model.device)
                fps.append(self._model.fingerprint(bmg).numpy(force=True))

        if not fps:
            dummy = self._model.fingerprint(
                BatchMolGraph([self._featurizer(MolFromSmiles("C"))]).to(device=self._model.device)
            )
            return np.full((len(smiles_list), dummy.shape[1]), np.nan, dtype=np.float32)

        valid_fps = np.concatenate(fps, axis=0)
        out = np.full((len(smiles_list), valid_fps.shape[1]), np.nan, dtype=np.float32)
        for result_i, orig_i in enumerate(valid_idx):
            out[orig_i] = valid_fps[result_i]
        return out
