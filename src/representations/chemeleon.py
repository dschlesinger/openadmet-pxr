"""CheMeleon learned molecular fingerprints via pretrained MPNN.

Wraps the CheMeleonFingerprint from JacksonBurns/chemeleon. On first use the
pretrained weights (~3.5 MB) are downloaded automatically to ~/.chemprop/.
Requires: pip install 'chemprop>=2.2.0'
"""

import tempfile
from pathlib import Path
from typing import ClassVar
from urllib.request import urlretrieve

import numpy as np
import polars as pl
import torch
from chemprop import featurizers, nn
from chemprop.data import BatchMolGraph
from chemprop.models import MPNN
from chemprop.nn import RegressionFFN
from rdkit.Chem import MolFromSmiles
from tqdm import tqdm

from representations.base import Representation

_WEIGHTS_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"


def _load_model(device: str | torch.device | None) -> tuple[MPNN, featurizers.SimpleMoleculeMolGraphFeaturizer]:
    ckpt_dir = Path.home() / ".chemprop"
    ckpt_dir.mkdir(exist_ok=True)
    mp_path = ckpt_dir / "chemeleon_mp.pt"
    if not mp_path.exists():
        print(f"Downloading CheMeleon weights -> {mp_path}")
        tmp = Path(tempfile.mktemp(dir=ckpt_dir, suffix=".tmp"))
        try:
            with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc="chemeleon_mp.pt") as bar:
                def _reporthook(count: int, block_size: int, total_size: int) -> None:
                    if total_size > 0 and bar.total is None:
                        bar.total = total_size
                    bar.update(block_size)
                urlretrieve(_WEIGHTS_URL, tmp, reporthook=_reporthook)
            tmp.rename(mp_path)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
    chemeleon_mp = torch.load(mp_path, weights_only=True)
    mp = nn.BondMessagePassing(**chemeleon_mp["hyper_parameters"])
    mp.load_state_dict(chemeleon_mp["state_dict"])
    model = MPNN(
        message_passing=mp,
        agg=nn.MeanAggregation(),
        predictor=RegressionFFN(input_dim=mp.output_dim),
    )
    model.eval()
    if device is not None:
        model.to(device=device)
    return model, featurizers.SimpleMoleculeMolGraphFeaturizer()


class ChemeleonFingerprint(Representation):
    """CheMeleon pretrained MPNN fingerprints.

    Invalid SMILES produce an all-NaN row (cleaned by drop_nan_rows downstream).
    """

    name: ClassVar[str] = "chemeleon"

    def __init__(self, device: str | torch.device | None = None) -> None:
        self._device = device
        self._model: MPNN | None = None
        self._featurizer: featurizers.SimpleMoleculeMolGraphFeaturizer | None = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self._model, self._featurizer = _load_model(self._device)

    def transform(self, smiles: pl.Series) -> np.ndarray:
        """Return a (n_molecules, embedding_dim) float32 array of CheMeleon fingerprints."""
        self._ensure_loaded()
        smiles_list = smiles.to_list()
        mols = [MolFromSmiles(s) for s in tqdm(smiles_list, desc="Parsing SMILES", unit="mol", leave=False)]

        valid_idx = [i for i, m in enumerate(mols) if m is not None]
        valid_mols = [mols[i] for i in valid_idx]

        if not valid_mols:
            dummy = self._run_batch([MolFromSmiles("C")])
            return np.full((len(smiles_list), dummy.shape[1]), np.nan, dtype=np.float32)

        valid_fps = self._run_batch(valid_mols)
        out = np.full((len(smiles_list), valid_fps.shape[1]), np.nan, dtype=np.float32)
        for result_i, orig_i in enumerate(valid_idx):
            out[orig_i] = valid_fps[result_i]
        return out

    def _run_batch(self, mols: list) -> np.ndarray:
        assert self._model is not None and self._featurizer is not None
        bmg = BatchMolGraph([self._featurizer(m) for m in tqdm(mols, desc="CheMeleon fingerprints", unit="mol", leave=True)])
        bmg.to(device=self._model.device)
        with torch.no_grad():
            return self._model.fingerprint(bmg).numpy(force=True)
