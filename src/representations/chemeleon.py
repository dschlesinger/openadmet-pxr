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
from chemprop import data, featurizers, nn
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


_DEFAULT_CHEMELEON_CKPT_DIR = Path("checkpoints/chemeleon")


def _find_best_checkpoint(ckpt_dir: Path) -> Path:
    candidates = sorted(ckpt_dir.glob("best-epoch*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No best-epoch*.ckpt found in {ckpt_dir}")
    return candidates[-1]


def _load_finetuned_model(
    checkpoint_dir: Path, device: str | torch.device | None
) -> tuple[MPNN, featurizers.SimpleMoleculeMolGraphFeaturizer]:
    ckpt = _find_best_checkpoint(checkpoint_dir)
    model = MPNN.load_from_checkpoint(ckpt)
    model.eval()
    if device is not None:
        model.to(device=device)
    return model, featurizers.SimpleMoleculeMolGraphFeaturizer()


class FinetunedChemeleonFingerprint(Representation):
    """CheMeleon fingerprints from a locally finetuned Lightning checkpoint.

    Loads the full MPNN from a checkpoint saved by finetune_chemeleon.py and
    extracts the message-passing fingerprint, identical interface to ChemeleonFingerprint.
    """

    name: ClassVar[str] = "chemeleon_finetuned"

    def __init__(
        self,
        checkpoint_dir: Path | str = _DEFAULT_CHEMELEON_CKPT_DIR,
        device: str | torch.device | None = None,
        batch_size: int = 64,
        num_workers: int = 0,
    ) -> None:
        self._checkpoint_dir = Path(checkpoint_dir)
        self._device = device
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._model: MPNN | None = None
        self._featurizer: featurizers.SimpleMoleculeMolGraphFeaturizer | None = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self._model, self._featurizer = _load_finetuned_model(self._checkpoint_dir, self._device)

    def transform(self, smiles: pl.Series) -> np.ndarray:
        """Return a (n_molecules, embedding_dim) float32 array of finetuned CheMeleon fingerprints."""
        self._ensure_loaded()
        assert self._model is not None and self._featurizer is not None

        smiles_list = smiles.to_list()
        valid_idx = [i for i, s in enumerate(smiles_list) if MolFromSmiles(s) is not None]
        valid_smiles = [smiles_list[i] for i in valid_idx]

        datapoints = [data.MoleculeDatapoint.from_smi(s) for s in valid_smiles]
        dset = data.MoleculeDataset(datapoints, self._featurizer)
        loader = data.build_dataloader(dset, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=False, drop_last=False)

        fps: list[np.ndarray] = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Finetuned CheMeleon fingerprints", unit="batch", leave=True):
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


class ChemeleonFingerprint(Representation):
    """CheMeleon pretrained MPNN fingerprints.

    Invalid SMILES produce an all-NaN row (cleaned by drop_nan_rows downstream).
    """

    name: ClassVar[str] = "chemeleon"

    def __init__(
        self,
        device: str | torch.device | None = None,
        batch_size: int = 64,
        num_workers: int = 0,
    ) -> None:
        self._device = device
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._model: MPNN | None = None
        self._featurizer: featurizers.SimpleMoleculeMolGraphFeaturizer | None = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self._model, self._featurizer = _load_model(self._device)

    def transform(self, smiles: pl.Series) -> np.ndarray:
        """Return a (n_molecules, embedding_dim) float32 array of CheMeleon fingerprints."""
        self._ensure_loaded()
        assert self._model is not None and self._featurizer is not None

        smiles_list = smiles.to_list()
        valid_idx = [i for i, s in enumerate(smiles_list) if MolFromSmiles(s) is not None]
        valid_smiles = [smiles_list[i] for i in valid_idx]

        datapoints = [data.MoleculeDatapoint.from_smi(s) for s in valid_smiles]
        dset = data.MoleculeDataset(datapoints, self._featurizer)
        loader = data.build_dataloader(dset, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=False, drop_last=False)

        fps: list[np.ndarray] = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="CheMeleon fingerprints", unit="batch", leave=True):
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
