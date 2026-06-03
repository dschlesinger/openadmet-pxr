"""Finetuned UniMol 3D molecular representation.

Loads model_*.pth checkpoints saved by scripts/finetune_unimol.py and extracts
512-dimensional CLS-token embeddings. When multiple checkpoint files exist
(from internal kfold CV), embeddings are averaged across all fold models.

Requires: pip install unimol_tools
Run scripts/finetune_unimol.py first to produce checkpoints.
"""

import shutil
from pathlib import Path
from typing import ClassVar

import numpy as np
import polars as pl

from representations.base import Representation

_DEFAULT_UNIMOL_CKPT_DIR = Path("checkpoints/unimol/fold0")


class FinetunedUniMolRepresentation(Representation):
    """Finetuned UniMol CLS-token embeddings (512-dim), ensembled across CV fold checkpoints.

    Invalid SMILES produce an all-NaN row (cleaned by drop_nan_rows downstream).
    Raises FileNotFoundError on first transform() if no checkpoints are found.
    """

    name: ClassVar[str] = "unimol_finetuned"

    def __init__(
        self,
        checkpoint_dir: Path | str = _DEFAULT_UNIMOL_CKPT_DIR,
        remove_hs: bool = False,
        use_gpu: bool = True,
        batch_size: int = 32,
    ) -> None:
        self._checkpoint_dir = Path(checkpoint_dir)
        self._remove_hs = remove_hs
        self._use_gpu = use_gpu
        self._batch_size = batch_size
        self._models: list | None = None

    def _ensure_loaded(self) -> None:
        if self._models is not None:
            return
        from unimol_tools import UniMolRepr  # type: ignore[import]
        from unimol_tools.weights.weighthub import get_weight_dir  # type: ignore[import]

        ckpt_files = sorted(self._checkpoint_dir.glob("model_*.pth"))
        if not ckpt_files:
            raise FileNotFoundError(
                f"No model_*.pth checkpoints found in {self._checkpoint_dir}. "
                "Run scripts/finetune_unimol.py first."
            )

        # UniMolModel expects dict.txt alongside each .pth — copy from weights dir if absent
        dict_dst = self._checkpoint_dir / "dict.txt"
        if not dict_dst.exists():
            dict_src = Path(get_weight_dir()) / "mol.dict.txt"
            if dict_src.exists():
                shutil.copy(dict_src, dict_dst)

        import torch

        self._models = []
        for ckpt in ckpt_files:
            # UniMolRepr hardcodes output_dim=1; construct with base weights then
            # overwrite encoder weights from the finetuned checkpoint (strict=False
            # silently skips the head size mismatch — head is unused during get_repr)
            model = UniMolRepr(
                data_type="molecule",
                batch_size=self._batch_size,
                remove_hs=self._remove_hs,
                use_cuda=self._use_gpu,
            )
            state = torch.load(str(ckpt), map_location=str(model.device), weights_only=True)
            inner = state.get("model_state_dict", state)
            current = model.model.state_dict()
            compatible = {k: v for k, v in inner.items() if k in current and v.shape == current[k].shape}
            model.model.load_state_dict(compatible, strict=False)
            self._models.append(model)
        print(f"Loaded {len(self._models)} UniMol checkpoint(s) from {self._checkpoint_dir}")

    def transform(self, smiles: pl.Series) -> np.ndarray:
        """Return a (n_molecules, 512) float32 array of finetuned UniMol CLS embeddings."""
        import torch
        from rdkit.Chem import MolFromSmiles  # type: ignore[import]

        self._ensure_loaded()
        assert self._models is not None

        smiles_list = smiles.to_list()
        valid_idx = [i for i, s in enumerate(smiles_list) if MolFromSmiles(s) is not None]
        valid_smiles = [smiles_list[i] for i in valid_idx]

        if not valid_smiles:
            return np.full((len(smiles_list), 512), np.nan, dtype=np.float32)

        all_embeddings: list[np.ndarray] = []
        for model in self._models:
            result = model.get_repr(valid_smiles, return_atomic_reprs=False)
            torch.cuda.empty_cache()
            all_embeddings.append(np.array(result, dtype=np.float32))

        # Average embeddings across internal CV fold checkpoints
        valid_fps = np.mean(all_embeddings, axis=0)

        out = np.full((len(smiles_list), valid_fps.shape[1]), np.nan, dtype=np.float32)
        for result_i, orig_i in enumerate(valid_idx):
            out[orig_i] = valid_fps[result_i]
        return out
