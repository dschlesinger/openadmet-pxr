"""UniMol 3D molecular representations via unimol_tools.

Uses the pretrained UniMol model to generate 512-dimensional CLS-token
embeddings that encode 3D geometry from SMILES. Weights are downloaded
automatically on first use (~500 MB) to ~/.unimol/.

Requires: pip install unimol_tools
"""

from typing import ClassVar

import numpy as np
import polars as pl

from representations.base import Representation


class UniMolRepresentation(Representation):
    """UniMol pretrained 3D molecular fingerprints (CLS-token, 512-dim).

    Invalid SMILES produce an all-NaN row (cleaned by drop_nan_rows downstream).
    """

    name: ClassVar[str] = "unimol"

    def __init__(
        self,
        remove_hs: bool = False,
        use_gpu: bool = True,
        batch_size: int = 64,
    ) -> None:
        self._remove_hs = remove_hs
        self._use_gpu = use_gpu
        self._batch_size = batch_size
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from unimol_tools import UniMolRepr  # type: ignore[import]
        self._model = UniMolRepr(
            data_type="molecule",
            remove_hs=self._remove_hs,
            use_gpu=self._use_gpu,
        )

    def transform(self, smiles: pl.Series) -> np.ndarray:
        """Return a (n_molecules, 512) float32 array of UniMol CLS embeddings."""
        import torch
        self._ensure_loaded()
        assert self._model is not None

        smiles_list = smiles.to_list()

        from rdkit.Chem import MolFromSmiles
        valid_idx = [i for i, s in enumerate(smiles_list) if MolFromSmiles(s) is not None]
        valid_smiles = [smiles_list[i] for i in valid_idx]

        if not valid_smiles:
            return np.full((len(smiles_list), 512), np.nan, dtype=np.float32)

        # Pass all valid SMILES in one call — unimol_tools handles its own
        # internal batching. Calling get_repr in a Python loop leaks GPU memory
        # because each call allocates a new GPU context without releasing the last.
        result = self._model.get_repr(valid_smiles, return_atomic_reprs=False)
        torch.cuda.empty_cache()

        valid_fps = np.array(result, dtype=np.float32)
        out = np.full((len(smiles_list), valid_fps.shape[1]), np.nan, dtype=np.float32)
        for result_i, orig_i in enumerate(valid_idx):
            out[orig_i] = valid_fps[result_i]
        return out
