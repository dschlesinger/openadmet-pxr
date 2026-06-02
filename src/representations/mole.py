"""MolE (Recursion) graph-transformer molecular representations.

Calls third_party/mole/gather_representation.py via subprocess in the
moleß conda environment. SMILES are written to a temp CSV; embeddings are
read back from the output TSV.
"""

import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import polars as pl

from representations.base import Representation

_MOLE_DIR = Path(__file__).parents[2] / "third_party" / "mole"
_MOLE_CONDA_ENV = "mole"
_DEFAULT_REPRESENTATION = "gin_concat_R1000_E8000_lambda0.0001"
_ZENODO_MODEL_URL = "https://zenodo.org/api/records/10803099/files/model.pth/content"


def _ensure_weights(representation: str) -> None:
    model_pth = _MOLE_DIR / "ckpt" / representation / "checkpoints" / "model.pth"
    if model_pth.exists():
        return
    model_pth.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MolE weights to {model_pth} ...")
    urllib.request.urlretrieve(_ZENODO_MODEL_URL, model_pth)
    print("Download complete.")


class MolERepresentation(Representation):
    """MolE pretrained graph-transformer embeddings via subprocess."""

    name: ClassVar[str] = "mole"

    def __init__(
        self,
        representation: str = _DEFAULT_REPRESENTATION,
        accelerator: str = "cuda:0",
    ) -> None:
        self._representation = representation
        self._accelerator = accelerator

    def transform(self, smiles: pl.Series) -> np.ndarray:
        """Return a (n_molecules, embedding_dim) float32 array of MolE embeddings."""
        from rdkit.Chem import MolFromSmiles

        _ensure_weights(self._representation)

        smiles_list = smiles.to_list()
        valid_idx = [i for i, s in enumerate(smiles_list) if MolFromSmiles(s) is not None]
        valid_smiles = [smiles_list[i] for i in valid_idx]

        if not valid_smiles:
            return np.full((len(smiles_list), 1), np.nan, dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            input_csv = tmp / "smiles.csv"
            output_tsv = tmp / "output.tsv.gz"

            pd.DataFrame({"smiles": valid_smiles}).to_csv(input_csv, index=False)

            result = subprocess.run(
                [
                    "conda", "run", "-n", _MOLE_CONDA_ENV, "--no-capture-output",
                    "python", "gather_representation.py",
                    "--smiles_filepath", str(input_csv),
                    "--smiles_colname", "smiles",
                    "--representation", self._representation,
                    "--gpu", self._accelerator,
                    "--output_filepath", str(output_tsv),
                ],
                cwd=str(_MOLE_DIR),
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"MolE subprocess failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
                )

            emb_df = pd.read_csv(output_tsv, sep="\t", index_col=0)

        valid_emb = emb_df.values.astype(np.float32)
        out = np.full((len(smiles_list), valid_emb.shape[1]), np.nan, dtype=np.float32)
        for result_i, orig_i in enumerate(valid_idx):
            out[orig_i] = valid_emb[result_i]
        return out
