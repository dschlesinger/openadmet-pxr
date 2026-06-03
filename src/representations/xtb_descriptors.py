"""xTB (GFN2-xTB) semi-empirical quantum chemistry descriptor representation.

Calls third_party/xtb/compute_xtb.py via subprocess in the conda environment
that has xtb-python installed (default: base). SMILES are written to a temp CSV;
feature arrays are read back from the output .npy file.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import polars as pl

from representations.base import Representation

_XTB_DIR = Path(__file__).parents[2] / "third_party" / "xtb"
_XTB_CONDA_ENV = "xtb"

# total_energy, homo_energy, lumo_energy, homo_lumo_gap,
# dipole_x, dipole_y, dipole_z, dipole_norm,
# charges_min, charges_max, charges_mean, charges_std
N_XTB_FEATURES = 12

_FEATURE_NAMES = [
    "total_energy",
    "homo_energy",
    "lumo_energy",
    "homo_lumo_gap",
    "dipole_x",
    "dipole_y",
    "dipole_z",
    "dipole_norm",
    "charges_min",
    "charges_max",
    "charges_mean",
    "charges_std",
]


def _find_conda_base() -> Path | None:
    """Return the conda base prefix by asking conda itself, or by searching common paths."""
    import shutil

    conda_exe = shutil.which("conda")
    if conda_exe:
        result = subprocess.run([conda_exe, "info", "--base"], capture_output=True, text=True)
        if result.returncode == 0:
            return Path(result.stdout.strip())
    home = Path.home()
    for name in ["miniconda3", "anaconda3", "miniforge3", "mambaforge", "miniconda", "anaconda"]:
        candidate = home / name
        if (candidate / "bin" / "conda").exists():
            return candidate
    return None


def _find_conda_env_python(env_name: str) -> str:
    """Return the Python executable for a named conda env."""
    base = _find_conda_base()
    if base is not None:
        candidate = base / "bin" / "python" if env_name == "base" else base / "envs" / env_name / "bin" / "python"
        if candidate.exists():
            return str(candidate)
    raise RuntimeError(
        f"Cannot find Python for conda env '{env_name}'. "
        f"Create it with: conda env create -f third_party/xtb/environment.yml"
    )


class XTBDescriptors(Representation):
    """GFN2-xTB semi-empirical quantum chemistry descriptors via subprocess.

    Requires xtb-python in the target conda environment:
        conda install -c conda-forge xtb-python
    Invalid SMILES produce an all-NaN row.
    """

    name: ClassVar[str] = "xtb"

    def __init__(self, conda_env: str = _XTB_CONDA_ENV) -> None:
        self._conda_env = conda_env

    @property
    def feature_names(self) -> list[str]:
        return list(_FEATURE_NAMES)

    def transform(self, smiles: pl.Series) -> np.ndarray:
        """Return a (n_molecules, N_XTB_FEATURES) float64 array of xTB descriptors."""
        python = _find_conda_env_python(self._conda_env)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            input_csv = tmp / "smiles.csv"
            output_npy = tmp / "features.npy"

            pd.DataFrame({"SMILES": smiles.to_list()}).to_csv(input_csv, index=False)

            result = subprocess.run(
                [
                    python,
                    str(_XTB_DIR / "compute_xtb.py"),
                    "--input", str(input_csv),
                    "--output", str(output_npy),
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"xTB subprocess failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
                )

            return np.load(str(output_npy))
