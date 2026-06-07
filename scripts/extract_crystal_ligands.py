"""Extract co-crystal ligand SMILES from the bound nuclear-receptor PDB structures.

Walks ``structures/<receptor>/bound/*.pdb``, collects every distinct HET (HETATM)
3-letter chemical-component code, drops common crystallization additives / ions /
solvents / modified residues, then resolves each remaining code to a canonical
SMILES via the RCSB ``chemcomp`` REST API.

The result is cached to ``data/crystal_ligands.csv`` (columns: ``receptor``,
``het_code``, ``smiles``, ``name``) so the downstream representation works offline.
Re-running only fetches codes not already present in the cache.

Usage:
    python scripts/extract_crystal_ligands.py
    python scripts/extract_crystal_ligands.py --structures-dir structures --out data/crystal_ligands.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import polars as pl
import requests
from rdkit import Chem

# HET codes that are crystallization additives, ions, solvents, sugars, or
# modified amino acids rather than functional receptor ligands. These dominate
# the HETATM records but carry no SAR signal, so they are excluded outright.
EXCLUDED_HET_CODES: frozenset[str] = frozenset(
    {
        # water / common solvents
        "HOH",
        "DOD",
        "GOL",
        "EDO",
        "PEG",
        "PG4",
        "PGE",
        "1PE",
        "2PE",
        "P6G",
        "MPD",
        "IPA",
        "DMS",
        "DMF",
        "ACT",
        "ACY",
        "EOH",
        "MOH",
        "TRS",
        "BME",
        "MES",
        "EPE",
        "FMT",
        "CIT",
        "FLC",
        "TLA",
        "MLI",
        "BTB",
        "POL",
        "DTT",
        "PE4",
        "PEU",
        "12P",
        "15P",
        "33O",
        "DIO",
        "BU3",
        "OCT",
        "SIN",
        "TAR",
        # ions
        "SO4",
        "PO4",
        "NA",
        "CL",
        "MG",
        "ZN",
        "CA",
        "K",
        "MN",
        "FE",
        "FE2",
        "CD",
        "NI",
        "CO",
        "CU",
        "CU1",
        "HG",
        "BR",
        "IOD",
        "F",
        "NO3",
        "NH4",
        "CAC",
        "SCN",
        "AZI",
        "BR3",
        "GA",
        "SR",
        "CS",
        "RB",
        "LI",
        "BA",
        # cryoprotectants / detergents / lipids commonly co-modeled
        "BOG",
        "LDA",
        "C8E",
        "OLC",
        "OLA",
        "PLM",
        "MYR",
        "STE",
        "PEE",
        "PCW",
        "LMT",
        "LMN",
        "DDQ",
        "HTO",
        "BNG",
        "12M",
        "F09",
        "JEF",
        # sugars
        "NAG",
        "BMA",
        "MAN",
        "FUC",
        "GAL",
        "GLC",
        "BGC",
        "XYP",
        "SUC",
        "FRU",
        # modified / non-standard residues frequently flagged as HETATM
        "MSE",
        "SEP",
        "TPO",
        "PTR",
        "CSO",
        "CSD",
        "CME",
        "KCX",
        "LLP",
        "PCA",
        "ACE",
        "NH2",
        "NME",
        "FME",
    }
)

RCSB_CHEMCOMP_URL = "https://data.rcsb.org/rest/v1/core/chemcomp/{code}"
RECEPTORS: tuple[str, ...] = ("pxr", "fxr", "rxra", "vdr", "car")


def collect_het_codes(structures_dir: Path) -> dict[str, set[str]]:
    """Return {receptor: {het_code, ...}} for every bound PDB, minus excluded codes."""
    per_receptor: dict[str, set[str]] = {}
    for receptor in RECEPTORS:
        bound_dir = structures_dir / receptor / "bound"
        codes: set[str] = set()
        for pdb_path in sorted(bound_dir.glob("*.pdb")):
            for line in pdb_path.read_text().splitlines():
                if line.startswith("HETATM"):
                    code = line[17:20].strip()
                    if code and code not in EXCLUDED_HET_CODES:
                        codes.add(code)
        per_receptor[receptor] = codes
        print(f"{receptor}: {len(codes)} candidate ligand codes from {bound_dir}")
    return per_receptor


def fetch_smiles(code: str, session: requests.Session) -> tuple[str, str] | None:
    """Resolve a HET code to (canonical_smiles, name) via RCSB; None on failure."""
    try:
        resp = session.get(RCSB_CHEMCOMP_URL.format(code=code), timeout=15)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"  ! {code}: request failed ({exc})")
        return None
    data = resp.json()
    descriptor = data.get("rcsb_chem_comp_descriptor", {})
    # Prefer stereo SMILES when present; fall back to the flat form.
    raw = descriptor.get("SMILES_stereo") or descriptor.get("SMILES")
    if not raw:
        return None
    mol = Chem.MolFromSmiles(raw)
    if mol is None:
        print(f"  ! {code}: RDKit could not parse SMILES {raw!r}")
        return None
    name = data.get("chem_comp", {}).get("name", "")
    return Chem.MolToSmiles(mol), name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structures-dir", type=Path, default=Path("structures"))
    parser.add_argument("--out", type=Path, default=Path("data/crystal_ligands.csv"))
    parser.add_argument(
        "--min-heavy-atoms", type=int, default=6, help="Drop fetched ligands with fewer heavy atoms than this."
    )
    args = parser.parse_args()

    cached: dict[tuple[str, str], dict[str, str]] = {}
    if args.out.exists():
        existing = pl.read_csv(args.out)
        for row in existing.iter_rows(named=True):
            cached[(row["receptor"], row["het_code"])] = row
        print(f"Loaded {len(cached)} cached ligand rows from {args.out}")

    per_receptor = collect_het_codes(args.structures_dir)
    session = requests.Session()
    rows: list[dict[str, str]] = list(cached.values())

    for receptor, codes in per_receptor.items():
        for code in sorted(codes):
            if (receptor, code) in cached:
                continue
            result = fetch_smiles(code, session)
            time.sleep(0.05)  # be polite to the RCSB API
            if result is None:
                continue
            smiles, name = result
            mol = Chem.MolFromSmiles(smiles)
            if mol is None or mol.GetNumHeavyAtoms() < args.min_heavy_atoms:
                continue
            rows.append({"receptor": receptor, "het_code": code, "smiles": smiles, "name": name})
            print(f"  + {receptor}/{code}: {smiles}")

    out_df = pl.DataFrame(rows, schema={"receptor": pl.Utf8, "het_code": pl.Utf8, "smiles": pl.Utf8, "name": pl.Utf8})
    out_df = out_df.unique(subset=["receptor", "het_code"]).sort(["receptor", "het_code"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_csv(args.out)
    print(f"\nWrote {len(out_df)} ligand rows to {args.out}")
    print(out_df.group_by("receptor").len().sort("receptor"))


if __name__ == "__main__":
    main()
