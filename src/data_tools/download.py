"""Download PXR challenge dataset splits from HuggingFace to a local data folder."""

import argparse
import urllib.error
import urllib.request
from pathlib import Path

import polars as pl

_HF_BASE = "hf://datasets/openadmet/pxr-challenge-train-test/"
_HF_TRAIN = "pxr-challenge_TRAIN.csv"
_HF_TEST = "pxr-challenge_TEST_BLINDED.csv"
_HF_TEST_UNBLINDED = "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv"
_HF_COUNTER_TRAIN = "pxr-challenge_counter-assay_TRAIN.csv"
_HF_SINGLE_CONC_TRAIN = "pxr-challenge_single_concentration_TRAIN.csv"
_TRAIN_FILE = "train.csv"
_TEST_FILE = "test.csv"
_TEST_UNBLINDED_FILE = "test_unblinded.csv"
_COUNTER_TRAIN_FILE = "counter_train.csv"
_SINGLE_CONC_TRAIN_FILE = "single_concentration_train.csv"
_DEFAULT_DATA_DIR = Path("data")
_DEFAULT_STRUCTURES_DIR = Path("structures")
_PDB_DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"

# Unbound (apo) structures per protein — saved to structures/<protein>/apo/
_APO_STRUCTURES: dict[str, list[str]] = {
    # PXR (NR1I2, UniProt O75469)
    "pxr": [
        "1ILG",  # Apo human PXR LBD (canonical apo)
        "3CTB",  # Tethered PXR-LBD/SRC-1p apoprotein
        "4J5W",  # Apo PXR/RXRalpha LBD heterotetramer
        "4S0S",  # PXR LBD with Adnectin-1 (no small-molecule ligand)
        "7AX8",  # hPXR-LBD apo form (P43212 SG)
        "8SVN",  # Apo PXR LBD (recent)
        "8SVU",  # L428V mutant apo
        "9O41",  # L411A mutant apo
        "9O44",  # L411W mutant apo
    ],
    # VDR (NR1I1, UniProt P11473) — all apo entries are DBD; no apo LBD exists in PDB
    "vdr": [
        "1KB2",  # VDR DBD bound to mouse osteopontin response element
        "1KB4",  # VDR DBD bound to canonical DR3 response element
        "1KB6",  # VDR DBD bound to rat osteocalcin response element
        "1YNW",  # VDR/RXRalpha DBD heterodimer on DR3
    ],
    # FXR (NR1H4, UniProt Q96RI1)
    "fxr": [
        "5Q0K",  # FXR LBD apo
        "6HL0",  # FXR LBD with NCoA-2 coactivator peptide (no small-molecule ligand)
        "8HBM",  # FXR/RXRalpha heterodimer on IR DNA (no small-molecule ligand)
    ],
    # RXRalpha (NR2B1, UniProt P19793) — LBD apo structures only
    "rxra": [
        "1G1U",  # RXRalpha LBD tetramer, no ligand (canonical apo)
        "3NSP",  # Tetrameric RXRalpha LBD
        "3R29",  # RXRalpha LBD with corepressor SMRT2 (no small-molecule ligand)
    ],
    # CAR (NR1I3, UniProt Q14994) — no apo structures in PDB
}

# Ligand-bound structures per protein — saved to structures/<protein>/bound/
_BOUND_STRUCTURES: dict[str, list[str]] = {
    # PXR (NR1I2, UniProt O75469) — 70 structures
    "pxr": [
        "1ILH", "1M13", "1NRL", "1SKX", "2O9I", "2QNV", "3HVL", "3R8D", "4J5X", "4NY9",
        "4S0T", "4X1F", "4X1G", "4XHD", "5A86", "5X0R", "6BNS", "6DUP", "6HJ2", "6HTY",
        "6NX1", "6P2B", "6S41", "6TFI", "6XP9", "7AX9", "7AXA", "7AXB", "7AXC", "7AXD",
        "7AXE", "7AXF", "7AXG", "7AXH", "7AXI", "7AXJ", "7AXK", "7AXL", "7N2A", "7RIO",
        "7RIU", "7RIV", "7YFK", "8CCT", "8CF9", "8CH8", "8E3N", "8EQZ", "8F5Y", "8FPE",
        "8R00", "8R81", "8R82", "8SVO", "8SVP", "8SVQ", "8SVR", "8SVS", "8SVT", "8SVX",
        "8SZV", "9BEQ", "9FZG", "9FZH", "9FZI", "9FZJ", "9O42", "9O43", "9O45", "9O46",
    ],
    # VDR (NR1I1, UniProt P11473) — 48 structures
    "vdr": [
        "1DB1", "1IE8", "1IE9", "1S0Z", "1S19", "1TXI", "2HAM", "2HAR", "2HAS", "2HB7",
        "2HB8", "3A2I", "3A2J", "3A3Z", "3A40", "3A78", "3AUQ", "3AUR", "3AX8", "3AZ1",
        "3AZ2", "3AZ3", "3B0T", "3CS4", "3CS6", "3KPZ", "3M7R", "3OGT", "3P8X", "3TKC",
        "3VHW", "3W0A", "3W0C", "3W0Y", "3WGP", "3WWR", "3X31", "3X36", "4G2I", "4ITE",
        "4ITF", "5GT4", "5V39", "5YSY", "5YT2", "7QPP", "8IQN", "8IQT",
    ],
    # FXR (NR1H4, UniProt Q96RI1) — 86 structures
    "fxr": [
        "1OSH", "3BEJ", "3DCT", "3DCU", "3FLI", "3FXV", "3GD2", "3HC5", "3HC6", "3L1B",
        "3OKH", "3OKI", "3OLF", "3OMK", "3OMM", "3OOF", "3OOK", "3P88", "3P89", "3RUT",
        "3RUU", "3RVF", "4OIV", "4QE8", "4WVD", "5IAW", "5ICK", "5Q0I", "5Q0J", "5Q0L",
        "5Q0M", "5Q0N", "5Q0O", "5Q0P", "5Q0Q", "5Q0R", "5Q0S", "5Q0T", "5Q0U", "5Q0V",
        "5Q0W", "5Q0X", "5Q0Y", "5Q0Z", "5Q10", "5Q11", "5Q12", "5Q13", "5Q14", "5Q15",
        "5Q16", "5Q17", "5Q18", "5Q19", "5Q1A", "5Q1B", "5Q1C", "5Q1D", "5Q1E", "5Q1F",
        "5Q1G", "5Q1H", "5Q1I", "5WZX", "5Y1J", "5Y44", "5Y49", "5YXB", "5YXD", "5YXJ",
        "5YXL", "5Z12", "6A5W", "6A5X", "6A5Y", "6A5Z", "6A60", "6HL1", "6ITM", "7D42",
        "7TRB", "7VUE", "8HD3", "9H65", "9H66", "9LQ3",
    ],
    # RXRalpha (NR2B1, UniProt P19793) — 91 structures
    "rxra": [
        "1FBY", "1FM6", "1FM9", "1G5Y", "1K74", "1MV9", "1MVC", "1MZN", "1RDT", "1XLS",
        "1XV9", "1XVP", "2ACL", "2P1T", "2P1U", "2P1V", "2ZXZ", "2ZY0", "3DZU", "3DZY",
        "3E00", "3E94", "3FAL", "3FC6", "3FUG", "3H0A", "3KWY", "3NSQ", "3OAP", "3OZJ",
        "3PCU", "3R2A", "3R5M", "3UVV", "4J5X", "4K4J", "4K6I", "4M8E", "4M8H", "4N5G",
        "4N8R", "4NQA", "4OC7", "4POH", "4POJ", "4PP3", "4PP5", "4RFW", "4RMC", "4RMD",
        "4RME", "4ZO1", "4ZSH", "5EC9", "5JI0", "5LYQ", "5MJ5", "5MK4", "5MKJ", "5MKU",
        "5MMW", "5TBP", "5UAN", "5Z12", "5ZQU", "6A5Y", "6A5Z", "6A60", "6FBQ", "6FBR",
        "6HN6", "6JNO", "6JNR", "6L6K", "6LB4", "6LB5", "6LB6", "6SJM", "6STI", "7B88",
        "7B9O", "7BK4", "7CFO", "7NKE", "7UW2", "7UW4", "8PP0", "9QX6", "9RMR", "9S70",
        "9S71",
    ],
    # CAR (NR1I3, UniProt Q14994) — 2 structures (all bound; no apo exists)
    "car": [
        "1XV9",  # CAR/RXR heterodimer with SRC1, fatty acid, and 5b-pregnane-3,20-dione
        "1XVP",  # CAR/RXR heterodimer with SRC1, fatty acid, and CITCO
    ],
}


def _download_split(hf_path: str, local_path: Path) -> None:
    if local_path.exists():
        print(f"Skipping (already exists): {local_path}")
        return
    local_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {hf_path} ...")
    df = pl.read_csv(hf_path)
    df.write_csv(local_path)
    print(f"Saved {len(df)} rows -> {local_path}")


def _download_pdb(pdb_id: str, dest: Path) -> None:
    if dest.exists():
        print(f"Skipping (already exists): {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = _PDB_DOWNLOAD_URL.format(pdb_id=pdb_id)
    print(f"Downloading {url} ...")
    try:
        urllib.request.urlretrieve(url, dest)
    except urllib.error.HTTPError as e:
        print(f"Warning: {pdb_id} unavailable as PDB ({e.code}), skipping")
        return
    print(f"Saved -> {dest}")


def _download_protein_structures(protein: str, pdb_ids: list[str], structures_dir: Path, subfolder: str) -> None:
    dest_dir = structures_dir / protein / subfolder
    for pdb_id in pdb_ids:
        _download_pdb(pdb_id, dest_dir / f"{pdb_id}.pdb")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PXR challenge dataset splits.")
    parser.add_argument("--no-train", action="store_true", help="Skip the train split.")
    parser.add_argument("--no-test", action="store_true", help="Skip the blinded test split.")
    parser.add_argument("--no-test-unblinded", action="store_true", help="Skip the phase-1 unblinded test split.")
    parser.add_argument("--no-counter-train", action="store_true", help="Skip the counter-assay train split.")
    parser.add_argument("--no-single-concentration", action="store_true", help="Skip the single-concentration train split.")
    parser.add_argument(
        "--train-location",
        type=Path,
        default=_DEFAULT_DATA_DIR / _TRAIN_FILE,
        help="Destination path for the train CSV.",
    )
    parser.add_argument(
        "--test-location",
        type=Path,
        default=_DEFAULT_DATA_DIR / _TEST_FILE,
        help="Destination path for the blinded test CSV.",
    )
    parser.add_argument(
        "--test-unblinded-location",
        type=Path,
        default=_DEFAULT_DATA_DIR / _TEST_UNBLINDED_FILE,
        help="Destination path for the phase-1 unblinded test CSV.",
    )
    parser.add_argument(
        "--counter-train-location",
        type=Path,
        default=_DEFAULT_DATA_DIR / _COUNTER_TRAIN_FILE,
        help="Destination path for the counter-assay train CSV.",
    )
    parser.add_argument(
        "--single-concentration-location",
        type=Path,
        default=_DEFAULT_DATA_DIR / _SINGLE_CONC_TRAIN_FILE,
        help="Destination path for the single-concentration train CSV.",
    )
    parser.add_argument("--no-structures", action="store_true", help="Skip downloading protein structures.")
    parser.add_argument(
        "--structures-dir",
        type=Path,
        default=_DEFAULT_STRUCTURES_DIR,
        help="Root directory for downloaded protein structures.",
    )
    args = parser.parse_args()

    if not args.no_train:
        _download_split(_HF_BASE + _HF_TRAIN, args.train_location)

    if not args.no_test:
        _download_split(_HF_BASE + _HF_TEST, args.test_location)

    if not args.no_test_unblinded:
        _download_split(_HF_BASE + _HF_TEST_UNBLINDED, args.test_unblinded_location)

    if not args.no_counter_train:
        _download_split(_HF_BASE + _HF_COUNTER_TRAIN, args.counter_train_location)

    if not args.no_single_concentration:
        _download_split(_HF_BASE + _HF_SINGLE_CONC_TRAIN, args.single_concentration_location)

    if not args.no_structures:
        for protein, pdb_ids in _APO_STRUCTURES.items():
            _download_protein_structures(protein, pdb_ids, args.structures_dir, "apo")
        for protein, pdb_ids in _BOUND_STRUCTURES.items():
            _download_protein_structures(protein, pdb_ids, args.structures_dir, "bound")


if __name__ == "__main__":
    main()
