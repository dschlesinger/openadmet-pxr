"""Migrate hash-keyed feature cache files to the molecule-keyed format.

Old format:  data/features/{rep}_{sha256}.npy  +  data/features/smiles_{sha256}.txt
New format:  data/features/{rep}.npy            +  data/features/{rep}.idx

Run once after upgrading inputs.py:
    python scripts/migrate_feature_cache.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

CACHE_DIR = Path("data/features")


def _load_index(idx_path: Path) -> list[str]:
    return idx_path.read_text().splitlines()


def migrate(cache_dir: Path = CACHE_DIR) -> None:
    if not cache_dir.exists():
        print(f"Cache directory {cache_dir} does not exist — nothing to migrate.")
        return

    # Find all old-style {rep}_{hash}.npy files
    old_npy = [p for p in cache_dir.glob("*.npy") if re.match(r".+_[0-9a-f]{16}$", p.stem)]
    if not old_npy:
        print("No old-style hash-keyed .npy files found — nothing to migrate.")
        return

    # Group by representation name
    by_rep: dict[str, list[Path]] = {}
    for p in old_npy:
        rep = p.stem.rsplit("_", 1)[0]
        by_rep.setdefault(rep, []).append(p)

    for rep, candidates in sorted(by_rep.items()):
        new_npy = cache_dir / f"{rep}.npy"
        new_idx = cache_dir / f"{rep}.idx"

        if new_npy.exists() and new_idx.exists():
            print(f"  {rep}: already migrated — skipping")
            continue

        # Pick the candidate with the most rows (largest superset)
        best: Path | None = None
        best_smiles: list[str] = []
        for candidate in candidates:
            h = candidate.stem.rsplit("_", 1)[1]
            idx_path = cache_dir / f"smiles_{h}.txt"
            if not idx_path.exists():
                print(f"  {rep}: no index for {candidate.name} — skipping this candidate")
                continue
            smiles = _load_index(idx_path)
            if len(smiles) > len(best_smiles):
                best = candidate
                best_smiles = smiles

        if best is None:
            print(f"  {rep}: no indexed candidate found — cannot migrate; run featurize on full train set first")
            continue

        arr = np.load(str(best))
        if arr.shape[0] != len(best_smiles):
            print(f"  {rep}: row count mismatch in {best.name} ({arr.shape[0]} rows vs {len(best_smiles)} SMILES) — skipping")
            continue

        np.save(str(new_npy), arr)
        new_idx.write_text("\n".join(best_smiles))
        print(f"  {rep}: migrated {len(best_smiles)} molecules from {best.name}")


if __name__ == "__main__":
    cache_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else CACHE_DIR
    print(f"Migrating cache in {cache_dir} ...")
    migrate(cache_dir)
    print("Done.")
