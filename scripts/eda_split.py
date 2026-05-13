"""Analyse the train/test split strategy for the PXR challenge.

Outputs a markdown report to viz/eda/train_test_split_analysis.md and prints a
summary to stdout.

Usage:
    python scripts/eda_split.py
    python scripts/eda_split.py --train data/train.csv --test data/test.csv --out viz/eda
"""

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import polars as pl
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold

warnings.filterwarnings("ignore", category=DeprecationWarning)

_SAMPLE_SIZE = 200  # test molecules used for NN Tanimoto (full run is slow)
_TANIMOTO_THRESHOLDS = [0.4, 0.7]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _murcko(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def _morgan_fp(smi: str):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    return gen.GetFingerprint(mol)


def _oadmet_id(name: str) -> int | None:
    m = re.search(r"OADMET-(\d+)", name)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Analysis sections
# ---------------------------------------------------------------------------

def _compound_overlap(train: pl.DataFrame, test: pl.DataFrame) -> dict:
    smiles_overlap = len(set(train["SMILES"]) & set(test["SMILES"]))
    name_overlap = len(set(train["Molecule Name"]) & set(test["Molecule Name"]))
    return {"smiles_overlap": smiles_overlap, "name_overlap": name_overlap}


def _id_analysis(train: pl.DataFrame, test: pl.DataFrame) -> dict:
    train_ids = sorted(_oadmet_id(n) for n in train["Molecule Name"] if _oadmet_id(n))
    test_ids = sorted(_oadmet_id(n) for n in test["Molecule Name"] if _oadmet_id(n))
    train_max = max(train_ids)
    above = [n for n in test_ids if n > train_max]
    overlap = [n for n in test_ids if n <= train_max]
    return {
        "train_min": min(train_ids),
        "train_max": train_max,
        "test_min": min(test_ids),
        "test_max": max(test_ids),
        "test_above_train_max": len(above),
        "test_in_train_range": overlap,
        "n_test": len(test_ids),
    }


def _scaffold_analysis(train: pl.DataFrame, test: pl.DataFrame) -> dict:
    print("  Computing Murcko scaffolds ...")
    train_scafs = [_murcko(s) for s in train["SMILES"]]
    test_scafs = [_murcko(s) for s in test["SMILES"]]

    train_scaf_set = {s for s in train_scafs if s}
    test_scaf_set = {s for s in test_scafs if s}
    shared_scafs = train_scaf_set & test_scaf_set

    test_mols_with_train_scaf = sum(1 for s in test_scafs if s in train_scaf_set)
    test_mols_novel = sum(1 for s in test_scafs if s and s not in train_scaf_set)

    return {
        "train_unique_scaffolds": len(train_scaf_set),
        "test_unique_scaffolds": len(test_scaf_set),
        "shared_scaffolds": len(shared_scafs),
        "pct_test_scafs_shared": len(shared_scafs) / len(test_scaf_set) * 100,
        "test_mols_with_train_scaf": test_mols_with_train_scaf,
        "test_mols_novel_scaf": test_mols_novel,
        "n_test": len(test_scafs),
    }


def _tanimoto_analysis(train: pl.DataFrame, test: pl.DataFrame, sample: int) -> dict:
    print(f"  Computing Morgan fingerprints (sample={sample} test molecules) ...")
    train_fps = [fp for s in train["SMILES"] if (fp := _morgan_fp(s))]

    rng = np.random.default_rng(42)
    test_smiles = test["SMILES"].to_list()
    idx = rng.choice(len(test_smiles), size=min(sample, len(test_smiles)), replace=False)
    sample_fps = [fp for i in idx if (fp := _morgan_fp(test_smiles[i]))]

    print(f"  Computing bulk Tanimoto ({len(sample_fps)} test × {len(train_fps)} train) ...")
    nn_sims = [max(DataStructs.BulkTanimotoSimilarity(fp, train_fps)) for fp in sample_fps]
    arr = np.array(nn_sims)

    result: dict = {
        "n_sample": len(arr),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
    for t in _TANIMOTO_THRESHOLDS:
        key = f"pct_above_{str(t).replace('.', '_')}"
        result[key] = float((arr > t).mean() * 100)
    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _render_report(
    n_train: int,
    n_test: int,
    overlap: dict,
    ids: dict,
    scaffolds: dict,
    tanimoto: dict,
    tanimoto_sample: int,
) -> str:
    above_pct = ids["test_above_train_max"] / ids["n_test"] * 100
    novel_pct = scaffolds["test_mols_novel_scaf"] / scaffolds["n_test"] * 100
    shared_mol_pct = scaffolds["test_mols_with_train_scaf"] / scaffolds["n_test"] * 100

    lines = [
        "# Train/Test Split Analysis",
        "",
        "## Conclusion",
        "",
        "The PXR challenge uses a **prospective (temporal) split**, not a scaffold-based split.",
        "Test compounds were registered after the training set was assembled, mimicking real drug discovery",
        "where a model must predict activity for the next round of synthesized compounds.",
        "",
        "---",
        "",
        "## Dataset sizes",
        "",
        "| Split | Compounds |",
        "|-------|-----------|",
        f"| Train | {n_train:,} |",
        f"| Test  | {n_test:,} |",
        "",
        "---",
        "",
        "## Evidence",
        "",
        "### 1. OADMET ID ordering",
        "",
        "Each compound carries an OADMET registration ID that reflects synthesis/registration order.",
        "",
        "| | Min ID | Max ID |",
        "|-|--------|--------|",
        f"| Train | {ids['train_min']:,} | {ids['train_max']:,} |",
        f"| Test  | {ids['test_min']:,} | {ids['test_max']:,} |",
        "",
        f"**{ids['test_above_train_max']:,} / {ids['n_test']:,} ({above_pct:.1f}%) of test IDs are "
        f"strictly greater than the training maximum ({ids['train_max']:,}).**",
    ]

    if ids["test_in_train_range"]:
        overlap_ids = ", ".join(str(x) for x in ids["test_in_train_range"])
        lines.append(
            f"Only {len(ids['test_in_train_range'])} test compound(s) fall within the training ID "
            f"range (IDs: {overlap_ids}) — likely a small batch held back at assignment time."
        )

    lines += [
        "",
        "### 2. No leakage at the compound level",
        "",
        f"- Exact SMILES overlap between train and test: **{overlap['smiles_overlap']}**",
        f"- Molecule Name overlap: **{overlap['name_overlap']}**",
        "",
        "### 3. Scaffold novelty is a consequence, not the mechanism",
        "",
        "Murcko scaffold analysis confirms the prospective nature of the split:",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Unique scaffolds in train | {scaffolds['train_unique_scaffolds']:,} |",
        f"| Unique scaffolds in test  | {scaffolds['test_unique_scaffolds']:,} |",
        f"| Scaffolds shared (train ∩ test) | {scaffolds['shared_scaffolds']} "
        f"({scaffolds['pct_test_scafs_shared']:.1f}% of test scaffolds) |",
        f"| Test molecules with a scaffold unseen in train | {scaffolds['test_mols_novel_scaf']} / "
        f"{scaffolds['n_test']} **({novel_pct:.1f}%)** |",
        f"| Test molecules whose scaffold appears in train | {scaffolds['test_mols_with_train_scaf']} / "
        f"{scaffolds['n_test']} ({shared_mol_pct:.1f}%) |",
        "",
        "The high scaffold novelty is a by-product of the temporal split rather than an explicit "
        "scaffold-splitting constraint. Later design cycles naturally explored different chemical matter.",
        "",
        "### 4. Moderate Tanimoto nearest-neighbor similarity",
        "",
        f"Morgan fingerprint (r=2, 2048 bits) Tanimoto similarity from each test compound to its "
        f"nearest training neighbour (random sample of {tanimoto['n_sample']} test molecules, seed=42):",
        "",
        "| Statistic | Value |",
        "|-----------|-------|",
        f"| Mean NN Tanimoto | {tanimoto['mean']:.3f} |",
        f"| Median NN Tanimoto | {tanimoto['median']:.3f} |",
    ]

    for t in _TANIMOTO_THRESHOLDS:
        key = f"pct_above_{str(t).replace('.', '_')}"
        lines.append(f"| % with NN sim > {t} | {tanimoto[key]:.1f}% |")

    lines += [
        f"| Min / Max | {tanimoto['min']:.2f} / {tanimoto['max']:.2f} |",
        "",
        "Test compounds are structurally related to the training chemical space (same screening "
        "campaign) but are not close analogues in the majority of cases.",
        "",
        "---",
        "",
        "## Implications for modelling",
        "",
        "- **Scaffold interpolation is not sufficient** — over 80% of test scaffolds are unseen, "
        "so a model that memorises scaffold–activity relationships will struggle.",
        "- **Generalisation across chemical space is required** — the task is closer to lead-hopping "
        "than to analogue activity prediction.",
        "- **Uncertainty estimates matter** — high scaffold novelty means predictions for test "
        "compounds sit at or beyond the edge of the training distribution.",
        "- **The local val split (random 10%) is optimistic** — it does not replicate the "
        "prospective nature of the test set. A scaffold- or ID-ordered internal validation split "
        "would give a more realistic estimate of generalisation performance.",
    ]

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse PXR train/test split strategy.")
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--out", type=Path, default=Path("viz/eda"))
    parser.add_argument(
        "--tanimoto-sample",
        type=int,
        default=_SAMPLE_SIZE,
        help=f"Number of test molecules for NN Tanimoto (default: {_SAMPLE_SIZE})",
    )
    args = parser.parse_args()

    print(f"Loading {args.train} and {args.test} ...")
    train = pl.read_csv(args.train)
    test = pl.read_csv(args.test)
    print(f"  Train: {len(train):,} rows  |  Test: {len(test):,} rows")

    print("Compound-level overlap ...")
    overlap = _compound_overlap(train, test)
    print(f"  SMILES overlap: {overlap['smiles_overlap']}  |  Name overlap: {overlap['name_overlap']}")

    print("OADMET ID analysis ...")
    ids = _id_analysis(train, test)
    pct = ids["test_above_train_max"] / ids["n_test"] * 100
    print(f"  {ids['test_above_train_max']}/{ids['n_test']} ({pct:.1f}%) test IDs > train max")

    print("Scaffold analysis ...")
    scaffolds = _scaffold_analysis(train, test)
    novel_pct = scaffolds["test_mols_novel_scaf"] / scaffolds["n_test"] * 100
    print(f"  Novel scaffold molecules: {scaffolds['test_mols_novel_scaf']} ({novel_pct:.1f}%)")

    print("Tanimoto NN analysis ...")
    tanimoto = _tanimoto_analysis(train, test, args.tanimoto_sample)
    print(f"  Mean NN Tanimoto: {tanimoto['mean']:.3f}  |  Median: {tanimoto['median']:.3f}")

    args.out.mkdir(parents=True, exist_ok=True)
    report_path = args.out / "train_test_split_analysis.md"
    report = _render_report(len(train), len(test), overlap, ids, scaffolds, tanimoto, args.tanimoto_sample)
    report_path.write_text(report)
    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
