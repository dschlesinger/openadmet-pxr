"""EDA script for the PXR challenge training set.

Outputs PNG plots to viz/eda/.

Usage:
    python scripts/eda_train.py
    python scripts/eda_train.py --data data/train.csv --out viz/eda
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

sns.set_theme(style="whitegrid", palette="muted")
_FIG_DPI = 150


# ---------------------------------------------------------------------------
# Chemistry helpers
# ---------------------------------------------------------------------------

def _compute_mol_props(smiles_series: pl.Series) -> pl.DataFrame:
    records = []
    for smi in smiles_series:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            records.append(None)
            continue
        records.append({
            "MW": Descriptors.MolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "TPSA": Descriptors.TPSA(mol),
            "HBD": rdMolDescriptors.CalcNumHBD(mol),
            "HBA": rdMolDescriptors.CalcNumHBA(mol),
            "RotBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "AromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
            "HeavyAtoms": mol.GetNumHeavyAtoms(),
        })
    valid = [r for r in records if r is not None]
    return pl.DataFrame(valid)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _savefig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=_FIG_DPI)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_response_distributions(df: pl.DataFrame, out: Path) -> None:
    targets = [
        ("pEC50", "pEC50  (−log₁₀ M)"),
        ("Emax_estimate (log2FC vs. baseline)", "Emax  (log₂FC vs. baseline)"),
        ("Emax.vs.pos.ctrl_estimate (dimensionless)", "Emax vs. positive control  (dimensionless)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (col, label) in zip(axes, targets):
        vals = df[col].to_numpy()
        sns.histplot(vals, kde=True, ax=ax, bins=40)
        ax.axvline(float(df[col].mean()), color="red", linestyle="--", linewidth=1.2, label=f"mean={df[col].mean():.2f}")
        ax.axvline(float(df[col].median()), color="orange", linestyle=":", linewidth=1.2, label=f"median={df[col].median():.2f}")
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution of {col.split(' ')[0]}")
        ax.legend(fontsize=8)

    fig.suptitle("Response variable distributions  (train set, n=4 139)", fontsize=12)
    _savefig(fig, out / "response_distributions.png")


def plot_response_pairplot(df: pl.DataFrame, out: Path) -> None:
    cols = [
        "pEC50",
        "Emax_estimate (log2FC vs. baseline)",
        "Emax.vs.pos.ctrl_estimate (dimensionless)",
    ]
    pdf = df.select(cols).to_pandas()
    pdf.columns = ["pEC50", "Emax (log2FC)", "Emax vs. ctrl"]
    g = sns.pairplot(pdf, diag_kind="kde", plot_kws={"alpha": 0.3, "s": 8})
    g.figure.suptitle("Response variable pairplot  (train set)", y=1.01, fontsize=11)
    g.figure.tight_layout()
    g.figure.savefig(out / "response_pairplot.png", dpi=_FIG_DPI)
    plt.close(g.figure)
    print(f"Saved: {out / 'response_pairplot.png'}")


def plot_mol_property_distributions(props: pl.DataFrame, out: Path) -> None:
    prop_meta = [
        ("MW", "Molecular weight  (Da)"),
        ("LogP", "cLogP"),
        ("TPSA", "TPSA  (Å²)"),
        ("HBD", "H-bond donors"),
        ("HBA", "H-bond acceptors"),
        ("RotBonds", "Rotatable bonds"),
        ("AromaticRings", "Aromatic rings"),
        ("HeavyAtoms", "Heavy atom count"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, (col, label) in zip(axes.flat, prop_meta):
        vals = props[col].to_numpy()
        discrete = col in {"HBD", "HBA", "RotBonds", "AromaticRings", "HeavyAtoms"}
        sns.histplot(vals, kde=not discrete, discrete=discrete, ax=ax, bins=None if discrete else 40)
        ax.axvline(float(props[col].mean()), color="red", linestyle="--", linewidth=1.1,
                   label=f"μ={props[col].mean():.1f}")
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(label)
        ax.legend(fontsize=7)

    fig.suptitle("Molecular property distributions  (train set)", fontsize=12)
    _savefig(fig, out / "mol_property_distributions.png")


def plot_lipinski_ro5(props: pl.DataFrame, out: Path) -> None:
    ro5_rules = {
        "MW ≤ 500": (props["MW"] <= 500).sum(),
        "LogP ≤ 5": (props["LogP"] <= 5).sum(),
        "HBD ≤ 5": (props["HBD"] <= 5).sum(),
        "HBA ≤ 10": (props["HBA"] <= 10).sum(),
    }
    n = len(props)
    labels = list(ro5_rules.keys())
    pcts = [v / n * 100 for v in ro5_rules.values()]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, pcts, color=sns.color_palette("muted", 4))
    ax.axhline(100, color="grey", linestyle="--", linewidth=0.8)
    ax.set_ylim(0, 110)
    ax.set_ylabel("% compounds passing")
    ax.set_title("Lipinski Ro5 compliance  (train set)")
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, pct + 1, f"{pct:.1f}%", ha="center", fontsize=9)
    _savefig(fig, out / "lipinski_ro5.png")


def plot_mw_vs_logp(props: pl.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(
        props["MW"].to_numpy(),
        props["LogP"].to_numpy(),
        alpha=0.3, s=8, c=props["TPSA"].to_numpy(), cmap="viridis"
    )
    plt.colorbar(sc, ax=ax, label="TPSA (Å²)")
    ax.axvline(500, color="red", linestyle="--", linewidth=1, label="MW=500")
    ax.axhline(5, color="orange", linestyle="--", linewidth=1, label="LogP=5")
    ax.set_xlabel("Molecular weight  (Da)")
    ax.set_ylabel("cLogP")
    ax.set_title("MW vs. cLogP coloured by TPSA  (train set)")
    ax.legend(fontsize=8)
    _savefig(fig, out / "mw_vs_logp.png")


def plot_uncertainty_distributions(df: pl.DataFrame, out: Path) -> None:
    se_cols = [
        ("pEC50_std.error (-log10(molarity))", "pEC50 SE"),
        ("Emax_std.error (log2FC vs. baseline)", "Emax SE (log2FC)"),
        ("Emax.vs.pos.ctrl_std.error (dimensionless)", "Emax vs. ctrl SE"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (col, label) in zip(axes, se_cols):
        vals = df[col].to_numpy()
        sns.histplot(vals, kde=True, ax=ax, bins=40)
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(f"Curve-fit SE: {label}")

    fig.suptitle("Curve-fit standard errors  (train set)", fontsize=12)
    _savefig(fig, out / "uncertainty_distributions.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="EDA for PXR train set.")
    parser.add_argument("--data", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--out", type=Path, default=Path("viz/eda"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.data} ...")
    df = pl.read_csv(args.data)
    print(f"  {df.shape[0]} rows, {df.shape[1]} columns")

    print("Computing molecular properties ...")
    props = _compute_mol_props(df["SMILES"])
    print(f"  {len(props)} valid molecules")

    print("Plotting ...")
    plot_response_distributions(df, args.out)
    plot_response_pairplot(df, args.out)
    plot_mol_property_distributions(props, args.out)
    plot_lipinski_ro5(props, args.out)
    plot_mw_vs_logp(props, args.out)
    plot_uncertainty_distributions(df, args.out)

    print("Done.")


if __name__ == "__main__":
    main()
