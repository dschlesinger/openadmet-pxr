"""EDA script for the PXR challenge counter (background) assay.

Focus: how molecular properties relate to background noise levels.
Compounds with no curve fit above noise appear as null pEC50/Emax.

Outputs PNG plots to viz/eda_counter/.

Usage:
    python scripts/eda_counter.py
    python scripts/eda_counter.py --data data/counter_train.csv --out viz/eda_counter
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

sns.set_theme(style="whitegrid", palette="muted")
_FIG_DPI = 150

MOL_PROPS = ["MW", "LogP", "TPSA", "HBD", "HBA", "RotBonds", "AromaticRings", "HeavyAtoms"]
RESPONSE_COLS = [
    ("pEC50", "pEC50  (−log₁₀ M)"),
    ("Emax_estimate (log2FC vs. baseline)", "Emax  (log₂FC)"),
    ("Emax.vs.pos.ctrl_estimate (dimensionless)", "Emax vs. ctrl"),
]


# ---------------------------------------------------------------------------
# Chemistry helpers
# ---------------------------------------------------------------------------

def _compute_mol_props(smiles_series: pl.Series) -> pl.DataFrame:
    records = []
    for smi in smiles_series:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            records.append({p: None for p in MOL_PROPS})
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
    return pl.DataFrame(records)


def _savefig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=_FIG_DPI)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_null_by_mol_props(props_fit: pl.DataFrame, props_null: pl.DataFrame, out: Path) -> None:
    """KDE: mol properties of fitted vs. null (background-only) compounds."""
    prop_labels = {
        "MW": "Molecular weight (Da)", "LogP": "cLogP", "TPSA": "TPSA (Å²)",
        "HBD": "H-bond donors", "HBA": "H-bond acceptors", "RotBonds": "Rotatable bonds",
        "AromaticRings": "Aromatic rings", "HeavyAtoms": "Heavy atoms",
    }
    palette = sns.color_palette("muted", 2)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, prop in zip(axes.flat, MOL_PROPS):
        fit_vals = props_fit[prop].drop_nulls().to_numpy()
        null_vals = props_null[prop].drop_nulls().to_numpy()
        discrete = prop in {"HBD", "HBA", "RotBonds", "AromaticRings", "HeavyAtoms"}
        if discrete:
            bins = np.arange(int(min(fit_vals.min(), null_vals.min())),
                             int(max(fit_vals.max(), null_vals.max())) + 2) - 0.5
            ax.hist(fit_vals, bins=bins, density=True, alpha=0.5, color=palette[0],
                    label=f"Fitted (n={len(fit_vals)})")
            ax.hist(null_vals, bins=bins, density=True, alpha=0.5, color=palette[1],
                    label=f"Null (n={len(null_vals)})")
        else:
            sns.kdeplot(fit_vals, ax=ax, color=palette[0], linewidth=1.5,
                        label=f"Fitted (n={len(fit_vals)})")
            sns.kdeplot(null_vals, ax=ax, color=palette[1], linewidth=1.5, linestyle="--",
                        label=f"Null (n={len(null_vals)})")
        ax.set_xlabel(prop_labels[prop])
        ax.set_ylabel("Density")
        ax.set_title(prop_labels[prop])
        ax.legend(fontsize=7)

    fig.suptitle("Mol. properties: fitted vs. null (no curve fit above background)", fontsize=12)
    _savefig(fig, out / "null_vs_fit_mol_props.png")


def plot_props_vs_response(props: pl.DataFrame, df_fit: pl.DataFrame, out: Path) -> None:
    """Scatter: each mol property vs. counter-assay pEC50 and Emax."""
    # Join props (same row order as df_fit) with response cols
    resp_cols = [c for c, _ in RESPONSE_COLS]
    combined = pl.concat([props, df_fit.select(resp_cols)], how="horizontal")

    fig, axes = plt.subplots(len(MOL_PROPS), len(RESPONSE_COLS), figsize=(14, 22))
    palette = sns.color_palette("muted")

    for row, prop in enumerate(MOL_PROPS):
        for col_idx, (resp_col, resp_label) in enumerate(RESPONSE_COLS):
            ax = axes[row][col_idx]
            x = combined[prop].drop_nulls()
            # align after any mol parse failures
            sub = combined.filter(pl.col(prop).is_not_null() & pl.col(resp_col).is_not_null())
            x_vals = sub[prop].to_numpy()
            y_vals = sub[resp_col].to_numpy()
            ax.scatter(x_vals, y_vals, alpha=0.25, s=6, color=palette[col_idx % len(palette)])
            # regression line
            if len(x_vals) > 2:
                m, b = np.polyfit(x_vals, y_vals, 1)
                x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
                ax.plot(x_line, m * x_line + b, color="red", linewidth=1,
                        label=f"r={np.corrcoef(x_vals, y_vals)[0, 1]:.2f}")
                ax.legend(fontsize=7)
            ax.set_xlabel(prop)
            ax.set_ylabel(resp_label)
            if row == 0:
                ax.set_title(resp_label)

    fig.suptitle("Mol. properties vs. counter-assay response (background signal)", fontsize=13, y=1.001)
    _savefig(fig, out / "props_vs_response.png")


def plot_correlation_heatmap(props: pl.DataFrame, df_fit: pl.DataFrame, out: Path) -> None:
    """Correlation heatmap: mol properties × counter-assay response variables."""
    resp_cols = [c for c, _ in RESPONSE_COLS]
    combined = pl.concat([props, df_fit.select(resp_cols)], how="horizontal").drop_nulls()
    pdf = combined.to_pandas()

    corr = pdf[MOL_PROPS + resp_cols].corr().loc[MOL_PROPS, resp_cols]
    resp_short = [l for _, l in RESPONSE_COLS]

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        vmin=-1, vmax=1, ax=ax,
        xticklabels=resp_short, yticklabels=MOL_PROPS,
        linewidths=0.5,
    )
    ax.set_title("Mol. properties × background response  (Pearson r)")
    _savefig(fig, out / "correlation_heatmap.png")


def plot_logp_tpsa_vs_noise(props: pl.DataFrame, df_fit: pl.DataFrame, out: Path) -> None:
    """2-D scatter of LogP vs. TPSA coloured by pEC50 background level."""
    resp_col = "pEC50"
    combined = pl.concat([props, df_fit.select([resp_col])], how="horizontal").drop_nulls()

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(
        combined["LogP"].to_numpy(),
        combined["TPSA"].to_numpy(),
        c=combined[resp_col].to_numpy(),
        cmap="plasma", alpha=0.4, s=10,
    )
    plt.colorbar(sc, ax=ax, label="pEC50 (background)")
    ax.set_xlabel("cLogP")
    ax.set_ylabel("TPSA (Å²)")
    ax.set_title("LogP vs. TPSA coloured by counter-assay pEC50")
    _savefig(fig, out / "logp_tpsa_vs_noise.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="EDA for PXR counter (background) assay.")
    parser.add_argument("--data", type=Path, default=Path("data/counter_train.csv"))
    parser.add_argument("--out", type=Path, default=Path("viz/eda_counter"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.data} ...")
    df = pl.read_csv(args.data)
    n_total = len(df)
    n_null = df["pEC50"].null_count()
    print(f"  {n_total} rows  |  {n_null} null / no curve fit ({n_null / n_total * 100:.1f}%)")

    df_fit = df.filter(pl.col("pEC50").is_not_null())
    df_null = df.filter(pl.col("pEC50").is_null())

    print("Computing molecular properties ...")
    props_fit = _compute_mol_props(df_fit["SMILES"])
    props_null = _compute_mol_props(df_null["SMILES"])
    props_all = _compute_mol_props(df["SMILES"])
    print(f"  {len(props_fit)} fitted  |  {len(props_null)} null")

    print("Plotting ...")
    plot_null_by_mol_props(props_fit, props_null, args.out)
    plot_props_vs_response(props_fit, df_fit, args.out)
    plot_correlation_heatmap(props_fit, df_fit, args.out)
    plot_logp_tpsa_vs_noise(props_fit, df_fit, args.out)

    print("Done.")


if __name__ == "__main__":
    main()
