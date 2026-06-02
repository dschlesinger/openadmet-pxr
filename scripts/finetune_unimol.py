"""Finetune UniMol on PXR training data end-to-end.

Uses unimol_tools.MolTrain with task='multilabel_regression' (MAE loss), scaffold
5-fold internal CV, and optional counter-assay multitask head. Outer n-folds loop
trains with different seeds and averages predictions.

Inference supports SMILES augmentation (averaging predictions across randomized
atom orderings) to reduce variance, following discoverybytes (rank 11).

Run download-data first to produce the raw data CSVs under data/.

Usage:
    python scripts/finetune_unimol.py
    python scripts/finetune_unimol.py --epochs 30 --lr 2e-4 --n-folds 3
    python scripts/finetune_unimol.py \\
        --epochs 30 --lr 2e-4 \\
        --split-type butina --butina-cutoff 0.4 \\
        --include-counter-assay \\
        --n-folds 3

Overfitting notes:
  Lower LR (5e-5) and early stopping are the primary levers.
  --pooler-dropout controls dropout before the linear head (default 0.3).
  Do NOT freeze the encoder — that degenerates to evaluate-models --input unimol.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit.Chem import MolFromSmiles, MolToSmiles  # type: ignore[import]

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_tools.load import load_data, load_test as _load_test_df  # noqa: E402


def _randomize_smiles(smiles: str, seed: int) -> str:
    mol = MolFromSmiles(smiles)
    if mol is None:
        return smiles
    return MolToSmiles(mol, doRandom=True, randomSeed=seed % (2**31))


def _predict_with_augmentation(
    predictor: "MolPredict",
    smiles_list: list[str],
    n_aug: int,
    seed: int,
) -> np.ndarray:
    """Return mean predictions across n_aug randomized SMILES orderings."""
    all_preds: list[np.ndarray] = []
    for i in range(n_aug):
        aug_smiles = [_randomize_smiles(s, seed + i * 1337) for s in smiles_list]
        preds = predictor.predict(aug_smiles)
        preds = np.asarray(preds, dtype=np.float64)
        if preds.ndim == 2:
            preds = preds[:, 0]  # primary task (pEC50)
        all_preds.append(preds.flatten())
    return np.mean(all_preds, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Finetune UniMol on PXR data.")
    # Data
    parser.add_argument("--test", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--split-type",
        choices=["random", "scaffold", "butina", "unblinded"],
        default="butina",
        help="Outer train/val split strategy (default: butina cluster split)",
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--butina-cutoff", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-counter-assay", action="store_true",
                        help="Add counter-assay pEC50 as a second regression task")
    parser.add_argument("--include-single-concentration", action="store_true",
                        help="Add aggregated single-conc log2FC as a third regression task")
    # Output
    parser.add_argument("--out", type=Path, default=Path("results/unimol_finetuned_submission.csv"))
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints/unimol"))
    # Training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--kfold", type=int, default=5, help="UniMol internal scaffold kfold")
    parser.add_argument("--early-stopping", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--n-folds", type=int, default=1,
                        help="Outer loop: number of seeds to train and average predictions over")
    parser.add_argument("--n-aug", type=int, default=10,
                        help="SMILES augmentation orderings for inference averaging (0 to disable)")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU training")
    parser.add_argument(
        "--pooler-dropout",
        type=float,
        default=0.3,
        help="Dropout rate applied before the linear regression head (default 0.3). "
             "UniMol pretrained default is 0.2.",
    )
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoints.mkdir(parents=True, exist_ok=True)

    from unimol_tools import MolPredict, MolTrain  # type: ignore[import]

    test_df = pd.read_csv(args.test)
    test_smiles = test_df["SMILES"].tolist()

    target_cols = ["pEC50"]
    if args.include_counter_assay:
        target_cols.append("pEC50_counter")
    if args.include_single_concentration:
        target_cols.append("log2_fc_single")
    print(f"Tasks: {target_cols}")

    all_fold_preds: list[np.ndarray] = []
    pec50_range: tuple[float, float] | None = None

    for fold_idx in range(args.n_folds):
        fold_seed = args.seed + fold_idx
        train_df, val_df = load_data(
            split_type=args.split_type,
            val_fraction=args.val_fraction,
            butina_cutoff=args.butina_cutoff,
            seed=fold_seed,
            include_counter_assay=args.include_counter_assay,
            include_single_concentration=args.include_single_concentration,
            data_dir=args.data_dir,
        )
        train_pdf = train_df.to_pandas()
        val_pdf = val_df.to_pandas()

        if pec50_range is None:
            pec50_range = (float(train_pdf["pEC50"].min()), float(train_pdf["pEC50"].max()))

        print(f"\n[Fold {fold_idx}] Train: {len(train_pdf)} | Val: {len(val_pdf)} | Test: {len(test_smiles)}")
        if args.include_counter_assay:
            n_ca = train_pdf["pEC50_counter"].notna().sum()
            print(f"[Fold {fold_idx}] Counter-assay labels: {n_ca}/{len(train_pdf)}")

        # Select only SMILES + task columns; keep NaN for missing auxiliary tasks
        available_cols = ["SMILES"] + [c for c in target_cols if c in train_pdf.columns]
        train_input = train_pdf[available_cols]

        ckpt_path = args.checkpoints / f"fold{fold_idx}"
        ckpt_path.mkdir(parents=True, exist_ok=True)

        print(
            f"[Fold {fold_idx}] Training UniMol — "
            f"epochs={args.epochs}, lr={args.lr}, pooler_dropout={args.pooler_dropout}, "
            f"kfold={args.kfold}"
        )
        trainer = MolTrain(
            task="multilabel_regression",
            data_type="molecule",
            epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            early_stopping=args.early_stopping,
            metrics="mae",
            split="scaffold",
            kfold=args.kfold,
            save_path=str(ckpt_path),
            remove_hs=False,
            target_cols=target_cols,
            target_normalize="auto",
            use_cuda=not args.no_gpu,
            pooler_dropout=args.pooler_dropout,
        )
        trainer.fit(train_input)

        predictor = MolPredict(load_model=str(ckpt_path))
        n_aug = args.n_aug if args.n_aug > 0 else 1
        fold_preds = _predict_with_augmentation(predictor, test_smiles, n_aug, fold_seed)
        all_fold_preds.append(fold_preds)
        print(f"[Fold {fold_idx}] Test pEC50 mean={fold_preds.mean():.3f} std={fold_preds.std():.3f}")

    # Average across outer folds and clip to training range
    pec50_preds = np.mean(all_fold_preds, axis=0)
    pec50_min, pec50_max = pec50_range  # type: ignore[misc]
    pec50_preds = np.clip(pec50_preds, pec50_min, pec50_max)

    submission = pd.DataFrame({
        "Molecule Name": test_df["Molecule Name"],
        "SMILES": test_df["SMILES"],
        "pEC50": pec50_preds,
    })
    submission.to_csv(args.out, index=False)

    stats = submission["pEC50"].describe()
    print(f"\npEC50  mean={stats['mean']:.3f}  std={stats['std']:.3f}  "
          f"min={stats['min']:.3f}  max={stats['max']:.3f}")
    print(f"Submission saved -> {args.out}")

    # Prediction distribution plot
    fig, ax = plt.subplots()
    ax.hist(pec50_preds, bins=30, edgecolor="black")
    ax.set_xlabel("Predicted pEC50")
    ax.set_ylabel("Count")
    ax.set_title(f"UniMol finetuned — {args.n_folds} outer fold(s), {args.split_type} split")
    fig.tight_layout()
    plot_path = args.out.parent / (args.out.stem + "_dist.png")
    fig.savefig(plot_path, dpi=150)
    print(f"Distribution plot saved -> {plot_path}")


if __name__ == "__main__":
    main()
