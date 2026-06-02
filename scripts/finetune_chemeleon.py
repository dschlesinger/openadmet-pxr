"""Finetune the CheMeleon pretrained MPNN on PXR training data.

Downloads chemeleon_mp.pt from Zenodo (cached to ~/.chemprop/) on first run.
Trains a chemprop MPNN with the pretrained message-passing backbone end-to-end
using MAE loss, optional multitask heads (counter-assay, single-concentration),
Butina-cluster CV split, and optional fold averaging across multiple seeds.

Run download-data first to produce the raw data CSVs under data/.

Usage:
    python scripts/finetune_chemeleon.py
    python scripts/finetune_chemeleon.py --epochs 30 --n-folds 3
    python scripts/finetune_chemeleon.py \\
        --epochs 30 --lr 5e-4 \\
        --split-type butina --butina-cutoff 0.4 \\
        --include-counter-assay --include-single-concentration \\
        --n-folds 5
"""

import argparse
import sys
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from chemprop import data, featurizers, nn
from chemprop import models
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_tools.load import load_data, load_test as _load_test_df  # noqa: E402

_WEIGHTS_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"
_WEIGHTS_CACHE = Path.home() / ".chemprop" / "chemeleon_mp.pt"


class _LossLogger(pl.Callback):
    def __init__(self) -> None:
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        if "train_loss" in metrics:
            self.train_losses.append(float(metrics["train_loss"]))
        if "val_loss" in metrics:
            self.val_losses.append(float(metrics["val_loss"]))


def _download_weights() -> Path:
    _WEIGHTS_CACHE.parent.mkdir(exist_ok=True)
    if not _WEIGHTS_CACHE.exists():
        print(f"Downloading CheMeleon weights -> {_WEIGHTS_CACHE}")
        tmp = Path(tempfile.mktemp(dir=_WEIGHTS_CACHE.parent, suffix=".tmp"))
        try:
            with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc="chemeleon_mp.pt") as bar:
                def _hook(count: int, block_size: int, total_size: int) -> None:
                    if total_size > 0 and bar.total is None:
                        bar.total = total_size
                    bar.update(block_size)
                urlretrieve(_WEIGHTS_URL, tmp, reporthook=_hook)
            tmp.rename(_WEIGHTS_CACHE)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
    return _WEIGHTS_CACHE


def _load_mp() -> nn.BondMessagePassing:
    ckpt = torch.load(_download_weights(), weights_only=True)
    mp = nn.BondMessagePassing(**ckpt["hyper_parameters"])
    mp.load_state_dict(ckpt["state_dict"])
    return mp


def _build_datapoints(
    smiles: list[str], targets_2d: np.ndarray | None
) -> list[data.MoleculeDatapoint]:
    """Build MoleculeDatapoints with optional multitask targets (NaN allowed)."""
    if targets_2d is not None:
        return [data.MoleculeDatapoint.from_smi(smi, list(row)) for smi, row in zip(smiles, targets_2d)]
    return [data.MoleculeDatapoint.from_smi(smi) for smi in smiles]


def _train_fold(
    args: argparse.Namespace,
    train_df: "pd.DataFrame",
    val_df: "pd.DataFrame",
    tasks: list[str],
    fold_idx: int,
) -> tuple[np.ndarray, list[float], list[float], str]:
    """Train one fold; return (val_preds_task0, train_losses, val_losses, best_ckpt_path)."""
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    n_tasks = len(tasks)

    train_smiles = train_df["SMILES"].tolist()
    val_smiles = val_df["SMILES"].tolist()

    train_targets = train_df[tasks].to_numpy(dtype=float)  # (n_train, n_tasks)
    val_targets = val_df[[t for t in tasks if t in val_df.columns]].to_numpy(dtype=float)

    # Pad val_targets with NaN columns for tasks not in val (e.g. counter-assay on test-unblinded)
    if val_targets.shape[1] < n_tasks:
        pad = np.full((len(val_smiles), n_tasks - val_targets.shape[1]), np.nan)
        val_targets = np.hstack([val_targets, pad])

    train_datapoints = _build_datapoints(train_smiles, train_targets)
    val_datapoints = _build_datapoints(val_smiles, val_targets)

    train_dset = data.MoleculeDataset(train_datapoints, featurizer)
    scaler = train_dset.normalize_targets()

    val_dset = data.MoleculeDataset(val_datapoints, featurizer)
    val_dset.normalize_targets(scaler)

    train_loader = data.build_dataloader(train_dset, batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader = data.build_dataloader(
        val_dset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
    )

    mp = _load_mp()
    if args.freeze_mp:
        mp.requires_grad_(False)
    agg = nn.MeanAggregation()
    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
    ffn = nn.RegressionFFN(
        n_tasks=n_tasks,
        input_dim=mp.output_dim,
        dropout=args.dropout,
        output_transform=output_transform,
        criterion=nn.MAE(),
    )
    metric_list = [nn.metrics.RMSE(), nn.metrics.MAE()]
    mpnn = models.MPNN(
        mp, agg, ffn,
        batch_norm=False,
        metrics=metric_list,
        warmup_epochs=args.warmup_epochs,
        init_lr=args.init_lr,
        max_lr=args.lr,
        final_lr=args.final_lr,
    )

    ckpt_dir = args.checkpoints / f"fold{fold_idx}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpointing = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best-{epoch}-{val_loss:.3f}",
        monitor="val_loss",
        mode="min",
        save_last=True,
    )
    loss_logger = _LossLogger()
    callbacks = [checkpointing, loss_logger]
    if args.early_stopping > 0:
        callbacks.append(EarlyStopping(monitor="val_loss", patience=args.early_stopping, mode="min"))
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=True,
        accelerator="auto",
        devices=1,
        max_epochs=args.epochs,
        callbacks=callbacks,
    )

    print(f"\n[Fold {fold_idx}] Training for {args.epochs} epochs ...")
    trainer.fit(mpnn, train_loader, val_loader)
    print(f"[Fold {fold_idx}] Best checkpoint: {checkpointing.best_model_path}")

    return loss_logger.train_losses, loss_logger.val_losses, checkpointing.best_model_path


def _predict(best_ckpt: str, test_smiles: list[str], args: argparse.Namespace) -> np.ndarray:
    """Load best checkpoint, run inference on test SMILES, return task-0 predictions."""
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    best_mpnn = models.MPNN.load_from_checkpoint(best_ckpt)
    best_mpnn.eval()

    test_datapoints = _build_datapoints(test_smiles, None)
    test_dset = data.MoleculeDataset(test_datapoints, featurizer)
    test_loader = data.build_dataloader(
        test_dset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
    )

    trainer = pl.Trainer(logger=False, enable_progress_bar=False, accelerator="auto", devices=1)
    preds_list = trainer.predict(best_mpnn, test_loader)
    preds = np.concatenate([p.numpy() for p in preds_list])
    if preds.ndim == 2:
        preds = preds[:, 0]  # task 0 = primary pEC50
    return preds.flatten()


def main() -> None:
    parser = argparse.ArgumentParser(description="Finetune CheMeleon on PXR data.")
    # Data
    parser.add_argument("--test", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--split-type",
        choices=["random", "scaffold", "butina", "unblinded"],
        default="butina",
        help="Train/val split strategy (default: butina cluster split)",
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--butina-cutoff", type=float, default=0.4, help="Tanimoto distance cutoff for Butina split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-counter-assay", action="store_true", help="Add counter-assay pEC50 as task 2")
    parser.add_argument(
        "--include-single-concentration",
        action="store_true",
        help="Add aggregated single-concentration log2FC as task 3",
    )
    # Output
    parser.add_argument("--out", type=Path, default=Path("results/chemeleon_finetuned_submission.csv"))
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints/chemeleon"))
    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-4, help="Peak learning rate (max_lr)")
    parser.add_argument("--init-lr", type=float, default=1e-4, help="Starting LR before warmup")
    parser.add_argument("--final-lr", type=float, default=1e-4, help="Final LR after exponential decay")
    parser.add_argument("--warmup-epochs", type=int, default=2, help="Epochs for linear LR warmup")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate on the FFN head")
    parser.add_argument(
        "--freeze-mp",
        action="store_true",
        help="Freeze the message-passing backbone; only train the FFN head (reduces overfitting on small data)",
    )
    parser.add_argument("--n-folds", type=int, default=1, help="Number of seeds to train and average over")
    parser.add_argument("--early-stopping", type=int, default=10,
                        help="Stop after this many epochs with no val_loss improvement (0 to disable)")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoints.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(args.test)
    test_smiles = test_df["SMILES"].tolist()

    # Determine task columns
    tasks = ["pEC50"]
    if args.include_counter_assay:
        tasks.append("pEC50_counter")
    if args.include_single_concentration:
        tasks.append("log2_fc_single")
    print(f"Tasks: {tasks}")

    all_fold_preds: list[np.ndarray] = []
    all_train_losses: list[list[float]] = []
    all_val_losses: list[list[float]] = []
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

        print(f"\n[Fold {fold_idx}] Train: {len(train_pdf)} | Val: {len(val_pdf)} | Test: {len(test_smiles)}")
        if args.include_counter_assay:
            n_ca = train_pdf["pEC50_counter"].notna().sum()
            print(f"[Fold {fold_idx}] Counter-assay labels: {n_ca}/{len(train_pdf)}")
        if args.include_single_concentration:
            n_sc = train_pdf["log2_fc_single"].notna().sum()
            print(f"[Fold {fold_idx}] Single-conc labels: {n_sc}/{len(train_pdf)}")

        if pec50_range is None:
            pec50_range = (float(train_pdf["pEC50"].min()), float(train_pdf["pEC50"].max()))

        train_losses, val_losses, best_ckpt = _train_fold(args, train_pdf, val_pdf, tasks, fold_idx)
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)

        fold_preds = _predict(best_ckpt, test_smiles, args)
        all_fold_preds.append(fold_preds)
        print(f"[Fold {fold_idx}] Test pEC50 mean={fold_preds.mean():.3f} std={fold_preds.std():.3f}")

    # Average across folds and clip to training range
    pec50_preds = np.mean(all_fold_preds, axis=0)
    pec50_min, pec50_max = pec50_range  # type: ignore[misc]
    pec50_preds = np.clip(pec50_preds, pec50_min, pec50_max)

    # Loss curve (averaged across folds)
    fig, ax = plt.subplots()
    max_epochs = max(len(t) for t in all_train_losses)
    epochs = range(1, max_epochs + 1)
    mean_train = np.mean([np.pad(t, (0, max_epochs - len(t)), constant_values=np.nan) for t in all_train_losses], axis=0)
    mean_val = np.mean([np.pad(v, (0, max_epochs - len(v)), constant_values=np.nan) for v in all_val_losses], axis=0)
    ax.plot(epochs, mean_train, label="train (mean)")
    ax.plot(epochs, mean_val, label="val (mean)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MAE)")
    ax.set_title(f"CheMeleon finetune — {args.n_folds} fold(s), {args.split_type} split")
    ax.legend()
    fig.tight_layout()
    plot_path = args.out.parent / (args.out.stem + "_loss.png")
    fig.savefig(plot_path, dpi=150)
    print(f"\nLoss curve saved -> {plot_path}")

    submission = pd.DataFrame({
        "Molecule Name": test_df["Molecule Name"],
        "SMILES": test_df["SMILES"],
        "pEC50": pec50_preds,
    })
    submission.to_csv(args.out, index=False)

    stats = submission["pEC50"].describe()
    print(f"pEC50  mean={stats['mean']:.3f}  std={stats['std']:.3f}  "
          f"min={stats['min']:.3f}  max={stats['max']:.3f}")
    print(f"Submission saved -> {args.out}")


if __name__ == "__main__":
    main()
