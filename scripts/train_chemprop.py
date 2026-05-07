"""Train a vanilla chemprop MPNN from scratch on PXR training data.

Saves the best checkpoint to --checkpoints dir and copies it to
~/.chemprop/chemprop_pxr.ckpt so it is picked up by ChempropFingerprint.

Run download-data first to produce data/train_split.csv and data/val_split.csv.

Usage:
    python scripts/train_chemprop.py
    python scripts/train_chemprop.py --train data/train_split.csv --val data/val_split.csv --test data/test.csv
    python scripts/train_chemprop.py --epochs 50 --batch-size 64
"""

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from chemprop import data, featurizers, nn
from chemprop import models
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from tqdm import tqdm

_DEFAULT_CKPT_DEST = Path.home() / ".chemprop" / "chemprop_pxr.ckpt"


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


def _build_datapoints(smiles: list[str], targets: list[float] | None) -> list[data.MoleculeDatapoint]:
    if targets is not None:
        return [data.MoleculeDatapoint.from_smi(smi, [y]) for smi, y in zip(smiles, targets)]
    return [data.MoleculeDatapoint.from_smi(smi) for smi in smiles]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a vanilla chemprop MPNN on PXR data.")
    parser.add_argument("--train", type=Path, default=Path("data/train_split.csv"))
    parser.add_argument("--val", type=Path, default=Path("data/val_split.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/chemprop_submission.csv"))
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints/chemprop"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate")
    parser.add_argument("--init-lr", type=float, default=1e-4, help="Starting LR before warmup")
    parser.add_argument("--final-lr", type=float, default=1e-4, help="Final LR after decay")
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=300, help="MPNN hidden dimension")
    parser.add_argument("--depth", type=int, default=3, help="Number of message-passing steps")
    parser.add_argument(
        "--no-copy-ckpt",
        action="store_true",
        help=f"Skip copying best checkpoint to {_DEFAULT_CKPT_DEST}",
    )
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoints.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train)
    val_df = pd.read_csv(args.val)
    test_df = pd.read_csv(args.test)

    train_smiles = train_df["SMILES"].tolist()
    train_targets = train_df["pEC50"].tolist()
    val_smiles_list = val_df["SMILES"].tolist()
    val_targets = val_df["pEC50"].tolist()
    test_smiles = test_df["SMILES"].tolist()

    print(f"Train: {len(train_smiles)} | Val: {len(val_smiles_list)} | Test: {len(test_smiles)} molecules")

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    train_datapoints = _build_datapoints(train_smiles, train_targets)
    val_datapoints = _build_datapoints(val_smiles_list, val_targets)

    train_dset = data.MoleculeDataset(train_datapoints, featurizer)
    scaler = train_dset.normalize_targets()

    val_dset = data.MoleculeDataset(val_datapoints, featurizer)
    val_dset.normalize_targets(scaler)

    train_loader = data.build_dataloader(train_dset, batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader = data.build_dataloader(
        val_dset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
    )

    mp = nn.BondMessagePassing(d_h=args.hidden_size, depth=args.depth)
    agg = nn.MeanAggregation()
    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
    ffn = nn.RegressionFFN(output_transform=output_transform, input_dim=mp.output_dim)
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

    checkpointing = ModelCheckpoint(
        dirpath=args.checkpoints,
        filename="best-{epoch}-{val_loss:.3f}",
        monitor="val_loss",
        mode="min",
        save_last=True,
    )
    loss_logger = _LossLogger()
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=True,
        accelerator="auto",
        devices=1,
        max_epochs=args.epochs,
        callbacks=[checkpointing, loss_logger],
    )

    print(f"\nTraining for {args.epochs} epochs ...")
    trainer.fit(mpnn, train_loader, val_loader)
    best_path = Path(checkpointing.best_model_path)
    print(f"Best checkpoint: {best_path}")

    epochs = range(1, len(loss_logger.train_losses) + 1)
    fig, ax = plt.subplots()
    ax.plot(epochs, loss_logger.train_losses, label="train")
    ax.plot(epochs, loss_logger.val_losses, label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Chemprop — train vs val loss")
    ax.legend()
    fig.tight_layout()
    plot_path = args.out.parent / (args.out.stem + "_loss.png")
    fig.savefig(plot_path, dpi=150)
    print(f"Loss curve saved -> {plot_path}")

    if not args.no_copy_ckpt:
        _DEFAULT_CKPT_DEST.parent.mkdir(exist_ok=True)
        shutil.copy2(best_path, _DEFAULT_CKPT_DEST)
        print(f"Copied checkpoint -> {_DEFAULT_CKPT_DEST}")

    best_mpnn = models.MPNN.load_from_checkpoint(str(best_path))
    best_mpnn.eval()

    test_datapoints = _build_datapoints(test_smiles, None)
    test_dset = data.MoleculeDataset(test_datapoints, featurizer)
    test_loader = data.build_dataloader(
        test_dset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
    )

    preds = trainer.predict(best_mpnn, test_loader)
    pec50_preds = np.concatenate([p.numpy() for p in preds]).flatten()

    submission = pd.DataFrame({
        "Molecule Name": test_df["Molecule Name"],
        "SMILES": test_df["SMILES"],
        "pEC50": pec50_preds,
    })

    stats = submission["pEC50"].describe()
    print(f"pEC50  mean={stats['mean']:.3f}  std={stats['std']:.3f}  "
          f"min={stats['min']:.3f}  max={stats['max']:.3f}")


if __name__ == "__main__":
    main()
