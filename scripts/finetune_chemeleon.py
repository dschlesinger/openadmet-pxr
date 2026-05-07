"""Finetune the CheMeleon pretrained MPNN on PXR training data.

Downloads chemeleon_mp.pt from Zenodo (cached to ~/.chemprop/) on first run.
Trains a chemprop MPNN with the pretrained message-passing backbone end-to-end,
then generates pEC50 predictions on the test set.

Run download-data first to produce data/train_split.csv and data/val_split.csv.

Usage:
    python scripts/finetune_chemeleon.py
    python scripts/finetune_chemeleon.py --train data/train_split.csv --val data/val_split.csv --test data/test.csv
    python scripts/finetune_chemeleon.py --epochs 30 --batch-size 64
"""

import argparse
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
from lightning.pytorch.callbacks import ModelCheckpoint
from tqdm import tqdm

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


def _build_datapoints(smiles: list[str], targets: list[float] | None) -> list[data.MoleculeDatapoint]:
    if targets is not None:
        return [data.MoleculeDatapoint.from_smi(smi, [y]) for smi, y in zip(smiles, targets)]
    return [data.MoleculeDatapoint.from_smi(smi) for smi in smiles]


def main() -> None:
    parser = argparse.ArgumentParser(description="Finetune CheMeleon on PXR data.")
    parser.add_argument("--train", type=Path, default=Path("data/train_split.csv"))
    parser.add_argument("--val", type=Path, default=Path("data/val_split.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--out", type=Path, default=Path("viz/chemeleon_submission.csv"))
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints/chemeleon"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate (max_lr)")
    parser.add_argument("--init-lr", type=float, default=1e-4, help="Starting LR before warmup")
    parser.add_argument("--final-lr", type=float, default=1e-4, help="Final LR after exponential decay")
    parser.add_argument("--warmup-epochs", type=int, default=2, help="Epochs for linear LR warmup")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoints.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    train_df = pd.read_csv(args.train)
    val_df = pd.read_csv(args.val)
    test_df = pd.read_csv(args.test)

    train_smiles = train_df["SMILES"].tolist()
    train_targets = train_df["pEC50"].tolist()
    val_smiles_list = val_df["SMILES"].tolist()
    val_targets = val_df["pEC50"].tolist()
    test_smiles = test_df["SMILES"].tolist()

    print(f"Train: {len(train_smiles)} | Val: {len(val_smiles_list)} | Test: {len(test_smiles)} molecules")

    # -------------------------------------------------------------------------
    # Build datasets
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Build model
    # -------------------------------------------------------------------------
    mp = _load_mp()
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

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------
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
    print(f"Best checkpoint: {checkpointing.best_model_path}")

    epochs = range(1, len(loss_logger.train_losses) + 1)
    fig, ax = plt.subplots()
    ax.plot(epochs, loss_logger.train_losses, label="train")
    ax.plot(epochs, loss_logger.val_losses, label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("CheMeleon finetune — train vs val loss")
    ax.legend()
    fig.tight_layout()
    plot_path = args.out.parent / (args.out.stem + "_loss.png")
    fig.savefig(plot_path, dpi=150)
    print(f"Loss curve saved -> {plot_path}")

    # -------------------------------------------------------------------------
    # Predict on test set
    # -------------------------------------------------------------------------
    best_mpnn = models.MPNN.load_from_checkpoint(checkpointing.best_model_path)
    best_mpnn.eval()

    test_datapoints = _build_datapoints(test_smiles, None)
    test_dset = data.MoleculeDataset(test_datapoints, featurizer)
    test_loader = data.build_dataloader(
        test_dset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
    )

    preds = trainer.predict(best_mpnn, test_loader)
    pec50_preds = np.concatenate([p.numpy() for p in preds]).flatten()

    # -------------------------------------------------------------------------
    # Save submission
    # -------------------------------------------------------------------------
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
