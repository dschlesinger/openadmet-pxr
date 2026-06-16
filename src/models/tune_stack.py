"""CLI for Optuna tuning of a level-1 model on cross-fitted stacked-ensemble base predictions.

This mirrors ``tune-model`` (:mod:`models.tune`) but, instead of featurizing the
molecules directly, builds the feature matrix from the StackedEnsemble base
learners' *cross-fitted* predictions:

  for each base learner j (DEFAULT_ROSTER) and each Butina fold f, fit base_j on
  the fold-f training portion of the train pool and predict on **all rows**
  (train pool + val). Stacking gives an (n, n_base * n_splits) matrix — 7*5 = 35
  columns by default.

The val rows are never used to fit any base learner, so the val metric Optuna
optimizes is honest. The generated prediction matrices are cached to an ``.npz``
keyed by split config and reused on later runs unless ``--refresh`` is passed,
decoupling the expensive base-learner fits from cheap re-tuning of level-1 models.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import numpy as np
import optuna
import polars as pl
from optuna.trial import FixedTrial

from data_tools.inputs import featurize
from data_tools.load import butina_kfold_indices, load_data, load_test
from models import REGISTRY
from models.evaluate import _compute_metrics, _evaluate_model
from models.optimize import SEARCH_SPACES
from models.stacked_ensemble import DEFAULT_ROSTER, _impute

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _make_callback(metric: str) -> Callable:
    def _cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        print(f"  Trial {trial.number:4d}  {metric}={trial.value:.4f}  best={study.best_value:.4f}", flush=True)

    return _cb


def _build_predictions(  # pylint: disable=too-many-locals
    train_df,
    eval_dfs: dict,
    target: str,
    n_splits: int,
    butina_cutoff: float,
    seed: int,
    cache_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, dict, list[str]]:
    """Cross-fit every base learner over Butina folds; predict on train + each eval frame.

    Returns (X_train, y_train, eval_mats, col_labels) where eval_mats maps each key
    in eval_dfs to its (n_rows, n_base * n_splits) cross-fitted prediction matrix.
    """
    y_train = train_df[target].to_numpy().astype(np.float64)
    n_train = len(train_df)
    n_base = len(DEFAULT_ROSTER)

    # Featurize each base learner's representation block once on train + every eval frame,
    # then impute with per-column medians from the train pool (matches StackedEnsemble.fit).
    mats_tr: list[np.ndarray] = []
    mats_eval: dict[str, list[np.ndarray]] = {k: [] for k in eval_dfs}
    for label, _, reps in DEFAULT_ROSTER:
        raw_tr = featurize(train_df, reps, cache_dir).astype(np.float64)
        medians = np.nan_to_num(np.nanmedian(np.where(np.isfinite(raw_tr), raw_tr, np.nan), axis=0))
        mats_tr.append(_impute(raw_tr, medians))
        for key, df in eval_dfs.items():
            mats_eval[key].append(_impute(featurize(df, reps, cache_dir).astype(np.float64), medians))
        print(f"[tune_stack] featurized {label} ({'+'.join(reps)})", flush=True)

    folds = butina_kfold_indices(train_df, n_splits=n_splits, cutoff=butina_cutoff, seed=seed)

    p_train = np.zeros((n_train, n_base, n_splits), dtype=np.float64)
    p_eval = {k: np.zeros((len(df), n_base, n_splits), dtype=np.float64) for k, df in eval_dfs.items()}
    all_idx = np.arange(n_train)
    for j, (label, factory, _) in enumerate(DEFAULT_ROSTER):
        for f, val_idx in enumerate(folds):
            tr_idx = np.setdiff1d(all_idx, np.asarray(val_idx, dtype=int), assume_unique=False)
            model = factory()
            model.fit(mats_tr[j][tr_idx], y_train[tr_idx])
            p_train[:, j, f] = model.predict(mats_tr[j])
            for key in eval_dfs:
                p_eval[key][:, j, f] = model.predict(mats_eval[key][j])
            print(f"[tune_stack] {label} fold {f + 1}/{n_splits} done", flush=True)

    x_train = p_train.reshape(n_train, n_base * n_splits)
    eval_mats = {k: p.reshape(len(eval_dfs[k]), n_base * n_splits) for k, p in p_eval.items()}
    col_labels = [f"{label}_f{f}" for label, _, _ in DEFAULT_ROSTER for f in range(n_splits)]
    return x_train, y_train, eval_mats, col_labels


def _drop_nan_target(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Drop rows with a non-finite target (predictions themselves are finite)."""
    mask = np.isfinite(y)
    dropped = int((~mask).sum())
    if dropped:
        print(f"Dropped {dropped} of {len(y)} rows with non-finite target")
    return x[mask], y[mask]


def _holdout_mask(n: int, frac: float, seed: int) -> np.ndarray:
    """Seeded boolean mask marking the held-out final rows (all False when frac <= 0)."""
    mask = np.zeros(n, dtype=bool)
    n_final = int(round(n * frac))
    if n_final > 0:
        mask[np.random.default_rng(seed).permutation(n)[:n_final]] = True
    return mask


def _metrics(model, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Metrics for an already-fitted model on (x, y)."""
    preds = pl.Series("prediction", model.predict(x).tolist())
    return _compute_metrics(pl.Series("actual", y.tolist()), preds)


def _rmse(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def _permutation_importance(
    model, x: np.ndarray, y: np.ndarray, seed: int, n_repeats: int
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Model-agnostic permutation importance: mean RMSE increase when each column is shuffled.

    Returns (baseline_rmse, mean_increase[n_features], std_increase[n_features]).
    Works for any level-1 model since it only calls ``model.predict``.
    """
    rng = np.random.default_rng(seed)
    baseline = _rmse(model.predict(x), y)
    n_features = x.shape[1]
    deltas = np.zeros((n_features, n_repeats), dtype=np.float64)
    for j in range(n_features):
        col = x[:, j].copy()
        for r in range(n_repeats):
            x[:, j] = rng.permutation(col)
            deltas[j, r] = _rmse(model.predict(x), y) - baseline
        x[:, j] = col
    return baseline, deltas.mean(axis=1), deltas.std(axis=1)


def _native_importance(model, n_features: int) -> np.ndarray | None:
    """Native importances (tree gains or |coef|) when they align with the input columns.

    Returns ``None`` for models whose pipeline transforms the feature space (e.g. PCA),
    so the values would not map back onto the base-prediction columns.
    """
    est = getattr(model, "_model", model)
    if hasattr(est, "steps"):  # sklearn Pipeline — importances live on the final estimator
        est = est.steps[-1][1]
    imp = getattr(est, "feature_importances_", None)
    if imp is None:
        coef = getattr(est, "coef_", None)
        imp = None if coef is None else np.abs(np.asarray(coef, dtype=np.float64).ravel())
    if imp is None or np.asarray(imp).shape != (n_features,):
        return None
    return np.asarray(imp, dtype=np.float64)


def _report_importance(  # pylint: disable=too-many-locals
    model, x: np.ndarray, y: np.ndarray, col_labels: list[str], seed: int, n_repeats: int
) -> Dict[str, Any]:
    """Print and return per-feature and per-base-learner importances for the fitted level-1 model."""
    baseline, perm_mean, perm_std = _permutation_importance(model, x, y, seed, n_repeats)
    native = _native_importance(model, x.shape[1])

    order = np.argsort(perm_mean)[::-1]
    has_native = native is not None
    print(f"\nFeature importance (permutation, {n_repeats} repeats, baseline RMSE={baseline:.4f}):")
    header = f"  {'feature':<22}{'perm ΔRMSE':>14}{'± std':>10}"
    if has_native:
        header += f"{'native':>12}"
    print(header)
    for j in order:
        line = f"  {col_labels[j]:<22}{perm_mean[j]:>14.4f}{perm_std[j]:>10.4f}"
        if has_native:
            line += f"{native[j]:>12.4f}"
        print(line)

    # Aggregate across folds: collapse "<label>_f<k>" columns into one row per base learner.
    bases = [lbl.rsplit("_f", 1)[0] for lbl in col_labels]
    per_base: Dict[str, float] = {}
    per_base_native: Dict[str, float] = {}
    for b, pm in zip(bases, perm_mean):
        per_base[b] = per_base.get(b, 0.0) + float(pm)
    if has_native:
        for b, nv in zip(bases, native):
            per_base_native[b] = per_base_native.get(b, 0.0) + float(nv)
    print("\nPer-base-learner importance (summed over folds):")
    for b in sorted(per_base, key=per_base.get, reverse=True):
        line = f"  {b:<22}{per_base[b]:>14.4f}"
        if has_native:
            line += f"{per_base_native[b]:>12.4f}"
        print(line)

    return {
        "baseline_rmse": baseline,
        "permutation": {lbl: float(perm_mean[j]) for j, lbl in enumerate(col_labels)},
        "permutation_std": {lbl: float(perm_std[j]) for j, lbl in enumerate(col_labels)},
        "native": None if not has_native else {lbl: float(native[j]) for j, lbl in enumerate(col_labels)},
        "per_base_learner": per_base,
    }


def main() -> None:  # pylint: disable=too-many-locals,too-many-statements,too-many-branches
    """Entry point for the tune-stack CLI."""
    parser = argparse.ArgumentParser(
        description="Tune a level-1 model with Optuna on cross-fitted stacked-ensemble base predictions."
    )
    parser.add_argument("--model", required=True, choices=list(SEARCH_SPACES.keys()), help="Level-1 model to tune")
    parser.add_argument("--n-trials", type=int, default=100, help="Number of Optuna trials (default: 100)")
    parser.add_argument("--timeout", type=float, default=None, help="Stop after N seconds regardless of trial count")
    parser.add_argument("--metric", default="RMSE", choices=["RMSE", "MAE", "R2"], help="Metric to optimize")
    parser.add_argument("--output", type=Path, default=None, help="Write best params + metrics to JSON")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing data CSVs")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/features"), help="Feature cache directory")
    parser.add_argument("--val-split", type=float, default=0.2, help="Val fraction for scaffold/butina split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampler, data split, and folds")
    parser.add_argument("--target", default="pEC50", help="Target column")
    parser.add_argument("--n-splits", type=int, default=5, help="Butina folds for the cross-fit (default: 5)")
    parser.add_argument(
        "--final-frac",
        type=float,
        default=0.2,
        help="Fraction of the val/test set held out (untuned) for final validation of the best model (default: 0.2).",
    )
    parser.add_argument(
        "--scaffold-split",
        action="store_true",
        help="Use scaffold-based train/val split (no scaffold leakage).",
    )
    parser.add_argument(
        "--butina-split",
        action="store_true",
        help="Use Butina cluster split on Morgan fps (no similar molecules span both splits).",
    )
    parser.add_argument(
        "--butina-cutoff",
        type=float,
        default=0.4,
        help="Tanimoto distance cutoff for Butina clustering / folds (default: 0.4).",
    )
    parser.add_argument(
        "--include-unblinded",
        action="store_true",
        help="Add phase-1 unblinded test molecules to the training pool.",
    )
    parser.add_argument(
        "--include-counter-assay",
        action="store_true",
        help="Add pEC50_counter column from counter-assay data.",
    )
    parser.add_argument(
        "--include-single-conc",
        action="store_true",
        help="Add log2_fc_single column from single-concentration screen.",
    )
    parser.add_argument(
        "--pred-cache", type=Path, default=None, help="Override path for the cached prediction matrices"
    )
    parser.add_argument("--refresh", action="store_true", help="Regenerate predictions even if a cache exists")
    parser.add_argument(
        "--importance-repeats",
        type=int,
        default=10,
        help="Permutation-importance shuffle repeats per feature; 0 disables the importance report (default: 10).",
    )
    parser.add_argument(
        "--pred-csv",
        type=Path,
        default=None,
        help="Where to write the best model's predictions on the val/unblinded set "
        "(default: results/tune_stack_<model>_<split>_preds.csv).",
    )
    parser.add_argument(
        "--blind-csv",
        type=Path,
        default=None,
        help="Where to write the best model's predictions on the blinded test.csv submission set "
        "(default: results/tune_stack_<model>_blinded_preds.csv).",
    )
    args = parser.parse_args()

    if (args.scaffold_split or args.butina_split) and not 0.0 < args.val_split < 1.0:
        print(f"--val-split must be in (0, 1), got {args.val_split}", file=sys.stderr)
        sys.exit(1)

    if not 0.0 <= args.final_frac < 1.0:
        print(f"--final-frac must be in [0, 1), got {args.final_frac}", file=sys.stderr)
        sys.exit(1)

    if args.scaffold_split:
        split_type = "scaffold"
    elif args.butina_split:
        split_type = "butina"
    else:
        split_type = "unblinded"

    cache_path = args.pred_cache or Path(
        f"results/stack_xfold_{split_type}_vs{args.val_split}_seed{args.seed}_ns{args.n_splits}.npz"
    )

    if cache_path.exists() and not args.refresh:
        data = np.load(cache_path, allow_pickle=True)
        x_train, x_val = data["X_train"], data["X_val"]
        y_train, y_val = data["y_train"], data["y_val"]
        col_labels = list(data["col_labels"])
        val_names = (
            data["val_names"] if "val_names" in data.files else np.array([f"row_{i}" for i in range(len(y_val))])
        )
        val_smiles = data["val_smiles"] if "val_smiles" in data.files else np.array([""] * len(y_val))
        print(f"[tune_stack] loaded cached predictions from {cache_path} (use --refresh to regenerate)")
    else:
        train_df, val_df = load_data(
            include_unblinded=args.include_unblinded,
            include_counter_assay=args.include_counter_assay,
            include_single_concentration=args.include_single_conc,
            split_type=split_type,
            val_fraction=args.val_split,
            seed=args.seed,
            butina_cutoff=args.butina_cutoff,
            data_dir=args.data_dir,
        )
        print(
            f"load_data(split_type={split_type!r}): {len(val_df)} val / {len(train_df)} train molecules",
            file=sys.stderr,
        )
        if args.target not in train_df.columns:
            print(f"Target '{args.target}' not found. Columns: {list(train_df.columns)}", file=sys.stderr)
            sys.exit(1)

        x_train, x_val, y_train, y_val, col_labels = _build_predictions(
            train_df, val_df, args.target, args.n_splits, args.butina_cutoff, args.seed, args.cache_dir
        )
        val_names = val_df["Molecule Name"].to_numpy()
        val_smiles = val_df["SMILES"].to_numpy()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            X_train=x_train,
            X_val=x_val,
            y_train=y_train,
            y_val=y_val,
            col_labels=np.array(col_labels),
            val_names=val_names,
            val_smiles=val_smiles,
        )
        print(f"[tune_stack] saved predictions to {cache_path}")

    x_train, y_train = _drop_nan_target(x_train, y_train)
    # Keep the test/val identifiers aligned with rows through the NaN-target drop.
    val_finite = np.isfinite(y_val)
    x_val, y_val = x_val[val_finite], y_val[val_finite]
    val_names, val_smiles = np.asarray(val_names)[val_finite], np.asarray(val_smiles)[val_finite]

    # Hold out a fraction of the val/test set, untouched by Optuna, for an honest
    # final estimate of the tuned model (the rest is the optimization val set).
    is_final = _holdout_mask(len(y_val), args.final_frac, args.seed)
    x_opt, y_opt = x_val[~is_final], y_val[~is_final]
    x_final, y_final = x_val[is_final], y_val[is_final]
    print(
        f"Val/test split: {len(y_opt)} optimization rows / {len(y_final)} held-out final rows"
        f" (final_frac={args.final_frac})"
    )

    model_cls = REGISTRY[args.model]
    direction = "maximize" if args.metric == "R2" else "minimize"
    study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=args.seed))

    def objective(trial: optuna.Trial) -> float:
        params: Dict[str, Any] = SEARCH_SPACES[args.model](trial)
        metrics = _evaluate_model(model_cls(**params), x_train, y_train, x_opt, y_opt)
        return float(metrics[args.metric])

    print(
        f"Tuning {args.model}  trials={args.n_trials}  metric={args.metric}"
        f"  n_features={x_train.shape[1]}  n_train={len(x_train)}  n_opt={len(x_opt)}"
    )
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, callbacks=[_make_callback(args.metric)])

    best_params = study.best_params
    best_value = study.best_value
    best_model_params: Dict[str, Any] = SEARCH_SPACES[args.model](FixedTrial(best_params))

    # Refit the best model once, then score on both the optimization and held-out final sets.
    best_model = model_cls(**best_model_params)
    best_model.fit(x_train, y_train)
    opt_metrics = _metrics(best_model, x_opt, y_opt)
    final_metrics = _metrics(best_model, x_final, y_final) if len(y_final) else opt_metrics

    print(f"\nBest {args.metric}: {best_value:.4f}  (after {len(study.trials)} trials)")
    print("Best params:")
    for k, v in best_params.items():
        print(f"  {k}: {v:.6g}" if isinstance(v, float) else f"  {k}: {v}")
    print(
        f"\nOptimization val — RMSE={opt_metrics['RMSE']:.4f}"
        f"  MAE={opt_metrics['MAE']:.4f}  R2={opt_metrics['R2']:.4f}"
    )
    final_tag = "Held-out final" if len(y_final) else "Final (no holdout)"
    print(
        f"{final_tag} — RMSE={final_metrics['RMSE']:.4f}"
        f"  MAE={final_metrics['MAE']:.4f}  R2={final_metrics['R2']:.4f}"
    )

    # Feature importance for the fitted level-1 model, scored on the held-out final set
    # (untouched by Optuna) when available, else the optimization set.
    importance: Dict[str, Any] | None = None
    if args.importance_repeats > 0:
        imp_x, imp_y = (x_final, y_final) if len(y_final) else (x_opt, y_opt)
        importance = _report_importance(best_model, imp_x.copy(), imp_y, col_labels, args.seed, args.importance_repeats)

    # Save the best model's predictions on the entire test/val set.
    pred_csv = args.pred_csv or Path(f"results/tune_stack_{args.model}_{split_type}_preds.csv")
    pred_csv.parent.mkdir(parents=True, exist_ok=True)
    preds_all = best_model.predict(x_val)
    pl.DataFrame(
        {
            "Molecule Name": [str(v) for v in val_names],
            "SMILES": [str(v) for v in val_smiles],
            args.target: preds_all.tolist(),
            f"{args.target}_actual": y_val.tolist(),
            "subset": np.where(is_final, "held_out_final", "optimization").tolist(),
        }
    ).write_csv(pred_csv)
    print(f"Saved test-set predictions ({len(preds_all)} rows) to {pred_csv}")

    if args.output:
        result: Dict[str, Any] = {
            "model": args.model,
            "metric": args.metric,
            "best_value": best_value,
            "n_trials": len(study.trials),
            "params": best_params,
            "opt_metrics": opt_metrics,
            "final_metrics": final_metrics,
            "final_frac": args.final_frac,
            "n_held_out": int(len(y_final)),
            "n_features": int(x_train.shape[1]),
            "split_type": split_type,
            "importance": importance,
        }
        args.output.write_text(json.dumps(result, indent=2))
        print(f"Saved to {args.output}")
