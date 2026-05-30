# openadmet-pxr

ML pipeline for predicting Pregnane X Receptor (PXR) activity (pEC50) from molecular SMILES, built for the [OpenADMET PXR challenge](https://huggingface.co/datasets/openadmet/pxr-challenge-train-test).

4,139 training compounds, 513 blinded test compounds, prospective (temporal) split.

## Installation

```bash
git clone --recurse-submodules <repo>
pip install -e .
```

GPU acceleration (optional): uncomment `cuml-cu12` in `requirements.txt` before installing.

MolE representation: requires a separate `mole` conda environment and the `third_party/mole` submodule (see [Representations](#representations)).

## Pipeline

Run these five steps in order:

```bash
# 1. Fetch data from HuggingFace; write 90/10 train/val split (seed 42)
download-data

# 2. Fit each model on train features, score on val, print RMSE/MAE/R² table
evaluate-models --models all --input morgan
evaluate-models --preconfigs all              # for PreConfig ensemble models

# 3. Cross-tabulate all input representations × all models into a 2-D MAE table
score-table

# 4. Train on full train.csv, write results/<model>_<input>_submission.csv + distribution PNG
generate-results --model xgboost --input morgan
generate-results --preconfig OneofEachDim     # for PreConfig ensemble models

# 5. Validate submission format (513 rows, required columns, no NaNs/duplicates)
validate-results results/xgboost_morgan_submission.csv
```

Other useful commands:

```bash
# UMAP plot of train/test chemical space
visualize-space --input morgan

# Restrict training set to the top-30% Tanimoto-similar molecules to the test set
evaluate-models --models xgboost --input morgan --filter tanimoto_top30
generate-results --model xgboost --input morgan --filter tanimoto_top30
```

Multiple representations can be stacked: `--input morgan rdkit`.

## Models

Pass `--model <name>` / `--models <name> [name ...]` to `evaluate-models` and `generate-results`.

| Name | Description |
|---|---|
| `mean_baseline` | Predict the training mean |
| `median_baseline` | Predict the training median |
| `knn` | scikit-learn KNN regressor |
| `decision_tree` | scikit-learn DecisionTreeRegressor |
| `linear_regression` | Ridge regression with StandardScaler |
| `mlp` | 3-layer PyTorch net (256-128-64), BatchNorm, Dropout, early stopping |
| `symbolic_regression` | Genetic programming via PySR (requires Julia) |
| `tabicl` | TabICL in-context learner; PCA-200 preprocessing |
| `tabpfn` | TabPFN in-context learner; PCA-200 preprocessing |
| `kan` | Kolmogorov-Arnold Network via pykan; auto-PCA to 128 input dims |
| `xgboost` | XGBRegressor (100 trees, depth 6, lr 0.1) |

**PreConfig models** bundle a fixed set of representations; pass `--preconfig <name>` / `--preconfigs all`.

| Name | Description |
|---|---|
| `OneofEachDim` | Ensemble of XGBoost + 3×TabICL across rdkit / chemprop\_finetuned / chemeleon\_finetuned / unimol |

To add a new model: subclass `PXRModel` in `src/models/`, set a unique `name` class variable, add it to `REGISTRY` in `src/models/__init__.py`. For a model with a fixed representation, subclass `PXRPreConfigModel` and add to `PRECONFIG_REGISTRY`.

## Representations

Pass `--input <name> [name ...]` to `evaluate-models`, `generate-results`, and `score-table`. Feature matrices are cached as `.npy` files under `data/features/` keyed by a SHA-256 of the SMILES list.

| Name | Description |
|---|---|
| `morgan` | Binary ECFP4 Morgan fingerprints, 2048 bits (RDKit) |
| `rdkit` | 217 RDKit 2D physicochemical descriptors |
| `chemeleon` | Pretrained CheMeleon MPNN fingerprints; weights auto-downloaded from Zenodo on first use |
| `chemeleon_finetuned` | CheMeleon fine-tuned on PXR labels; run `scripts/finetune_chemeleon.py` first |
| `chemprop_finetuned` | chemprop MPNN trained on PXR data; run `scripts/train_chemprop.py` first |
| `unimol` | UniMol 3D CLS-token embeddings, 512-dim; ~500 MB weights auto-downloaded to `~/.unimol/` on first use |
| `mole` | MolE graph-transformer embeddings; requires a separate `mole` conda env and `third_party/mole` submodule |

To add a new representation: subclass `Representation` in `src/representations/`, instantiate it and add a featurize function to `INPUT_REGISTRY` in `src/data_tools/inputs.py`.

## Dataset

- **Source**: `hf://datasets/openadmet/pxr-challenge-train-test/`
- **Files**: `train.csv` (4,139 compounds), `test.csv` (513 compounds, blinded), `counter_train.csv`
- **Target**: `pEC50` — continuous measure of PXR activation potency (–log₁₀ EC50)
- **Split type**: prospective/temporal — 99.4% of test OADMET registration IDs exceed the training maximum; 81% of test Murcko scaffolds are unseen in training. Models must generalise across chemical space, not merely interpolate within known scaffolds.
- **Internal validation**: random 90/10 split (seed 42); note this is optimistic relative to the prospective test split.

## Fine-tuning scripts

These must be run before using `chemeleon_finetuned` or `chemprop_finetuned` representations.

```bash
# Fine-tune pretrained CheMeleon on PXR labels; saves to checkpoints/chemeleon/
python scripts/finetune_chemeleon.py [--epochs N] [--batch-size N]

# Train chemprop MPNN from scratch on PXR data; saves to checkpoints/chemprop/
python scripts/train_chemprop.py [--epochs N]
```

## Project structure

```
src/
  data_tools/         # download, load, featurize, validate, visualize
  models/             # REGISTRY of PXRModel subclasses + CLI tools
  representations/    # REGISTRY of Representation subclasses
scripts/
  finetune_chemeleon.py
  train_chemprop.py
  eda_*.py
third_party/mole/     # git submodule: MolE graph-transformer
checkpoints/          # fine-tuned weights (chemeleon/, chemprop/)
data/                 # created by download-data (gitignored)
results/              # submission CSVs and prediction distribution PNGs
viz/                  # UMAP and EDA plots
```

## Development

```bash
pip install -e ".[test]"
pytest                      # unit tests
pytest -m integration       # integration tests (requires data/)
black src/ tests/
flake8 src/ tests/
pyright
```
