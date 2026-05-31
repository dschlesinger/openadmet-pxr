# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install with dev dependencies
pip install -e ".[test]"

# Format
black src/ tests/

# Lint
flake8 src/ tests/
pylint src/

# Type check
pyright

# Run unit tests (default — excludes integration/spark)
pytest

# Run a single test
pytest tests/test_models.py::test_mean_baseline -v

# Run integration tests
pytest -m integration

# Run all tests via tox
tox          # unit only
tox -e integration
tox -e spark
tox -e all

# Data pipeline (run in order)
download-data                         # fetch from HuggingFace (train, blinded test, unblinded test, counter-assay), create train/val split
evaluate-models --models all          # eval on train_split/val_split, print RMSE/MAE/R2
score-table                           # cross-tabulate every input × every model
generate-results --model xgboost      # train on full train.csv, write results/<model>_submission.csv
validate-results <submission>.csv     # check submission format before uploading
```

## Architecture

This is the **OpenADMET PXR challenge** ML pipeline for predicting Pregnane X Receptor (PXR) activity (pEC50) from molecular SMILES. The codebase is a full experiment harness: download → featurize → train → evaluate → submit.

**Source layout** (`src/` on `PYTHONPATH` via pytest):

- `src/data_tools/` — data I/O layer
  - `download.py` — pulls train/test/counter-assay CSVs from `hf://datasets/openadmet/pxr-challenge-train-test/`, then writes a random train/val split (default 90/10, seed 42) to `data/train_split.csv` / `data/val_split.csv`
  - `load.py` — thin helpers (`load_train`, `load_test`, `load_val_split`, …) that read local CSVs into Polars DataFrames
  - `inputs.py` — `INPUT_REGISTRY` maps representation name → featurize function; `featurize()` handles disk caching of `.npy` arrays under `data/features/` keyed by SHA-256 of the SMILES list
  - `validate_submission.py` — checks a results CSV for required columns, row count (513), no NaNs/infs, no duplicate molecule names

- `src/models/` — model layer
  - `base.py` — `PXRModel` (ABC): subclasses must set `name: ClassVar[str]` and implement `fit(X, y)` / `predict(X)`. `PXRPreConfigModel` is a subclass for models that fix their own representation via `required_representations: ClassVar[list[str]]`.
  - `__init__.py` — `REGISTRY` and `PRECONFIG_REGISTRY` dicts mapping name → class; add new models here
  - Individual model files: `baseline.py`, `decision_tree.py`, `knn.py`, `linear_regression.py`, `mlp.py`, `symbolic_regression.py`, `tabicl_model.py`, `tabpfn_model.py`, `xgboost_model.py`, `one_each_dim.py`
  - `evaluate.py` — CLI (`evaluate-models`): fits each model on train features, scores on val, prints a metrics table
  - `predict.py` — CLI (`generate-results`): fits on full train, predicts test, writes `results/<model>_submission.csv` + a distribution plot PNG
  - `score_table.py` — CLI (`score-table`): cross-tabulates every input × every model

- `src/representations/` — molecular featurizers
  - `base.py` — `Representation` (ABC): must set `name` and implement `transform(smiles: pl.Series) -> np.ndarray`
  - Concrete implementations: `fingerprints.py` (Morgan), `rdkit_descriptors.py`, `chemeleon.py`, `chemprop_rep.py`, `unimol.py`, `mole.py`
  - New representations must be instantiated and registered in `src/data_tools/inputs.py`

**Extending the pipeline**:

- **New model**: subclass `PXRModel` in a new file under `src/models/`, set a unique `name`, add to `REGISTRY` in `src/models/__init__.py`.
- **New representation**: subclass `Representation` in `src/representations/`, add an instance + featurize function to `INPUT_REGISTRY` in `src/data_tools/inputs.py`.
- **PreConfig model** (fixed representation): subclass `PXRPreConfigModel`, set `required_representations`, add to `PRECONFIG_REGISTRY`.

**Data files** (not committed, created by `download-data`):
- `data/train.csv`, `data/test.csv`, `data/test_unblinded.csv`, `data/counter_train.csv`, `data/single_concentration_train.csv` — raw HuggingFace splits
  - `test.csv` = blinded submission target (no labels); `test_unblinded.csv` = phase-1 labels released (use for evaluation)
- `data/train_split.csv`, `data/val_split.csv` — canonical 90/10 split used by `evaluate-models`
- `data/features/` — cached `.npy` feature arrays

**Dependencies**: Runtime deps in `requirements.txt` (torch, rdkit, scikit-learn, xgboost, chemprop, tabicl, tabpfn, unimol_tools, etc.). The `cuml-cu12` GPU dep is commented out and not installed by default.

**Testing constraints**:
- Test markers: `unit`, `integration`, `spark`, `gpu`, `slow`, `notebooks`. Default run excludes `integration` and `spark`.
- Files with `_int_` in the name are auto-marked `integration`.
- Coverage threshold is currently set to 0 in `pyproject.toml` (`--cov-fail-under 0`).

**Code style**:
- Max line length: 120 characters (Black, Flake8, Pylint all agree).
- Pylint max function args: 5; max attributes per class: 7.
- Pyright type checking enabled — add type annotations to all public APIs.
