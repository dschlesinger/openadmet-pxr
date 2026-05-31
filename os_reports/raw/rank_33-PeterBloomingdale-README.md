# OpenADMET PXR Blind Challenge

Codebase for the [OpenADMET PXR Blind Challenge](https://huggingface.co/spaces/openadmet/pxr-challenge) — pEC50 regression on 513 Enamine analogs of 63 potent PXR training hits.

**Metric:** RAE = MAE / dynamic_range (lower is better) | **Phase 2 deadline:** July 1, 2026

---

## Results

| Submission | Description | MAE | RAE | Spearman | Rank |
|---|---|---|---|---|---|
| Sub 18 | LightGBM + Chemprop + TabPFN + TabICL + UniMol (target_std=0.70) | 0.4690 | 0.5887 | 0.7971 | 67 |
| Sub 19 | + TabICL-Chemprop embeddings (no diversity gain) | 0.4690 | 0.5888 | 0.7972 | 67 |
| **Sub 20** | **+ UniMol s3 (LR=2e-4) — highest diversity lever** | **0.4682** | **0.5876** | **0.8051** | **68** |
| Sub 21 | + UniMol s4 (LR=5e-4), OOF MAE 0.4690 | *pending* | *pending* | *pending* | *pending* |

Sub 20 is the current Phase 1 best. Sub 21 adds a fourth UniMol run (LR=5e-4, r=0.904 vs LR=5e-5) which earned 7.5% SLSQP blend weight and improved OOF MAE by 0.0005.

---

## Methodology

Full write-up: [`manuscript/pxr_challenge.qmd`](manuscript/pxr_challenge.qmd) ([PDF](manuscript/pxr_challenge.pdf))

**Scaffold-split CV:** Butina clustering (threshold=0.30), 5-fold leave-cluster-out. OOF predictions are a pessimistic lower bound vs the leaderboard.

**Models (SLSQP blend weights, Sub 20):**

| Model | Weight |
|---|---|
| UniMol v1 fine-tune, LR=2e-4 (s3) | 24.8% |
| LightGBM + docking features | 23.5% |
| TabPFN | 20.0% |
| UniMol v1, LR=5e-5 (s1 / s2) | 21.6% |
| TabICL + TabICL-Chemprop | 10.2% |

**Calibration:** Dynamic variance recalibration (`target_std=0.70`). Training pEC50 std=1.121; test analogs cluster in a narrower, higher-potency distribution (estimated std≈0.70).

**Key negative results:**
- MMP-delta model: Spearman=−0.071 on LB; 90.8% of test compounds have no close training neighbor
- TabICL-HTS embeddings: Pearson r=1.000 vs base TabICL (PCA collapses HTS+CheMeleon to same latent space)
- UniMol s5 (LR=1e-3): r=0.689 diversity but OOF MAE=0.683 — too noisy, zeroed out by SLSQP

---

## Pipeline

```
scripts/01_download_data.py          # pull from HuggingFace
scripts/02_curate_data.py            # standardize, dedup, curate labels
scripts/04_build_features.py         # Morgan FP, RDKit 2D, Mordred, CheMeleon embeddings
scripts/05_build_cv_splits.py        # Butina 5-fold scaffold CV

scripts/06_train_lgbm.py             # LightGBM baseline
scripts/07_train_chemprop.py         # Chemprop v2 multitask (GPU, ~2h)
scripts/07b_pretrain_chemprop_hts.py # HTS pretraining on Tox21 + NCATS PXR
scripts/10_train_tabpfn.py           # TabPFN in-context
scripts/16_extract_chemeleon_embeddings.py
scripts/17_extract_unimol_embeddings.py
scripts/20_prepare_docking_receptor.py
scripts/21_dock_compounds.py         # AutoDock-GPU
scripts/22_extract_docking_features.py
scripts/23_train_lgbm_docking.py     # LightGBM + docking features
scripts/24_train_unimol2.py          # UniMol v1 fine-tune, LR=5e-5
scripts/26_train_tabicl.py
scripts/30_extract_chemprop_finetuned_emb.py
scripts/31_train_tabicl_chemprop.py
scripts/32_train_unimol2_s3.py       # UniMol v1 fine-tune, LR=2e-4
scripts/33_extract_chemprop_hts_pretrained_emb.py
scripts/34_train_tabicl_hts.py       # HTS embeddings (negative result: r=1.0 vs tabicl)
scripts/35_train_unimol2_s4.py       # UniMol v1 fine-tune, LR=5e-4
scripts/36_train_unimol2_s5.py       # UniMol v1 fine-tune, LR=1e-3 (zeroed out)
scripts/37_residual_analysis.py      # OOF residual diagnostics by quartile

scripts/11_ensemble.py               # SLSQP blend + variance recalibration
scripts/13_validate_submission.py    # format + leakage check before upload
scripts/12_phase2_calibrate.py       # Phase 2: linear calibration on Analog Set 1 labels
```

---

## Repository Structure

```
src/openadmet/      Python package (cv, ensemble, features, utils)
scripts/            Numbered pipeline scripts (01-37)
configs/            YAML configs (ensemble.yaml, model configs)
data/               Pipeline artifacts (raw + splits tracked; features gitignored)
data-challenge/     Original challenge data files
models/             Model checkpoints and OOF/test prediction arrays
submissions/        Competition CSV files (phase1/)
manuscript/         QMD source + rendered PDF/HTML
tests/              Unit tests
```

---

## Reproduce

```bash
pip install -e ".[dev]"
```

Then run each script in the pipeline order above. Most steps are fast (seconds to minutes); the GPU-intensive steps are:
- `07_train_chemprop.py` — ~2h on GTX 1080
- `24/32/35/36_train_unimol2*.py` — ~2.5h each on GTX 1080
- `21_dock_compounds.py` — ~4h (AutoDock-GPU required)

Once all model outputs exist under `models/`, run:

```bash
python scripts/11_ensemble.py
python scripts/13_validate_submission.py submissions/phase1/ensemble_recal.csv
```

`11_ensemble.py` writes `submissions/phase1/ensemble_recal.csv`. The numbered copies (e.g. `ensemble_recal070_sub21.csv`) are manual archives of submitted files.

Requires: Python 3.12, CUDA GPU (GTX 1080 8GB tested), AutoDock-GPU for docking step.
