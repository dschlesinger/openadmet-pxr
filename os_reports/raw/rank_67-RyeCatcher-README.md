---
license: apache-2.0
language:
- en
tags:
- chemistry
- drug-discovery
- admet
- pxr
- nr1i2
- openadmet
- molecular-property-prediction
- pharmaceutical
- chemprop
- autogluon
library_name: pytorch
pipeline_tag: tabular-regression
base_model:
- openadmet/chemeleon
datasets:
- openadmet/pxr-challenge-train-test
metrics:
- rae
- mae
- spearmanr
- kendalltau
model-index:
- name: openadmet-pxr-challenge-2026 (v43 final blend)
  results:
  - task:
      type: tabular-regression
      name: PXR pEC50 prediction
    dataset:
      type: openadmet/pxr-challenge-train-test
      name: OpenADMET PXR Induction Blind Challenge (Activity track)
    metrics:
    - type: rae
      value: 0.586
      name: Relative Absolute Error (LB)
    - type: mae
      value: 0.467
      name: Mean Absolute Error (LB)
    - type: spearmanr
      value: 0.802
      name: Spearman correlation (LB)
---

# OpenADMET PXR Induction Blind Challenge — v43 Final Blend

**Track:** Activity Prediction
**Best leaderboard RAE:** 0.586 (rank ~40 / 211, top 19%)
**Team:** BioInfo / RyeCatcher
**Author:** Justin Johnson <justin@rundatarun.io>
**Methodology report:** [`docs/METHODOLOGY_REPORT.pdf`](docs/METHODOLOGY_REPORT.pdf)

This repository contains the full pipeline, trained model checkpoints, out-of-fold predictions, and final submission for the [OpenADMET PXR Induction Blind Challenge](https://huggingface.co/spaces/openadmet/pxr-challenge), held March–July 2026.

## TL;DR

`v43 = 0.78 · v31 + 0.22 · T1v5`, isotonic-calibrated per fold.

- **v31** is itself a cascade: `v7 → v22 → v26 → v28 → v29 → v31`, blending five Chemprop multitask models (T1, T1v5, T1v2, T13, T14), an AutoGluon-CheMeleon tabular model (T2), and two LightGBM baselines (v3, v4) on FCFP4-count + Mordred + CheMeleon-2048d features.
- **T1v5** is a 5-head Chemprop D-MPNN multitask model trained jointly on the OpenADMET DRC, the counter-screen, NCATS qHTS PXR (AID 1346982 / 1346985), Tox21 SR-ARE, and ChEMBL NR1I2.
- Final calibration is **honest per-fold isotonic regression** — fit on out-of-fold predictions only, leave-one-fold-out. Isotonic fit on full training OOFs over-predicts improvement by ~0.01 RAE; honest fit was the gate that prevented submitting overfit garbage.

## Repository layout

```
.
├── README.md                          # this card
├── docs/
│   ├── METHODOLOGY_REPORT.md          # required contest methodology report (source)
│   └── METHODOLOGY_REPORT.pdf         # rendered PDF
├── code/
│   ├── requirements.txt               # pinned deps
│   ├── featurization/                 # CheMeleon, Mordred, FCFP4 wrappers
│   ├── baseline/                      # v2 XGBoost, v3/v4 LightGBM
│   ├── multitask/                     # T1, T1v5 (Chemprop), T2 (AutoGluon)
│   ├── ensemble/                      # v26, v31, v43 cascade blends
│   ├── submit/                        # Gradio API submission script
│   └── utils/                         # validation utilities
├── models/
│   ├── T1_chemprop/                   # 5-seed multitask Chemprop (12 MB)
│   ├── T1v5_chemprop/                 # 5-fold + pretrained 5-head Chemprop (9.5 MB)
│   └── T2_autogluon_chemeleon/        # AutoGluon-tabular on CheMeleon emb (1.6 GB)
├── data/
│   ├── raw/                           # OpenADMET-provided CSVs (also at openadmet/pxr-challenge-train-test)
│   ├── parent_clusters_fcfp4.csv      # FCFP4-NN parent-cluster assignment (LOCO grouping)
│   └── oof_predictions/               # per-track OOF + test predictions (inputs to v43 blend)
└── submission/
    ├── v43_final.parquet              # 513-row final submission
    ├── v43_final.csv
    └── v43_final.json                 # manifest with CV metrics + LB outcome
```

## Reproducing v43 from the OOF predictions

Fastest path — the blend itself is pure numpy on OOFs we've published:

```bash
git clone https://huggingface.co/RyeCatcher/openadmet-pxr-challenge-2026
cd openadmet-pxr-challenge-2026
python -m venv .venv && source .venv/bin/activate
pip install -r code/requirements.txt
python code/ensemble/v43_final_blend.py    # produces submission/v43_final.parquet
```

This recomputes `v43_final.parquet` from the stored OOF CSVs in `data/oof_predictions/`.

## Reproducing the trained models from scratch

Heavier path — full retraining of every component:

1. Pull the OpenADMET data: `data/raw/pxr-challenge_TRAIN.csv`, `pxr-challenge_TEST_BLINDED.csv`, the counter-screen and single-concentration files. (Also at [`openadmet/pxr-challenge-train-test`](https://huggingface.co/datasets/openadmet/pxr-challenge-train-test).)
2. Generate features: `python code/featurization/featurize_mordred.py` and `code/featurization/featurize_chemeleon.py`.
3. Train baselines: `python code/baseline/v3_v4_lgbm.py` → produces v3 (Butina-LOCO LGBM) and v4 (kitchen-sink LGBM) OOFs.
4. Train multitask models: `code/multitask/T1_chemprop_external_pretrain.py`, `T1v5_chemprop_chembl_nr1i2.py`, `T2_autogluon_chemeleon.py`.
5. Run cascade: `code/ensemble/v26_cascade_blend.py` → `v31_v29_t2_blend.py` → `v43_final_blend.py`.

Expect a few hours of GPU time on a single ~A100/H100-equivalent and several hours of CPU time for AutoGluon and Mordred featurization. Methodology report has the full hyperparameter list.

## Final-leaderboard metrics (live leaderboard, partial test set)

| Metric | v43 (this model) | Top-1 (Yan) |
|---|---|---|
| RAE | 0.586 | 0.496 |
| MAE | 0.467 | 0.394 |
| RMSE | 0.542 | 0.46 |
| Spearman | 0.802 | 0.82 |
| Kendall | 0.615 | 0.63 |

Reported on the live (partial blind) leaderboard as of 2026-05-10. Final blinded leaderboard publishes after challenge close (Phase 1: 2026-05-25, Phase 2 + Structure track: 2026-07-01).

## What worked

1. **Diversity over scale.** A 300-parameter Chemprop blended with a 4,900-feature LightGBM beat single models 10× larger. The error modes of D-MPNN graph models and gradient-boosted trees on descriptors are decorrelated; combining them captures variance neither covers alone.
2. **Honest per-fold isotonic.** In-sample isotonic fit on full OOFs is optimistic by ~0.009 RAE; chained iso-on-iso compounds at ~0.01 per step. Per-fold (leave-one-fold-out) isotonic was the only protocol that produced CV→LB shifts consistent with the actual leaderboard.
3. **Parent-cluster LOCO.** Random folds leak parent-series into validation. The test set is 513 analogs of ~89 parent compounds; FCFP4-NN-based grouping reproduces that structure for CV.
4. **External multitask aux.** ChEMBL NR1I2 (907 compounds, T1v5 head 5) and NCATS qHTS PXR (AID 1346982 / 1346985, ~10K compounds, T1 heads 2-3) added measurable lift even though individual heads had Spearman ≤ 0.4 against the primary task. Multitask transfer works through shared representations, not direct correlation.

## What did not work (in case it saves you time)

- **NN-transfer of any biological-assay readout from train compounds to test.** This caused a catastrophic LB regression (v50: OOF 0.4536 → LB 0.658, rank 87). Test compounds include "SAR misses" — high structural similarity to active train compounds but biologically inactive. Nearest-neighbor copying of the train assay readout assigns high activity to inactive misses and overshoots pEC50.
- **TabPFN on CheMeleon.** Spearman 0.93 against v43 → no orthogonal signal. Foundation-model embeddings dominate the representation regardless of the tabular learner on top.
- **ChemBERTa, GIN-ACtriplet, UniMolV2-310M, MaskMol, ImageNet ViT on molecular images.** All converged to Spearman > 0.88 against v43. The 2D-graph representation space is saturated for this task and this data size.
- **Tail-weighted loss / quantile regression heads.** Cannot fix the data ceiling: only 11 training compounds have pEC50 ≥ 6.5.

## License

Apache 2.0 for code, model weights, and predictions. OpenADMET-provided data in `data/raw/` is redistributed under the OpenADMET license (Apache 2.0, see [original dataset](https://huggingface.co/datasets/openadmet/pxr-challenge-train-test)). External datasets referenced in the methodology report (Tox21, NCATS qHTS, ChEMBL, BindingDB) are under their respective public licenses.

No proprietary or confidential data was used.

## Citation

```bibtex
@misc{johnson2026openadmetpxr,
  author = {Johnson, Justin},
  title  = {OpenADMET PXR Induction Blind Challenge — v43 Final Blend (rank 40/211)},
  year   = {2026},
  publisher = {Hugging Face},
  url    = {https://huggingface.co/RyeCatcher/openadmet-pxr-challenge-2026},
}
```

## Acknowledgements

Thanks to the OpenADMET team (Open Molecular Software Foundation) for the assay data, the public leaderboard, the structural cleanup, and a contest design that rewards methodology over compute. Thanks to the Chemprop, AutoGluon, RDKit, and CheMeleon authors for the open-source primitives that made this possible in two weeks.
