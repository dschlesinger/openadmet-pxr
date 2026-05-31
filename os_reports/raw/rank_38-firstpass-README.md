---
license: mit
tags:
  - chemistry
  - admet
  - drug-discovery
  - pxr
  - regression
language:
  - en
---

# OpenADMET PXR Activity Challenge — Public Model Summary

A short summary of the modelling approach used for our submission to the [OpenADMET PXR Activity Prediction track](https://huggingface.co/spaces/openadmet/pxr-challenge) — predicting pEC50 against human pregnane X receptor (PXR / NR1I2) for the held-out blinded compounds.

## Approach

A **stacked ensemble** of complementary base learners, each looking at the chemistry through a different lens (learned 1D/2D/3D representations, with auxiliary pharmacophore and docking signal. Predictions are combined by a regularised linear stacker fit on cluster-disjoint out-of-fold predictions; the submitted column is the **raw stacker output**, with no post-hoc calibration to the training distribution.

### Base learners

- **Representation-diverse fine-tuned learners.** Three independent fine-tunes spanning complementary molecular representations — SMILES sequence, molecular graph (with single-concentration HTS pretraining), and 3D conformer geometry — each contributing an independent input to the meta-learner. **These supply the bulk of the predictive signal in the final stack.**
- **Auxiliary structure-based features.** Exhaustive custom alignment and shared pharmacophore (Phase) hypothesis generation procedure, coupled with full dataset screening. Top-performing ensembles distilled from millions of combinations from thousands of selected hypotheses (per-fold combinatorial best-K subset selection) and a Glide docking ensemble across multiple PXR co-crystal structures — both produced with the Schrödinger commercial suite — contributed non-redundant inputs to the meta-learner. They are minority contributors to the blend, included for tail-behaviour coverage rather than as primary signal.

### Splitting

Repeated **Butina k-fold cross-validation** over calibrated similarity cutoff (~0.5) — whole clusters assigned to single folds via greedy round-robin. MCS splitting was tried but did not appear to help performance. The OOF matrix that the stacker sees is used to fit model weights to improve generalizability.

### Calibration

Predictions are submitted raw. Variance-matching and quartile mapping dramatically hurt blind-test performance.

## Leaderboard history

Iterative progression from RAE ~0.94 (rank ~220, V1 baseline) to **RAE 0.57, R² 0.55, ρ 0.82, rank 27** (current best).

## Acknowledgements

- **NN Digital Chemistry and MolAI Teams** - Internal team for advice and contributions
- All comntributors to the challenge - with special mention to **[Liz Yurkewych](https://github.com/lizyurkewych-git/pxr-challenge)** and  **[jeremycheminf](https://github.com/jeremycheminf/openadmet_scripts/tree/main/PXR/2D)** whose early work provided some inspiration
- **The OpenADMET team** — challenge, data, leaderboard, [openadmet_scripts](https://github.com/openadmet/openadmet) reference.
- **ChEMBL** — augmentation labels, indispensable public data resource
- **[Chemprop](https://github.com/chemprop/chemprop)** and **[CheMeleon](https://github.com/JacksonBurns/chemeleon)** — open-source models used as base learners.


## Tools

Built on the open-source cheminformatics and ML stack: **RDKit**, **scikit-learn**, **PyTorch**, and the broader open-source cheminformatics-ML ecosystem.
- Pharmacophore generation and ensemble docking via **Schrödinger Phase and Glide**. 
- **Internal LLM-based coding assistants** — notebook scaffolding and refactoring support.

---

*Last updated: 2026-05-24*
