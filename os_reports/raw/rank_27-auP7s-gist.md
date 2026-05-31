# PXR OpenADMET Challenge — Methodology Report

Submission for the [OpenADMET PXR Challenge](https://huggingface.co/spaces/openadmet/pxr-challenge): predicting PXR activation (pEC50) from SMILES on a held-out blind set, ranked by RAE.

---

## Overview

The pipeline is a mix of classical descriptors and fine-tuned embeddings from open chemistry foundation models, fed into TabICL (tabular in-context learner) as the main predictor. TabPFN-3 runs alongside it for an orthogonal read. Out-of-fold predictions from every (featurizer × CV-axis) combination get stacked into a candidate matrix, and an NSGA-II search picks subsets that trade off OOF MAE against Spearman ρ. The shipped submission is one of the picks — usually an NNLS blend with isotonic post-calibration.

Nothing particularly novel here, but the combination of (a) sweeping many CV axes and (b) restricting the ensemble pool to portal-aligned axes seems to do most of the work.

---

## Featurization

Two tracks:

**Classical descriptors**
- RDKit 2D descriptors
- Mordred (the slow one, ran once and cached)
- MACCS keys
- Morgan and Avalon count fingerprints
- jazzy QM interaction descriptors
- xTB quantum descriptors

**scikit-fingerprints** add-ons (these were a late addition but mostly pulled their weight):
- MAP4
- AtomPair
- Topological Torsion
- PubChem
- Pharmacophore

**Fine-tuned chemistry foundation models** — pretrained models adapted on the challenge training data, then used as fixed feature extractors. For each encoder we fine-tune the primary multi-concentration regression head.

- CheMeleon (chemprop MPNN backbone)
- MoLFormer-XL (IBM SMILES transformer)
- UniMol v1 + v2 (3D MPNN)
- MolCLR (GIN contrastive)
- ChemBERTa-2 (SMILES BERT)
- SELFormer (SELFIES transformer)

---

## Models

- **TabICL v2** (`soda-inria/tabicl`) — primary
- **TabPFN-3** (Prior Labs) — secondary, run on the same featurizer × axis grid

Both are run per (featurizer × axis), producing 5-fold OOF predictions on train and an averaged prediction on blind. Truncated SVD reduces to 768 dimensions when features get bigger than that — mostly an OOM safety net for TabICL, occasionally fires on the hybrid concats.

We default to **single seed × 5 folds** for the main sweep. Multi-seed (3 seeds averaged) is available as a toggle for seed-stability checks but didn't move the needle enough on this dataset to be worth the 3× compute.

---

## Cross-validation strategy

Six split axes get computed:

- random (i.i.d. baseline)
- Butina cluster, Tanimoto cutoff 0.4
- Murcko scaffold
- near-duplicate grouping (Tanimoto > 0.7)
- MaxMin centrality-graded


Only scaffold + Butina 0.4 + MaxMin enter the ensemble pool. Random and near-dup are still computed but treated as diagnostic axes only.


---

## Ensembling

OOF + blind predictions across (featurizer × axis) stack into a candidate matrix. From there:

- Subset selection by NSGA-II (via Optuna), Pareto on (OOF MAE, Spearman ρ)
- Subset size K from 3 to 12
- Three blending methods evaluated per Pareto point:
  - NNLS
  - NNLS + isotonic post-calibration
  - Caruana 2004 hill-climb with replacement


---

## Data preparation

- RDKit canonicalization + standardization
- Molecular-weight and heavy-atom filters
- Per-compound deduplication on canonical SMILES
- Promiscuous-compound filter: hit on primary AND has counter-assay AND counter/primary ratio ≥ 1 (the looser "fold > X" rule we used earlier produced too many false positives)
- Tanimoto-leakage check against the blind set (very small fraction flagged)

---

## Performance comments

- Fine-tuning the foundation models on the challenge target was the biggest single lift over using their pretrained features as-is. Not surprising in retrospect; PXR is far enough from generic activity tasks that pretrained-only features under-call the harder series.
- CV-axis diversity > seed diversity. Adding scaffold/Butina axes to a featurizer gave a bigger ensemble lift than running the same axis with multiple random seeds.
- In-distribution OOF MAE consistently under-estimates portal MAE. Filtering the ensemble pool to strict axes closed most of that gap.
- Calibrating against an external public ADMET-adjacent dataset (we tried ChEMBL early on) backfired — distribution mismatch between that source and the blind set inflated portal MAE by a lot. Switched back to OOF-only fitting on the challenge training data.


---

## Reproducibility

The whole thing runs as a single notebook on Colab GPU (T4). Featurizers, fine-tuned encoder checkpoints, and intermediate predictions all cache to disk and resume across kernel restarts. Per-(model × featurizer × axis) predictions are written atomically, so partial reruns are safe.

---

## Data availability

Only the challenge-provided data plus open public chemistry datasets were used. No proprietary data. Pre-trained foundation models loaded from public Hugging Face / GitHub releases.

---

## Acknowledgments

- **OpenADMET** for the challenge, dataset, and submission portal
- **Prior Labs** (TabPFN) and **soda-inria** (TabICL) for the tabular foundation models and academic access
- Author teams of **chemprop**, **CheMeleon**, **MoLFormer**, **UniMol**, **MolCLR**, **ChemBERTa-2**, **SELFormer**
- Open-source ecosystem: **RDKit**, **scikit-learn**, **scikit-fingerprints**, **Optuna**, **Mordred**, **jazzy**, **datamol**, **molfeat**
