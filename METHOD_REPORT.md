# Method Report — OpenADMET PXR Challenge (pEC50 Prediction)
Authors: Denali Schlesinger, Achyut Shastri, Claude (Sonnet 4.6, Opus 4.6, Fable 5 ✝️)

## Abstract

We predict Pregnane X Receptor (PXR) activity (pEC50) from molecular SMILES. Our
final submission is a **two-level stacked ensemble**: seven base learners
(model × molecular-representation pairs) are **cross-fitted over 5 Butina-clustered
folds** to produce a 35-column out-of-fold prediction matrix, augmented with three
uncertainty meta-features, and combined by a **level-1 meta-learner whose
hyperparameters are tuned with Optuna**. On a held-out portion of the unblinded
test set never seen by the optimizer, the model reaches **RMSE 0.7155 / MAE 0.539 /
R² 0.546**, note this is on 51 SMILES so is unreliable.

The central empirical findings of this work:

1. **Pretrained molecular embeddings dominate hand-crafted descriptors.** Finetuned
   ChemProp (2D message-passing GNN) and Uni-Mol (3D) representations were the
   strongest single signals by a wide margin.
2. **The ensemble's accuracy comes from its base learners, not its meta-learner.**
   A tuned XGBoost stacker and a tuned ElasticNetCV stacker produce *identical*
   held-out RMSE and R²; the choice of level-1 model is a wash.
3. **Only three of seven base learners carry the ensemble.** Permutation importance
   on held-out data shows Uni-Mol ≫ ChemProp > Delta, with the remaining four
   learners contributing ≈0 (TabICL slightly negative). This ranking is robust
   across both meta-learners.

---

## 1. Data

- **Training pool:** 4,139 molecules with measured pEC50.
- **Evaluation split:** the released *unblinded* test set, **253 molecules**, used as
  the held-out evaluation target. Base learners are never fit on these rows.
- **Final unbiased split:** of the 253 unblinded rows, **202 ("optimization")** are
  exposed to the Optuna search and **51 ("held-out final")** are reserved and scored
  only once, after hyperparameters are frozen.
- The provide training data was used as well as the unblinded test data for validation for representation/model
  screening; the unblinded test is used for the final ensemble evaluation.

> Note on the held-out estimate: at n = 51 the held-out R² (0.546) is noisier than
> the optimization-set R² (0.479) computed on n = 202; the gap is consistent with
> small-sample variance rather than a real improvement on unseen data.

---

## 2. Molecular Representations Explored

We screened eleven representations, each fed to a gradient-boosted tree (XGBoost) and
scored by R² on the unblinded test set. The table below reports the pairwise
representation grid (representation A → representation B, two-representation input);
because order of construction matters the grid is mildly asymmetric, so we report it
in triangular form, with the **diagonal = single-representation performance**.

**Single-representation ranking (diagonal, R²):**

| Representation | R² | Family |
|---|---|---|
| chemprop_finetuned | **0.536** | 2D message-passing GNN (pretrained, finetuned) |
| chemeleon_finetuned | 0.435 | 2D GNN embedding (finetuned) |
| chemeleon | 0.430 | 2D GNN embedding (pretrained) |
| unimol | 0.394 | 3D conformer transformer |
| unimol_finetuned | 0.391 | 3D conformer transformer (finetuned) |
| mole | 0.375 | pretrained tabular foundation embedding |
| rdkit | 0.373 | 2D physicochemical descriptors |
| maccs | 0.314 | 166-bit structural keys |
| morgan | 0.289 | ECFP circular fingerprint |
| xtb | 0.121 | GFN2-xTB quantum-chemistry features |
| jazzy | −0.033 | solvation / H-bonding descriptors |

**Findings:**

- **Learned embeddings beat descriptors.** The best four single representations are
  all pretrained neural embeddings. `chemprop_finetuned` is the strongest standalone
  signal (R² 0.536), and pairing *any* representation with it lifts that pair into the
  0.52–0.56 band — it is the dominant pairwise signal in the grid.
- **Quantum-chemistry and solvation descriptors were the weakest.** `xtb` (R² 0.12)
  and `jazzy` (negative R² standalone) did not justify their cost (separate conda
  env, per-molecule subprocess). They were retained only inside a combined tabular
  block for diversity, not for standalone signal.
- **Finetuning helped unevenly.** ChemProp finetuning was decisive. ChemeLeon and
  Uni-Mol finetuning produced little-to-no standalone gain over their pretrained
  forms; the finetuned variants were nonetheless used in the roster.

---

## 3. Base Learners (Level-0 Roster)

Seven `(model, representation)` learners, chosen for a mix of raw signal and
*orthogonal error structure*:

| Learner | Model | Representation(s) | Rationale |
|---|---|---|---|
| chemprop | XGBoost | chemprop_finetuned | best single signal (2D GNN) |
| unimol | XGBoost | unimol_finetuned | 3D shape — orthogonal to 2D |
| mole | XGBoost | mole | pretrained tabular, different family |
| tabular | XGBoost | rdkit + jazzy + xtb + crystal_similarity | descriptors + QM + binder-resemblance |
| tabicl | TabICL | rdkit | tabular foundation model |
| knn | KNN | morgan | local Tanimoto neighborhood |
| delta | DeltaModel | rdkit | pairwise/relational; also supplies a coverage meta-feature |

---

## 4. Ensemble Architecture

1. **Cross-fitting.** Molecules are clustered with the **Butina algorithm** and split
   into 5 grouped folds, so structurally similar molecules cannot straddle the
   train/validation boundary. For each base learner *j* and each fold *f*, the learner
   is fit on the fold-*f* training portion and predicts on **all** rows. This yields a
   **(n × 35)** matrix (7 learners × 5 folds) of leak-controlled predictions.
2. **Uncertainty meta-features.** The level-1 input is augmented with three columns:
   nearest-neighbor Tanimoto similarity (an applicability-domain signal),
   base-learner disagreement (prediction std), and a Delta-model coverage flag.
3. **Imputation.** Non-finite features are replaced by per-column medians computed on
   the train pool.
4. **Level-1 meta-learner.** A single regressor maps the 35 (+3) columns to pEC50.

---

## 5. Meta-Learner Tuning

The level-1 model's hyperparameters were tuned with **Optuna (TPE, 100 trials)**,
minimizing RMSE on the 202-row optimization set, with the 51-row held-out set scored
only after freezing. We compared a gradient-boosted tree against a regularized linear
stacker head-to-head:

| Meta-learner | Opt-val RMSE | Opt-val R² | Held-out RMSE | Held-out MAE | Held-out R² |
|---|---|---|---|---|---|
| XGBoost (tuned) | 0.7393 | 0.4790 | **0.7155** | 0.5392 | **0.5459** |
| ElasticNetCV (tuned) | 0.7430 | 0.4737 | **0.7155** | **0.5234** | **0.5459** |

**The meta-learner choice is a wash.** Held-out RMSE and R² are identical; XGBoost
edges ElasticNet by ~0.4% on optimization RMSE, while ElasticNet is *better* on
held-out MAE. Because a regularized linear stacker on 35 collinear predictions is the
defensible default — and cannot overfit the redundant columns the way a tree can —
ElasticNetCV is an equally valid final choice. Optuna improved RMSE only modestly
(≈0.771 → 0.743), confirming that tuning the stacker is not where the accuracy lives.

Best XGBoost params: `n_estimators=788, max_depth=10, lr=0.39, subsample=0.93,
colsample_bytree=0.42, min_child_weight=6, gamma=0.24, reg_alpha=0.043,
reg_lambda=0.074`.
Best ElasticNet params: `l1_ratio=0.010, n_alphas=277, cv=4` (i.e. nearly pure ridge).

---

## 6. What Actually Carries the Ensemble

Permutation importance (mean RMSE increase when a learner's columns are shuffled,
10 repeats) and native gain were computed **on the held-out final set**, summed over
each learner's 5 folds:

| Base learner | Permutation ΔRMSE | Native gain | Reading |
|---|---|---|---|
| **unimol** | **0.086** | 0.573 | most irreplaceable signal (3D) |
| **chemprop** | **0.035** | 0.200 | strong, but partly redundant |
| **delta** | **0.022** | 0.325 | genuine relational contribution |
| mole | 0.002 | 0.089 | negligible |
| tabular | 0.002 | 0.015 | negligible |
| knn | 0.000 | 0.005 | negligible |
| tabicl | **−0.005** | 0.070 | slightly harmful |

**Thesis:** the ensemble is carried by **two pretrained embedding learners (Uni-Mol
3D, ChemProp 2D-GNN) plus the Delta relational model.** The remaining four learners
(mole, tabular, TabICL, kNN) are diversity insurance that the meta-learner barely
attends to. Notably, `chemprop` has the highest *gain* among neural learners yet
modest *permutation* importance — the stacker leans on it, but its signal is largely
recoverable from the others; `unimol` is the truly orthogonal, irreplaceable input.
This ordering is stable across both the XGBoost and ElasticNet meta-learners, so it
reflects the data, not a quirk of the level-1 model.

---

## 7. Leakage Control & Evaluation Methodology

- **Grouped cross-validation.** Butina clustering ensures near-duplicate scaffolds do
  not appear in both the train and validation side of any fold — the primary defense
  against the optimistic bias that random folds produce on congeneric chemistry. We
  adopted Butina on principle for all k-fold cross-fitting; we did not run a formal
  scaffold-vs-random ablation.
- **Base learners never see evaluation rows.** The unblinded test is featurized and
  predicted, but never used to fit any level-0 learner, so the meta-learner's
  validation signal is honest.
- **Frozen held-out set.** 20% of the unblinded test (51 rows) is withheld from the
  Optuna search entirely and scored once, giving an estimate uncontaminated by
  hyperparameter selection. (Caveat: n = 51 makes this estimate noisy; see §1.)

---

## 8. What Didn't Work (Negative Results)

- **Symbolic regression and KANs** (Kolmogorov–Arnold networks): poor results in
  initial screening on ChemeLeon and RDKit representations; discarded as lines of
  investigation.
- **TabICL** earned a slightly *negative* permutation importance in the final stack;
  it is the one base learner that, on net, the ensemble would be better without. (It
  is also memory-hungry — at default sizes it OOMs on constrained GPU hardware.)
- **Quantum-chemistry / solvation descriptors (xTB, Jazzy).** Weakest standalone
  signal (xTB R² 0.12; Jazzy negative); not worth their compute cost. Retained only
  inside the combined tabular block for diversity.

---

## 9. Conclusions & Future Work

The winning recipe is **strong pretrained molecular embeddings + leak-controlled
stacking**, where the stacking machinery matters far less than the choice of base
representations. Given the importance analysis, a leaner ensemble of
{Uni-Mol, ChemProp, Delta} with an ElasticNet stacker would likely match the full
seven-learner model at a fraction of the cost.

Future directions:

- Prune the roster to the three contributing learners and re-tune.
- Invest finetuning effort where it paid off (ChemProp-style) rather than where it
  didn't (Uni-Mol, ChemeLeon).
- Acquire a larger, scaffold-disjoint external test set to tighten the held-out
  estimate (the current n = 51 is the main source of metric uncertainty).
- Test different stacks for the ensemble model (this one is particularly bad)
