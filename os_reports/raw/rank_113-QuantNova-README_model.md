# PXR pEC50 Prediction Pipeline

This project provides a complete molecular activity prediction workflow for the OpenADMET PXR induction task. The goal is to predict compound-level pEC50 values from molecular structure and available assay-derived context while keeping the modelling process reproducible, conservative, and interpretable.

The workflow is designed around one central idea:

> PXR activity is not determined by a single descriptor or a single nearest analogue. Reliable prediction requires combining molecular fingerprints, physicochemical descriptors, assay context, local structure similarity, and validation-aware model fusion.

---

## 1. Task Overview

Pregnane X Receptor (PXR) activation is a structure-sensitive ADMET endpoint. Small changes in scaffold, linker geometry, hydrophobic surface, heteroatom placement, or aromatic substitution can cause non-linear changes in measured potency.

This project treats pEC50 prediction as a supervised molecular regression problem, but with several important constraints:

- compounds may belong to related analogue series;
- nearby molecules can still show activity cliffs;
- assay noise and confidence vary between records;
- external activity data may be useful but should not be trusted equally;
- leaderboard-style test compounds may not follow the same distribution as random validation folds.

Because of this, the pipeline avoids relying on a single model family. Instead, it builds several complementary views of each molecule and combines model outputs using validation-driven weighting.

---

## 2. Methodological Principles

### 2.1 Structure first, not metadata first

The primary signal is the molecular structure. Each compound is represented through multiple molecular views, including circular fingerprints, substructure keys, physicochemical descriptors, and optional learned molecular embeddings.

The reason for using multiple molecular representations is that different descriptors capture different aspects of SAR:

- circular fingerprints capture local atom environments and substituent patterns;
- count-based fingerprints preserve repeated motif information;
- chirality-aware fingerprints retain stereochemical distinctions;
- MACCS-style keys capture common medicinal chemistry fragments;
- RDKit descriptors summarize size, polarity, lipophilicity, hydrogen bonding, ring systems, and other global properties;
- mol2vec-style embeddings provide a learned fragment-level representation.

No single representation is assumed to be sufficient. The final model uses agreement and complementarity between these views.

---

### 2.2 Assay context is auxiliary, not a replacement for structure

The pipeline uses assay-derived auxiliary information when available, such as single-concentration response patterns and counter-assay related signals. These features help describe whether a molecule tends to behave like a true PXR-active compound, a weak responder, or a possible assay-interfering compound.

However, these signals are treated as supporting evidence rather than the main target. The final pEC50 prediction remains structure-centered.

This matters because assay metadata can improve local ranking but may also introduce leakage or overfitting if used too aggressively. The intended logic is:

1. use structure to estimate intrinsic potency;
2. use auxiliary assay context to adjust confidence and relative ordering;
3. avoid letting noisy assay-derived signals dominate the final prediction.

---

### 2.3 Similar compounds are useful, but not always safe

The project includes a local structure-similarity component based on Tanimoto-nearest neighbors. This captures an intuitive medicinal chemistry rule: if a test compound is close to known compounds, nearby activities provide useful local information.

At the same time, PXR SAR often contains activity cliffs. A high Tanimoto similarity does not guarantee the same pEC50. Therefore, the local-neighbor signal is used as one component in the ensemble, not as the sole predictor.

The expected role of the similarity component is:

- improve predictions for compounds inside known analogue neighborhoods;
- provide a diagnostic signal for local SAR consistency;
- help identify compounds where model predictions disagree with nearest-neighbor evidence;
- avoid over-correcting when the structure is close but the pharmacophore relationship is uncertain.

---

### 2.4 Validation must respect chemical similarity

Random train/validation splitting can overestimate performance when related analogues appear on both sides of the split. To reduce this problem, the project uses structure-aware validation based on molecular similarity clustering.

The validation design attempts to answer a more realistic question:

> Can the model generalize to related but not identical chemical neighborhoods, rather than simply memorizing highly similar molecules?

This is especially important for PXR because the test set may contain analogue expansions around selected active scaffolds. A useful validation scheme should therefore test both interpolation within known chemistry and robustness across scaffold-adjacent regions.

---

### 2.5 External data is useful but lower-trust

The workflow can incorporate curated external PXR activity records. These records expand chemical coverage, but they may differ from the official assay in protocol, measurement conditions, endpoint definitions, and noise level.

For this reason, external rows are treated as lower-confidence evidence. They can help the model learn broader PXR-relevant structure-activity patterns, but they should not override the official data distribution.

The intended use of external data is:

- improve broad chemical coverage;
- expose the model to additional PXR-active motifs;
- reduce overfitting to a small official set;
- keep the official assay as the main calibration anchor.

---

## 3. Overall Workflow

The pipeline follows a staged process:

```text
raw data
  ↓
compound standardization and quality filtering
  ↓
structure-aware train/validation splitting
  ↓
molecular and assay-derived feature construction
  ↓
cross-validated model training
  ↓
out-of-fold prediction analysis
  ↓
validation-aware ensemble fusion
  ↓
test-set prediction and diagnostics
  ↓
submission file generation
```

Each stage is designed to be reproducible and auditable.

---

## 4. Data Preparation Logic

### 4.1 Molecular standardization

Before modelling, compounds are standardized to reduce duplicated or inconsistent molecular records. The pipeline uses canonical molecular representations and molecule-level identifiers to align records that represent the same compound.

This step is important because duplicate structures can otherwise create overly optimistic validation results or inconsistent target values.

---

### 4.2 Quality filtering

The training data may contain records with high uncertainty, abnormal values, or weak reliability. The curation step removes or downweights records that are likely to harm model calibration.

The filtering logic is based on the following principles:

- remove invalid molecules;
- remove unusable target values;
- reduce duplicate measurements;
- prefer more reliable measurements when repeated records exist;
- preserve the official distribution as much as possible.

The aim is not to make the dataset artificially clean, but to avoid training on records that are clearly inconsistent with the regression objective.

---

### 4.3 Applicability-domain awareness

Compounds that are extremely far from the test-set chemistry provide weaker information for test prediction. The pipeline therefore tracks structural proximity between training and test compounds.

This does not mean distant compounds are always useless. They can still help models understand general PXR chemistry. But when a compound is outside the relevant chemical neighborhood, its influence should be interpreted carefully.

---

## 5. Feature Strategy

The feature strategy intentionally combines several molecular views.

### 5.1 Circular fingerprints

Circular fingerprints are used because they are strong general-purpose descriptors for ligand-based QSAR. They encode local atom environments and are especially useful for recognizing analogue-level modifications.

Both binary and count-style fingerprints are useful:

- binary fingerprints indicate whether a substructure exists;
- count fingerprints preserve repeated motif frequency;
- different radii capture different local environment sizes.

---

### 5.2 Chirality-aware features

Stereochemistry can matter in receptor activation, especially when ligand binding depends on three-dimensional orientation. Chirality-aware variants are included to preserve this information where available.

Even when stereochemistry is not the dominant signal, including it prevents the model from collapsing stereoisomers into identical representations.

---

### 5.3 Substructure keys

Substructure-key descriptors provide a more interpretable fragment-level view. They are less detailed than high-dimensional circular fingerprints, but they can stabilize predictions by capturing common medicinal chemistry motifs.

---

### 5.4 Physicochemical descriptors

Global molecular descriptors are included to represent properties that are not fully captured by local fingerprints, such as:

- molecular size;
- lipophilicity;
- polar surface area;
- hydrogen-bond donor and acceptor balance;
- ring count and aromaticity;
- rotatable bonds;
- charge-related properties.

For PXR, these features are especially relevant because activation is often associated with hydrophobic volume, aromatic surface, and suitable polarity balance.

---

### 5.5 Learned fragment embeddings

When available, mol2vec-style embeddings provide a learned representation of molecular fragments. These embeddings can capture similarities that are not identical at the fingerprint-bit level.

They are used as an auxiliary molecular view, not as a replacement for chemically interpretable descriptors.

---

### 5.6 Assay-derived auxiliary features

The pipeline can use auxiliary features derived from related assay information. These features are intended to represent experimental response context and possible screening behavior.

Their role is mainly to improve ranking and calibration where the structural model alone is uncertain.

---

## 6. Model Strategy

The final predictor is an ensemble of complementary regression models.

The model pool is intentionally diverse:

- tree ensembles capture non-linear SAR and feature interactions;
- gradient boosting models capture structured non-linear effects;
- regularized linear models provide a smoother counterbalance;
- nearest-neighbor similarity provides local analogue evidence.

This diversity is useful because PXR prediction contains both global trends and local exceptions. A single model family may perform well on one chemical region and poorly on another.

---

## 7. Cross-Validation and Out-of-Fold Predictions

The project uses out-of-fold prediction as the main internal evidence for model selection and fusion.

For each validation fold:

1. the model is trained only on the training portion;
2. predictions are produced for the held-out fold;
3. all held-out predictions are combined into one full out-of-fold prediction vector;
4. metrics are calculated on these out-of-fold predictions.

This gives a more realistic estimate of how each model behaves on unseen molecules.

Out-of-fold predictions are also used to:

- compare model families;
- estimate ensemble weights;
- check whether a model is overconfident;
- inspect prediction distribution shifts;
- diagnose disagreement between global models and local-neighbor evidence.

---

## 8. Ensemble Logic

The final prediction is not selected from one individual model. Instead, model outputs are combined using validation-aware weighting.

The ensemble has three main goals:

1. reward models that perform well in out-of-fold validation;
2. preserve complementary signals from different feature views;
3. avoid allowing any brittle component to dominate.

The local nearest-neighbor component is deliberately constrained because it can be very strong for close analogues but unsafe near activity cliffs.

The final blend combines a validation-weighted ensemble with a robust average of strong models. This gives a conservative prediction that is less sensitive to one model's failure mode.

---

## 9. Calibration Logic

After model fusion, the prediction distribution is lightly calibrated using out-of-fold behavior.

The goal is not to force the test distribution to match the training distribution. Instead, calibration is used to correct systematic shrinkage or expansion observed during validation.

This step is intentionally conservative. It should improve stability without introducing large artificial offsets.

## 10. Method Summary

In compact form, the method is:

1. standardize molecules and curate the training set;
2. build chemically meaningful validation folds;
3. generate multiple molecular descriptor views;
4. add assay-derived auxiliary context where available;
5. train diverse regression models with out-of-fold validation;
6. compute local analogue predictions as a constrained diagnostic signal;
7. combine models using validation-aware ensemble weighting;
8. apply light calibration based on out-of-fold behavior;
9. export final test predictions and detailed diagnostics.

The central philosophy is that PXR pEC50 prediction should be treated as a structure-aware, assay-aware, uncertainty-conscious regression problem rather than a simple descriptor-to-target fitting task.
