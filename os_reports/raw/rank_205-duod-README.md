# PXR_report

## 1. Model Overview

This report describes an optimized machine learning workflow for the OpenADMET PXR Blind Challenge Activity Prediction Track. The objective is to predict the primary PXR induction potency of unseen compounds from molecular structure, expressed as **pEC50**.

The model uses molecular structure as the only direct chemical input. Each compound is represented by its SMILES string, which is converted into numerical features using RDKit fingerprints, physicochemical descriptors, PXR-aware pharmacophore features, and SAR-informed rule features. The final prediction is produced by an ensemble of tree-based and linear regression models, followed by a conservative nearest-neighbor SAR calibration step.

The design follows a hybrid strategy:

1. **Data-driven QSAR modeling** from molecular fingerprints and descriptors.
2. **PXR-specific feature engineering** based on known physicochemical preferences of the PXR ligand-binding pocket.
3. **Structure-activity relationship calibration** derived from 99 analog-hit molecular pairs.
4. **Uncertainty-aware training** using assay standard errors and confidence intervals when available.

The model is intended to improve robustness over a purely fingerprint-based baseline by injecting chemically interpretable PXR-specific rules, without using test labels or any leaderboard feedback.

## 2. Prediction Target

The primary target is:

```text
pEC50 = -log10(EC50 in mol/L)
```

Higher pEC50 means stronger potency. For example:

```text
pEC50 = 6.0  -> EC50 = 1 µM
pEC50 = 5.0  -> EC50 = 10 µM
pEC50 = 7.0  -> EC50 = 0.1 µM
```

The model predicts pEC50 directly instead of EC50 because EC50 values span several orders of magnitude, while pEC50 is on a more stable logarithmic scale. This improves regression stability and makes error magnitudes chemically interpretable.

## 3. Input Data

The core input files are expected under the `data/` directory:

```text
data/
  pxr-challenge_TRAIN.csv
  pxr-challenge_TEST_BLINDED.csv
```

Optional auxiliary files may also be present:

```text
data/
  pxr-challenge_counter-assay_TRAIN.csv
  pxr-challenge_single_concentration_TRAIN.csv
  pxr-challenge_structure_TEST_BLINDED.csv
```

The main training file contains SMILES strings and experimentally fitted pEC50 labels. The blinded test file contains compound identifiers and SMILES strings only. Compound identifiers such as `Molecule Name`, `OCNT_ID`, `OCNT Batch`, `Split`, and `source` are used for tracking and submission alignment only. They are not used as chemical model features.

## 4. Molecular Representation

The model converts each SMILES string into several complementary feature blocks.

### 4.1 Morgan Fingerprints

The notebook generates multiple Morgan fingerprint variants:

```text
Morgan bit fingerprint, radius = 2, 2048 bits
Morgan bit fingerprint, radius = 3, 2048 bits
Morgan count fingerprint, radius = 2, 2048 dimensions
```

These fingerprints encode local atomic environments and substructures. Radius 2 captures common ECFP4-like structural patterns, while radius 3 captures larger local environments. Count fingerprints preserve repeated substructure frequency, which can matter for repeated aromatic groups, halogens, and functional motifs.

### 4.2 RDKit Physicochemical Descriptors

The model computes RDKit descriptors that describe global molecular properties, including:

```text
Molecular weight
LogP
Topological polar surface area
Heavy atom count
Rotatable bond count
Ring count
Aromatic ring count
Aromatic carbocycle count
Aromatic heterocycle count
Hydrogen-bond donor count
Hydrogen-bond acceptor count
Fraction sp3 carbon
Stereocenter count
Formal charge
```

These descriptors are especially important for PXR because its ligand-binding pocket is large, flexible, and hydrophobic. The model therefore explicitly captures size, lipophilicity, aromaticity, polarity, and molecular flexibility.

### 4.3 PXR-Aware SMARTS Features

The workflow includes PXR-aware SMARTS features. For each SMARTS pattern, two features are generated:

```text
n_<pattern_name>    = substructure count
has_<pattern_name>  = binary presence/absence indicator
```

The SMARTS features cover several PXR-relevant structural classes:

```text
Fused aromatic systems
Biaryl and triphenyl-like systems
tert-Butyl and isopropyl groups
Trifluoromethyl groups
Sulfonamide, phosphonate, amide, carbamate, and urea motifs
Macrocyclic ring patterns
Pyridine, imidazole, and pyrimidine rings
```

These features make the model more chemically interpretable than a purely hashed fingerprint representation.

## 5. SAR-Informed Feature Engineering

A key improvement in this version is the use of SAR rules derived from 99 analog-hit molecular pairs. These rules are encoded as additional numerical features and are also used in the post-model calibration step.

### 5.1 Encoded SAR Rules

The model encodes the following rule groups:

#### Hydrophobic aromatic enrichment

PXR tends to favor large, lipophilic, flat, multi-ring aromatic systems. The model therefore encodes:

```text
Number of aromatic rings
Number of aromatic carbocycles
Presence of biaryl systems
Presence of fused aromatic systems
Presence of naphthalene-like motifs
Presence of fluorene-like motifs
Aromatic halogen count
Hydrophobic aromatic score
```

#### Charged group penalty

Charged or highly ionizable groups are generally unfavorable for the hydrophobic PXR pocket. The model encodes:

```text
Free carboxylic acid indicator
Acylsulfonamide indicator
Strong acid indicator
Positive formal charge indicator
Net formal charge
Charged group penalty score
```

#### COOH-to-amide transformation logic

The SAR analysis identified the conversion from a free carboxylic acid to an amide with a lipophilic aromatic partner as one of the most reliable positive transformations. The model encodes:

```text
Amide count
Aromatic amide indicator
Amide plus aromatic partner indicator
COOH and amide coexistence indicator
COOH penalty score
Amide aromatic bonus score
```

#### Small-molecule activity floor

Very small molecules often cannot fill the large PXR pocket and tend to cluster near the weak-activity floor. The model encodes:

```text
MW < 150 indicator
MW < 200 indicator
Single-aromatic-small-molecule indicator
Small-molecule floor risk score
```

#### Heteroaromatic polarity penalty

Several SAR comparisons indicate that replacing phenyl or thiophene-like hydrophobic aromatic systems with more polar heteroaromatics can reduce activity. The model encodes:

```text
Pyridine count
Isoxazole count
Triazole count
Tetrazole count
N-rich aromatic score
Phenyl-to-pyridine penalty proxy
Phenyl-to-isoxazole penalty proxy
```

#### Dead scaffold and weak anchor patterns

Several recurrent weak scaffolds were identified in the SAR analysis. The model encodes approximate substructure matches for known weak anchors, including:

```text
4-methoxyphenol-like anchor
Benzyloxyphenol-like anchor
Simple benzoic-acid-like anchor
Small aliphatic acid anchor
Aminopyridine anomaly indicator
Dead-scaffold risk proxy
```

These rules are not used as hard labels. They are soft model features and conservative calibration signals.

## 6. Assay-Uncertainty Weighting

The training labels include uncertainty fields when available:

```text
pEC50 standard error
pEC50 confidence interval lower bound
pEC50 confidence interval upper bound
```

The model uses these values to construct sample weights. Samples with narrower confidence intervals or smaller standard errors are assigned higher training weight. Weights are clipped to avoid a small number of low-uncertainty samples dominating training.

The general form is:

```text
weight = 1 / (uncertainty^2 + epsilon)
```

A conservative additional adjustment is applied for compounds flagged as likely assay artifacts or structurally unreliable anchors. This adjustment is intentionally mild because the challenge evaluates the primary experimental readout, including possible assay-specific effects.

## 7. Cross-Validation Strategy

The notebook supports scaffold-aware validation. Bemis-Murcko scaffolds are computed from SMILES strings and used as grouping variables when possible.

The validation strategy is:

```text
Preferred: GroupKFold by molecular scaffold
Fallback: KFold if scaffold grouping is not feasible
```

Scaffold-based validation is stricter than random splitting because structurally similar analogs are less likely to be split across training and validation folds. This provides a more realistic estimate of model generalization to novel or partially novel chemical series.

For each fold, nearest-neighbor SAR calibration is applied in a leakage-safe way: validation molecules search only among that fold's training molecules, never among validation molecules themselves.

## 8. Model Ensemble

The notebook trains multiple models when the required packages are available. The ensemble can include:

```text
LightGBM Regressor
XGBoost Regressor
ExtraTrees Regressor
RandomForest Regressor
Ridge Regression
```

Tree-based models are expected to perform well on sparse fingerprints and nonlinear descriptor interactions. Ridge regression provides a simpler linear baseline that can stabilize ensemble behavior.

Model predictions are averaged using predefined weights. If a package is missing, that model is skipped automatically and the remaining weights are renormalized.

Default ensemble logic:

```text
LightGBM      high weight when available
XGBoost       high weight when available
ExtraTrees    medium weight
RandomForest  medium weight
Ridge         low stabilizing weight
```

The final model prediction is the weighted average of all available fold-averaged model outputs.

## 9. Nearest-Neighbor SAR Calibration

After the ensemble produces base pEC50 predictions, the notebook applies a conservative SAR calibration step.

For each test molecule:

1. Compute Morgan fingerprint similarity to all training molecules.
2. Identify the nearest training neighbor by Tanimoto similarity.
3. Compare SAR features between the test molecule and nearest neighbor.
4. Apply a small rule-based adjustment if the structural difference matches a high-confidence SAR pattern.

The adjustment is intentionally conservative and is blended with the data-driven model prediction:

```text
final_prediction = (1 - alpha) * ensemble_prediction + alpha * SAR_calibrated_prediction
```

The default calibration weight is mild. This avoids overcorrecting uncertain cases, while still using strong SAR evidence when the nearest neighbor is highly informative.

### 9.1 Example Calibration Rules

The post-processing function includes rules such as:

```text
If test has COOH and nearest hit does not:
    apply negative adjustment, especially if test is small or polar.

If test has amide plus lipophilic aromatic partner and nearest hit has COOH:
    apply positive adjustment.

If test is very small and predicted above the likely activity floor:
    shrink prediction toward a lower pEC50 range.

If test has a large hydrophobic aromatic system absent in nearest neighbor:
    apply mild positive adjustment.

If test introduces many polar heteroaromatic atoms:
    apply mild negative adjustment.
```

These rules are not meant to replace the learned model. They are used to correct predictable failure modes of fingerprint-based regression.

## 10. Prediction Output

The notebook writes outputs to:

```text
outputs_sar_optimized/
```

Expected output files:

```text
submission_sar_optimized.csv
```

Main challenge submission file. It contains:

```text
Molecule Name
pEC50
```

Additional diagnostic files:

```text
test_predictions_detail_sar_optimized.csv
```

Contains base prediction, SAR-calibrated prediction, nearest-neighbor similarity, nearest-neighbor pEC50, and selected SAR feature values.

```text
oof_predictions_sar_optimized.csv
```

Contains out-of-fold predictions and validation diagnostics.

```text
cv_model_scores.csv
```

Contains cross-validation scores for each model and fold.

## 11. Performance Evaluation

The notebook evaluates model performance using cross-validation on the training set. Reported metrics include:

```text
RMSE
MAE
Pearson correlation
Spearman correlation
```

No leaderboard or blinded test labels are used during model training or validation.

If the notebook has not yet been run in the final environment, this report should be considered a **method report**, not a completed empirical performance report. After execution, the actual cross-validation values should be copied from:

```text
outputs_sar_optimized/cv_model_scores.csv
```

Recommended reporting format after running:

```text
OOF RMSE:      <fill after execution>
OOF MAE:       <fill after execution>
OOF Pearson:   <fill after execution>
OOF Spearman:  <fill after execution>
```

This avoids inventing performance numbers, a habit apparently popular among both humans and badly supervised models.

## 12. Model Strengths

The main strengths of this approach are:

1. **Chemically interpretable features**  
   The model does not rely only on hashed fingerprints. It includes explicit descriptors and SAR rule features.

2. **PXR-specific inductive bias**  
   The feature set reflects known PXR preferences for large, hydrophobic, aromatic ligands and penalties for highly charged structures.

3. **Robust ensemble learning**  
   Multiple model families reduce dependence on one learner's bias.

4. **Uncertainty-aware training**  
   Label standard error and confidence interval information are used to reduce the impact of noisy fitted pEC50 values.

5. **Leakage-safe SAR calibration**  
   Nearest-neighbor correction is applied fold-safely during validation and uses all training molecules only for final test prediction.

6. **Automatic fallback behavior**  
   If LightGBM or XGBoost is unavailable, the notebook still runs with scikit-learn models.

## 13. Model Limitations

The model also has several limitations:

1. **No 3D docking or conformational modeling**  
   PXR has a large flexible binding pocket, and some SAR effects depend on 3D shape, stereochemistry, and conformational fit. Fingerprints and 2D descriptors cannot fully capture these.

2. **Limited stereochemical sensitivity**  
   Morgan fingerprints may miss some stereochemical or E/Z differences. The SAR analysis observed cases where highly similar or fingerprint-identical molecules still differed by around 0.5 pEC50 units.

3. **Rule-based calibration can be imperfect**  
   SAR rules are useful but not universally additive. Multiple simultaneous structural changes can interact nonlinearly.

4. **Assay artifacts remain difficult**  
   Some molecules may produce non-PXR-specific reporter effects. The model can flag likely artifact patterns, but the challenge target is still the primary assay readout.

5. **Scaffold extrapolation is hard**  
   Even scaffold validation may not fully represent the difficulty of truly novel chemotypes in the blinded test set.

## 14. Reproducibility

### 14.1 Required Python Packages

Core requirements:

```text
numpy
pandas
scikit-learn
rdkit
```

Recommended packages:

```text
lightgbm
xgboost
catboost
```

RDKit installation is usually most stable through conda:

```bash
conda install -c conda-forge rdkit lightgbm xgboost -y
```

### 14.2 How to Run

Place the notebook in the project directory:

```text
PXR(317)/
  data/
  PXR_SAR_optimized_prediction_notebook.ipynb
```

Open Jupyter Notebook and run all cells from top to bottom.

The final submission will be written to:

```text
outputs_sar_optimized/submission_sar_optimized.csv
```

## 15. Summary

This model combines molecular fingerprints, RDKit descriptors, PXR-aware SMARTS features, SAR-derived chemical rules, uncertainty-aware weighting, scaffold-based validation, and ensemble regression. The key design principle is to keep the model data-driven while encoding enough medicinal chemistry knowledge to avoid obvious QSAR failure modes.

In practical terms, the model predicts pEC50 from SMILES by learning broad structure-potency relationships and then lightly correcting predictions using high-confidence SAR transformations such as COOH penalties, amide-aromatic bonuses, hydrophobic aromatic enrichment, small-molecule activity floors, and known weak-scaffold anchors.

The resulting workflow is suitable as an interpretable and competition-ready PXR pEC50 prediction pipeline.
