# Model Report: 
## 1. Overview

This report describes a primary-only machine learning baseline developed for the OpenADMET PXR Activity Prediction track. The objective of this model is to predict the **pEC50** value of blinded test compounds from their molecular structures represented as SMILES strings.

The model uses only the official primary PXR dose-response training data. No proprietary data, external bioactivity databases, counter-assay labels, single-concentration labels, or structure-track information were used in this baseline submission.

## 2. Prediction Task

The task is formulated as a molecular regression problem:

```text
Input:  Molecular structure represented by SMILES
Output: Predicted primary PXR induction pEC50
```

The target variable, **pEC50**, is defined as:

```text
pEC50 = -log10(EC50 in molar concentration)
```

A larger pEC50 value indicates that a compound reaches half-maximal induction at a lower concentration and is therefore more potent in the primary PXR induction assay.

## 3. Data Used

### 3.1 Primary Training Data

The primary training set contains 4,139 compounds. Each row corresponds to one compound with a fitted dose-response result from the primary PXR induction assay. The main training target is `pEC50`. Additional primary assay fields such as Emax estimates and uncertainty fields were retained during preprocessing, but this baseline model only predicts pEC50.

### 3.2 Blinded Test Data

The blinded activity test set contains 513 compounds. Only `Molecule Name` and `SMILES` are provided. No pEC50 or Emax labels are available for the test set.

### 3.3 Data Not Used in This Baseline

The following official files were processed or inspected during the project, but were not used in this primary-only baseline model:

- `pxr-challenge_counter-assay_TRAIN.csv`
- `pxr-challenge_single_concentration_TRAIN.csv`
- `pxr-challenge_structure_TEST_BLINDED.csv`

The counter-assay data can be used in future versions to model non-PXR-specific reporter signals or artifact risk. The single-concentration data can be used as weak supervision for concentration-dependent induction trends. The structure-track data was not used because this submission is for the activity prediction track.

## 4. Data Preprocessing

### 4.1 SMILES Standardization

All molecules were processed using RDKit. For each SMILES string, the following steps were applied:

1. Strip leading and trailing whitespace.
2. Parse the molecule using RDKit.
3. Convert valid molecules into canonical SMILES.
4. Mark invalid molecules if parsing fails.
5. Compute basic molecular quality-control fields such as heavy atom count.

All primary training and test molecules were successfully parsed by RDKit. No invalid SMILES were found.

### 4.2 Training Set Filtering

The primary training set was filtered using two criteria:

```text
mol_ok == True
pEC50 is not missing
```

All 4,139 primary training compounds passed these checks. Therefore, no primary training compounds were removed during this baseline preprocessing step.

### 4.3 Duplicate Check

Canonical SMILES were used to check for exact duplicate molecular structures. No duplicate canonical SMILES were found in the primary training set.

The blinded test set was also compared against the primary training set by canonical SMILES. No exact train-test duplicate structures were found.

### 4.4 Sample Weighting

A sample reliability weight was generated from the pEC50 standard error. The purpose was to assign lower training weight to compounds with less certain dose-response fits and higher weight to compounds with more stable pEC50 estimates.

The baseline weighting formula was:

```text
raw_weight = 1 / (pEC50_se^2 + 0.05^2)
```

The weights were then normalized by their mean and clipped to avoid extreme values:

```text
sample_weight = clip(raw_weight / mean(raw_weight), 0.2, 5.0)
```

These sample weights were used during model fitting when the regressor supported weighted training.

## 5. Molecular Feature Engineering

The final input feature matrix contained **2,070 molecular features** for each compound.

### 5.1 Morgan Fingerprints

A 2,048-bit Morgan fingerprint was generated for each molecule using RDKit:

```text
Morgan fingerprint radius = 2
Number of bits = 2048
```

This corresponds to an ECFP4-like circular fingerprint representation. It captures local molecular substructures and is commonly used in ligand-based molecular property prediction.

### 5.2 RDKit 2D Molecular Descriptors

In addition to fingerprints, a set of 2D RDKit descriptors was computed for each compound. These descriptors describe global physicochemical and topological properties such as molecular weight, lipophilicity, polar surface area, hydrogen bonding capacity, ring count, aromaticity, rotatable bonds, and molecular complexity.

Missing descriptor values were imputed using the median value from the training set. Descriptor scaling was applied consistently using training-set statistics and then applied to the test set.

### 5.3 Final Feature Matrix

The final model input was constructed by concatenating:

```text
2048-bit Morgan fingerprint + RDKit 2D descriptors
```

The resulting matrices had the following dimensions:

```text
Training feature matrix: 4139 × 2070
Test feature matrix:      513 × 2070
```

## 6. Cross-Validation Strategy

Two validation strategies were used to evaluate model performance before submission.

### 6.1 Random 5-Fold Cross-Validation

The training set was randomly split into five folds. This setting measures general interpolation performance but may be optimistic because structurally similar molecules can appear in both training and validation folds.

### 6.2 Scaffold 5-Fold Cross-Validation

A scaffold-based split was also used. Bemis-Murcko scaffolds were computed from canonical SMILES, and compounds sharing the same scaffold were assigned to the same fold. This split is more stringent and better reflects performance on new molecular cores.

## 7. Models Trained

Three regression models were trained:

1. Ridge regression
2. Random forest regression
3. Extra trees regression

For each cross-validation fold, each model was trained on the training folds and evaluated on the held-out fold. Predictions were clipped to the observed pEC50 range of the primary training set to prevent unrealistic extrapolation.

The final baseline prediction was generated using a simple mean ensemble:

```text
blend_mean = mean(ridge prediction, random forest prediction, extra trees prediction)
```

This ensemble was selected because it achieved the best validation performance across both random and scaffold cross-validation.

## 8. Local Validation Results

### 8.1 Random 5-Fold Cross-Validation

| Model | MAE | RAE | R2 | Spearman ρ | Kendall τ |
|---|---:|---:|---:|---:|---:|
| Ridge | 0.5972 | 0.6564 | 0.4693 | 0.6853 | not reported |
| Random Forest | 0.5504 | 0.6050 | 0.5432 | 0.7189 | not reported |
| Extra Trees | 0.5702 | 0.6267 | 0.5239 | 0.7125 | not reported |
| Mean Ensemble | **0.5329** | **0.5857** | **0.5597** | **0.7350** | not reported |

### 8.2 Scaffold 5-Fold Cross-Validation

| Model | MAE | RAE | R2 | Spearman ρ | Kendall τ |
|---|---:|---:|---:|---:|---:|
| Ridge | 0.6035 | 0.6633 | 0.4605 | 0.6784 | not reported |
| Random Forest | 0.5533 | 0.6082 | 0.5402 | 0.7160 | not reported |
| Extra Trees | 0.5757 | 0.6327 | 0.5172 | 0.7089 | not reported |
| Mean Ensemble | **0.5377** | **0.5910** | **0.5539** | **0.7298** | not reported |

The mean ensemble was consistently better than each individual model in both random and scaffold validation. The scaffold validation results were close to the random validation results, suggesting that the baseline generalizes reasonably well across molecular scaffolds.

## 9. Submission File

The submitted file was:

```text
submission_primary_blend.csv
```

It contained 513 rows and the following columns:

```text
SMILES, Molecule Name, pEC50
```

The submitted predictions were generated using the mean ensemble trained on the full primary training set.

## 10. Live Leaderboard Result

The submitted baseline achieved the following live leaderboard performance:

| Metric | Value |
|---|---:|
| Rank | 180 |
| MAE | 0.5741 |
| RAE | 0.7202 |
| R2 | 0.3249 |
| Spearman ρ | 0.7456 |
| Kendall τ | 0.5473 |

The leaderboard result confirms that the model provides a valid and useful primary-only baseline. The Spearman correlation is relatively strong, indicating that the model captures the ranking of compound potency reasonably well. However, the RAE and R2 indicate that the numerical pEC50 predictions are still conservative and can be improved.

## 11. Interpretation

The model appears to learn meaningful structure-activity relationships from molecular fingerprints and simple physicochemical descriptors. The ensemble improves over individual models, likely because ridge regression, random forests, and extra trees capture different aspects of the feature space.

The prediction distribution was relatively narrow compared with the full pEC50 range in the training set. This suggests a tendency toward mean regression, especially for highly potent or weakly active compounds. This conservative behavior likely contributed to the gap between local validation performance and live leaderboard performance.

## 12. Limitations

This baseline has several limitations:

1. It uses only primary training data.
2. It does not use counter-assay information to model non-PXR-specific reporter artifacts.
3. It does not use single-concentration data as weak supervision.
4. It does not include PXR-specific pharmacophore or pocket-aware features.
5. It does not use external molecular pretraining or graph neural networks.
6. It tends to shrink predictions toward the center of the training pEC50 distribution.

## 13. Planned Improvements

Future versions may improve performance by adding:

1. Gradient boosting models such as LightGBM and XGBoost.
2. PXR-aware physicochemical and SMARTS-based pharmacophore features.
3. Counter-assay-derived artifact risk features or sample reweighting.
4. Single-concentration weak-supervision features.
5. Prediction calibration to reduce mean shrinkage.
6. Scaffold-specific or nearest-neighbor correction strategies.
7. Model stacking or validation-weighted ensembling.

## 14. Reproducibility Summary

The complete baseline workflow consists of:

```text
1. Load official primary train and blinded test data.
2. Canonicalize SMILES with RDKit.
3. Generate Morgan fingerprints and RDKit descriptors.
4. Train ridge, random forest, and extra trees regressors.
5. Evaluate by random and scaffold 5-fold cross-validation.
6. Train final models on all primary training compounds.
7. Average model predictions to generate the final pEC50 submission.
```

No proprietary data were used.

