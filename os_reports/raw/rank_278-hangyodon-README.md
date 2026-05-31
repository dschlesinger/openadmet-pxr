# Baseline Method Report: OpenADMET PXR Induction Challenge

## 1. Overview

This repository documents the baseline method used for the OpenADMET PXR induction challenge. The task is to predict compound activity, reported as `pEC50`, for the official test compounds.

The baseline follows a simple and reproducible cheminformatics machine-learning workflow. It uses molecular structure information derived from SMILES, trains standard regression models, evaluates them with cross-validation, and exports predictions in the official submission format.

This method is intended as a basic reference pipeline. It provides a stable starting point for comparison with stronger models, auxiliary assay features, ensembling strategies, and uncertainty quantification methods.

## 2. Data Used

The method uses the official challenge-provided data, including:

- the training set with compound SMILES and observed `pEC50` values;
- the official test set with compound SMILES.

No private, proprietary, or non-public external data were used. The test-set activity labels were not used at any stage.

## 3. Molecular Preprocessing

The input molecular representation is based on canonical SMILES.

The preprocessing procedure includes:

1. loading the official training and test files;
2. reading the SMILES column;
3. parsing each SMILES string with RDKit;
4. converting valid molecules to canonical SMILES;
5. handling invalid or missing molecules safely;
6. keeping the feature matrix aligned with the original compound order.

This step ensures that the same molecule is represented consistently and that the final predictions match the official test-set row order.

## 4. Feature Construction

The baseline mainly uses structure-based molecular features.

### 4.1 Morgan Fingerprints

Morgan fingerprints are used as the primary molecular representation. These fingerprints encode circular atom environments and are widely used in ligand-based machine learning.

They are useful for this task because biological activity often depends on local chemical substructures.

### 4.2 Optional Descriptor Features

In some baseline runs, simple RDKit descriptors may also be included. These descriptors capture basic molecular properties such as molecular weight, polarity, ring count, and other physicochemical or topological information.

The baseline can run with fingerprints alone, or with fingerprints plus descriptors.

## 5. Model Training

The baseline trains standard regression models to predict `pEC50` from molecular features.

Typical baseline models include:

- Random Forest regression;
- Extra Trees regression;
- Ridge regression;
- K-nearest-neighbor regression.

The exact model choice can be adjusted, but the baseline does not rely on complex neural networks or external pretrained models.

Tree-based models are useful because they can capture nonlinear relationships between structural features and activity. Ridge regression gives a simple linear reference. K-nearest-neighbor regression provides a similarity-based reference.

## 6. Cross-Validation

Model performance is estimated using cross-validation on the training set.

For each fold:

1. the model is trained on the training portion;
2. predictions are generated for the validation portion;
3. validation performance is recorded;
4. predictions are generated for the official test set.

The validation metrics considered include:

- Mean Absolute Error (MAE);
- Root Mean Squared Error (RMSE);
- R²;
- Spearman correlation;
- Kendall correlation.

Cross-validation is used to estimate how well the model may generalize to unseen compounds.

## 7. Prediction Strategy

The simplest baseline uses one trained regression model to generate test predictions.

A slightly stronger baseline averages predictions across cross-validation folds. In this setting, each fold model predicts the test set, and the final prediction is the mean of the fold-level predictions.

This fold-averaging strategy reduces sensitivity to one specific train-validation split.

Predictions may be clipped to a reasonable `pEC50` range based on the training data distribution. This prevents unrealistic extreme values outside the observed activity range.

## 8. Final Submission

The final output is a CSV file following the official submission format.

Before submission, the following checks are performed:

1. the number of rows matches the official test set;
2. all required columns are present;
3. column names match the official format;
4. row order matches the official test file;
5. predicted `pEC50` values are numeric;
6. there are no missing prediction values.

## 9. Difference from the Better Baseline

This baseline is intentionally simple.

Compared with the better baseline, it usually uses fewer feature groups, fewer auxiliary assay inputs, and a simpler prediction strategy. It may rely mostly on molecular fingerprints and standard regression models.

The better baseline improves on this by adding more feature types, using assay-derived auxiliary features, combining multiple model families, and using validation-based ensembling.

Therefore, this baseline should be interpreted as a reference method rather than the final optimized solution.

## 10. Reproducibility

The workflow can be reproduced using the following general steps:

1. install the required Python packages, including RDKit, NumPy, pandas, and scikit-learn;
2. place the official challenge data in the expected data directory;
3. load training and test SMILES;
4. generate Morgan fingerprints and optional RDKit descriptors;
5. train baseline regression models with cross-validation;
6. average fold-level test predictions when applicable;
7. export the final CSV in the official format.

## 11. Limitations

The main limitations of this baseline are:

- it relies mostly on structure-derived features;
- it does not fully use all available biological assay information;
- it does not explicitly model prediction uncertainty;
- it may perform poorly on compounds far from the training chemical space;
- it may not capture assay noise, batch effects, or complex biological mechanisms.

These limitations motivate the development of better baseline models, auxiliary-feature models, ensembling methods, and uncertainty-aware prediction methods.

## 12. Summary

The baseline method provides a simple, transparent, and reproducible workflow for predicting PXR induction activity. It uses official challenge data, RDKit molecular features, standard regression models, cross-validation, and official-format CSV export.

It is designed as a clean starting point for comparison with more advanced methods.
