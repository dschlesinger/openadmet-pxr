# Method Report: OpenADMET PXR Induction Challenge

## 1. Task Overview

This submission addresses the activity prediction task for the OpenADMET PXR induction challenge. The target variable is `pEC50`, a continuous measure of compound activity. The final submission reports predicted `pEC50` values for the official test compounds, together with an uncertainty estimate for each prediction.

The goal of the method is not only to produce point predictions, but also to provide a simple and interpretable estimate of prediction uncertainty. The uncertainty estimate is intended to reflect how stable and reliable each prediction is, based on model disagreement, cross-validation residual behavior, and chemical similarity to the training data.

## 2. Data Used

The method uses the official challenge data provided for Phase 1, including:

- Training compounds with canonical SMILES and observed `pEC50` values.
- Official test compounds with canonical SMILES.
- High-throughput screening response features where available.
- Assay-related auxiliary information, including primary assay and counter-assay response profiles when available.

No test-set activity labels were used during model training, model selection, or uncertainty construction.

## 3. Molecular Representation

Each compound was represented using structure-based molecular features derived from canonical SMILES. The main feature groups were:

1. **Morgan fingerprints**  
   Circular fingerprints were calculated from molecular graphs to encode local chemical environments.

2. **MACCS keys**  
   MACCS structural keys were used as a compact binary representation of common chemical substructures.

3. **RDKit molecular descriptors**  
   Standard physicochemical and topological descriptors were calculated using RDKit.

4. **Assay auxiliary features**  
   When available, HTS response and related assay-derived features were included as additional predictors. These features help capture biological response patterns that may not be fully encoded by molecular structure alone.

Invalid SMILES were handled defensively. Molecules that could not be parsed were either excluded from feature construction or assigned missing/default feature values, depending on the feature type.

## 4. Model Training

The prediction model was built as an ensemble of several standard regression models. The core models included tree-based models and linear or distance-based baselines, such as:

- ExtraTreesRegressor
- RandomForestRegressor
- Ridge regression
- K-nearest-neighbor regression
- Additional gradient-boosting or baseline models when available in the working pipeline

The training process used cross-validation to generate out-of-fold predictions. These out-of-fold predictions were used to estimate model performance and to construct ensemble weights.

The final prediction was obtained by combining the predictions of multiple base models. The ensemble combination was based on validation behavior, using either weighted averaging or non-negative least squares on out-of-fold predictions. This avoids relying on a single model family and reduces sensitivity to one model's inductive bias.

Predictions were clipped to a chemically and biologically reasonable `pEC50` range based on the observed training distribution.

## 5. Use of Auxiliary Assay Information

The dataset contains assay-related information that is useful because PXR induction is not determined only by simple structural similarity. Biological response profiles from earlier assays can provide indirect evidence about compound behavior.

The auxiliary assay features were therefore used as additional inputs alongside molecular descriptors. The motivation is that two molecules with different structures may still show related assay response patterns, and two molecules with similar structures may behave differently if their assay profiles diverge.

This auxiliary information was used only as predictor information and not as hidden activity labels.

## 6. Uncertainty Estimation

The submitted uncertainty value was constructed from several complementary signals.

### 6.1 Ensemble Disagreement

For each test compound, predictions from multiple base models were collected. Larger disagreement among models was interpreted as higher predictive uncertainty. This is a direct model-based uncertainty signal: if several reasonable models produce different answers, the final prediction should be treated as less reliable.

### 6.2 Cross-Validation Residual Scale

Out-of-fold residuals on the training set were used to estimate the typical prediction error scale. This provides a global calibration reference for uncertainty. The final uncertainty should not be smaller than the empirical error level observed during validation.

### 6.3 Chemical Applicability Domain

Chemical similarity between each test compound and the training compounds was used as an applicability-domain signal. A test compound that is far from the training set in fingerprint space is more likely to be difficult to predict. Such compounds were assigned larger uncertainty.

Similarity was estimated using fingerprint-based nearest-neighbor similarity, such as Tanimoto similarity on Morgan fingerprints.

### 6.4 Final Uncertainty Score

The final uncertainty estimate combined the following components:

- model disagreement across ensemble members;
- validation residual scale from cross-validation;
- distance or dissimilarity from the training chemical space.

The resulting uncertainty score was normalized to a practical range and included in the final submission file.

This uncertainty estimate should be interpreted as a relative confidence score rather than a fully calibrated predictive standard deviation. Larger values indicate lower confidence in the predicted `pEC50`.

## 7. Validation Strategy

Model performance was evaluated using cross-validation on the training set. The main validation metrics considered were:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of determination (R²)
- Spearman rank correlation
- Kendall rank correlation

Cross-validation was used both for model comparison and for ensemble construction. This reduces the risk of overfitting model weights to the full training set.

## 8. Final Submission

The final submission contains the official test compounds and the following prediction outputs:

- predicted `pEC50`;
- associated uncertainty estimate.

The column names and ordering were aligned with the official submission format. The final file was checked to ensure that the required columns were present and that the number of rows matched the official test set.

## 9. Reproducibility Notes

The workflow can be summarized as follows:

1. Load official training and test files.
2. Standardize and parse canonical SMILES.
3. Generate molecular fingerprints and descriptors.
4. Add available assay auxiliary features.
5. Train multiple regression models using cross-validation.
6. Generate out-of-fold predictions and test predictions.
7. Combine base models using validation-based ensemble weighting.
8. Estimate uncertainty using model disagreement, residual scale, and applicability-domain distance.
9. Export the final submission CSV in the official format.

The method uses only official challenge-provided data and standard open-source cheminformatics and machine-learning tools, including RDKit, NumPy, pandas, scikit-learn, and related Python packages.
