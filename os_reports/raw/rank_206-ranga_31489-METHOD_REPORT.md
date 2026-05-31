# Method Report: PXR pEC50 Prediction

## Overview

This report summarizes the modeling workflow used to generate the final PXR activity submission. The task was to predict continuous `pEC50` values for a blinded test set of compounds. The final submission model is an uncertainty-weighted XGBoost regressor trained on a `v2` feature matrix combining 2D molecular descriptors, charge-derived features, and Morgan fingerprint bits.

The final submission file is:

- [xgboost_champion_fulltrain_submission.csv](/home/avranga1008/pxr-challenge/activity/outputs/finalist_predictions/xgboost_champion_fulltrain_submission.csv)

## Data

The primary labeled training matrix used for final model development was:

- [train_modeling_matrix_v2_with_fingerprints.csv](/home/avranga1008/pxr-challenge/activity/data/features/train_modeling_matrix_v2_with_fingerprints.csv)

This table contains:

- `4140` labeled training compounds
- target column `pEC50`
- assay uncertainty fields including `pEC50_std.error (-log10(molarity))`
- split annotations for `train`, `val`, and `holdout`
- descriptor features (`desc_...`)
- charge features
- Morgan fingerprint features (`fp_morgan_r2_....`)

The blinded test set was taken from:

- [pxr-challenge_TEST_BLINDED.csv](/home/avranga1008/pxr-challenge/activity/data/raw/pxr-challenge_TEST_BLINDED.csv)

Descriptor and charge features for the test compounds came from:

- [test_scikit_mol_features.csv](/home/avranga1008/pxr-challenge/activity/data/features/test_scikit_mol_features.csv)

Fingerprint bits for the test set were generated from the blinded test `SMILES` using the same Morgan radius and bit length as the training `v2` matrix.

## Feature Representation

The final feature set was `desc_plus_charge_fp`, consisting of:

- `217` descriptor columns
- `6` charge-derived columns
- `2048` Morgan fingerprint bits

Total feature count:

- `2271`

Morgan fingerprints were generated with:

- radius `2`
- fingerprint size `2048`

The rationale for this representation was that descriptors and charge terms provide compact physicochemical summaries, while binary fingerprints preserve higher-resolution substructure information. A matched ablation showed that adding fingerprints improved predictive performance relative to the same descriptor-plus-charge model without fingerprint bits.

## Uncertainty Handling

Target-quality analysis showed clear heteroscedasticity: lower-potency compounds were systematically noisier than higher-potency compounds. Instead of fitting all labels equally, the final models used uncertainty-aware sample weighting.

The weighting scheme was:

```text
w_i = 1 / (SE_i^2 + sigma_model^2)
```

where:

- `SE_i` is `pEC50_std.error (-log10(molarity))`
- `sigma_model = 0.60`

This `total_error` weighting was chosen instead of pure inverse-variance weighting because it incorporates both experimental uncertainty and a model-error floor, preventing a small number of very precise measurements from dominating training.

## Model Development

### Baseline Models

Initial baselines used the earlier `v1` matrix and included ridge regression, random forest, and histogram gradient boosting. Random forest was the strongest early baseline among the saved `v1` runs.

### HGB Development on `v2`

The next branch moved to the `v2` matrix with fingerprint features and tested `HistGradientBoostingRegressor` under `total_error` weighting. A constrained Optuna sweep improved the `v2 + fingerprints` HGB model relative to descriptor-only variants and confirmed that fingerprints added useful signal.

Best `v2 + fingerprints` HGB run:

- [pxr_v2_with_fingerprints_hgb_sweep2__hist_gbm__desc_plus_charge_fp__total_error](/home/avranga1008/pxr-challenge/activity/outputs/model_runs/pxr_v2_with_fingerprints_hgb_sweep2__hist_gbm__desc_plus_charge_fp__total_error)
- validation RMSE `0.809598`
- holdout RMSE `0.818593`

Matched no-fingerprint ablation:

- [pxr_v2_without_fingerprints__hist_gbm__desc_plus_charge__total_error](/home/avranga1008/pxr-challenge/activity/outputs/model_runs/pxr_v2_without_fingerprints__hist_gbm__desc_plus_charge__total_error)
- validation RMSE `0.816075`
- holdout RMSE `0.827677`

This established that fingerprint features should be retained.

### XGBoost Champion

After the feature representation was stabilized, a fixed XGBoost baseline was evaluated on the same `v2 + desc_plus_charge_fp + total_error` setup. This produced a large improvement over the HGB champion and over the best saved random-forest baseline.

Primary XGBoost run:

- [pxr_v2_with_fingerprints__xgboost__desc_plus_charge_fp__total_error](/home/avranga1008/pxr-challenge/activity/outputs/model_runs/pxr_v2_with_fingerprints__xgboost__desc_plus_charge_fp__total_error)
- validation RMSE `0.750065`
- holdout RMSE `0.784433`

Confirmation run with a different seed:

- [pxr_v2_with_fingerprints__xgboost__desc_plus_charge_fp__total_error__seed123](/home/avranga1008/pxr-challenge/activity/outputs/model_runs/pxr_v2_with_fingerprints__xgboost__desc_plus_charge_fp__total_error__seed123)
- validation RMSE `0.752048`
- holdout RMSE `0.778844`

The small seed-to-seed variation supported promoting XGBoost as the final model family.

An additional constrained XGBoost Optuna sweep was also tested:

- [pxr_v2_with_fingerprints_xgboost_sweep1__xgboost__desc_plus_charge_fp__total_error](/home/avranga1008/pxr-challenge/activity/outputs/model_runs/pxr_v2_with_fingerprints_xgboost_sweep1__xgboost__desc_plus_charge_fp__total_error)

That sweep reduced overfitting but did not outperform the fixed XGBoost baseline on holdout, so the fixed model was retained as the champion.

## Final Model

The final submission model was trained on all labeled rows in the `v2` training matrix using the confirmed fixed XGBoost configuration:

- `n_estimators = 1200`
- `learning_rate = 0.03`
- `max_depth = 6`
- `min_child_weight = 8.0`
- `subsample = 0.8`
- `colsample_bytree = 0.5`
- `reg_alpha = 0.0`
- `reg_lambda = 2.0`
- `gamma = 0.0`
- `tree_method = "hist"`
- `random_state = 123`

The final full-training submission artifacts were generated with:

- [generate_xgboost_champion_submission.py](/home/avranga1008/pxr-challenge/activity/scripts/generate_xgboost_champion_submission.py)

The run metadata for the full-train submission is:

- [xgboost_champion_fulltrain_run_config.json](/home/avranga1008/pxr-challenge/activity/outputs/finalist_predictions/xgboost_champion_fulltrain_run_config.json)

## Submission Format

The submission CSV contains exactly three columns:

- `Molecule Name`
- `SMILES`
- `pEC50`

Final submission file:

- [xgboost_champion_fulltrain_submission.csv](/home/avranga1008/pxr-challenge/activity/outputs/finalist_predictions/xgboost_champion_fulltrain_submission.csv)

## Conclusion

The final workflow converged on three important conclusions:

1. The PXR endpoint is heteroscedastic and benefits from uncertainty-aware weighting.
2. Morgan fingerprint features add useful predictive signal beyond descriptors plus charge features alone.
3. XGBoost on the `v2 + desc_plus_charge_fp + total_error` setup outperformed both the tuned HGB branch and the saved random-forest baselines.

For these reasons, the final submission was generated with the fixed XGBoost champion trained on all labeled training data.
