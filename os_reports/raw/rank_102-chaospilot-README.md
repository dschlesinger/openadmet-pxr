# PXR

This project predicts PXR activation potency (`pEC50`) from molecular structures.  
The main idea is to combine a strong descriptor-based regression model with a local analog / activity-cliff correction model.

## Method Overview

The model uses three kinds of information:

1. **Global molecular features**
   - Morgan fingerprints
   - Chiral Morgan fingerprints
   - Morgan count fingerprints
   - RDKit molecular descriptors

2. **Direct pEC50 regression**
   - Several tree-based regression models are trained on the molecular features.
   - The models learn the overall relationship between chemical structure and `pEC50`.
   - Their predictions are blended to produce a stable main prediction.

3. **Pair-delta / activity-cliff correction**
   - For each molecule, the pipeline finds similar molecules by Morgan fingerprint Tanimoto similarity.
   - Instead of only predicting absolute `pEC50`, it also learns local changes:
     ```text
     delta = pEC50(query molecule) - pEC50(neighbor molecule)
     ```
   - This helps the model capture analog-series effects and activity cliffs, where small structural changes may cause large potency changes.

Finally, the direct regression prediction and the pair-delta prediction are blended together.  
A conservative blended submission is generated as the main output.

## Why This Method

PXR activity is difficult to predict because structurally similar compounds can sometimes have very different potency.  
A pure global model is usually stable, but it may miss local SAR changes.  
The pair-delta branch adds local analog information, while the final blend keeps the prediction from becoming too aggressive.

## Pipeline

The full pipeline has five stages:

```text
1. make splits
2. generate molecular features
3. train boosting ensemble
4. train pair-delta / activity-cliff model
5. blend and calibrate predictions
```

## Notes

This model is designed to be conservative.  
The boosting branch provides the stable base prediction, while the pair-delta branch contributes local SAR correction.  
If the pair-delta model is noisy, the final blend should still remain close to the stable global prediction.
