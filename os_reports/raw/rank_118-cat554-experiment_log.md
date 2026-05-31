# PXR Stability Stack

This package is an experiment for predicting PXR pEC50 by combining a strong structure-based model with a smaller block of PXR-specific stability and analog-neighbor features.

The main idea is simple:

> Use broad molecular representations to learn general activity, then add single-concentration and analog-stability signals as a controlled auxiliary layer.

---

## What this model tries to fix

A pure structural model can rank many molecules reasonably well, but it does not always know whether a structure is experimentally stable, noisy, or close to a known activity cliff.

This package therefore adds a second source of information:

1. **Single-concentration PXR screening signal**
2. **Dose-response label quality / uncertainty signal**
3. **Nearest-neighbor analog behavior from ECFP4 Tanimoto similarity**
4. **PXR-oriented substructure and physicochemical descriptors**

The goal is not to replace the main molecular model. The goal is to make the main model more aware of experimental reliability and local SAR context.

---

## Pipeline map

```text
official train/test
        │
        ├── single-concentration aggregation
        │       └── stability / uncertainty features
        │
        ├── PXR53 descriptors
        │       └── PXR-relevant motifs + physicochemical features
        │
        ├── analog-neighbor features
        │       └── ECFP4 Tanimoto local pEC50 statistics
        │
        ├── selected low-dimensional auxiliary block
        │
        └── broad molecular feature stack
                └── Morgan / MACCS / RDKit / mol2vec / HTS-response features

final feature matrix
        │
        ├── tree models
        ├── linear robust models
        ├── boosting models
        └── OOF-weighted blend

submission
```

---

## Data treatment

The package separates two different problems:

### 1. Bad dose-response labels

Some training rows have unstable dose-response fits, such as high pEC50 standard error or very wide confidence intervals. These are treated as label-quality problems and can be removed from training.

### 2. Single-screen mismatch

Some compounds disagree between single-concentration signal and fitted pEC50. These are **not automatically removed**. They may represent real SAR cliffs, assay-stage mismatch, or useful borderline behavior. Instead of deleting them, the pipeline keeps them and adds uncertainty/stability flags.

This distinction is important: the model should reduce obvious label noise, but it should not erase difficult chemistry.

---

## Feature blocks

### Broad structural block

The model feature base:

- Morgan fingerprints
- Chiral Morgan variants
- MACCS keys
- compact RDKit descriptors
- RDKit2D descriptors
- mol2vec embeddings
- HTS response features

These features provide the main activity-prediction capacity.

### Stability / PXR auxiliary block

The auxiliary block is built from:

- aggregated single-concentration measurements
- inferred stability/uncertainty values for molecules without direct overlap
- PXR53 hand-designed descriptors
- leave-one-out train analog-neighbor features
- train-to-test analog-neighbor features for blinded test molecules

This block is intentionally low-dimensional. It is meant to add signal, not dominate the structural model.

---

## Model design

The training step uses multiple simple regressors rather than a single complex model.

Typical model families include:

- LightGBM
- XGBoost
- CatBoost
- ExtraTrees
- RandomForest
- HistGradientBoosting
- Ridge / Elastic / Huber regression

Each model is evaluated with out-of-fold predictions. The final ensemble weights are assigned from OOF MAE, so models with better validation behavior contribute more to the final prediction.

The final package also keeps several prediction variants:

- CV ensemble prediction
- full-refit prediction
- mixed full-refit / CV prediction

This is useful because public leaderboard behavior may differ from local OOF behavior.


## Method summary

This model is best understood as a two-layer system.

The first layer learns general molecular activity from broad chemical representations. The second layer adds experiment-aware information: whether the molecule has reliable dose-response behavior, whether it resembles stable or unstable training compounds, and whether its local analog neighborhood suggests stronger or weaker PXR activity.

The method is conservative by design. It does not assume that analog similarity always means similar pEC50, and it does not aggressively delete all inconsistent compounds. Instead, it gives the model extra context and lets the OOF ensemble decide how much to trust each signal.

---

## Practical note

The auxiliary feature model alone is not expected to be the strongest model. Its value is mainly in combination with the broad structural stack. In this package, the stability/PXR block should be treated as a calibration and local-context layer, not as a complete replacement for the main molecular representation.
