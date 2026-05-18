# TODO

## Hyperparameter Optimization
- [ ] Integrate Optuna for automated hyperparameter search across models

## Train/Test Split Strategies
- [ ] Closest 30% to test set — use chemical similarity (Tanimoto/embedding distance) to carve out a training split that mirrors test distribution
- [ ] Finetune with test/train merge constraint — take pretrained repersentation model and finetune with the constaint that the training and test distribution should be similar

## Models
- [ ] Pretrained property prediction model (e.g. ChemBERTa, MolBERT, Uni-Mol fine-tune on PXR labels)
- [ ] KANs (Kolmogorov-Arnold Networks) — drop-in replacement for MLP layers, bc why not

## Data
- [ ] Find additional PXR datasets to augment training (ChEMBL, PubChem bioassay, literature)

## Structure-Based
- [ ] Use PXR bound structure to inform prediction via Boltz2 (structure-conditioned scoring)
