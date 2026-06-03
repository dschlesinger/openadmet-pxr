# TODO

## Hyperparameter Optimization
- [ ] Integrate Optuna for automated hyperparameter search across models

## Train/Test Split Strategies
- [x] Closest 30% to test set — `filters.py::tanimoto_top30()`, keeps training molecules most similar to test by Tanimoto
- [x] Butina cluster split — `load.py::_butina_split()`, ECFP4 Tanimoto 0.4 cutoff, `split_type="butina"`
- [ ] Sweep CV axes (Butina + scaffold + MaxMin) for ensemble diversity rather than repeating the same axis with different seeds
- [ ] Finetune with test/train merge constraint — take pretrained representation model and finetune with the constraint that training and test distributions should be similar

## Chemprop Training
- [ ] HTS single-concentration pretraining before pEC50 fine-tuning (`single_concentration_train.csv` already downloaded) — most widely replicated lift across competitors
- [ ] Use MAE loss instead of MSE for training
- [ ] Concentration-aware pretraining: pass all four dose-response points as rows with `log10[conc_M]` as `x_d` (~21K rows vs. ~5.5K from Hill-fitting)
- [ ] Multitask training: pEC50 + counter-assay + single-conc + FXR heads — largest single ablation-documented lift (+0.070 RAE per discoverybytes)
- [ ] PMI shape descriptors (19 RDKit conformer-derived features) as `x_d` — captures 3D shape orthogonal to 2D topology

## Ensemble / Stacking
- [ ] Ridge or ElasticNetCV meta-learner on OOF predictions — replace simple average in `OneofEachDim`; fit stacker on out-of-fold preds only
- [ ] NNLS stacking for large candidate pools — non-negative weights, automatically zeros out redundant models
- [ ] Uncertainty meta-features in stacker: Tanimoto NN distance to train, overall model disagreement, Chemprop protocol disagreement

## Representations
- [x] CheMeleon finetuning — `scripts/finetune_chemeleon.py` + `chemeleon.py::FinetunedChemeleonFingerprint`
- [x] UniMol finetuning — `scripts/finetune_unimol.py` exists; representation class not yet wired up
- [x] MACCS keys (167-bit) — ~90% non-overlapping activity cliff pairs vs. Morgan alone
- [ ] Count-based and chiral Morgan fingerprints — `GetHashedMorganFingerprint(useCounts=True, useChirality=True)`
- [ ] PXR-specific SMARTS features — ~17–53 binary/continuous features encoding PXR pocket preferences (sulfonamide, CF3, hydrophobic volume, PSA)
- [ ] PXR co-crystal ligand Tanimoto similarity — 5 ECFP4 features (max, mean, std, top2, top3) vs. 56 PDB co-crystal ligands
- [ ] Cofolding structural features from dargason's dataset — mean/max ipTM, pTM, pLDDT, contact geometry at 3.5/4.5/5.0 Å per ligand across Chai-1/OpenFold3/Boltz2
- [ ] UniMol randomized SMILES augmentation at inference — average predictions over 10 SMILES orderings per compound
- [x] jazzy QM interaction descriptors — hydrogen bond donor/acceptor atom-pair energies from `jazzy` library; used by auP7s (rank 27)
- [x] xTB quantum descriptors — semiempirical GFN2-xTB atomic charges, chemical hardness, dipole, HOMO/LUMO gap; used by auP7s (rank 27)

## Auxiliary Data
- [ ] ChEMBL nuclear receptor data (PXR, FXR, LXRa, LXRb EC50) as Chemprop multitask pretraining head
- [ ] NCATS qHTS PXR assay (AID 1346982/1346985, ~10K compounds) as additional training head
- [ ] ADMET-AI features (41 endpoints: CYP, hepatotoxicity, BBB, solubility) as LightGBM sub-model input to meta-learner

## Post-Hoc Processing
- [ ] Activity gate for inactives — LightGBM binary classifier (active = pEC50 ≥ 4.0, threshold 0.30) with specialist model for predicted inactives; ~0.016 RAE reduction
- [ ] Tautomer correction for test compounds via RDKit MolStandardize — ~0.002 RAE at near-zero compute cost

## Avoid
- **In-sample isotonic calibration on full-training OOFs** — inflates OOF by ~0.009 RAE per step; does not generalize to blind test
- **Variance scaling of test predictions** — worsened RAE by +0.01 in controlled experiments
- **Nearest-neighbor activity transfer to test** — SAR misses invalidate this; RyeCatcher v50: OOF 0.45 → LB 0.66
- **Knowledge distillation of ensemble into single model** — destroys ensemble diversity (ΔRAE = +0.013)
- **External data for post-hoc calibration** — distribution mismatch inflates portal MAE; external data is safe for pretraining only
- **PAINS/REOS filtering of training data** — removes inactive anchors; LightGBM degraded from RAE 0.609 → 0.724
- **MolFormer-XL** — Spearman ρ = −0.007
- **Multiple seeds of same architecture** — seed-to-seed correlation > 0.97; use representation diversity instead
