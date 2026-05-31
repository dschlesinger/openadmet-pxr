# GAP Analysis — What This Pipeline Is Missing or Should Try

Based on `SUMMARY.md` and the competitor evidence. Items are roughly ordered by expected impact, highest first. Each entry names the source and the measured or estimated improvement where available.

---

## Cross-Validation

**Switch default CV to Butina clustering (Tanimoto 0.4, ECFP4).** Every top-10 open-source submission uses Butina or parent-cluster LOCO instead of Murcko scaffold. The current default (random 90/10 scaffold split in `load.py`) systematically overestimates validation performance relative to the leaderboard because analog pairs leak across the split boundary. Butina holds out entire clusters of similar molecules per fold, simulating the leaderboard's scaffold-novelty regime and narrowing the OOF→LB gap substantially. Source: discoverybytes (rank 11), ldbc1999 (rank 65), PeterBloomingdale (rank 33), auP7s (rank 27), RyeCatcher (rank 67).

**CV-axis diversity over seed diversity.** AuP7s (rank 27) found that adding Butina + scaffold + MaxMin axes to the sweep gave a larger ensemble lift than running the same axis with multiple random seeds. Once Butina CV is in place, sweeping multiple split axes is more valuable than re-running identical splits.

---

## Chemprop Training

**MAE loss instead of MSE.** Discoverybytes (rank 11) measured ~0.01 RAE gain switching Chemprop's training objective from MSE to MAE, attributing it to better gradient signal on noisy pEC50 labels and reduced sensitivity to outliers. The `chemprop_finetuned` representation currently uses default MSE. This is a one-line config change.

**HTS single-concentration pretraining before pEC50 fine-tuning.** Training the Chemprop backbone to predict log2_fc_estimate from ~21K single-concentration rows, then fine-tuning on the 4,139 pEC50 labels, is the most widely replicated lift in the field. Ldbc1999 (rank 65) showed Sub 4→5 improvement from LB RAE 0.7643 to 0.6615 attributable primarily to HTS pretraining. Schnappi (rank 127), hangyodon (rank 278), and mp-alex (rank 64) all used this pattern. The data is already downloaded (`single_concentration_train.csv`). The encoder-only transfer (copy message-passing weights, reinitialize FFN) is required when the FFN dimensionality differs between pretraining and fine-tuning.

**Concentration-aware pretraining variant.** Instead of Hill-fitting to pseudo-pEC50 (which discards borderline-active compounds with R² < 0.5), pass all four dose-response points per compound as separate rows with `log10[concentration_M]` as a molecule-level descriptor (`x_d`). Ldbc1999 Sub 6 used this, yielding ~21K training rows vs. ~5,500 from Hill-fitting. Implemented as `x_d` in Chemprop v2.

**Multitask training with auxiliary targets.** Discoverybytes (rank 11) showed a 0.070 RAE advantage for four-task training (pEC50 + counter-assay + single-conc + FXR) over two-task in a controlled ablation — the largest single modeling improvement documented with an ablation. Ldbc1999's Sub 8 added Emax as a second head (partial agonists carry complementary SAR info). All auxiliary data is already present in the pipeline. The key constraint: only transfer message-passing encoder weights; reinitialize the FFN for the fine-tuning task count.

**PMI shape descriptors as molecule-level features.** Discoverybytes appended 19 principal-moment-of-inertia descriptors (rod/disc/sphere shape from RDKit conformers) as `x_d` to the Chemprop FFN, capturing global 3D shape orthogonally to 2D topology and without the full UniMol conformer pipeline.

---

## Ensemble / Stacking

**Ridge or ElasticNet meta-learner on OOF predictions** instead of the current simple average in `OneofEachDim`. Discoverybytes' Ridge stack (Chemprop 0.706, MolE 0.206, UniMol 0.166) outperformed naive averaging by ~0.01–0.03 RAE. ElasticNetCV with L1 regularization automatically zeroes out redundant models. The stacker should be fitted on out-of-fold predictions, not on held-out test performance.

**NNLS (non-negative least squares) stacking.** Ldbc1999 (rank 65) used NNLS to select 5 survivors from 87 candidate models. Cheaper than ElasticNet when the candidate pool is large; non-negativity prevents destabilizing negative weights. Equally applicable here.

**Uncertainty features in the meta-learner.** Mp-alex (rank 64) added three meta-features to their ElasticNet: Tanimoto NN distance to training data (coefficient −0.382), overall model disagreement (−0.594), and Chemprop protocol disagreement (+0.161). These allow the stacker to adaptively reduce predictions for out-of-distribution or high-disagreement compounds without needing separate calibration.

---

## Representations to Add

**MACCS keys (167-bit).** Ldbc1999 (rank 65) showed MACCS and ECFP4 identify ~90% non-overlapping activity cliff pairs — their combination is meaningfully richer than Morgan alone. Not currently a registered representation. Cheap to compute via RDKit.

**Count-based and chiral Morgan fingerprints.** Chaospilot (rank 102) and QuantNova (rank 113) used count Morgan (preserves repeated substructure frequency) and chiral Morgan (retains stereochemical distinctions). Both available in RDKit as `GetHashedMorganFingerprint` with `useChirality=True` and `useCounts=True`.

**PXR-specific SMARTS features.** Multiple competitors (duoduo6 rank 168, wuhicky rank 148, duod rank 205, cat554 rank 118) added hand-designed features encoding known PXR pocket preferences: sulfonamide, trifluoromethyl, aromatic surface, hydrophobic volume, polar surface area thresholds. These are ~17–53 binary/continuous features. Provides interpretable bias toward PXR-relevant chemistry and some orthogonal signal to the learned representations.

**PXR co-crystal ligand Tanimoto similarity.** Ldbc1999 (rank 65) computed five ECFP4 Tanimoto features against 56 PXR co-crystal ligands from the PDB (max, mean, std, top2, top3 similarity). These fixed reference ligands encode proximity to confirmed PXR binders in a way orthogonal to within-training pairwise similarity. Script: `fetch_crystal_ligands.py` from the ldbc1999 repo.

**Cofolding structural features from dargason's dataset.** The consolidated metrics CSV (`cofolding_structure_metrics_one_row_per_structure.csv`, ~70 MB, 46,075 rows × 87 cols) contains per-prediction features for all 513 activity-challenge test ligands across three methods (Chai-1, OpenFold3, Boltz2), 5 seeds × 5 samples = 25 structures per method per ligand. ML-usable feature block after aggregating over samples per ligand per method: mean/max/std ipTM, mean/max pTM, mean/max pLDDT, mean contact atom pairs at 3.5/4.5/5.0 Å, mean clash count (2.0 Å), fraction of anchor-contact passes, mean ligand-protein distance, radius of gyration. These cofolding confidence and contact geometry features encode protein-ligand compatibility in a way orthogonal to any 2D or sequence representation. No published submission has reported results using them as ML inputs. Download into `data/` separately; schema documented in `os_reports/raw/rank_14-dargason-cofolding-consolidated-README.md`.

**UniMol with multiple LR fine-tuning variants.** PeterBloomingdale (rank 33) trained UniMol at three learning rates (5e-5, 2e-4, 5e-4) and found the variants provided complementary signal (SLSQP assigned 24.8% to LR=2e-4; LR=1e-3 was zeroed out as too noisy). A single LR fine-tune leaves ensemble diversity on the table.

**UniMol with randomized SMILES augmentation at inference.** Discoverybytes averaged UniMol predictions across 10 randomized SMILES orderings per test compound (different orderings → different RDKit conformers → different 3D geometries → variance reduction). Free inference-time improvement once UniMol is trained.

---

## Auxiliary Data / External Sources

**ChEMBL nuclear receptor data as a pretraining head.** Mp-alex (rank 64) pretrained Chemprop-NR on ChEMBL EC50 data across PXR, FXR, LXRa, LXRb; OOF RAE 0.549 (stronger than scratch, weaker than HTS-pretrained). RyeCatcher (rank 67) used ChEMBL NR1I2 (907 compounds) and NCATS qHTS PXR (AID 1346982/1346985, ~10K compounds) as additional multitask heads. Both are free public datasets. Even individual external heads with Spearman ≤ 0.4 against the primary task lift performance through shared representations.

**ADMET-AI features as a meta-learner input.** Ldbc1999 trained a LightGBM sub-model on 41 ADMET-AI endpoint predictions (CYP activity, hepatotoxicity, BBB permeability, solubility) and it received 3.1% NNLS weight in their final ensemble. Low-effort addition once ADMET-AI is installed.

---

## Post-Hoc Processing

**Activity gate for inactives.** Ldbc1999 (rank 65) found 37 of 253 revealed test compounds are truly inactive (pEC50 < 3.5; true mean 2.670), and regression models predict them near the training mean (~4.6), errors of 1.3–1.9 log units. Their fix: a LightGBM binary classifier on CheMeleon HTS embeddings (active = pEC50 ≥ 4.0) with a threshold of 0.30; compounds below the threshold have their predictions swapped to a specialist LGB model trained on inactive-like space. This reduced revealed-RAE from 0.5424 to 0.5259 (+0.016). The classifier requires the HTS-pretrained CheMeleon encoder.

**Tautomer correction for test compounds.** Discoverybytes identified three test compounds with likely incorrect tautomeric SMILES. Manual or RDKit-MolStandardize tautomer enumeration + a model sensitivity check can identify these; corrections yield ~0.002 RAE gain at essentially zero compute cost.

---

## Things Tried by Competitors That Should Be Avoided

**In-sample isotonic calibration on full-training OOFs.** Overestimates improvement by ~0.009 RAE per step; "chained iso-on-iso" compounds the bias. Discoverybytes, firstpass (rank 38), and auP7s (rank 27) all found this harmful. RyeCatcher's honest per-fold (leave-one-fold-out) variant is the only calibration approach shown to generalize.

**Variance scaling of test predictions.** Discoverybytes tested k=1.10 scaling; RAE worsened by +0.01. Firstpass submitted raw stacker output, explicitly noting "variance-matching and quartile mapping dramatically hurt blind-test performance."

**Nearest-neighbor activity transfer (copying training pEC50 to similar test compounds).** RyeCatcher v50: OOF 0.4536 → LB 0.658. The test set contains SAR misses — structurally similar to actives but biologically inactive. Do not copy training labels to test based on structural similarity.

**Knowledge distillation of the ensemble into a single model.** Discoverybytes trained a new Chemprop on ensemble pseudo-labels; ΔRAE = +0.0126. Ensemble diversity was destroyed because the distilled model had absorbed UniMol and MolE signal, making them redundant. Do not train any model on another model's predictions if you intend to ensemble them.

**External data for calibration (not pretraining).** AuP7s tried calibrating against a public ChEMBL ADMET-adjacent dataset; distribution mismatch inflated portal MAE substantially. External data is safe for pretraining; dangerous as a calibration anchor.

**Delta/pairwise learning without training coverage.** Ldbc1999's pairwise delta Chemprop delivered Spearman = −0.071 on the leaderboard because 90.8% of test compounds had no training neighbor within Tanimoto ≥ 0.7. The approach works in principle but requires dense structural coverage of the test set.

**PAINS/REOS filtering of training data.** Discoverybytes: removing 841 PAINS-flagged compounds degraded LightGBM from OOF RAE 0.609 to 0.724. Those compounds anchor the inactive end of the landscape.

**MolFormer-XL (IBM SMILES transformer).** Spearman ρ = −0.007 (discoverybytes). Not worth the compute.

**Multiple seeds of the same architecture without representation diversity.** Discoverybytes found seed-to-seed Chemprop correlations > 0.97 — negligible ensemble diversity gain. Representation diversity (2D vs. 3D vs. pretrained) is far more valuable than architectural repetition.
