# OpenADMET PXR Challenge — What Has Been Tried
#### Written by Claude (Sonnet 4.6)
A narrative synthesis of methodology across all open-sourced competitor reports (ranked 11–279). Sources are cited by username and leaderboard rank. All leaderboard metrics are Phase 1 MAE and Spearman ρ from `activity_leaderboard_tgyvb15g.csv`. RAE = MAE / MAD(training mean).

---

## 1. Data: What Goes In, What Gets Filtered

Every submission begins with the same four HuggingFace files: 4,139 training compounds with dose-response pEC50 labels, 513 blinded test compounds, ~2,859 counter-assay compounds (PXR-null cell line), and ~21,000 single-concentration screen rows. The universally applied preprocessing is RDKit canonicalization of SMILES, largest-fragment selection to strip salts and counter-ions, deduplication on canonical SMILES, and a sanity check that all molecules parse correctly.

Beyond that baseline, approaches diverge sharply. The most careful curation came from discoverybytes (rank 11, MAE 0.4268), who ran a two-stage protocol. First, 754 compounds containing reactive electrophilic groups — acrylamides (n=236), acrylates (n=229), and aldehydes (n=87) — were excluded, improving cross-validation RAE by approximately 0.019. The logic is that the test set contains none of these compounds, so including them trains on noise not signal. Second, and instructively, PAINS/REOS filtering was evaluated and *rejected*. Removing 841 flagged compounds actually degraded LightGBM from OOF RAE 0.609 to 0.724: despite being structurally "problematic," those compounds anchor the inactive end of the activity landscape and help calibrate the dynamic range. The lesson across reports is to remove synthesis noise (reactive groups not present in test), but to retain structural diversity even when it looks messy.

Uncertainty-aware training was explored by ranga_31489 (rank 206), who used sample weights w = 1 / (SE² + σ_model²) with σ_model = 0.60 to account for heteroscedastic pEC50 measurement error. Discoverybytes downweighted 474 high-SE compounds to 0.3–0.5 weight and removed three compounds with zero structural similarity to any test compound — reasoning that fully out-of-domain training points contribute noise rather than signal for analog-series prediction.

Two edge cases that affected only test-set predictions: tautomer correction and stereochemistry. Discoverybytes identified three test compounds with apparently incorrect tautomeric SMILES (enol forms instead of the likely keto tautomers), applied post-prediction pEC50 offsets (−0.356, −0.411, −0.009), and estimated ~0.002 RAE gain — small but free. On stereochemistry, discoverybytes found ~39.5% of training and 38.2% of test compounds have undefined stereocenters, so filtering would bias the training distribution relative to test; all undefined-stereocenter compounds were retained.

After Phase 1 labels were released for 253 test compounds, discoverybytes and ldbc1999 (rank 65) both retrained on the expanded 4,392-compound set and, in discoverybytes' case, directly substituted known true pEC50 values for those 253 compounds in the final submission, leaving the model to predict only the 260 still-blinded compounds. This is legitimate under the challenge rules since Phase 2 evaluation covers only the blinded 260.

---

## 2. Molecular Representations: 2D, 3D, and Pretrained

The field organized into three tracks: classical 2D fingerprints and descriptors, 3D conformer-based representations, and pretrained neural embeddings. The most competitive submissions combine at least two orthogonal tracks.

**Classical 2D.** Morgan ECFP4 (2048-bit, radius=2) is the universal starting fingerprint — every submission uses it. MACCS keys (167-bit structural fragment keys) add complementary signal: ldbc1999 (rank 65) found MACCS and ECFP4 identify ~90% non-overlapping activity cliff pairs, making their combination richer than either alone. RDKit 2D descriptors (~200–217 properties: MW, LogP, TPSA, HBA/HBD, rings, RotBonds, FractionCSP3, QED) provide global physicochemical context that fingerprints miss. Mordred descriptors (used by auP7s rank 27, RyeCatcher rank 67, PeterBloomingdale rank 33) extend descriptor coverage but introduce overfitting risk: discoverybytes observed Mordred OOF RAE 0.574 against test RAE 0.67, a gap that indicates scaffold-overfitting. Count-based Morgan and chiral Morgan variants (chaospilot rank 102, QuantNova rank 113) preserve repeated substructure frequency and stereochemical distinctions. A subset of competitors added PXR-specific SMARTS features — 17 pharmacophore motifs and physicochemical rules encoding known PXR pocket preferences (sulfonamide groups, trifluoromethyl, aromatic surface area, hydrophobic volume) — including duoduo6 (rank 168), wuhicky (rank 148), duod (rank 205), and cat554 (rank 118). AuP7s (rank 27) pushed furthest into exotic classical descriptors, adding jazzy QM interaction descriptors, xTB quantum descriptors, and MAP4/AtomPair/Topological Torsion/Pharmacophore fingerprints via scikit-fingerprints.

**Pretrained neural fingerprints.** CheMeleon — a Chemprop D-MPNN pretrained on large-scale molecular data — was the single largest improvement reported across the field. Discoverybytes (rank 11) measured a direct comparison: test RAE dropped from ~0.62 to ~0.59 when switching from random initialization to CheMeleon pretrained weights as the starting point for PXR fine-tuning. Whitebox (rank 175) fine-tuned CheMeleon end-to-end on the full PXR training set with scaffold 5-fold CV. The improvement is driven by the pretrained backbone having already learned PXR-relevant chemical space structure before seeing any dose-response labels.

**Chemprop D-MPNN trained from scratch** remains competitive as a baseline GNN but consistently underperforms CheMeleon-initialized variants. Discoverybytes' scratch models reached test RAE ~0.62; CheMeleon-initialized reached ~0.59.

**3D representations.** UniMol (Zhou et al., ICLR 2023) is a transformer pretrained on 209 million 3D conformers from ZINC and PubChem, encoding pairwise inter-atomic distances via SE(3)-equivariant attention. Fine-tuned on PXR pEC50, it consistently provides signal orthogonal to 2D graph models. Discoverybytes reported standalone OOF RAE ~0.645; PeterBloomingdale (rank 33) broke through a persistent CV RAE plateau at ~0.527 when UniMol was added, which they attribute to PXR's large, flexible buried binding pocket — shape complementarity matters here in ways invisible to bond-graph topology. Discoverybytes augmented UniMol inference by averaging predictions across 10 randomized SMILES orderings (different atom orderings produce different RDKit conformers), reducing prediction variance. Multiple learning-rate fine-tuning variants were explored by PeterBloomingdale: LR 5e-5, 2e-4, and 5e-4, with the highest-LR model contributing to SLSQP with weight 24.8% (LR=2e-4) but the LR=1e-3 variant assigned zero weight as too noisy.

**MolE.** A graph-transformer foundation model pretrained on large-scale unlabeled chemical data. Discoverybytes reported OOF RAE ~0.705 — the weakest standalone of their three base models — but it earned Ridge coefficient 0.206 in the final stack because its pretrained embeddings generalized well to the test set's novel scaffolds, providing architectural diversity the two GNN approaches lacked.

**PMI shape descriptors.** Discoverybytes appended 19 principal-moment-of-inertia descriptors (capturing rod-like, disc-like, and spherical molecular shape from RDKit conformers) as molecule-level features (`x_d`) to the Chemprop FFN head. These capture global 3D shape orthogonally to both 2D topology and pairwise-distance UniMol representations.

**What failed on representations.** MolFormer-XL (IBM's SMILES transformer) produced Spearman ρ = −0.007 in discoverybytes' experiments — worse than random ordering. AuP7s (rank 27) used MolFormer with limited gains. RyeCatcher (rank 67) tested ChemBERTa, GIN-ACtriplet, UniMolV2-310M, MaskMol, and an ImageNet ViT applied to molecular images: all converged to Spearman > 0.88 against their primary ensemble, meaning the 2D molecular graph representation space is saturated for this task and dataset size — additional models from the same family add compute, not predictive signal.

Two competitors also explored biology-anchored features. Ldbc1999 (rank 65) computed ECFP4 Tanimoto similarity against 56 PXR co-crystal ligands from the PDB (max, mean, std, top2, top3 similarity), reasoning these experimentally confirmed binders encode proximity to the known active pharmacophore in a way orthogonal to within-training pairwise similarity. Ldbc1999 also used 41 ADMET-AI endpoint predictions (CYP activity, hepatotoxicity, BBB permeability, solubility) as features for a LightGBM sub-model. Firstpass (rank 38) and PeterBloomingdale (rank 33) incorporated AutoDock-GPU or Glide docking scores across multiple PXR co-crystal structures as minority ensemble contributions for tail-behavior coverage. Dargason (rank 14) used GNINA docking across 8 PXR crystal structures in their competition submission.

---

## 3. Model Architectures

**Classical regressors** appear across the board but in different roles. At the lower end of the leaderboard (ranks 100–280) they are the primary models; in the top decile they are ensemble members only. Ranga_31489 (rank 206) ran a clean ablation on the same v2 feature matrix: XGBoost (uncertainty-weighted, val RMSE 0.750) outperformed a tuned HistGradientBoosting (val RMSE 0.809) and random forest (val RMSE ~0.83). LightGBM and CatBoost are interchangeable with XGBoost for most practical purposes in this dataset. Ridge regression contributes a smooth linear signal that complements nonlinear tree ensembles. The shared trait of top tree ensembles is that they are used only as ensemble contributors, with weights learned from OOF rather than set by hand.

**Chemprop D-MPNN** is the workhorse GNN. Competitive depth: 3–5 message-passing layers; hidden dim 250–600; dropout 0.1–0.3; 60–100 training epochs on scaffold-split CV. The most impactful finding about Chemprop training is that MAE loss consistently outperforms MSE loss: discoverybytes measured ~0.01 RAE gain, attributed to MAE's robustness to outliers and better gradient signal on the noisy pEC50 labels. The second most impactful finding is that multitask training — predicting pEC50 jointly with counter-assay, single-concentration screen, and external nuclear receptor activity — consistently outperforms single-task training. Discoverybytes showed a four-task vs. two-task advantage of 0.070 RAE in a controlled comparison.

**TabICL and TabPFN** are tabular in-context learners that perform no gradient training: the full (X_train, y_train) pair is passed to a transformer at inference time, and the transformer's pretrained prior over tabular datasets does the work. When paired with CheMeleon embeddings compressed via PCA-200, they are highly competitive. In ldbc1999's (rank 65) Sub 7, TabPFN was the dominant ensemble contributor (ElasticNet coef=0.42). AuP7s (rank 27) built an entire competition pipeline around TabICL with no GNN at all, reaching rank 27 (MAE 0.4352). The mechanism that makes this work: unlike tree models that forget training labels at prediction time, TabICL/TabPFN see all 4,139 training labels during every prediction, allowing them to capture analog-series patterns by direct comparison. Ben Hicham et al. (2025) showed TabPFN+CheMeleon achieves up to 100% win rate on the MoleculeACE activity cliff benchmark, which has identical structure to this test set. One failure mode: when TabPFN is given CheMeleon embeddings from OOF-fold-trained models (which differ from all-data CheMeleon embeddings), a distribution shift degrades test performance — discoverybytes identified this and ruled TabPFN out of their final stack.

**Graph transformers.** Mp-alex (rank 64, MAE 0.4567) assembled the most architecturally diverse single submission. Their GPS graph transformer (GATv2 local MPNN + global multi-head attention, 128 hidden, 4 heads) was pretrained in three stages: (1) 20 epochs on 3.08M ChEMBL+ZINC molecules with dual objectives — 10 RDKit property regression plus GraphMAE attribute masking — (2) 20 epochs fine-tuning on 21K single-concentration data, (3) 10 epochs fine-tuning on PXR pEC50 with discriminative learning rates; achieving OOF RAE 0.534. Their TorchMD-NET equivariant transformer (O(3)-equivariant, pretrained on PCQM4MV2 3.4M molecules via self-conditioned denoising, then fine-tuned on ChEMBL PXR, then on PXR pEC50) achieved OOF RAE 0.591. Ldbc1999 also trained a GraphGPS model (script 121) that received 16.8% NNLS weight in their final ensemble.

**Foundation model fine-tuning (non-Chemprop).** JacksonBurns (rank 202, MAE 0.5292) fine-tuned the MIST 1.8B-parameter SMILES language model on kekulized PXR SMILES (10 epochs, HuggingFace Trainer, fp16, AdamW-8bit). At rank 202 it is not competitive against GNN approaches but demonstrates that pure sequence-based foundation models are viable, just weaker here. Whitebox (rank 175) fine-tuned CheMeleon end-to-end with scaffold CV and averaged five fold models.

**Pairwise/delta models.** Ldbc1999 trained a Chemprop D-MPNN on all pairwise activity differences (input: two SMILES; target: pEC50_i − pEC50_j), oversampling activity cliff pairs (Tanimoto ≥ 0.7, |Δ pEC50| ≥ 1.0) 3×, and anchoring test predictions to 10 nearest training neighbors via the learned delta. This is theoretically appealing for analog-series prediction but ran into a data-coverage problem: 90.8% of test compounds had no training neighbor within Tanimoto ≥ 0.7, so the model extrapolated outside its intended regime and delivered Spearman = −0.071 on the leaderboard — the single most cautionary negative result across all reports. The delta approach works only when training coverage is dense enough to find true structural neighbors.

---

## 4. Auxiliary Data and Transfer Learning

**Single-concentration screen.** The ~21,000 single-concentration rows (four concentrations per compound → log2_fc_estimate) are the most widely leveraged auxiliary signal. In the naive approach, Hill-curve fitting converts them to pseudo-pEC50 values (with an R² ≥ 0.5 filter) and the resulting ~5,500 qualified compounds are used to pretrain Chemprop before fine-tuning on pEC50. This pattern appears in Schnappi (rank 127), hangyodon (rank 278), mp-alex (rank 64), and ldbc1999 (rank 65). Ldbc1999's Sub 6 introduced a concentration-aware variant: all four dose-response points per compound are kept as separate training rows, with log10[concentration_M] passed as a molecule-level descriptor to the Chemprop FFN head, yielding ~21K training rows (~4× more than Hill-fitting) without discarding borderline-active compounds that fail the R² filter. The encoder-only transfer (copying message-passing weights but reinitializing the FFN) prevents size mismatches when the FFN dimensionality changes between pretraining and fine-tuning.

**Counter-assay.** The PXR-null counter-assay identifies compounds active in the reporter system regardless of PXR activation — i.e., assay artifacts or promiscuous binders. Discoverybytes included it as a fourth multitask head. Ldbc1999 used counter-assay pEC50 as a feature alongside the primary label for downweighting non-specific compounds. Simpler submissions used it as a 0.3× training-weight flag. All approaches improve over ignoring it.

**Emax as auxiliary target.** Ldbc1999's Sub 8 jointly predicted pEC50 and Emax (maximum efficacy, ranging ~0–1) as a two-task Chemprop, arguing that partial agonists (low Emax, structurally distinct from full agonists) carry complementary SAR information sharing the same molecular encoder. The ElasticNet meta-learner assigned RF higher weight (0.127 → 0.153) when MACCS features were added alongside ECFP4/6, confirming the architecture found additional orthogonal signal.

**ChEMBL and external nuclear receptor data.** Mp-alex pretrained Chemprop-NR on ChEMBL EC50 data across PXR, FXR, LXRa, and LXRb before fine-tuning on the challenge pEC50 (OOF RAE 0.549, stronger than scratch). RyeCatcher's final model T1v5 trained five heads simultaneously: primary pEC50, counter-screen, NCATS qHTS PXR (AID 1346982/1346985, ~10K compounds), Tox21 SR-ARE, and ChEMBL NR1I2 (907 compounds). Individual external heads had Spearman ≤ 0.4 against the primary task — but multitask transfer works through shared representations, not direct label correlation. Discoverybytes included FXR activity as a fifth cross-receptor regularizer in their four-task Chemprop.

**A calibration failure with external data.** AuP7s (rank 27) tried calibrating their OOF ensemble against a public ChEMBL ADMET-adjacent dataset; the distribution mismatch between that source and the challenge's blinded set inflated portal MAE substantially. The lesson is consistent across all reports: external data is useful for *pretraining representations* and *broadening task coverage*; it is dangerous as a *calibration anchor* unless it matches the test distribution closely.

---

## 5. Cross-Validation: The Honest Estimate Problem

The test set is 513 analogs of ~63 potent training scaffolds. This structural relationship means that random train/validation splitting, where related analogs appear on both sides of the split, systematically overestimates leaderboard performance. Bridging the OOF→LB gap was the central methodological challenge of the challenge.

Random KFold is the baseline — it severely overestimates, and essentially every report that tried it observed large gaps to the leaderboard. Murcko scaffold stratification (used in ldbc1999's early submissions) is better but still leaks analogues across folds. Butina clustering on ECFP4 at Tanimoto threshold 0.3–0.4 emerged as the dominant CV approach in top submissions: entire clusters of structurally similar molecules are held out per fold, simulating the scaffold-novelty regime of the leaderboard. AuP7s (rank 27) used Butina 0.4 + Murcko scaffold + MaxMin centrality-graded axes as three parallel CV dimensions, finding that CV-axis diversity (different ways of splitting the data) gives a larger ensemble lift than using multiple random seeds of the same split axis.

RyeCatcher (rank 67) went furthest in CV design. They built parent-cluster LOCO (leave-one-cluster-out) splits using FCFP4 nearest-neighbor clustering that explicitly mirrors the test set's structure — 513 analogs of ~89 parent compounds. This was the only protocol that produced CV→LB shifts consistent with actual leaderboard outcomes. Their honest per-fold isotonic calibration, fit only on left-out fold predictions (not on full-training OOFs), was the key: fitting isotonic regression on full-training OOFs overestimates improvement by ~0.009 RAE per step, and compounding isotonic fits compounds this bias. Discoverybytes and firstpass (rank 38) both tried isotonic calibration, found it harmful or neutral, and abandoned it. The general finding is that isotonic calibration can work if and only if the calibration set closely matches the test distribution — which requires the honest per-fold approach.

---

## 6. Ensembling

Virtually all competitive submissions are ensembles. The question is how to combine model outputs without overfitting the combination weights.

Simple inverse-RAE weighting (earlier ldbc1999 submissions) was replaced by learned stacking once OOF predictions were available. NNLS (non-negative least squares) became the most widely adopted stacking method: the non-negativity constraint prevents destabilizing negative weights, naturally produces sparse solutions (only strong/diverse models receive non-zero weight), and has no hyperparameters to tune. Ldbc1999's final NNLS pipeline filtered 147 trained models down to 5 survivors through quality filtering (R² > 0.3 on revealed set), anti-leakage filtering (drop models where revealed-RAE / OOF-RAE > 2.0), and correlation-based deduplication (cluster models with Pearson ρ > 0.95 and replace each cluster with its mean).

SLSQP (quadratic constrained optimization) was used by PeterBloomingdale (rank 33) for their final blend, with UniMol at LR=2e-4 receiving 24.8% weight and LightGBM+docking 23.5%, TabPFN 20.0%. Ridge meta-learning on OOF predictions was used by discoverybytes: their final Ridge stack assigned Chemprop (JG_MAE) coefficient 0.706, MolE 0.206, UniMol 0.166, intercept −0.383. ElasticNetCV (L1+L2 regularization) was used by mp-alex and ldbc1999; the L1 component zeros out redundant models. Mp-alex's 9-feature meta-learner added three adaptive uncertainty inputs — Tanimoto NN distance to training data (coefficient −0.382, shrinking predictions for novel compounds), overall model disagreement (−0.594), and Chemprop protocol disagreement (+0.161) — allowing the meta-learner to reduce confidence on out-of-distribution predictions. AuP7s (rank 27) used NSGA-II Pareto optimization over (OOF MAE, Spearman ρ) to select ensemble subsets, evaluating NNLS, NNLS+isotonic, and Caruana 2004 hill-climbing as blending methods per Pareto point. RyeCatcher cascaded through seven intermediate ensemble versions, accumulating incremental improvements across model types.

The most important anti-pattern is the knowledge distillation backfire documented by discoverybytes: they attempted to train a new Chemprop on the full ensemble's predictions as pseudo-labels, expecting the new model to "absorb" what UniMol and MolE know. When Ridge then stacked the distilled Chemprop with UniMol and MolE, it found the latter two were now redundant (the new Chemprop had already absorbed their signal during distillation) and slashed their weights. Ensemble diversity was destroyed; ΔRAE = +0.0126 relative to the original three-model stack. The principle: training any model on another model's predictions makes it non-independent, eliminating the ensemble diversity that justifies combining them.

---

## 7. Post-Hoc Processing: What Helps and What Hurts

Prediction clipping to the observed training range ([1.5, 8.0] or similar) is nearly universal and uncontroversially useful — it prevents physically unreasonable extrapolation at the extremes with no downside.

Dynamic range recalibration is more contested. The test set analogs cluster in a narrower, higher-potency distribution than the full training set (training pEC50 std ≈ 1.121; test analogs estimated std ≈ 0.70), so raw model predictions are systematically compressed. PeterBloomingdale (rank 33) applied variance recalibration targeting std=0.70. Discoverybytes tested k=1.10 variance scaling and found test RAE *worsened* by +0.01 (Spearman also decreased): "The compression is a real phenomenon but scaling is the wrong fix." Firstpass (rank 38) submitted raw stacker output explicitly, reporting "variance-matching and quartile mapping dramatically hurt blind-test performance." The consensus is that compression is real but that post-hoc scaling, without a calibration set matched to the test distribution, makes it worse.

The most important structural post-processing finding is the inactive overestimation problem. Ldbc1999 (rank 65) identified that 37 of 253 revealed test compounds are truly inactive (pEC50 < 3.5; true mean 2.670), and ensemble models predict them near the training mean (~4.6), producing errors of 1.3–1.9 log units. Their solution — a LightGBM binary classifier on CheMeleon HTS embeddings identifying probable inactives (p(active) < 0.30), with predictions swapped to a specialist LGB model trained on inactive-like space — reduced revealed-RAE from 0.5424 to 0.5259. Even the gated predictions overestimate true inactives (predicted 3.4–3.7 vs. true mean 2.67); a full fix requires confirmed inactive labels at inference time.

The inverse failure mode is nearest-neighbor activity transfer. RyeCatcher's v50 model used high Tanimoto similarity to training actives as evidence to copy their pEC50 values to test compounds. The test set contains "SAR misses" — compounds with Tanimoto 0.50–0.58 to training actives that are biologically inactive because their binding mode is incompatible with the PXR pocket despite superficial structural similarity. The result: OOF 0.4536 → LB 0.658, regressing from rank ~40 to rank ~87. No other experiment across all reports produced a comparably catastrophic regression.

Tautomer corrections (discoverybytes: three test compounds corrected, ~0.002 RAE gain) and directly substituting revealed Phase 1 labels for the 253 unblinded test compounds (discoverybytes: replaces model predictions with known truth for that subset) are both useful when applicable — the former requires chemical expertise to identify, the latter is only available after Phase 1 unblinding.

---

## References

- discoverybytes (rank 11): [GitHub](https://github.com/discoverybytes/openadmet-pxr-blind-challenge/tree/main/activity-prediction)
- dargason (rank 14): [aetherark.com/openadmet-pxr_phase1.html](https://aetherark.com/openadmet-pxr_phase1.html)
- auP7s (rank 27): [Gist chemotica/a49b002eda2f7fd5eef2dca4f98f8ad7](https://gist.github.com/chemotica/a49b002eda2f7fd5eef2dca4f98f8ad7)
- PeterBloomingdale (rank 33): [GitHub](https://github.com/PeterBloomingdale/openadmet-pxr)
- firstpass (rank 38): [HuggingFace lachrymator/openadmet-pxr-challenge](https://huggingface.co/lachrymator/openadmet-pxr-challenge)
- jeremy (rank 60): [GitHub jeremycheminf/openadmet_scripts/PXR/2D](https://github.com/jeremycheminf/openadmet_scripts/blob/main/PXR/2D)
- mp-alex (rank 64): [GitHub mirror-physics/pxr_openadmet_blind_challenge](https://github.com/mirror-physics/pxr_openadmet_blind_challenge/blob/main/reports/track1_activity/submission_6model_adaptive_elasticnet.md)
- ldbc1999 / Liz Yurkewych (rank 65): [GitHub lizyurkewych-git/pxr-challenge](https://github.com/lizyurkewych-git/pxr-challenge)
- RyeCatcher (rank 67): [HuggingFace RyeCatcher/openadmet-pxr-challenge-2026](https://huggingface.co/RyeCatcher/openadmet-pxr-challenge-2026)
- chaospilot (rank 102): OneDrive (raw/rank_102-chaospilot-README.md)
- Schnappi (rank 127): [HuggingFace Schni-Schnappi/openadmet-pxr-challenge](https://huggingface.co/Schni-Schnappi/openadmet-pxr-challenge)
- cat554 (rank 118): [GitHub ttll1667/pxr_comp](https://github.com/ttll1667/pxr_comp/blob/main/README_pxr_stability_stack_experiment_log.md)
- leeherman99 (rank 121): Google Doc (raw/rank_121-leeherman99-report.pdf)
- QuantNova (rank 113): [GitHub chaospilot2003/openadmet](https://github.com/chaospilot2003/openadmet)
- wuhicky (rank 148): [GitHub wuhike61-cmd/pxr](https://github.com/wuhike61-cmd/pxr)
- axelrolov (rank 152): [GitHub AxelRolov/moltabfm_pxr](https://github.com/AxelRolov/moltabfm_pxr)
- duoduo6 (rank 168): [GitHub duoduo6660/PXR](https://github.com/duoduo6660/PXR)
- Whitebox (rank 175): [HuggingFace Whitebox2026/openadmet-pxr-challenge](https://huggingface.co/Whitebox2026/openadmet-pxr-challenge)
- nyota (rank 193): [GitHub junyigloria-collab/pxr-challenge-method-report](https://github.com/junyigloria-collab/pxr-challenge-method-report)
- BalamuruganThirukonda (rank 196): [GitHub BalamuruganThirukonda/openadmet-pxr-pec50](https://github.com/BalamuruganThirukonda/openadmet-pxr-pec50)
- JacksonBurns (rank 202): [Gist JacksonBurns/94cb5e7dda4d72bd876c947df92c5147](https://gist.github.com/JacksonBurns/94cb5e7dda4d72bd876c947df92c5147)
- duod (rank 205): [GitHub duoyou666/PXR_report](https://github.com/duoyou666/PXR_report)
- ranga_31489 (rank 206): [GitHub quantumdolphin/OPENADMET-PXR-activity-pred](https://github.com/quantumdolphin/OPENADMET-PXR-activity-pred/blob/c02fd2a86421f9a808327071c2a9d7894aaa4d02/METHOD_REPORT.md)
- lxduo (rank 218): [GitHub duo6660/PXR-report](https://github.com/duo6660/PXR-report)
- Shorku (rank 267): [GitHub Shorku/OpenADMET_Blind_Challenge_Predicting_PXR_Induction](https://github.com/Shorku/OpenADMET_Blind_Challenge_Predicting_PXR_Induction)
- hangyodon (rank 278): [GitHub hangyodon-ljy/pxr-challenge-baseline](https://github.com/hangyodon-ljy/pxr-challenge-baseline)
- k3785331526 (rank 279): [GitHub kenny3785331526/PXR](https://github.com/kenny3785331526/PXR)

**External citations:**
- Zhou et al. (2023). Uni-Mol: A Universal 3D Molecular Representation Learning Framework. *ICLR 2023*. https://doi.org/10.26434/chemrxiv-2022-jjm0j
- Heid et al. (2024). Chemprop. *J. Chem. Inf. Model.* https://doi.org/10.1021/acs.jcim.3c01250
- Ben Hicham et al. (2025). TabPFN+CheMeleon on MoleculeACE (referenced by ldbc1999 rank 65).
- CheMeleon pretrained weights: https://arxiv.org/abs/2506.15792
- MolE: https://doi.org/10.1038/s42256-024-00860-8
