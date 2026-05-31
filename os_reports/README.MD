# Open Source Reports — OpenADMET PXR Challenge

Other competitors' reports that they have open sourced.

I want to thank the participants below, and in the same spirit I will be open sourcing my own repo and report.

Rank and link as of sourcing (2026-05-31). Leaderboard MAE and Spearman ρ are from `activity_leaderboard_tgyvb15g.csv`.

## Cofolding Dataset

| Who | Link | Contents |
|-----|------|----------|
| dargason (rank 14) | [HuggingFace](https://huggingface.co/datasets/dargason/pxr-challenge-cofolding) | >45,000 cofolding predictions (Chai-1, OpenFold3, Boltz2) for all 513 activity-challenge ligands + 184 structure-challenge ligands + 64 crystal set. Consolidated CSV with 87 per-structure features: ipTM, pTM, pLDDT, PAE, contact atom pairs at 3.5/4.5/5.0 Å, clash counts, anchor-contact pass. 5 seeds × 5 samples per method per ligand. |

Raw READMEs saved in `raw/rank_14-dargason-cofolding-*.md`. Metrics CSV (`consolidated/cofolding_structure_metrics_one_row_per_structure.csv`, ~70 MB, 46,075 rows × 87 cols) is the ML-usable artifact; download separately into `data/` if needed.

---

## Open Source Submissions

| Rank | Username | MAE | Spearman ρ | Submitted | Link | Approach | Reported Score |
|------|----------|-----|-----------|-----------|------|----------|----------------|
| 11 | discoverybytes | 0.4268 | 0.832 | 2026-05-24 | [GitHub](https://github.com/discoverybytes/openadmet-pxr-blind-challenge/tree/main/activity-prediction) | Chemprop (CheMeleon-init) + UniMol + MolE/LightGBM → Ridge meta-learner | MAE 0.428, RAE 0.538 (unblinded) |
| 14 | dargason | 0.4291 | 0.795 | 2026-05-27 | [Report](https://aetherark.com/openadmet-pxr_phase1.html) | ChemProp+CheMeleon+TabICL/LGBM + analog perturbation + GNINA docking (8 PXR structures) | MAE 0.4291, R² 0.606, ρ 0.795 (2026-05-27) |
| 27 | auP7s | 0.4352 | 0.813 | 2026-05-26 | [Gist](https://gist.github.com/chemotica/a49b002eda2f7fd5eef2dca4f98f8ad7) | TabICL ensemble, 10+ descriptor blocks (MOE/Mordred/RDKit/UniMol/fingerprints), NSGA-II blend | Internal MAE 0.4609 (2026-05-05) |
| 33 | PeterBloomingdale | 0.4399 | 0.815 | 2026-05-25 | [GitHub](https://github.com/PeterBloomingdale/openadmet-pxr) | LightGBM + Chemprop v2 + TabPFN + UniMol + TabICL, scaffold (Butina) CV | — |
| 38 | firstpass | 0.4437 | 0.821 | 2026-05-26 | [HuggingFace](https://huggingface.co/lachrymator/openadmet-pxr-challenge) | SMILES seq + molecular graph + 3D conformer + docking, Butina CV, NNLS blend | RAE 0.57, R² 0.55, ρ 0.82 (2026-05-24) |
| 60 | jeremy | 0.4552 | 0.808 | 2026-05-26 | [GitHub](https://github.com/jeremycheminf/openadmet_scripts/blob/main/PXR/2D) | 2D descriptor pipeline; PDF + submission report in repo | — |
| 64 | mp-alex | 0.4567 | 0.801 | 2026-05-22 | [GitHub](https://github.com/mirror-physics/pxr_openadmet_blind_challenge/blob/main/reports/track1_activity/submission_6model_adaptive_elasticnet.md) | 6-model adaptive ElasticNet (Chemprop×3 + TorchMD-NET + GPS + ChemBERTa) | OOF RAE 0.4816 |
| 65 | ldbc1999 | 0.4571 | 0.806 | 2026-05-24 | [GitHub](https://github.com/lizyurkewych-git/pxr-challenge) | 12 submissions: KNN → LightGBM → ensemble → CheMeleon/ChemBERTa → TabPFN → UniMol + PXR crystal ligand features | Best leaderboard RAE 0.5998, CV RAE 0.5208 (Sub 12) |
| 67 | RyeCatcher | 0.4585 | 0.784 | 2026-05-19 | [HuggingFace](https://huggingface.co/RyeCatcher/openadmet-pxr-challenge-2026) | Chemprop multitask + AutoGluon + LGBM, honest per-fold isotonic (v43 = 0.78×v31 + 0.22×T1v5) | RAE 0.586, MAE 0.467, ρ 0.802 (2026-05-10) |
| 83 | adlvdl | 0.4696 | 0.790 | 2026-05-19 | [GitHub](https://github.com/adlvdl/pxr_challenge) | RF + XGBoost + ChemProp + multitask, marimo notebooks, educational workflow | — |
| 97 | Usagi | 0.4765 | 0.787 | 2026-05-24 | [GitHub](https://github.com/Usagi-ljy/pxr-challenge-better-baseline) | Morgan + MACCS + RDKit + multiple regressors, NNLS ensemble | — |
| 102 | chaospilot | 0.4787 | 0.764 | 2026-05-25 | [OneDrive](https://1drv.ms/t/c/a6144d17cf10bd2e/IQAiYJT7u4i4SaUD8h0K_SBgAWWOcR8yUpp6IgBuh-9APO8?e=kcHwyZ) | Morgan (regular + chiral + count) + RDKit descriptors → boosting ensemble + pair-delta activity-cliff correction, conservative blend | — |
| 103 | N1NC1O | 0.4790 | 0.769 | 2026-05-05 | [Gist](https://gist.github.com/dehaenw/f63f9c9bd42eacbddede706e90d8d8e9) | TabICL + MOE/Mordred/RDKit/fingerprints/UniMol, isotonic calibration + ridge | Internal MAE 0.4609 (2026-05-05) |
| 113 | QuantNova | 0.4845 | 0.751 | 2026-05-24 | [GitHub](https://github.com/chaospilot2003/openadmet) | (content failed to load) | — |
| 118 | cat554 | 0.4895 | 0.784 | 2026-05-24 | [GitHub](https://github.com/ttll1667/pxr_comp/blob/main/README_pxr_stability_stack_experiment_log.md) | Morgan + MACCS + RDKit + mol2vec + LightGBM/XGB/CatBoost/RF/Ridge stacking | — |
| 121 | leeherman99 | 0.4907 | 0.750 | 2026-05-26 | [Google Doc](https://docs.google.com/document/d/1Rhm161zVYxms7FYOXR6zVja5e62jtVTMXY1MXdyTqMM/edit?usp=sharing) | Chemprop v2: BondMessagePassing + CheMeleonFingerprint (2048-dim) multi-modal fusion, z-score norm per fold, 10-fold CV, MAE loss, CPU-only | — |
| 127 | Schnappi | 0.4962 | 0.750 | 2026-05-24 | [HuggingFace](https://huggingface.co/Schni-Schnappi/openadmet-pxr-challenge) | ChemProp two-stage: single-conc self-supervised pretraining → pEC50 finetune, 5-rep ensemble | — |
| 148 | wuhicky | 0.5063 | 0.759 | 2026-05-24 | [GitHub](https://github.com/wuhike61-cmd/pxr) | Morgan + MACCS + RDKit + PXR-SMARTS features + Ridge/ExtraTrees/RF/HistGB | — |
| 152 | axelrolov | 0.5082 | 0.747 | 2026-04-23 | [GitHub](https://github.com/AxelRolov/moltabfm_pxr) | TabICL + RDKit/MACCS + CheMeleon/ChemBERTa/MoLFormer, scaffold-grouped CV | — |
| 168 | duoduo6 | 0.5163 | 0.759 | 2026-05-25 | [GitHub](https://github.com/duoduo6660/PXR) | Morgan + RDKit + PXR-SMARTS + SAR calibration + LightGBM/XGB/ExtraTrees | — |
| 175 | Whitebox | 0.5192 | 0.711 | 2026-05-24 | [HuggingFace](https://huggingface.co/Whitebox2026/openadmet-pxr-challenge) | CheMeleon fine-tuned on PXR training data, scaffold 5-fold CV | — |
| 193 | nyota | 0.5264 | 0.716 | 2026-05-24 | [GitHub](https://github.com/junyigloria-collab/pxr-challenge-method-report) | Morgan + MACCS + RDKit + ExtraTrees/RF/Ridge/KNN + uncertainty QA | — |
| 196 | BalamuruganThirukonda | 0.5272 | 0.736 | 2026-05-17 | [GitHub](https://github.com/BalamuruganThirukonda/openadmet-pxr-pec50) | Morgan + RDKit path FP + MACCS + LightGBM, single-conc assay features, 5-fold CV | — |
| 202 | JacksonBurns | 0.5292 | 0.707 | 2026-05-18 | [Gist](https://gist.github.com/JacksonBurns/94cb5e7dda4d72bd876c947df92c5147) | MIST foundation model fine-tuned on PXR pEC50 (kekulized SMILES, 10 epochs) | — |
| 205 | duod | 0.5307 | 0.742 | 2026-05-25 | [GitHub](https://github.com/duoyou666/PXR_report) | Morgan + RDKit + Ridge/RF/ExtraTrees mean ensemble, scaffold CV | MAE 0.574, ρ 0.746 (live leaderboard) |
| 206 | ranga_31489 | 0.5315 | 0.716 | 2026-04-12 | [GitHub](https://github.com/quantumdolphin/OPENADMET-PXR-activity-pred/blob/c02fd2a86421f9a808327071c2a9d7894aaa4d02/METHOD_REPORT.md) | Uncertainty-weighted XGBoost (w=1/(SE²+σ²)), 2D descriptors + Morgan FP | Val RMSE 0.750 |
| 218 | lxduo | 0.5379 | 0.743 | 2026-05-25 | [GitHub](https://github.com/duo6660/PXR-report) | Morgan + RDKit + SAR rules (99 analog pairs) + LightGBM/XGB/ExtraTrees + NN SAR calibration | — |
| 267 | Shorku | 0.5782 | 0.624 | 2026-05-27 | [GitHub](https://github.com/Shorku/OpenADMET_Blind_Challenge_Predicting_PXR_Induction) | ChemProp D-MPNN two-stage (PXR-aware + ChEMBL-pretrained), arithmetic mean ensemble | — |
| 278 | hangyodon | 0.5946 | 0.707 | 2026-05-24 | [GitHub](https://github.com/hangyodon-ljy/pxr-challenge-baseline) | ChemProp v2, single-conc SC pretraining → pEC50 finetune, 5-replicate ensemble | — |
| 279 | k3785331526 | 0.5948 | 0.708 | 2026-05-24 | [GitHub](https://github.com/kenny3785331526/PXR) | Educational ML pipeline (RF + XGBoost + ChemProp baseline) | — |

## Not Reachable / Internal Version Names

The following entries had a `model_report_link` that was not a public URL:

| Rank | Username | MAE | Link (internal name) |
|------|----------|-----|----------------------|
| 9 | jaybirdy | 0.4257 | `caruana_full_catalog_diverse_resubmit_2026-05-26_pre-phase2` |
| 12 | rasayan-labs | 0.4274 | `qvs-chemeleon-rasayan-pxr` |
| 16 | aarshit-mittal | 0.4298 | `rasayan.ai` |
| 23 | Radi | 0.4325 | `exp135_phase1` |
| 35 | ADMETing | 0.4413 | `mtl-v3-5seed-ens-tta` |
| 55 | ashoori | 0.4508 | `phase_f_d1_replay` |
| 56 | ghazivakili | 0.4508 | `phase_f_rank20` |
| 58 | KalenJosifovski | 0.4543 | `prune_tabpfn_chemeleon_pca200_iso` |
| 69 | foobar | 0.4595 | `https://model-version.invalid/test1/68` (invalid) |
| 114 | HungryCapybara | 0.4858 | `butina_earlystop_3seed` |
| 207 | ChAndersen | 0.5316 | GitHub 404 |
| 219 | mathysgrapotte | 0.5381 | `posthoc-bbse-multitask trial-44` (internal note) |
| 221 | marco | 0.5389 | `ensv2-v35-xgb-stacked` |
| 238 | ADMETron | 0.5488 | `xgb10-stacked-xgbmeta` |
| 270 | xX-its-amit-Xx | 0.5800 | GitHub 404 |
| 289 | flowvector | 0.6174 | `lgbm-ecfp4-tanimoto-cv` |
| 298 | pz-hug | 0.6518 | `multi-task-#1` |
