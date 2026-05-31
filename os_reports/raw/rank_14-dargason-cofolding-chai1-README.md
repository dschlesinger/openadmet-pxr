# Activity Challenge Chai-1 MSA seed 11/25/50/67/68 archive package

This folder contains Activity513 blind/test-side Chai-1 cofolding prediction archives for seeds 11, 25, 50, 67, and 68.

Scope:
- Ligands: 513 Activity Challenge blind/test ligands
- Seeds: 11, 25, 50, 67, 68
- Samples per ligand per seed: 5
- Structures: 12,825 CIFs = 513 x 5 x 5
- Ligand copy count: 1

Manifest:
- `activity_challenge/manifests/chai1_msa_seed11_25_50_67/model_manifest.csv`
- `activity_challenge/manifests/chai1_msa_seed11_25_50_67/archive_manifest.csv`

Caveats:
- Blind/test labels are not included and must not be used for training.
- These are raw/diagnostic structural artifacts until downstream QC/scoring and Validator/Data Czar gates pass.
