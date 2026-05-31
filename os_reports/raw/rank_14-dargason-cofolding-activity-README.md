# Activity Challenge cofolding data

Current Activity Challenge remote packages in this dataset repository:

- `boltz_no_template`: Boltz 1-copy/no-template package, 513 ligands x 5 models = 2,565 structures.
- `chai1_msa_seed11_25_50_67`: Chai-1 MSA package, 513 ligands x 5 seeds (11,25,50,67,68) x 5 samples = 12,825 structures.
- `openfold3_msa_seed11_25_50_67`: OpenFold3 MSA package, 513 ligands x 5 seeds (11,25,50,67,68) x 5 samples = 12,825 structures.

Note: the Chai-1 and OpenFold3 folder names predate the seed68 completion upload; the manifests now include seed68 and are the authoritative inventory.

Blind/test caveat: activity labels are not included and must not be used for training. These structure packages are raw/diagnostic artifacts until campaign QC/scoring and Validator/Data Czar gates pass.
