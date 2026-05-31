---
license: cc-by-4.0
tags:
- protein-ligand
- cofolding
- PXR
- Boltz2
- Chai-1
- OpenFold3
---

# PXR Challenge Multi-method Cofolding

This dataset contains >45,000 cofolding predictions using three different methods (OpenFold3, Boltz2, Chai1) for the nuclear hormone receptor hPXR and with >600 ligands. 
Ligands are derived from the OpenADMET PXR Challenge and contain known binders, known agonists, and inactives.  More on the ligand set can be found here: https://huggingface.co/spaces/openadmet/pxr-challenge

Each cofolding method was run using default settings and untemplated, except where noted.  Each protein/ligand/method/parameter set generated 20-25 structures (5 samples * 4 or 5 random seeds)

Deposited is the raw cofolding output with all confidence predictions and other data files. 

### Contents:
- `input_data`: raw SMILES tables, PDB64 rerefined input structures, hPXR LBD sequence, and ColabFold MSA used for all cofolding runs.
- `pdb64`: cofolding predictions for a 70-system rerefined PXR crystal set. Untemplated and templated with the protein crystal coordinates.
- `structure_challenge`: cofolding predictions and inputs for 184 structure-challenge ligands.
- `activity_challenge`: cofolding predictions for 513 activity-challenge ligands.
- `consolidated`: Extracted and aligned .cif files from all cofolding runs.  A .csv file with various structure and cofolding-based features for each sample.

### Top-level metadata:
- `dataset_manifest.csv`: dataset/archive summary.
- `checksums.sha256`: SHA256 checksums for repository files, excluding itself and `.gitattributes`.
- `*/manifests/*/model_manifest.csv`: row-level prediction manifests.
- `*/manifests/*/archive_inventory.csv`: complete file inventories for prediction/input archives.


### Data to be deposited soon:
- 120 ligands with high-quality hPXR activity values of various ranges
- PXR cofolding with multiple copies of ligand (structure and activity sets)
