# PXR HF cofolding structure metrics CSV

This README documents the one-row-per-structure metrics table derived from the consolidated aligned Hugging Face cofolding dataset.

## Files

- Metrics CSV: `cofolding_structure_metrics_one_row_per_structure.csv`
- Data dictionary CSV: `cofolding_structure_metrics_data_dictionary.csv`
- Validation summary JSON: `cofolding_structure_metrics_validation.json`
- Checksums: `SHA256SUMS`

Metrics CSV path in this repository:

`consolidated/cofolding_structure_metrics_one_row_per_structure.csv`

## Dataset summary

- Rows: 46,075
- Columns / labels: 87
- Unique structure IDs: 46,075
- Unique canonical CIF names: 46,075
- Metrics CSV SHA256: `879a4b3a22c40b382e829ced88b2d6c0509dd15df8e244dfdf7dadcfb02bea50`
- Data dictionary SHA256: `b99bfb0970f27ee2a92514f4898547c8c278fa1b832392ea2a3e4083e9e85c70`

## Rows by ligand set

- activity_challenge: 28,215
- pdb64: 10,500
- structure_challenge: 7,360

## Rows by consolidated method/setup label

- boltz2::boltz2_no_template: 5,430
- boltz2::boltz2_self_template: 1,750
- boltz::boltz_no_template: 2,565
- chai1::chai1_msa: 12,825
- chai1::chai1_no_template: 1,750
- chai1::chai1_self_template: 1,750
- openfold3::openfold3_msa: 12,825
- openfold3::openfold3_no_template: 5,430
- openfold3::openfold3_self_template: 1,750

## Source method/setup labels preserved for provenance

- boltz2::boltz2_no_template: 5,430
- boltz2::boltz2_self_template: 1,750
- boltz::boltz_no_template: 2,565
- chai1::chai1_msa_seed11_25_50_67: 12,825
- chai1::chai1_no_template: 1,750
- chai1::chai1_self_template: 1,750
- openfold3::openfold3_msa_seed11_25_50_67: 12,825
- openfold3::openfold3_no_template: 5,430
- openfold3::openfold3_self_template: 1,750

## Label consolidation note

Historical/incomplete-run seed-list labels were removed from the public analysis columns `setup` and `setup_slug` now that the datasets are complete:

- `chai1_msa_seed11_25_50_67` -> `chai1_msa`
- `openfold3_msa_seed11_25_50_67` -> `openfold3_msa`

The original source labels are retained in `source_setup` and `source_setup_slug`. Individual structure seeds are retained in `seed`.

Seed-list rows remaining in consolidated setup labels: 0

## Primary score note

`cofold_primary_score` is a pragmatic cross-method sorting/QC surrogate, not a scientific assertion that all method scores are equivalent. It is selected deterministically from the first available field in this priority order:

1. `confidence_score` -> represented as `cofold_confidence`
2. `aggregate_score` -> represented as `cofold_aggregate_score`
3. `iptm` -> represented as `cofold_iptm`
4. `ptm` -> represented as `cofold_ptm`

`cofold_primary_score_source` records which source field was used.

Primary score source counts:

- aggregate_score: 16,325
- confidence_score: 9,745
- iptm: 20,005

## Critical non-missing counts

- cofold_aggregate_score: 16,325
- cofold_confidence: 9,745
- cofold_iptm: 46,075
- cofold_primary_score: 46,075
- cofold_ptm: 46,075
- structure_anchor_contact_pass: 46,075
- structure_contact_atom_pairs_4p5a: 46,075
- structure_min_ligand_protein_distance_a: 46,075

## Manifest of CSV labels

The table below lists every column label in `cofolding_structure_metrics_one_row_per_structure.csv`, in CSV order.

## Identity and dataset labels

| # | Column label | Description |
|---:|---|---|
| 1 | `structure_id` | Stable source prediction identifier from cofolding_info_all.csv. |
| 2 | `canonical_cif_name` | Canonical aligned CIF basename: ligset_ligandID_method_template_seedXX_model_N.cif. |
| 3 | `aligned_cif_relpath` | Path to aligned CIF relative to dataset output root. |
| 4 | `ligand_set` | Dataset stratum: pdb64, activity_challenge, or structure_challenge. |
| 5 | `ligand_id` | Ligand/system identifier in the source manifest. |
| 6 | `method` | Human-readable cofolding method label. |
| 7 | `method_slug` | Normalized method slug. |
| 8 | `setup` | Consolidated setup/template/MSA label with historical seed-list run labels collapsed after completion. |
| 9 | `setup_slug` | Normalized consolidated setup slug. |
| 10 | `seed` | Source seed token; 'na' when no explicit seed exists. |
| 11 | `sample_index0` | Zero-based sample index harmonized across source methods where available. |
| 12 | `model_index0` | Zero-based model index harmonized across source methods where available. |
| 13 | `rank_in_target` | Rank within ligand/method/setup target from source/derived confidence ordering. |
| 14 | `is_best_confidence` | Whether this row was marked best-confidence within target. |
| 15 | `smiles` | See source cofolding_info_all.csv/provenance manifest. |
| 16 | `system_id` | See source cofolding_info_all.csv/provenance manifest. |
| 17 | `pdb_id` | See source cofolding_info_all.csv/provenance manifest. |
| 18 | `ligand_ccd` | See source cofolding_info_all.csv/provenance manifest. |
| 19 | `ligand_chain` | See source cofolding_info_all.csv/provenance manifest. |
| 20 | `ligand_resseq` | See source cofolding_info_all.csv/provenance manifest. |
## Source provenance labels

| # | Column label | Description |
|---:|---|---|
| 1 | `source_repo` | See source cofolding_info_all.csv/provenance manifest. |
| 2 | `source_revision` | See source cofolding_info_all.csv/provenance manifest. |
| 3 | `source_setup` | Original source setup label before consolidation. |
| 4 | `source_setup_slug` | Original source setup slug before consolidation. |
| 5 | `source_manifest_path` | See source cofolding_info_all.csv/provenance manifest. |
| 6 | `source_manifest_row` | See source cofolding_info_all.csv/provenance manifest. |
| 7 | `source_archive_path` | See source cofolding_info_all.csv/provenance manifest. |
| 8 | `source_member_path` | See source cofolding_info_all.csv/provenance manifest. |
| 9 | `source_score_member_paths` | See source cofolding_info_all.csv/provenance manifest. |
| 10 | `raw_structure_format` | See source cofolding_info_all.csv/provenance manifest. |
| 11 | `score_source_type` | See source cofolding_info_all.csv/provenance manifest. |
| 12 | `score_extraction_status` | See source cofolding_info_all.csv/provenance manifest. |
| 13 | `collation_status` | See source cofolding_info_all.csv/provenance manifest. |
## Cofolding confidence / error metrics

| # | Column label | Description |
|---:|---|---|
| 1 | `cofold_primary_score` | Single best available source confidence/ranking surrogate for coarse sorting: confidence_score, else aggregate_score, else iptm, else ptm. |
| 2 | `cofold_primary_score_source` | Source field used for cofold_primary_score. |
| 3 | `cofold_confidence` | Primary method confidence score, harmonized from confidence_score. |
| 4 | `cofold_ranking_score` | Ranking score when provided by source method. |
| 5 | `cofold_aggregate_score` | Aggregate score when provided by source method. |
| 6 | `cofold_ptm` | Predicted TM-score / pTM when provided. |
| 7 | `cofold_iptm` | Interface pTM / ipTM when provided. |
| 8 | `cofold_ligand_iptm` | Ligand-side/interface ipTM when provided. |
| 9 | `cofold_protein_iptm` | Protein-side ipTM when provided. |
| 10 | `cofold_complex_plddt` | Complex-level pLDDT/confidence when provided. |
| 11 | `cofold_complex_iplddt` | Complex interface pLDDT when provided. |
| 12 | `cofold_complex_pde` | Complex predicted distance error when provided. |
| 13 | `cofold_complex_ipde` | Complex interface predicted distance error when provided. |
| 14 | `cofold_mean_plddt` | Mean pLDDT when provided by source method. |
| 15 | `cofold_mean_pae` | Mean predicted aligned error when provided. |
| 16 | `cofold_mean_ipae` | Mean interface PAE when provided. |
| 17 | `cofold_mean_pde` | Mean predicted distance error when provided. |
| 18 | `cofold_mean_ipde` | Mean interface predicted distance error when provided. |
| 19 | `cofold_chain_0_ptm` | Chain 0 pTM when provided. |
| 20 | `cofold_chain_1_ptm` | Chain 1 pTM when provided. |
| 21 | `cofold_iptm_0_1` | Directional interface pTM chain 0 to chain 1 when provided. |
| 22 | `cofold_iptm_1_0` | Directional interface pTM chain 1 to chain 0 when provided. |
| 23 | `cofold_pair_iptm_protein_ligand_mean` | Mean pair ipTM between protein and ligand chains when derived/provided. |
## Alignment labels

| # | Column label | Description |
|---:|---|---|
| 1 | `alignment_status` | See source cofolding_info_all.csv/provenance manifest. |
| 2 | `alignment_ca_count` | Number of CA atoms used in stable-backbone alignment. |
| 3 | `alignment_ca_rmsd_a` | Stable-backbone alignment RMSD in Angstrom. |
| 4 | `alignment_reference_id` | See source cofolding_info_all.csv/provenance manifest. |
## Derived structural geometry/contact features

| # | Column label | Description |
|---:|---|---|
| 1 | `structure_id` | Stable source prediction identifier from cofolding_info_all.csv. |
| 2 | `structure_tier0_status` | Status of basic structural feature extraction. |
| 3 | `structure_protein_heavy_atom_count` | Protein heavy atom count extracted from aligned CIF. |
| 4 | `structure_ligand_heavy_atom_count` | Ligand heavy atom count extracted from aligned CIF. |
| 5 | `structure_ligand_residue_count` | Ligand residue count extracted from aligned CIF. |
| 6 | `structure_ligand_centroid_x` | Ligand heavy-atom centroid x coordinate after alignment. |
| 7 | `structure_ligand_centroid_y` | Ligand heavy-atom centroid y coordinate after alignment. |
| 8 | `structure_ligand_centroid_z` | Ligand heavy-atom centroid z coordinate after alignment. |
| 9 | `structure_ligand_radius_of_gyration_a` | Ligand heavy-atom radius of gyration in Angstrom. |
| 10 | `structure_min_ligand_protein_distance_a` | Minimum ligand-protein heavy atom distance in Angstrom. |
| 11 | `structure_mean_nearest_ligand_protein_distance_a` | Mean nearest protein-heavy-atom distance over ligand heavy atoms in Angstrom. |
| 12 | `structure_contact_atom_pairs_3p5a` | Ligand-protein heavy atom pairs within 3.5 Angstrom. |
| 13 | `structure_contact_atom_pairs_4p5a` | Ligand-protein heavy atom pairs within 4.5 Angstrom. |
| 14 | `structure_contact_atom_pairs_5p0a` | Ligand-protein heavy atom pairs within 5.0 Angstrom. |
| 15 | `structure_clash_atom_pairs_2p0a` | Ligand-protein heavy atom pairs within 2.0 Angstrom. |
| 16 | `structure_contact_residue_count_4p5a` | Protein residue count with any ligand heavy atom contact within 4.5 Angstrom. |
| 17 | `structure_contact_residue_ids_4p5a` | Semicolon-delimited contacting residue IDs at 4.5 Angstrom. |
| 18 | `structure_anchor_atom_count` | Anchor-region atom count used for Tier1 contact summary. |
| 19 | `structure_anchor_min_distance_a` | Minimum distance to anchor-region atoms in Angstrom. |
| 20 | `structure_anchor_contact_atom_pairs_4p5a` | Ligand-anchor atom contact pairs within 4.5 Angstrom. |
| 21 | `structure_anchor_contact_residue_count_4p5a` | Anchor residue count contacted within 4.5 Angstrom. |
| 22 | `structure_anchor_contact_pass` | Boolean Tier1 anchor-contact pass flag. |
## Derived helper flags / normalized features

| # | Column label | Description |
|---:|---|---|
| 1 | `derived_has_confidence_score` | True if cofold_confidence is present. |
| 2 | `derived_has_iptm` | True if cofold_iptm is present. |
| 3 | `derived_has_structural_features` | True if Tier0 structural extraction status is ok. |
| 4 | `derived_contact_atoms_4p5_per_ligand_heavy_atom` | 4.5A contact atom-pair count normalized by ligand heavy atom count. |
| 5 | `derived_contact_residues_4p5_per_ligand_heavy_atom` | 4.5A contact residue count normalized by ligand heavy atom count. |
| 6 | `derived_clash_atoms_2p0_per_ligand_heavy_atom` | 2.0A clash atom-pair count normalized by ligand heavy atom count. |


## Hugging Face consolidated folder layout

For Hub compatibility, CIF files in this uploaded package are sharded under `consolidated/{ligand_set}/{setup_slug}/` and, for the large MSA activity sets, `consolidated/{ligand_set}/{setup_slug}/seed_{seed}/`. This avoids the Hugging Face/Git limit of 10,000 files per directory. The `aligned_cif_relpath` column in the uploaded CSV points to these sharded paths. Canonical CIF basenames are unchanged.
