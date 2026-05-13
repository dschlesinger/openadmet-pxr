# Train/Test Split Analysis (By Claude 4.6 Sonnet)

## Conclusion

The PXR challenge uses a **prospective (temporal) split**, not a scaffold-based split.
Test compounds were registered after the training set was assembled, mimicking real drug discovery
where a model must predict activity for the next round of synthesized compounds.

---

## Dataset sizes

| Split | Compounds |
|-------|-----------|
| Train | 4,139 |
| Test  | 513 |

---

## Evidence

### 1. OADMET ID ordering

Each compound carries an OADMET registration ID that reflects synthesis/registration order.

| | Min ID | Max ID |
|-|--------|--------|
| Train | 927 | 6,089 |
| Test  | 4,674 | 6,617 |

**510 / 513 (99.4%) of test IDs are strictly greater than the training maximum (6,089).**
Only three test compounds (IDs 4,674–4,676) fall within the training ID range — likely a small
batch held back at assignment time.

### 2. No leakage at the compound level

- Exact SMILES overlap between train and test: **0**
- Molecule Name overlap: **0**

### 3. Scaffold novelty is a consequence, not the mechanism

Murcko scaffold analysis confirms the prospective nature of the split:

| Metric | Value |
|--------|-------|
| Unique scaffolds in train | 3,670 |
| Unique scaffolds in test  | 369 |
| Scaffolds shared (train ∩ test) | 35 (9.5% of test scaffolds) |
| Test molecules with a scaffold unseen in train | 416 / 513 **(81.1%)** |
| Test molecules whose scaffold appears in train | 97 / 513 (18.9%) |

The 81% scaffold novelty is high, but it is a by-product of the temporal split rather than an
explicit scaffold-splitting constraint. Later design cycles naturally explored different chemical
matter.

### 4. Moderate Tanimoto nearest-neighbor similarity

Morgan fingerprint (r=2, 2048 bits) Tanimoto similarity from each test compound to its nearest
training neighbour (sample of 50 test molecules):

| Statistic | Value |
|-----------|-------|
| Mean NN Tanimoto | 0.553 |
| Median NN Tanimoto | 0.533 |
| % with NN sim > 0.4 | 100% |
| % with NN sim > 0.7 | 6% |
| Min / Max | 0.42 / 0.81 |

Test compounds are structurally related to the training chemical space (same screening campaign)
but are not close analogues in the majority of cases.

---

## Implications for modelling

- **Scaffold interpolation is not sufficient** — 81% of test scaffolds are unseen, so a model
  that memorises scaffold–activity relationships will struggle.
- **Generalisation across chemical space is required** — the task is closer to lead-hopping than
  to analogue activity prediction.
- **Uncertainty estimates matter** — high scaffold novelty means predictions for test compounds
  sit at or beyond the edge of the training distribution; well-calibrated uncertainty is valuable.
- **The local val split (random 10%) is optimistic** — it does not replicate the prospective
  nature of the test set. A scaffold- or time-ordered internal validation split would give a
  more realistic estimate of generalisation performance.
