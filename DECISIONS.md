# Decision Log

## Decision 1
- Date: 2026-10-02
- Decision: Use image-only input; do not use radiology reports.
- Reason: Keeps the two-week scope focused and tests the value of MRI images alone.

## Decision 2
- Date: 2026-10-02
- Decision: Use macro AUC as the primary metric.
- Reason: Matches the RSNA task and handles class imbalance better than accuracy.

## Decision 3
- Date: 2026-10-02
- Decision: Freeze the test set before model tuning.
- Reason: Prevents test-set leakage and overfitting to the test set.

 ## Decision 4 — Training on a representative subset

- Date: 2026-10-05
- Decision: Train initial models on a representative subset of studies instead of the full 500 GB training set.
- Rationale:
  - The full dataset is very large (~500 GB), making rapid iteration difficult within a two-week timeline.
  - A well-sampled subset allows faster experiments while preserving the multi-label nature of the task.
- Plan:
  - Sample at the study/patient level, not slice level.
  - Target ~10–20% of studies for training, with attention to preserving label distribution across the 12 abnormalities.
  - Keep fixed validation and test sets (e.g., 10% each).
  - Clearly state in the paper that models are trained on a subset of the RSNA training data.
 
  - ## Decision 5 — Revised split to ensure gold labels in all splits

- Date: 2026-10-06
- Problem:
  - Only 58 gold-labeled studies exist in the dataset.
  - Current stratified split placed all gold studies in train/val, leaving test with 0 gold-labeled studies.
  - This makes test-set evaluation meaningless for the multi-label task.
- Decision:
  - Redesign the split at the study level to ensure that train, validation, and test each contain gold-labeled studies.
  - Accept that perfect stratification across all 55 label-pattern strata is not possible with only 58 gold studies.
- Plan:
  - Explicitly allocate gold-labeled studies across splits.
  - Then distribute the remaining (non-gold) studies to reach desired split sizes.
 
  - ## Decision 6 — Approved split for Experiment 001

- Date: 2026-10-05
- Split summary:
  - Train: 3,085 studies (30 gold)
  - Val: 661 studies (14 gold)
  - Test: 661 studies (14 gold)
- All 12 abnormalities have at least 1 positive in each split.
- Some labels are rare in val/test (e.g., Lateral OA, MCL injury, Baker’s cyst), so AUROC estimates for those labels will have higher variance. This is acceptable for a first baseline.
- Decision: Use this split for Experiment 001 and subsequent experiments unless a major issue arises.

- ## Decision 7 — Use only gold-labeled studies in DataLoader and evaluation

- Date: 2026-10-10
- Problem: Non-gold studies have NaN labels, causing NaN loss and NaN weights during training.
- Decision: Filter each split to `is_gold == 1` inside `datasets.py` and the evaluation pipeline.
- Rationale: NaN labels cannot be used in binary cross-entropy. The split files remain unchanged.
- Consequence: Experiment 001 will train on 30 gold studies, validate on 14, and test on 14.
- Limitation: This is sufficient for pipeline validation but not for strong performance claims. The paper must clearly report the small labeled sample size.
