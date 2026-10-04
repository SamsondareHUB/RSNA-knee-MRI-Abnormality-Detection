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
