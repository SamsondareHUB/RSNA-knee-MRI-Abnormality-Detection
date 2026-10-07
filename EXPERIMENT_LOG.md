# Experiment Log

## Experiment 001
- Date:
- Run by:
- Model:
- Input:
- Data split:
- Hyperparameters:
- Training loss:
- Validation macro AUC:
- Validation per-label AUC:
- Notes:
- Result status: keep / discard / revise
- 
## Data pipeline v1 — Member 2

- Updated `run_member2.py` with CLI args:
  - `--labels-csv`
  - `--series-csv`
  - `--output-dir`
- Automatically generates `train_split.csv` and `val_split.csv` aligned with Member 3’s DataLoader.
- Added pydicom fallback in `datasets.py` for environments without pydicom (sanity-check mode).
- Core logic unchanged; can run with defaults or custom paths.
