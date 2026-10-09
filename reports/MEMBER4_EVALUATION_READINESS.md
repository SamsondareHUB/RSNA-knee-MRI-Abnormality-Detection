# Member 4 evaluation readiness

Status: **prepared; real validation and test AUROC PENDING**. No model predictions or selected checkpoint are committed to `main` as inspected on 2026-10-09. No DICOM images are needed by this evaluation path.

## Verified repository contract

- Target repository: `SamsondareHUB/RSNA-knee-MRI-Abnormality-Detection`, base `main` at `03a6ba2`.
- Canonical ID: `StudyInstanceUID`. Canonical label order from `Member2/member2_work/config.py`: `ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`, `Medial OA`, `Lateral OA`, `PF OA`, `Effusion`, `Synovitis`, `Baker's`, `Contusion`, `Fracture`.
- Gold source: `Member2/data/train.csv`; 4,407 unique study IDs, all 12 columns, 58 studies with 12 binary labels, 4,349 with blank labels. The `Report` column is ignored by evaluation.
- Approved assignments: `Member2/outputs/split.csv`; `StudyInstanceUID`, `split`, `is_gold`, all 12 labels. Its binary labels use `0.0`/`1.0` for gold rows. Its 4,407 IDs match the gold source. `is_gold` agrees with label presence. `train_split.csv`, `val_split.csv`, and `test_split.csv` match their partitions. `split_summary.json` reports train 3,085 (30 gold), val 661 (14 gold), test 661 (14 gold), and the CSV counts agree. This matches Decision 6.
- All 12 labels have positive and negative gold support in each of val and test. The smallest test positive support is one (`Lateral OA`), so AUROC will be particularly unstable. Validation and test each have only 14 gold studies. No patient ID exists in the approved artifacts. Study separation was verified; patient separation cannot be independently verified. `run_member2.py` substitutes study ID for patient ID when none exists, which is not patient identity evidence.
- An earlier 1,798-study figure appears in team discussion but is not explained by the approved artifacts. This discrepancy remains open; the checked-in Experiment 001 split is 4,407 studies.

## Implemented components

- `evaluation/member4_metrics.py` checks unique and matching IDs; one split per study; complete 12-column schemas; valid binary gold labels or explicit blanks; summary counts; split labels against the gold source when included; finite scores; probability bounds; full coverage of selected split's gold studies; non-gold exclusion; and patient overlap if a real patient ID column is supplied. It aligns by ID, never row order.
- It calculates per-label AUROC using scikit-learn, support and missing counts, and a complete 12-label macro AUROC only when all labels are evaluable. Undefined AUROC stays NaN with a reason. A separately named mean over evaluable labels is descriptive only.
- `evaluation/member4_evaluate.py` writes `per_label_auroc.csv`, `macro_auroc.json`, `label_support.csv`, `evaluation_integrity.json`, and `evaluation_summary.md` with split, file hashes, versions, labels, and support. Validation and final test are separate modes. Test mode requires a JSON selection record with `checkpoint_id`, `selection_protocol`, and `fixed_before_test: true`; the record is an auditable declaration, not independent proof of when selection occurred.
- Synthetic tests exercise 12-label scoring, perfect/inverted/tied predictions, undefined classes, missing labels, duplicate and missing IDs, ID alignment, invalid scores, non-gold exclusion, split counts, decimal binary encoding, label conflicts, patient overlap, CLI outputs, and the test gate. No MRI data, GPU, or training is needed.

## Prediction contract

Supply one CSV row per predicted study with `StudyInstanceUID` and the 12 **exact** canonical label columns above. Values are finite probabilities in `[0,1]` by default. For logits, pass `--input-type logits` explicitly. Scores for all 12 columns are mandatory. A file may contain only the 14 selected gold studies or additional non-gold studies from the same split. Every selected gold study needs a prediction. Study IDs from another split, duplicate IDs, and extra unknown IDs fail. No Member 3 prediction schema or saved prediction file is present on `main`; Member 3 should export this contract or agree on a documented adapter before evaluation.

Blank gold label cells mean missing and are excluded label by label, with a count. No blank prediction score is accepted. Non-gold studies never enter AUROC and their predicted/excluded count is recorded.

## Commands when Member 3 provides predictions

Run at repository root. The output paths below are examples and must point to real saved model predictions, never synthetic fixtures.

```bash
python3 -m evaluation.member4_evaluate \
  --gold-csv Member2/data/train.csv \
  --split-csv Member2/outputs/split.csv \
  --split-summary Member2/outputs/split_summary.json \
  --predictions-csv /path/to/validation_predictions.csv \
  --split val --output-dir /path/to/validation_results
```

After choosing and freezing the checkpoint and protocol using validation data, write a record such as:

```json
{"checkpoint_id":"<immutable checkpoint identifier>","selection_protocol":"<how validation selected it>","fixed_before_test":true}
```

Then run final test once:

```bash
python3 -m evaluation.member4_evaluate \
  --gold-csv Member2/data/train.csv \
  --split-csv Member2/outputs/split.csv \
  --split-summary Member2/outputs/split_summary.json \
  --predictions-csv /path/to/test_predictions.csv \
  --split test --selection-record /path/to/fixed_selection.json \
  --output-dir /path/to/final_test_results
```

## Remaining dependencies and blockers

- Member 3 must provide validation predictions, the checkpoint identifier, and the score column/export contract. After validation selection is fixed, Member 3 must provide final test predictions and the selection record. Real validation macro and per-label AUROC are **PENDING**; real test macro and per-label AUROC are **PENDING**.
- The current DataLoader in `Member2/member2_work/datasets.py` reads only seven labels; unlabeled rows yield NaN targets; missing DICOM files and read errors become black images; `pydicom=None` is allowed but then image reads fail into that fallback; paths are unverified here; and there is no test loader. `DataLoader_for Member3.py` also expects a `[B,7]` label shape. These are upstream training/export blockers, outside Member 4's edit scope.
- Current `run_member2.py` uses GroupShuffleSplit and does not reproduce the approved explicit 30/14/14 gold allocation from the checked-in artifact. Use the checked-in approved split, and have Member 2 resolve generator drift before any regeneration. Do not silently replace it with a 42/8/8 allocation.
- Patient overlap cannot be audited from the committed split because a verified patient ID is absent. Member 2 should provide a real patient mapping if available.
- The 14-study val/test gold sets have high statistical uncertainty; these estimates cannot establish clinical validity. Test labels must not guide epoch, threshold, or hyperparameter choices.
