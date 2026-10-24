# RSNA Knee Abnormality Detection - Member 2: Data Engineering & Pipeline

This repository contains the data preparation, stratified study-level splitting, DICOM processing pipeline, and PyTorch dataloaders for the **RSNA Knee Abnormality Detection** project (Member 2 module).

---

## 📂 Component Overview (Member 2 Responsibilities)
- **Study-Level Stratification:** Ensures patient-level separation without data leakage across splits, incorporating both gold-labeled and non-gold studies.
- **DICOM Middle-Slice Extraction:** Automatically traverses `/kaggle/input/rsna-knee-abnormality-detection/train_series/<study_id>/**/*.dcm`, extracts the middle slice of each study, normalizes pixel intensities, and converts them into standard RGB tensors.
- **12-Label Schema Management:** Focuses strictly on the 12 official abnormality labels (excluding the `Report` column).
- **Reproducible Splits (`study_splits.csv`):** Generates and exports the official split configuration (`study_id`, `split`, `is_gold`) for downstream model training (Member 3) and evaluation (Member 4).

---

## 📊 Dataset Split Plan
- **Total studies:** 1,798
- **Gold-Labeled Studies (58 total):**
  - **Train:** 30 studies
  - **Validation:** 14 studies
  - **Test:** 14 studies
- **Non-Gold Studies (1,740 total):**
  - **Train:** 894 studies
  - **Validation:** 425 studies
  - **Test:** 421 studies
- **Final Split Totals:**
  - **Train:** 924 studies (30 Gold + 894 Non-gold)
  - **Validation:** 439 studies (14 Gold + 425 Non-gold)
  - **Test:** 435 studies (14 Gold + 421 Non-gold)

---

## 🏷️ Official 12-Label Schema
1. `ACL`
2. `MCL`
3. `Medial Meniscus`
4. `Lateral Meniscus`
5. `Medial OA`
6. `Lateral OA`
7. `PF OA`
8. `Effusion`
9. `Synovitis`
10. `Baker's`
11. `Contusion`
12. `Fracture`

---

## 🚀 How to Use (For Team Members)

### 1. Kaggle Environment Setup
Attach the official Kaggle competition dataset to your notebook:
`/kaggle/input/rsna-knee-abnormality-detection/`

### 2. Generating Splits & Initializing DataLoaders
Run the complete pipeline script in your Kaggle notebook to automatically generate `study_splits.csv` and set up the PyTorch DataLoaders:

```python
# The script automatically detects the Kaggle path, generates study_splits.csv, 
# prepares the RSNAKneeDataset class, and creates train_loader, val_loader, and test_loader.
