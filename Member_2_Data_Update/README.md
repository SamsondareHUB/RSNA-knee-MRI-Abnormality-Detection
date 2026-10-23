# RSNA Knee Abnormality Detection - Data Engineering Documentation

## 📂 Directory & Paths (Kaggle Environment)
- Raw Dataset Path: `/kaggle/input/competitions/rsna-knee-abnormality-detection`
- Images Path: `/kaggle/input/competitions/rsna-knee-abnormality-detection/train_images`
- Metadata CSV: `/kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv`
- Processed Split CSV: `/kaggle/working/study_splits.csv`

## 📊 Study-Level Split Distribution
"To prevent data leakage, the data has been strictly partitioned at the patient/study level (study-level) as follows:"
- **Train Split**: 30 Gold + 894 Non-gold = **924 studies**
- **Val Split**: 14 Gold + 425 Non-gold = **439 studies**
- **Test Split**: 14 Gold + 421 Non-gold = **435 studies**
- **Total**: **1,798 studies** (4,407 total rows in metadata)

## 🩺 Target Abnormalities (12 Classes)
"Included abnormality labels and report columns:"
- ACL, MCL, Medial Meniscus, Lateral Meniscus
- Medial OA, Lateral OA, PF OA
- Effusion, Synovitis, Baker's, Contusion, Fracture
- Report (Text Description)
