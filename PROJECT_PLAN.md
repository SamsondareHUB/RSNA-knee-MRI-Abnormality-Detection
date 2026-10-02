# Project Plan

## Project Title
Image-Only Deep Learning Baselines for Multi-Label Knee MRI Abnormality Detection

## Research Question
How accurately can image-only deep-learning models detect 12 knee MRI
abnormalities without using radiology reports?

 ## Clinical Motivation
Knee MRI studies contain multiple sequences and many slices, making
interpretation time-consuming. An image-only model could help flag likely
abnormalities for earlier radiologist review. This project does not replace
radiologists; it explores whether imaging alone can support prioritization
and decision support.

## Dataset
RSNA Knee Abnormality Detection dataset.

## Task
Multi-label classification: predict probabilities for 12 knee abnormalities
from MRI images.

## Labels
1. ACL injury
2. MCL injury
3. Medial meniscus tear
4. Lateral meniscus tear
5. Medial osteoarthritis
6. Lateral osteoarthritis
7. Patellofemoral osteoarthritis
8. Effusion
9. Synovitis
10. Baker’s cyst
11. Contusion
12. Fracture

## Primary Metric
Macro-averaged ROC-AUC across all 12 labels.

## Secondary Metrics
- Per-label ROC-AUC
- Accuracy
- Macro F1
- Precision and recall per label
- Calibration analysis
- Grad-CAM or attention-based explanations

## Models
1. Baseline: pretrained CNN slice encoder + simple pooling
2. Improved model: pretrained CNN/ConvNeXt/ViT slice encoder + attention pooling
3. Ablation: single MRI series vs multiple MRI series
4. Ablation: CNN vs ConvNeXt/ViT backbone

## Data Policy
- Use a fixed train/validation/test split.
- Do not use the test set for model selection.
- Document all preprocessing steps.
- Keep random seeds fixed for reproducibility.

# Timeline
- Days 1–2: Project setup, data inspection, research plan
- Days 3–5: Baseline model and initial results
- Days 6–8: Improved model and ablations
- Days 9–10: Explainability and final evaluation
- Days 11–12: Manuscript drafting
- Days 13–14: Review, reproducibility check, submission preparation

## Target Output
- GitHub repository
- arXiv preprint
- IEEE ISBI 2027 submission if ready by the deadline
