"""config.py — Member 2 configuration."""

from pathlib import Path
# 1. PATHS
DATA_ROOT  = Path("./data")
LABELS_CSV = DATA_ROOT / "train.csv"
SERIES_CSV = DATA_ROOT / "train_series.csv"


# 2. COLUMN NAMES
STUDY_ID_COL     = "StudyInstanceUID"
PATIENT_ID_COL   = None
MANUFACTURER_COL = None
MODEL_COL        = None


# 3. LABELS — 12 target columns

LABELS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]


# 4. SPLIT SETTINGS

SEED        = 42
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

OUTPUT_DIR  = Path("./member2_work/outputs")