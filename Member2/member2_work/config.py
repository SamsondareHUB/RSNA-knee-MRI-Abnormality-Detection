"""config.py — Member 2 configuration (redone split, corrected counts).

Source: data/train.csv  (4,407 studies)

─────────────────────────────────────────────────────────────────────────
MATH NOTES
─────────────────────────────────────────────────────────────────────────

TARGET COUNTS (from Team Lead):

    gold:      Train 30    | Val 14    | Test 14    = 58
    non-gold:  Train 3055  | Val 647   | Test 647   = 4349
    total:     Train 3085  | Val 661   | Test 661   = 4407

RATIOS (relative to total 4,407):

    train = 3085 / 4407 ≈ 0.700
    val   =  661 / 4407 ≈ 0.150
    test  =  661 / 4407 ≈ 0.150

GOLD RATIOS (relative to 58 gold):

    train = 30 / 58 ≈ 0.517
    val   = 14 / 58 ≈ 0.241
    test  = 14 / 58 ≈ 0.241

NON-GOLD RATIOS (relative to 4,349):

    train = 3055 / 4349 ≈ 0.702
    val   =  647 / 4349 ≈ 0.149
    test  =  647 / 4349 ≈ 0.149

LABEL RATES:

    rate = n_positive / n_gold     (only gold studies count)

─────────────────────────────────────────────────────────────────────────
"""

from pathlib import Path


# AUTO-DETECT PROJECT ROOT

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATA_ROOT  = PROJECT_ROOT / "data"
OUTPUT_DIR = SCRIPT_DIR / "outputs"


# 1. PATHS

LABELS_CSV = DATA_ROOT / "train.csv"
SERIES_CSV = DATA_ROOT / "train_series.csv"

# 2. COLUMN NAMES

STUDY_ID_COL     = "StudyInstanceUID"
PATIENT_ID_COL   = None
MANUFACTURER_COL = None
MODEL_COL        = None

# 3. LABELS

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


# 4. EXACT TARGET COUNTS

# Gold-labeled studies (58 total)
GOLD_TRAIN = 30
GOLD_VAL   = 14
GOLD_TEST  = 14
# Sum = 58

# Non-gold studies (4,349 total)
NONGOLD_TRAIN = 3055
NONGOLD_VAL   = 647
NONGOLD_TEST  = 647
# Sum = 4,349

# Grand totals
TOTAL_STUDIES = 4407
TOTAL_GOLD    = GOLD_TRAIN + GOLD_VAL + GOLD_TEST          # 58
TOTAL_NONGOLD = NONGOLD_TRAIN + NONGOLD_VAL + NONGOLD_TEST  # 4349

SEED = 42