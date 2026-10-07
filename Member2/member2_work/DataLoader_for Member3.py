import os
import sys
import subprocess
import argparse

# ------------------------------------------------------------
# 1. CONFIGURE PATHS – add Member 2's working directory to sys.path
# ------------------------------------------------------------
# Determine the absolute path to the local Member 2 code (no cloning needed).
MEMBER2_WORK_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "member2_work")
)
if MEMBER2_WORK_PATH not in sys.path:
    sys.path.append(MEMBER2_WORK_PATH)
    print(f"✅ Added Member 2 work path to sys.path: {MEMBER2_WORK_PATH}")
else:
    print("✅ Member 2 work path already in sys.path")

# ------------------------------------------------------------
# 2. COMMAND‑LINE / ENV‑VAR configuration for CSV locations & data root
# ------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Initialize DataLoaders for Member 3 using Member 2's split CSVs"
)
parser.add_argument(
    "--train-csv",
    default=os.getenv(
        "TRAIN_CSV_PATH",
        os.path.join(MEMBER2_WORK_PATH, "outputs", "train_split.csv"),
    ),
    help="Path to the training split CSV (default: env TRAIN_CSV_PATH or ./outputs/train_split.csv)",
)
parser.add_argument(
    "--val-csv",
    default=os.getenv(
        "VAL_CSV_PATH",
        os.path.join(MEMBER2_WORK_PATH, "outputs", "val_split.csv"),
    ),
    help="Path to the validation split CSV (default: env VAL_CSV_PATH or ./outputs/val_split.csv)",
)
parser.add_argument(
    "--data-root",
    default=os.getenv(
        "DATA_ROOT",
        "/kaggle/input/rsna-knee-abnormality-detection/",
    ),
    help="Root directory containing the DICOM series folders",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=int(os.getenv("BATCH_SIZE", "16")),
    help="Batch size for the DataLoaders (default: 16)",
)
parser.add_argument(
    "--num-workers",
    type=int,
    default=int(os.getenv("NUM_WORKERS", "2")),
    help="Number of DataLoader worker processes (default: 2; set 0 on Windows/Colab)",
)
args = parser.parse_args()

# ------------------------------------------------------------
# 3. IMPORT MEMBER 2's DataLoader helper
# ------------------------------------------------------------
try:
    from datasets import get_dataloaders
    print("✅ Successfully imported 'get_dataloaders' from Member 2's code.")
except ImportError as e:
    print(f"❌ Failed to import get_dataloaders: {e}")
    sys.exit(1)

# ------------------------------------------------------------
# 4. INITIALIZE DataLoaders using the supplied arguments
# ------------------------------------------------------------
print("\n--- Initializing DataLoaders ---")
try:
    train_loader, val_loader = get_dataloaders(
        train_csv_path=args.train_csv,
        val_csv_path=args.val_csv,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Simple sanity‑check: fetch one batch
    for images, labels in train_loader:
        print("✅ DataLoader is working correctly.")
        print(f"Image batch shape: {images.shape}")  # Expected: [B, 1, 224, 224]
        print(f"Label batch shape: {labels.shape}")  # Expected: [B, 7]
        break
except Exception as e:
    print(f"❌ Error initializing DataLoaders: {e}")
    print("Check that the CSV files exist and that DATA_ROOT points to the DICOM series directory.")
