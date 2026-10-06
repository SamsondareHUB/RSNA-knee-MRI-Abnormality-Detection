import os
import sys
import subprocess


# 1. CLONE MEMBER 2's REPOSITORY

# GitHub repository URL provided by Member 2
REPO_URL = "https://github.com/SamsondareHUB/RSNA-knee-MRI-Abnormality-Detection.git"

# Define the clone directory based on the environment
# Use "/content/..." for Google Colab, or "/kaggle/working/..." for Kaggle Notebook
CLONE_DIR = "/content/RSNA-knee-MRI-Abnormality-Detection" 

if not os.path.exists(CLONE_DIR):
    print("📥 Cloning Member 2's repository...")
    subprocess.run(["git", "clone", REPO_URL, CLONE_DIR], check=True)
else:
    print("✅ Repository already exists. Pulling latest changes...")
    subprocess.run(["git", "-C", CLONE_DIR, "pull"], check=True)


# 2. ADD MEMBER 2's CODE TO SYSTEM PATH

# This is the folder containing Member 2's 'datasets.py' file
MEMBER2_WORK_PATH = os.path.join(CLONE_DIR, "Member2", "member2_work")

if MEMBER2_WORK_PATH not in sys.path:
    sys.path.append(MEMBER2_WORK_PATH)
    print(f"✅ Added to system path: {MEMBER2_WORK_PATH}")
else:
    print("✅ Path already in system path.")

# ==========================================
# 3. IMPORT MEMBER 2's DATALOADER FUNCTION
# ==========================================
try:
    from datasets import get_dataloaders
    print("✅ Successfully imported 'get_dataloaders' from Member 2's code.")
except ImportError as e:
    print(f"❌ Failed to import. Error: {e}")
    print("Please ensure Member 2 has uploaded 'datasets.py' to the GitHub repository.")

# ==========================================
# 4. SETUP PATHS FOR DATA AND CSV FILES
# ==========================================
# Path to the DICOM data (Kaggle Competition Data)
# If using Kaggle: "/kaggle/input/rsna-knee-abnormality-detection/"
# If using Colab: Update this to the path where you downloaded the data
DATA_ROOT = "/kaggle/input/rsna-knee-abnormality-detection/"

# Path to the split CSVs created by Member 2
TRAIN_CSV_PATH = os.path.join(MEMBER2_WORK_PATH, "outputs", "train_split.csv")
VAL_CSV_PATH = os.path.join(MEMBER2_WORK_PATH, "outputs", "val_split.csv")

# ==========================================
# 5. INITIALIZE DATALOADERS
# ==========================================
print("\n--- Initializing DataLoaders ---")
try:
    train_loader, val_loader = get_dataloaders(
        train_csv_path=TRAIN_CSV_PATH,
        val_csv_path=VAL_CSV_PATH,
        data_root=DATA_ROOT,
        batch_size=16,   # Lower this to 8 or 4 if you run out of memory
        num_workers=2    # Set to 0 if using Windows/Colab and encountering errors
    )
    
    # Test the DataLoader
    for images, labels in train_loader:
        print("✅ DataLoader is working correctly.")
        print("Image batch shape:", images.shape) # Expected: [16, 1, 224, 224]
        print("Label batch shape:", labels.shape) # Expected: [16, 7]
        break
        
except Exception as e:
    print(f"❌ Error initializing DataLoaders: {e}")
    print("Check if Member 2's CSV files exist at the specified paths.")
