import os
import pandas as pd
import numpy as np
import pydicom
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from config import LABELS


# 1. Configuration

# Use the full label list from config (LABELS).
# IMPORTANT: These names must exactly match the column names in your CSVs.
LABEL_COLUMNS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

STUDY_ID_COL = 'StudyInstanceUID'

KAGGLE_DATA_PATH = "/kaggle/input/rsna-knee-abnormality-detection/"


# 2. Custom Dataset Class

class KneeMRIDataset(Dataset):
    def __init__(self, dataframe, series_root_path, transform=None):
        self.dataframe = dataframe
        self.series_root_path = series_root_path
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        study_id = self.dataframe.iloc[idx][STUDY_ID_COL]
        labels = self.dataframe.iloc[idx][LABELS].values.astype(np.float32)
        labels = torch.tensor(labels, dtype=torch.float32)

        patient_folder = os.path.join(self.series_root_path, str(study_id))

        dicom_files = []
        if os.path.exists(patient_folder):
            for root, dirs, files in os.walk(patient_folder):
                for file in files:
                    if file.endswith(".dcm"):
                        dicom_files.append(os.path.join(root, file))

        if len(dicom_files) == 0:
            # No DICOM files – raise explicit error
            raise FileNotFoundError(f"No DICOM files found for study {study_id} in {patient_folder}")
            image = np.zeros((224, 224), dtype=np.float32)
        else:
            dicom_files.sort()
            middle_idx = len(dicom_files) // 2
            target_file = dicom_files[middle_idx]

            try:
                # Read DICOM and apply rescale
                dicom_data = pydicom.dcmread(target_file)
                image = dicom_data.pixel_array.astype(float)
                # Apply DICOM rescale slope/intercept if present
                slope = getattr(dicom_data, "RescaleSlope", 1.0)
                intercept = getattr(dicom_data, "RescaleIntercept", 0.0)
                image = image * slope + intercept
                # Clip intensities to robust percentiles (1‑99)
                low, high = np.percentile(image, [1, 99])
                image = np.clip(image, low, high)
                # Normalize to [0, 1]

                if image.max() > image.min():
                    image = (image - image.min()) / (image.max() - image.min())
                else:
                    image = np.zeros_like(image)
                # End of DICOM processing
            except Exception as e:
                print(f"Error reading {target_file}: {e}")
                image = np.zeros((224, 224), dtype=np.float32)

        image = Image.fromarray(image).convert('L')

        if self.transform:
            image = self.transform(image)

        return image, labels


# 3. Helper Function for Member 3

def get_dataloaders(train_csv_path, val_csv_path, test_csv_path=None, data_root=KAGGLE_DATA_PATH, batch_size=16, num_workers=2):
    """
    Creates and returns training and validation DataLoaders.
    This is the function Member 3 will import and use.
    """
    # Define Image Transformations
def get_dataloaders(train_csv_path, val_csv_path, test_csv_path=None,
                    data_root=KAGGLE_DATA_PATH, batch_size=16, num_workers=2):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Load and filter splits: keep only gold-labeled studies
    train_df = pd.read_csv(train_csv_path)
    train_df = train_df[train_df["is_gold"] == 1].reset_index(drop=True)

    val_df = pd.read_csv(val_csv_path)
    val_df = val_df[val_df["is_gold"] == 1].reset_index(drop=True)

    test_df = None
    if test_csv_path is not None:
        test_df = pd.read_csv(test_csv_path)
        test_df = test_df[test_df["is_gold"] == 1].reset_index(drop=True)

    train_series_path = os.path.join(data_root, "train_series")

    # Create Datasets
    # Optionally create test dataset and loader if a test CSV is provided
    test_loader = None
    if test_csv_path is not None:
        test_df = pd.read_csv(test_csv_path)
        test_dataset = KneeMRIDataset(test_df, train_series_path, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

    train_dataset = KneeMRIDataset(train_df, train_series_path, transform=transform)
    val_dataset = KneeMRIDataset(val_df, train_series_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, drop_last=False)

    print("DataLoaders ready")
    print(f"Train studies: {len(train_df)} | Val studies: {len(val_df)}")
    if test_df is not None:
        print(f"Test studies: {len(test_df)}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    print(f"✅ DataLoaders Ready!   Train:{len(train_loader)} Val:{len(val_loader)}" + (f" Test:{len(test_loader)}" if test_loader else ""))
    
    return train_loader, val_loader, test_loader
    return train_loader, val_loader


# 4. Testing block

if __name__ == "__main__":
    print("--- Testing DataLoader ---")

    LOCAL_DATA_ROOT = "./Member2/data"  # adjust if your DICOM folders are elsewhere
    TRAIN_CSV = "Member2/outputs/train_split.csv"
    VAL_CSV = "Member2/outputs/val_split.csv"

    try:
        train_loader, val_loader = get_dataloaders(
            train_csv_path=TRAIN_CSV,
            val_csv_path=VAL_CSV,
            data_root=LOCAL_DATA_ROOT,
            batch_size=4,
            num_workers=0
        )

        for images, labels in train_loader:
            print("DataLoader working")
            print("Image batch shape:", images.shape)
            print("Label batch shape:", labels.shape)
            print("NaN labels:", labels.isnan().sum().item())
            break

    except Exception as e:
        print(f"Error: {e}")