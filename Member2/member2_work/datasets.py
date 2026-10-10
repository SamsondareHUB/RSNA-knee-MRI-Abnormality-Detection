import os
import pandas as pd
import numpy as np
try:
    import pydicom
except ImportError:
    pydicom = None
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

# 1. Configuration

# The 7 labels found in terminal output
LABEL_COLUMNS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA'
]


STUDY_ID_COL = 'StudyInstanceUID'


KAGGLE_DATA_PATH = "/kaggle/input/rsna-knee-abnormality-detection/"


# 2. Custom Dataset Class

class KneeMRIDataset(Dataset):
    def __init__(self, dataframe, series_root_path, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): The train_split.csv or val_split.csv dataframe.
            series_root_path (str): Path to the folder containing DICOM folders.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataframe = dataframe
        self.series_root_path = series_root_path
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # 1. Extract Study ID and Labels
        study_id = self.dataframe.iloc[idx][STUDY_ID_COL]
        labels = self.dataframe.iloc[idx][LABEL_COLUMNS].values.astype(np.float32)
        labels = torch.tensor(labels, dtype=torch.float32)

        # 2. Construct the path to the patient's DICOM folder
        patient_folder = os.path.join(self.series_root_path, str(study_id))

        # 3. Find all DICOM files in the folder
        dicom_files = []
        if os.path.exists(patient_folder):
            for root, dirs, files in os.walk(patient_folder):
                for file in files:
                    if file.endswith(".dcm"):
                        dicom_files.append(os.path.join(root, file))

        # 4. Read the DICOM files and extract an image (Middle Slice approach)
        if len(dicom_files) == 0:
            # If no files found, return a black image to prevent crashing
            image = np.zeros((224, 224), dtype=np.float32)
        else:
            dicom_files.sort() # Sort files to ensure consistency
            middle_idx = len(dicom_files) // 2
            target_file = dicom_files[middle_idx]
            
            try:
                dicom_data = pydicom.dcmread(target_file)
                image = dicom_data.pixel_array.astype(float)
                
                # Normalize the image to 0-1 range
                if image.max() > image.min():
                    image = (image - image.min()) / (image.max() - image.min())
                else:
                    image = np.zeros_like(image)
            except Exception as e:
                print(f"Error reading {target_file}: {e}")
                image = np.zeros((224, 224), dtype=np.float32)

        # 5. Convert to PIL Image and apply transforms
        image = Image.fromarray(image).convert('L') # 'L' for grayscale
        
        if self.transform:
            image = self.transform(image)

        return image, labels


# 3. Helper Function for Member 3

def get_dataloaders(train_csv_path, val_csv_path, data_root=KAGGLE_DATA_PATH, batch_size=16, num_workers=2):
    """
    Creates and returns training and validation DataLoaders.
    This is the function Member 3 will import and use.
    """
    # Define Image Transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]) # Single channel for grayscale
    ])

    # Load the split CSVs
    train_df = pd.read_csv(train_csv_path)
    train_df = train_df[train_df["is_gold"] == 1].reset_index(drop=True)

    val_df = pd.read_csv(val_csv_path)
    val_df = val_df[val_df["is_gold"] == 1].reset_index(drop=True)

    test_df = pd.read_csv(test_csv_path)
    test_df = test_df[test_df["is_gold"] == 1].reset_index(drop=True)

    # Define the path to the DICOM series folder
    train_series_path = os.path.join(data_root, "train_series")

    # Create Datasets
    train_dataset = KneeMRIDataset(train_df, train_series_path, transform=transform)
    val_dataset = KneeMRIDataset(val_df, train_series_path, transform=transform)

    # Create DataLoaders

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

    print(f"✅ DataLoaders Ready!")
    print(f"   Train Batches: {len(train_loader)} | Val Batches: {len(val_loader)}")
    
    return train_loader, val_loader



# 4. Testing Block (Runs only if this file is executed directly)

if __name__ == "__main__":
    print("--- Testing DataLoader ---")
    
    # IMPORTANT: Update these paths to match your local project structure
    # If running locally, point data_root to the folder containing 'train_series'
    LOCAL_DATA_ROOT = "./data" # Change this to your actual local path if needed
    TRAIN_CSV = "outputs/train_split.csv"
    VAL_CSV = "outputs/val_split.csv"
    
    try:
        # Use batch_size=4 and num_workers=0 for local testing to avoid memory/OS errors
        train_loader, val_loader = get_dataloaders(
            train_csv_path=TRAIN_CSV,
            val_csv_path=VAL_CSV,
            data_root=LOCAL_DATA_ROOT, # Change to KAGGLE_DATA_PATH if testing on Kaggle
            batch_size=4,
            num_workers=0
        )
        
        # Fetch one batch to verify
        for images, labels in train_loader:
            print("✅ DataLoader is working correctly.")
            print("Image batch shape:", images.shape)
            print("Label batch shape:", labels.shape)
            break
            
    except Exception as e:
        print(f"❌ Error encountered: {e}")
