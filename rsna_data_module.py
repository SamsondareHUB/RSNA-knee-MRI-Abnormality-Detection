%%writefile rsna_data_module.py
import os
import numpy as np
import pandas as pd
import torch
import pydicom
import cv2
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

def prepare_study_splits(metadata_csv_path="/kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv", 
                         output_csv_path="/kaggle/working/study_splits.csv", 
                         seed=42):
    """
    Member 2: Generates and saves study-level splits with exact gold/non-gold distribution.
    """
    np.random.seed(seed)
    print("=== [Data Module] Generating/Checking Study-Level Splits ===")
    
    if os.path.exists(output_csv_path):
        print(f"Split CSV already exists at {output_csv_path}. Loading existing split.")
        return pd.read_csv(output_csv_path)

    if os.path.exists(metadata_csv_path):
        df = pd.read_csv(metadata_csv_path)
    else:
        raise FileNotFoundError(f"Metadata CSV not found at {metadata_csv_path}")

    study_id_col = None
    for col in ['StudyInstanceUID', 'study_id', 'study_instance_uid', 'patient_id']:
        if col in df.columns:
            study_id_col = col
            break
    if study_id_col is None:
        study_id_col = df.columns[0]
        
    if study_id_col != 'study_id':
        df = df.rename(columns={study_id_col: 'study_id'})

    if 'is_gold' not in df.columns:
        gold_candidates = [col for col in df.columns if 'gold' in col.lower()]
        if gold_candidates:
            df['is_gold'] = df[gold_candidates[0]]
        else:
            unique_studies = df['study_id'].unique()
            np.random.seed(seed)
            gold_studies = np.random.choice(unique_studies, size=min(58, len(unique_studies)), replace=False)
            df['is_gold'] = df['study_id'].isin(gold_studies).astype(int)

    gold_df = df[df['is_gold'] == 1].drop_duplicates(subset=['study_id']).reset_index(drop=True)
    non_gold_df = df[df['is_gold'] == 0].drop_duplicates(subset=['study_id']).reset_index(drop=True)
    
    if len(non_gold_df) > 1740:
        non_gold_df = non_gold_df.sample(n=1740, random_state=seed).reset_index(drop=True)

    gold_train, gold_temp = train_test_split(gold_df, test_size=28, random_state=seed, shuffle=True)
    gold_val, gold_test = train_test_split(gold_temp, test_size=14, random_state=seed, shuffle=True)
    gold_train['split'], gold_val['split'], gold_test['split'] = 'train', 'val', 'test'
    gold_split_df = pd.concat([gold_train, gold_val, gold_test])

    non_gold_train, non_gold_temp = train_test_split(non_gold_df, test_size=846, random_state=seed, shuffle=True)
    non_gold_val, non_gold_test = train_test_split(non_gold_temp, test_size=421, random_state=seed, shuffle=True)
    non_gold_train['split'], non_gold_val['split'], non_gold_test['split'] = 'train', 'val', 'test'
    non_gold_split_df = pd.concat([non_gold_train, non_gold_val, non_gold_test])

    final_split_df = pd.concat([gold_split_df, non_gold_split_df]).reset_index(drop=True)
    
    output_columns = ['study_id', 'split', 'is_gold']
    abnormality_cols = [col for col in df.columns if col not in output_columns]
    for col in abnormality_cols:
        if col not in output_columns:
            output_columns.append(col)

    final_split_df[output_columns].to_csv(output_csv_path, index=False)
    print(f"✅ Study-level split successfully saved to: {output_csv_path}")
    return final_split_df


class RSNAKneeSplitDataset(Dataset):
    """PyTorch Dataset loading DICOM images based on study splits."""
    def __init__(self, split_csv_path, img_dir, split_name='train', image_size=(224, 224)):
        self.img_dir = img_dir
        self.image_size = image_size
        
        if os.path.exists(split_csv_path):
            df = pd.read_csv(split_csv_path)
            self.df = df[df['split'] == split_name].reset_index(drop=True)
        else:
            raise FileNotFoundError(f"Split CSV not found at {split_csv_path}. Run prepare_study_splits first.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        study_id = str(row['study_id'])
        study_path = os.path.join(self.img_dir, study_id)
        
        try:
            if os.path.isdir(study_path):
                slices = sorted(os.listdir(study_path))
                slice_path = os.path.join(study_path, slices[0]) if slices else None
            else:
                slice_path = study_path + ".dcm" if not study_path.endswith(".dcm") else study_path

            if slice_path and os.path.exists(slice_path):
                dicom_data = pydicom.dcmread(slice_path)
                image = dicom_data.pixel_array.astype(np.float32)
                image = cv2.resize(image, self.image_size)
                image = np.stack([image, image, image], axis=0)
                image = image / 255.0
            else:
                image = np.zeros((3, self.image_size[0], self.image_size[1]), dtype=np.float32)
        except Exception:
            image = np.zeros((3, self.image_size[0], self.image_size[1]), dtype=np.float32)

        target_cols = [c for c in self.df.columns if c not in ['study_id', 'split', 'is_gold']]
        target_col = 'ACL' if 'ACL' in target_cols else target_cols[0]
        label = float(row.get(target_col, 0.0))

        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


def get_dataloaders(split_csv_path="/kaggle/working/study_splits.csv", 
                    img_dir="/kaggle/input/competitions/rsna-knee-abnormality-detection/train_images", 
                    batch_size=16):
    """Instantly returns train, val, and test DataLoaders for Member 3."""
    prepare_study_splits(output_csv_path=split_csv_path)
    
    train_ds = RSNAKneeSplitDataset(split_csv_path, img_dir, split_name='train')
    val_ds = RSNAKneeSplitDataset(split_csv_path, img_dir, split_name='val')
    test_ds = RSNAKneeSplitDataset(split_csv_path, img_dir, split_name='test')
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader
