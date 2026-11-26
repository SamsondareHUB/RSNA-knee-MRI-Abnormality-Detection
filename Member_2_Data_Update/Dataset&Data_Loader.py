
# Member 2: Complete Dataset & DataLoader Pipeline 

import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pydicom
from PIL import Image

# 1. Official 12-Label List 
LABELS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

# 2. Automatic Path Detection for Kaggle Environment
BASE_DIR = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'train.csv' in files:
        BASE_DIR = root
        break

if BASE_DIR is None:
    BASE_DIR = '/kaggle/input/rsna-knee-abnormality-detection'

img_dir = os.path.join(BASE_DIR, 'train_series')

# 3. Data Preparation & Stratified Study-Level Split Function
def prepare_and_save_splits(base_dir):
    csv_path = os.path.join(base_dir, 'train.csv')
    df = pd.read_csv(csv_path)
    
    if 'StudyInstanceUID' in df.columns:
        df['study_id'] = df['StudyInstanceUID']
        
    df['is_gold'] = df[LABELS].notnull().any(axis=1).astype(int)
    df[LABELS] = df[LABELS].fillna(0)
    
    gold_df = df[df['is_gold'] == 1].sample(frac=1, random_state=42).reset_index(drop=True)
    non_gold_df = df[df['is_gold'] == 0].sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Stratified Split counts
    gold_train = gold_df.iloc[:30].copy()
    gold_val = gold_df.iloc[30:44].copy()
    gold_test = gold_df.iloc[44:58].copy()
    
    ng_train = non_gold_df.iloc[:894].copy()
    ng_val = non_gold_df.iloc[894:1319].copy()
    ng_test = non_gold_df.iloc[1319:1740].copy()
    
    gold_train['split'], gold_val['split'], gold_test['split'] = 'train', 'val', 'test'
    ng_train['split'], ng_val['split'], ng_test['split'] = 'train', 'val', 'test'
    
    train_df = pd.concat([gold_train, ng_train]).sample(frac=1, random_state=42).reset_index(drop=True)
    val_df = pd.concat([gold_val, ng_val]).sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = pd.concat([gold_test, ng_test]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    full_splits = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)
    
    output_df = full_splits[['study_id', 'split', 'is_gold']].copy()
    output_df.to_csv('study_splits.csv', index=False)
    print("Successfully generated and saved: study_splits.csv")
    
    return full_splits

# Load or Generate splits
if os.path.exists('study_splits.csv'):
    study_splits = pd.read_csv('study_splits.csv')
    print("Loaded existing 'study_splits.csv'")
    orig_df = pd.read_csv(os.path.join(BASE_DIR, 'train.csv'))
    if 'StudyInstanceUID' in orig_df.columns:
        orig_df['study_id'] = orig_df['StudyInstanceUID']
    orig_df[LABELS] = orig_df[LABELS].fillna(0)
    full_splits = pd.merge(study_splits, orig_df[['study_id'] + LABELS], on='study_id', how='left')
else:
    full_splits = prepare_and_save_splits(BASE_DIR)

train_df = full_splits[full_splits['split'] == 'train'].reset_index(drop=True)
val_df = full_splits[full_splits['split'] == 'val'].reset_index(drop=True)
test_df = full_splits[full_splits['split'] == 'test'].reset_index(drop=True)

# 4. Custom PyTorch Dataset Class
class RSNAKneeDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        study_id = str(row['study_id'])
        
        study_path = os.path.join(self.img_dir, study_id)
        dcm_files = sorted(glob.glob(os.path.join(study_path, '**', '*.dcm'), recursive=True))
        
        if len(dcm_files) > 0:
            mid_idx = len(dcm_files) // 2
            dcm_path = dcm_files[mid_idx]
            try:
                dcm = pydicom.dcmread(dcm_path)
                img = dcm.pixel_array.astype(float)
                img = (np.maximum(img, 0) / img.max()) * 255.0 if img.max() > 0 else img
                img = img.astype(np.uint8)
                img = Image.fromarray(img).convert('RGB')
            except Exception:
                img = Image.new('RGB', (224, 224))
        else:
            img = Image.new('RGB', (224, 224))

        if self.transform:
            img = self.transform(img)

        labels = torch.tensor(row[LABELS].values.astype(np.float32))
        return img, labels

# 5. Transforms & DataLoaders Ready for Team
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = RSNAKneeDataset(train_df, img_dir, transform=transform)
val_dataset = RSNAKneeDataset(val_df, img_dir, transform=transform)
test_dataset = RSNAKneeDataset(test_df, img_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

print("\n=== DataLoaders Ready for Team Members ===")
print(f"Train batches: {len(train_loader)} (Total studies: {len(train_df)})")
print(f"Val batches:   {len(val_loader)} (Total studies: {len(val_df)})")
print(f"Test batches:  {len(test_loader)} (Total studies: {len(test_df)})")
