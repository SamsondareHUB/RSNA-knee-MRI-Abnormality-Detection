import os
import sys
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models

PROJECT_ROOT = os.getcwd()
sys.path.insert(0, PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "Member2", "member2_work"))

from datasets import get_dataloaders
from evaluation.member4_metrics import LABELS

VAL_CSV = "Member2/outputs/val_split.csv"
DATA_ROOT = "./Member2/data"
CHECKPOINT = "Member3/outputs/experiment001_resnet50.pth"
OUTPUT_CSV = "Member3/outputs/experiment001_val_predictions.csv"
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

val_split = pd.read_csv(VAL_CSV)
study_ids = val_split[val_split["is_gold"] == 1]["StudyInstanceUID"].tolist()

_, val_loader = get_dataloaders(
    train_csv_path="Member2/outputs/train_split.csv",
    val_csv_path=VAL_CSV,
    data_root=DATA_ROOT,
    batch_size=BATCH_SIZE,
    num_workers=0
)

model = models.resnet50(weights=None)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(model.fc.in_features, len(LABELS))
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

all_probs = []

with torch.no_grad():
    for images, _ in val_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        probs = torch.sigmoid(outputs).cpu().numpy()
        all_probs.extend(probs.tolist())

if len(all_probs) != len(study_ids):
    raise RuntimeError(
        f"Prediction/study count mismatch: {len(all_probs)} predictions "
        f"but {len(study_ids)} validation studies."
    )

predictions = pd.DataFrame(all_probs, columns=LABELS)
predictions.insert(0, "StudyInstanceUID", study_ids)

os.makedirs("Member3/outputs", exist_ok=True)
predictions.to_csv(OUTPUT_CSV, index=False)

print(f"Saved predictions for {len(predictions)} studies to {OUTPUT_CSV}")