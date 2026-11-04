import os
import sys
import torch
import torch.nn as nn
from torchvision import models

sys.path.append(os.path.join("Member2", "member2_work"))
from datasets import get_dataloaders

TRAIN_CSV = "Member2/outputs/train_split.csv"
VAL_CSV = "Member2/outputs/val_split.csv"
DATA_ROOT = "./Member2/data"
BATCH_SIZE = 4
EPOCHS = 10
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

train_loader, val_loader = get_dataloaders(
    train_csv_path=TRAIN_CSV,
    val_csv_path=VAL_CSV,
    data_root=DATA_ROOT,
    batch_size=BATCH_SIZE,
    num_workers=0
)

model = models.resnet50(weights="IMAGENET1K_V2")
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(model.fc.in_features, 12)
model = model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1} | Train loss: {total_loss / len(train_loader):.4f}")

os.makedirs("Member3/outputs", exist_ok=True)
torch.save(model.state_dict(), "Member3/outputs/experiment001_resnet50.pth")
print("Training complete. Model saved.")