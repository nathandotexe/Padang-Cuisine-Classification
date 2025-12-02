import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# ===========================
# CONFIG
# ===========================
DATA_DIR = "./valid"
MODEL_PATH = "./runs/train_003/model.pth"
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("[INFO] Device:", DEVICE)

# ===========================
# LOAD MODEL (EfficientNet)
# ===========================
def load_efficientnet(num_classes):
    model = efficientnet_b0(weights=None)        # no pretrained weights (we load ours)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes
    )
    return model

# ===========================
# LOAD CHECKPOINT
# ===========================
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
classes = ckpt["classes"]
num_classes = len(classes)

print("[INFO] Loaded classes:", classes)

model = load_efficientnet(num_classes).to(DEVICE)
model.load_state_dict(ckpt["model"], strict=True)
model.eval()

# ===========================
# DATASET
# ===========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

valid_ds = datasets.ImageFolder(DATA_DIR, transform=transform)
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)

# ===========================
# EVALUATION
# ===========================
criterion = nn.CrossEntropyLoss()

all_labels = []
all_preds = []
all_logits = []

total_loss = 0
correct = 0
total = 0

with torch.no_grad():
    for images, labels in valid_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)

        loss = criterion(outputs, labels)
        total_loss += loss.item()

        preds = torch.argmax(probs, dim=1)

        total += labels.size(0)
        correct += (preds == labels).sum().item()

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_logits.extend(outputs.cpu().numpy())

accuracy = correct / total
avg_loss = total_loss / len(valid_loader)

# ===========================
# RMSE
# ===========================
onehot = np.eye(num_classes)[np.array(all_labels)]
rmse = np.sqrt(np.mean((np.array(all_logits) - onehot) ** 2))

# ===========================
# rMAP50 (mean recall)
# ===========================
cm = confusion_matrix(all_labels, all_preds)

recalls = []
for i in range(num_classes):
    tp = cm[i, i]
    fn = cm[i].sum() - tp
    recalls.append(tp / (tp + fn + 1e-8))

rMAP50 = np.mean(recalls)

# ===========================
# PRINT EVALUATION TABLE
# ===========================
print("\n=== EVALUATION TABLE ===")
print("{:<15} {:<10}".format("Metric", "Value"))
print("-" * 30)
print("{:<15} {:.4f}".format("Accuracy", accuracy))
print("{:<15} {:.4f}".format("Avg Loss", avg_loss))
print("{:<15} {:.4f}".format("RMSE", rmse))
print("{:<15} {:.4f}".format("rMAP50", rMAP50))

# ===========================
# PRINT CONFUSION MATRIX TABLE
# ===========================
print("\n=== CONFUSION MATRIX (TABLE) ===")
print("{:<12}".format("True\\Pred") + " ".join(f"{c:<12}" for c in classes))

for i, cname in enumerate(classes):
    row = " ".join(f"{cm[i, j]:<12}" for j in range(num_classes))
    print(f"{cname:<12} {row}")

# ===========================
# TEXT REPORT
# ===========================
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(all_labels, all_preds, target_names=classes))

# ===========================
# PLOT CONFUSION MATRIX
# ===========================
plt.figure(figsize=(9, 7))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()

plt.xticks(np.arange(num_classes), classes, rotation=45)
plt.yticks(np.arange(num_classes), classes)

for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="red")

plt.tight_layout()
plt.show()
