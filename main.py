import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np

# =========================
# CONFIG
# =========================
DATA_DIR = "./"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 40
LR = 0.001
RUNS_DIR = "runs"


def create_run_folder():
    os.makedirs(RUNS_DIR, exist_ok=True)
    existing = [d for d in os.listdir(RUNS_DIR) if d.startswith("train_")]
    idx = len(existing) + 1
    run_name = f"train_{idx:03d}"
    path = os.path.join(RUNS_DIR, run_name)
    os.makedirs(path, exist_ok=True)
    return path


def get_class_names():
    train_path = os.path.join(DATA_DIR, "train")
    return sorted(os.listdir(train_path))


def train_model():
    classes = get_class_names()
    num_classes = len(classes)
    run_dir = create_run_folder()
    print("[INFO] Saving run at:", run_dir)

    # =========================
    # AUGMENTATION
    # =========================
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])

    valid_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), train_tf)
    valid_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "valid"), valid_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)

    # =========================
    # MODEL: EfficientNet-B0
    # =========================
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(1280, num_classes)

    model = model.to(DEVICE)

    # Freeze backbone (warmup)
    for param in model.features.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_acc = 0
    patience = 5
    patience_counter = 0

    # =========================
    # TRAINING LOOP
    # =========================
    print("[INFO] Training started...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for img, label in train_loader:
            img, label = img.to(DEVICE), label.to(DEVICE)

            optimizer.zero_grad()
            out = model(img)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for img, label in valid_loader:
                img, label = img.to(DEVICE), label.to(DEVICE)
                out = model(img)
                _, pred = torch.max(out, 1)
                correct += (pred == label).sum().item()
                total += label.size(0)

        acc = correct / total * 100
        scheduler.step()

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss={avg_loss:.4f} | Val Acc={acc:.2f}%")

        # Save best model
        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
            torch.save({"model": model.state_dict(), "classes": classes},
                       os.path.join(run_dir, "model.pth"))
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print("[INFO] Training complete — best acc:", best_acc)


if __name__ == "__main__":
    train_model()
