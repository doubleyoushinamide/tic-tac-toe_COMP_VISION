"""
Rock-Paper-Scissors CNN Model Training Script (Kaggle)
------------------------------------------------------
- Trains a transfer learning CNN (ResNet18) to classify RPS hand gestures.
- Saves the trained model weights as 'rps_model.pth' for local inference.
- Designed for Kaggle Notebook environment.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import numpy as np

# 1. Data Loading and Preprocessing
TRAIN_DIR = '/kaggle/working/rock-paper-scissors-dataset/train'
VAL_DIR = '/kaggle/input/d/glushko/rock-paper-scissors-dataset/val'
TEST_DIR = '/kaggle/input/d/glushko/rock-paper-scissors-dataset/test'
BATCH_SIZE = 16
IMG_SIZE = 224
VAL_SPLIT = 0.2
NUM_CLASSES = 3
EPOCHS = 10
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Replace dataset loading logic:
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transforms)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# 2. Model Definition (Transfer Learning)
def get_rps_model(num_classes=3, unfreeze_layers=['layer4']):
    model = models.resnet18(pretrained=True)
    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze specified layers
    for name, child in model.named_children():
        if name in unfreeze_layers or name == 'fc':
            for param in child.parameters():
                param.requires_grad = True
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    return model

from collections import defaultdict

def per_class_accuracy(preds, labels, num_classes):
    correct = [0] * num_classes
    total = [0] * num_classes
    for p, l in zip(preds, labels):
        total[l] += 1
        if p == l:
            correct[l] += 1
    acc = [c / t if t > 0 else 0.0 for c, t in zip(correct, total)]
    return acc, correct, total

# 3. Training Loop with Early Stopping and per-class accuracy
def train_model(model, train_loader, val_loader, epochs, device, patience=3, lr=0.001, unfreeze_layers=['layer4']):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_wts = None
    num_classes = model.fc[-1].out_features if isinstance(model.fc, nn.Sequential) else model.fc.out_features
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print('-' * 20)
        # Training phase
        model.train()
        running_loss, running_corrects = 0.0, 0
        all_preds, all_labels = [], []
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = float(running_corrects) / len(train_loader.dataset)
        class_acc, class_correct, class_total = per_class_accuracy(all_preds, all_labels, num_classes)
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        for i, (acc, c, t) in enumerate(zip(class_acc, class_correct, class_total)):
            print(f"  Class {i}: {acc:.4f} ({c}/{t})")
        # Validation phase
        model.eval()
        val_loss, val_corrects = 0.0, 0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = float(val_corrects) / len(val_loader.dataset)
        val_class_acc, val_class_correct, val_class_total = per_class_accuracy(val_preds, val_labels, num_classes)
        print(f"Val   Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        for i, (acc, c, t) in enumerate(zip(val_class_acc, val_class_correct, val_class_total)):
            print(f"  Class {i}: {acc:.4f} ({c}/{t})")
        # Early Stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_wts = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {patience} epochs with no improvement.")
                break
        scheduler.step()
    # Load best model weights before returning
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    print("Starting RPS model training...")
    # Try to load best lr from file
    best_lr = 0.001
    try:
        with open('best_lr.txt', 'r') as f:
            best_lr = float(f.read().strip())
            print(f"Using best learning rate from Optuna: {best_lr}")
    except Exception:
        print("Could not load best_lr.txt, using default lr=0.001")
    # Print class mapping
    print("Class to index mapping:", train_dataset.class_to_idx)
    model = get_rps_model(num_classes=NUM_CLASSES, unfreeze_layers=['layer4'])
    model = train_model(model, train_loader, val_loader, EPOCHS, DEVICE, patience=3, lr=best_lr, unfreeze_layers=['layer4'])
    # Save model weights for local inference
    torch.save(model.state_dict(), 'rps_model.pth')
    print("\nTraining complete. Model saved as 'rps_model.pth'. Download this file for local use.") 