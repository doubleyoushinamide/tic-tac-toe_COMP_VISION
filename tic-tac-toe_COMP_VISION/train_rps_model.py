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
DATA_DIR = '../input/rock-paper-scissors-dataset/' 
BATCH_SIZE = 32
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

dataset = datasets.ImageFolder(DATA_DIR, transform=train_transforms)
val_size = int(VAL_SPLIT * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
val_dataset.dataset.transform = val_transforms

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# 2. Model Definition (Transfer Learning)
def get_rps_model(num_classes=3):
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False  # Freeze all layers
    num_ftrs = model.fc.in_features
    # Add Dropout before the final layer for regularization
    model.fc = nn.Sequential(
        nn.Dropout(0.5),  # 50% dropout
        nn.Linear(num_ftrs, num_classes)
    )
    return model

# 3. Training Loop with Early Stopping
def train_model(model, train_loader, val_loader, epochs, device, patience=3):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_wts = None

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print('-' * 20)

        # Training phase
        model.train()
        running_loss, running_corrects = 0.0, 0
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
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Validation phase
        model.eval()
        val_loss, val_corrects = 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        print(f"Val   Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

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
    model = get_rps_model(num_classes=NUM_CLASSES)
    model = train_model(model, train_loader, val_loader, EPOCHS, DEVICE, patience=3)
    # Save model weights for local inference
    torch.save(model.state_dict(), 'rps_model.pth')
    print("\nTraining complete. Model saved as 'rps_model.pth'. Download this file for local use.") 