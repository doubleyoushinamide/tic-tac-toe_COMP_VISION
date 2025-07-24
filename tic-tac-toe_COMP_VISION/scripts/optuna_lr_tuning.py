"""
Optuna Learning Rate Tuning for RPS Model
-----------------------------------------
- Tunes only the learning rate (lr) using Optuna.
- Uses the same data and model as train_rps_model.py.
- Saves the best model as 'rps_model_optuna_best.pth'.

Usage:
    pip install optuna
    python optuna_lr_tuning.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import numpy as np
import optuna

# Data loading and preprocessing (same as train_rps_model.py)
TRAIN_DIR = '/kaggle/working/rock-paper-scissors-dataset/train'
VAL_DIR = '/kaggle/input/d/glushko/rock-paper-scissors-dataset/val'
BATCH_SIZE = 16
IMG_SIZE = 224
VAL_SPLIT = 0.2
NUM_CLASSES = 3
EPOCHS = 5  # Fewer epochs for faster tuning
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

# Model definition (with dropout)
def get_rps_model(num_classes=3, unfreeze_layers=['layer4']):
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
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

# Optuna objective function
def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    model = get_rps_model(num_classes=NUM_CLASSES, unfreeze_layers=['layer4']).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data).item()
        val_acc = val_corrects / len(val_loader.dataset)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save best model for this trial
            torch.save(model.state_dict(), 'rps_model_optuna_best.pth')
    return best_val_acc

if __name__ == '__main__':
    print("Starting Optuna tuning for learning rate...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)
    print(f"Best learning rate: {study.best_params['lr']}")
    print(f"Best validation accuracy: {study.best_value}")
    print("Best model weights saved as 'rps_model_optuna_best.pth'.")
    # Save best learning rate to file for use in main training script
    with open('best_lr.txt', 'w') as f:
        f.write(str(study.best_params['lr'])) 