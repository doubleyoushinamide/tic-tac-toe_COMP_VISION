"""
RPS Model Architecture Definition
---------------------------------
- Defines the CNN model architecture for RPS gesture classification.
- Must match the architecture used during training.
"""

import torch.nn as nn
from torchvision import models

def get_rps_model(num_classes=3):
    """
    Returns a ResNet18 model with the final layer modified for RPS classification.
    """
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model 