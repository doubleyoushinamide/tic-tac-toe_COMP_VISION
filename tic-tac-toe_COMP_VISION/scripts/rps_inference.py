"""
RPS Gesture Inference via Webcam
-------------------------------
- Loads the trained RPS model and performs live gesture prediction using webcam.
"""

import torch
import cv2
import numpy as np
from torchvision import transforms
from rps_model import get_rps_model

# Update this path if downloading the model from external source like Kaggle or google colab
MODEL_PATH = 'rps_model.pth'
CLASS_NAMES = ['paper', 'rock', 'scissors']

def load_rps_model(model_path=MODEL_PATH, device=None):
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = get_rps_model(num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    return model, device

def preprocess_image(frame):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return preprocess(frame)

def get_player_rps_move():
    """
    Captures a frame from webcam, predicts RPS gesture, and returns the result.
    """
    model, device = load_rps_model()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None

    print("Show your hand gesture (rock, paper, or scissors). Press 's' to capture, 'q' to quit.")
    predicted = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        cv2.imshow('RPS Gesture - Press s to capture', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            # Preprocess and predict
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_tensor = preprocess_image(img).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(img_tensor)
                _, pred = torch.max(outputs, 1)
                predicted = CLASS_NAMES[pred.item()]
            print(f"Predicted gesture: {predicted}")
            break
        elif key == ord('q'):
            print("Gesture capture cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return predicted

if __name__ == "__main__":
    result = get_player_rps_move()
    print(f"Detected gesture: {result}") 