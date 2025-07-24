# Tic-Tac-Toe with Computer Vision RPS Control

A modern, interactive Tic-Tac-Toe game where each move is decided by a real-time Rock-Paper-Scissors (RPS) gesture, recognized via your webcam using a deep learning model. Play against a smart AI in your browser with a beautiful Streamlit UI!

---

## 🕹️ Features
- **Streamlit Web App**: Play Tic-Tac-Toe in your browser with a graphical board and webcam-based RPS gesture capture.
- **RPS Gesture Recognition**: Uses a ResNet18-based CNN to classify your hand gesture (rock, paper, or scissors) in real time.
- **Smart AI**: The computer opponent uses a minimax algorithm for challenging gameplay.
- **Customizable UI**: Swap in your own images for X and O, enjoy animations, and a polished interface.
- **Modular Code**: Clean, industry-standard Python project structure for easy extension and maintenance.

---

## 📁 Directory Structure
```
tic-tac-toe_COMP_VISION/
│
├── app/                        # Streamlit app and game logic
│   ├── __init__.py
│   ├── streamlit_ttt_rps.py    # Main Streamlit UI
│   ├── main_game.py            # CLI/legacy entry point
│   ├── tic_tac_toe_game.py     # Game logic
│   └── assets/                 # Images for UI
│       ├── x.png
│       ├── o.png
│       └── empty.png
│
├── models/                     # Model code and weights
│   ├── __init__.py
│   ├── rps_model.py            # Model architecture
│   ├── rps_model.pth           # Trained model weights
│   └── rps_model_v0.pth        # (Optional) older weights
│
├── scripts/                    # Training, tuning, and preprocessing scripts
│   ├── __init__.py
│   ├── train_rps_model.py
│   ├── optuna_lr_tuning.py
│   ├── preprocess_merge_and_augment.py
│   └── rps_inference.py
│
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🚀 Setup Instructions

### 1. **Clone the Repository**
```bash
git clone <your-repo-url>
cd tic-tac-toe_COMP_VISION
```

### 2. **Install Dependencies**
We recommend using a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. **Prepare Model Weights**
- Download or train the RPS model (see below).
- Place the latest `rps_model.pth` in `models/`.

### 4. **(Optional) Train or Tune the Model**
- To preprocess and augment data:
  ```bash
  python scripts/preprocess_merge_and_augment.py
  ```
- To tune learning rate with Optuna:
  ```bash
  python scripts/optuna_lr_tuning.py
  ```
- To train the model:
  ```bash
  python scripts/train_rps_model.py
  ```

### 5. **Run the Streamlit App**
```bash
streamlit run app/streamlit_ttt_rps.py
```
- The app will open in your browser. Make sure your webcam is connected!

### 6. **(Optional) Run the CLI Game**
```bash
python app/main_game.py
```

---

## 🧠 Model Architecture
- **Base Model**: ResNet18 (from torchvision)
- **Head**: Dropout(0.5) + Linear layer (output: 3 classes)
- **Input**: 224x224 RGB images, normalized as per ImageNet
- **Training**: Only the head and optionally the last block are unfrozen for transfer learning. Data augmentation is used for robustness.
- **Output**: Classifies hand gesture as 'rock', 'paper', or 'scissors'.

---

## 🖼️ Customization
- Place your own `x.png`, `o.png`, and `empty.png` in `app/assets/` for a personalized board.
- Tweak the UI in `app/streamlit_ttt_rps.py` for more features or a new look.

---

## 🙏 Credits & Acknowledgments
- Built with [Streamlit](https://streamlit.io/), [PyTorch](https://pytorch.org/), and [OpenCV](https://opencv.org/).
- Data for RPS gesture recognition were taken from these Kaggle datasets:
  - [Rock-Paper-Scissors Dataset by alexandredj](https://www.kaggle.com/datasets/alexandredj/rock-paper-scissors-dataset)
  - [Rock-Paper-Scissors Dataset by glushko](https://www.kaggle.com/datasets/glushko/rock-paper-scissors-dataset)
- Project structure and code organization inspired by best practices in the Python ML community.

---

Enjoy your AI-powered, gesture-controlled Tic-Tac-Toe game! 