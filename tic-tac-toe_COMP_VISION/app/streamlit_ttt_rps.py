import streamlit as st
import numpy as np
from PIL import Image
import random
import cv2
import torch
from models.rps_model import get_rps_model
from torchvision import transforms
import time
import os

# --- CONFIG ---
BOARD_SIZE = 3
CLASS_NAMES = ['paper', 'rock', 'scissors']
MODEL_PATH = 'models/rps_model.pth'
# Custom image paths (make sure to place your images in the same directory or update the paths)
X_IMG_PATH = 'app/assets/x.png'
O_IMG_PATH = 'app/assets/o.png'
EMPTY_IMG_PATH = 'app/assets/empty.png'

# --- GAME LOGIC ---
def new_board():
    return [['' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

def check_winner(board, player):
    # Check rows, columns, and diagonals for 5 in a row
    for i in range(BOARD_SIZE):
        if all(board[i][j] == player for j in range(BOARD_SIZE)):
            return True
        if all(board[j][i] == player for j in range(BOARD_SIZE)):
            return True
    if all(board[i][i] == player for i in range(BOARD_SIZE)):
        return True
    if all(board[i][BOARD_SIZE-1-i] == player for i in range(BOARD_SIZE)):
        return True
    return False

def is_draw(board):
    return all(board[i][j] != '' for i in range(BOARD_SIZE) for j in range(BOARD_SIZE))

def minimax(board, depth, is_maximizing, ai_player, human_player):
    # Simple minimax for 5x5, depth-limited for performance
    if check_winner(board, ai_player):
        return 10 - depth
    if check_winner(board, human_player):
        return depth - 10
    if is_draw(board) or depth == 2:
        return 0
    if is_maximizing:
        best_score = -float('inf')
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board[i][j] == '':
                    board[i][j] = ai_player
                    score = minimax(board, depth+1, False, ai_player, human_player)
                    board[i][j] = ''
                    best_score = max(score, best_score)
        return best_score
    else:
        best_score = float('inf')
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board[i][j] == '':
                    board[i][j] = human_player
                    score = minimax(board, depth+1, True, ai_player, human_player)
                    board[i][j] = ''
                    best_score = min(score, best_score)
        return best_score

def get_best_move(board, ai_player):
    human_player = 'X'
    best_score = -float('inf')
    move = (None, None)
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board[i][j] == '':
                board[i][j] = ai_player
                score = minimax(board, 0, False, ai_player, human_player)
                board[i][j] = ''
                if score > best_score:
                    best_score = score
                    move = (i, j)
    return move

# --- RPS INFERENCE ---
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

def get_player_rps_move_streamlit():
    st.info('Show your hand gesture (rock, paper, or scissors) to the webcam and click "Capture Gesture".')
    frame_placeholder = st.empty()
    capture = st.button('Capture Gesture')
    if 'webcam_active' not in st.session_state:
        st.session_state.webcam_active = False
    if 'last_frame' not in st.session_state:
        st.session_state.last_frame = None
    if capture:
        st.session_state.webcam_active = False
    if not st.session_state.webcam_active:
        if st.button('Start Webcam Preview'):
            st.session_state.webcam_active = True
    if st.session_state.webcam_active:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error('Could not open webcam.')
            st.session_state.webcam_active = False
            return None
        start_time = time.time()
        while time.time() - start_time < 10:
            ret, frame = cap.read()
            if not ret:
                st.error('Failed to grab frame.')
                break
            st.session_state.last_frame = frame.copy()
            frame_placeholder.image(frame, channels='BGR', caption='Live Preview (click Capture Gesture to snap)')
            if capture:
                break
            time.sleep(0.05)
        cap.release()
        st.session_state.webcam_active = False
    if capture:
        frame = st.session_state.last_frame
        if frame is not None:
            st.image(frame, channels='BGR', caption='Captured Frame')
            model, device = load_rps_model()
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_tensor = preprocess_image(img).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(img_tensor)
                _, pred = torch.max(outputs, 1)
                predicted = CLASS_NAMES[pred.item()]
            st.success(f'Predicted gesture: {predicted}')
            return predicted
        else:
            st.error('No frame captured. Please start the webcam preview and try again.')
            return None
    return None

def rps_winner(human, computer):
    beats = {'rock': 'scissors', 'scissors': 'paper', 'paper': 'rock'}
    if human == computer:
        return 'draw'
    elif beats[human] == computer:
        return 'human'
    else:
        return 'computer'

# --- UI HELPERS ---
def render_board(board, last_move=None):
    emoji_map = {'X': X_IMG_PATH, 'O': O_IMG_PATH, '': EMPTY_IMG_PATH}
    st.write('')
    for i, row in enumerate(board):
        cols = st.columns(BOARD_SIZE)
        for j, cell in enumerate(row):
            img_path = emoji_map[cell]
            if os.path.exists(img_path):
                img = Image.open(img_path)
                if last_move == (i, j):
                    cols[j].image(img, width=120, caption='⬅️')
                else:
                    cols[j].image(img, width=120)
            else:
                # fallback to emoji
                emoji = '❌' if cell == 'X' else '⭕' if cell == 'O' else '⬜'
                if last_move == (i, j):
                    cols[j].markdown(f'<span style="font-size:60px;">{emoji} ⬅️</span>', unsafe_allow_html=True)
                else:
                    cols[j].markdown(f'<span style="font-size:60px;">{emoji}</span>', unsafe_allow_html=True)

# --- STREAMLIT APP ---
st.set_page_config(page_title='RPS Tic-Tac-Toe', layout='centered')
st.title('🤖 RPS-Controlled Tic-Tac-Toe')

# Sidebar for info
with st.sidebar:
    st.header('Game Info')
    st.write('Play Tic-Tac-Toe against the computer!')
    st.write('Before every move, play Rock-Paper-Scissors with your webcam to decide who moves.')
    st.write('Custom images and animations included!')
    st.write('To use your own images, place x.png, o.png, and empty.png in the app directory.')

if 'board' not in st.session_state:
    st.session_state.board = new_board()
if 'status' not in st.session_state:
    st.session_state.status = 'playing'
if 'last_rps' not in st.session_state:
    st.session_state.last_rps = None
if 'current_player' not in st.session_state:
    st.session_state.current_player = None
if 'move_pending' not in st.session_state:
    st.session_state.move_pending = True
if 'last_move' not in st.session_state:
    st.session_state.last_move = None

st.header('ENJOY THE GAME!')
render_board(st.session_state.board, st.session_state.last_move)

if st.session_state.status == 'playing':
    if st.session_state.move_pending:
        st.subheader('RPS Round for This Move')
        human_rps = get_player_rps_move_streamlit()
        if human_rps:
            computer_rps = random.choice(['rock', 'paper', 'scissors'])
            st.markdown(f'''**Your move:** `{human_rps}`  
**CPU move:** `{computer_rps}`''')
            winner = rps_winner(human_rps, computer_rps)
            if winner == 'draw':
                st.warning('RPS round is a draw. Replay RPS for this move.')
            else:
                st.session_state.current_player = 'human' if winner == 'human' else 'computer'
                st.session_state.move_pending = False
                st.session_state.last_rps = (human_rps, computer_rps, winner)
                st.rerun()
    else:
        if st.session_state.current_player == 'human':
            st.subheader('Your Move (Click a cell)')
            for i in range(BOARD_SIZE):
                cols = st.columns(BOARD_SIZE)
                for j in range(BOARD_SIZE):
                    if st.session_state.board[i][j] == '':
                        if cols[j].button(' ', key=f'{i}-{j}'):
                            st.session_state.board[i][j] = 'X'
                            st.session_state.last_move = (i, j)
                            # Check for winner or draw immediately
                            for player, label in [('X', 'You'), ('O', 'Computer')]:
                                if check_winner(st.session_state.board, player):
                                    st.session_state.status = f'{label} win!'
                                    st.balloons()
                                    st.rerun()
                            if is_draw(st.session_state.board):
                                st.session_state.status = 'Draw!'
                                st.snow()
                                st.rerun()
                            st.session_state.move_pending = True
                            st.rerun()
        else:
            st.subheader('Computer is making a move...')
            row, col = get_best_move(st.session_state.board, 'O')
            if row is not None:
                st.session_state.board[row][col] = 'O'
                st.session_state.last_move = (row, col)
                # Check for winner or draw immediately
                for player, label in [('X', 'You'), ('O', 'Computer')]:
                    if check_winner(st.session_state.board, player):
                        st.session_state.status = f'{label} win!'
                        st.balloons()
                        st.rerun()
                if is_draw(st.session_state.board):
                    st.session_state.status = 'Draw!'
                    st.snow()
                    st.rerun()
            st.session_state.move_pending = True
            st.rerun()
else:
    st.header(st.session_state.status)
    if st.button('Play Again'):
        st.session_state.board = new_board()
        st.session_state.status = 'playing'
        st.session_state.last_rps = None
        st.session_state.current_player = None
        st.session_state.move_pending = True
        st.session_state.last_move = None
        st.rerun() 