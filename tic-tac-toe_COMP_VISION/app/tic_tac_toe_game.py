"""
Tic-Tac-Toe Game Logic
----------------------
- Board representation, move validation, win/draw checking, and unbeatable AI (Minimax).
"""

import numpy as np

class TicTacToe:
    def __init__(self):
        self.board = np.full((3, 3), ' ')
        self.current_winner = None

    def display(self):
        print("\n  0 1 2")
        for i, row in enumerate(self.board):
            print(f"{i} " + "|".join(row))
            if i < 2:
                print("  -----")
        print()

    def is_valid_move(self, row, col):
        return 0 <= row < 3 and 0 <= col < 3 and self.board[row, col] == ' '

    def make_move(self, row, col, player):
        if self.is_valid_move(row, col):
            self.board[row, col] = player
            return True
        return False

    def check_winner(self, player):
        # Rows, columns, diagonals
        for i in range(3):
            if all(self.board[i, :] == player) or all(self.board[:, i] == player):
                return True
        if all([self.board[i, i] == player for i in range(3)]) or \
           all([self.board[i, 2 - i] == player for i in range(3)]):
            return True
        return False

    def is_draw(self):
        return np.all(self.board != ' ') and not self.check_winner('X') and not self.check_winner('O')

    def available_moves(self):
        return [(r, c) for r in range(3) for c in range(3) if self.board[r, c] == ' ']

    def reset(self):
        self.board = np.full((3, 3), ' ')
        self.current_winner = None

def minimax(board, depth, is_maximizing, player, opponent):
    # Terminal states
    if check_win(board, player):
        return 1
    if check_win(board, opponent):
        return -1
    if np.all(board != ' '):
        return 0

    if is_maximizing:
        best_score = -float('inf')
        for (r, c) in [(r, c) for r in range(3) for c in range(3) if board[r, c] == ' ']:
            board[r, c] = player
            score = minimax(board, depth + 1, False, player, opponent)
            board[r, c] = ' '
            best_score = max(score, best_score)
        return best_score
    else:
        best_score = float('inf')
        for (r, c) in [(r, c) for r in range(3) for c in range(3) if board[r, c] == ' ']:
            board[r, c] = opponent
            score = minimax(board, depth + 1, True, player, opponent)
            board[r, c] = ' '
            best_score = min(score, best_score)
        return best_score

def check_win(board, player):
    for i in range(3):
        if all(board[i, :] == player) or all(board[:, i] == player):
            return True
    if all([board[i, i] == player for i in range(3)]) or \
       all([board[i, 2 - i] == player for i in range(3)]):
        return True
    return False

def get_best_move(board, player):
    opponent = 'O' if player == 'X' else 'X'
    best_score = -float('inf')
    move = None
    for (r, c) in [(r, c) for r in range(3) for c in range(3) if board[r, c] == ' ']:
        board[r, c] = player
        score = minimax(board, 0, False, player, opponent)
        board[r, c] = ' '
        if score > best_score:
            best_score = score
            move = (r, c)
    return move 