"""
Main Game Orchestration
-----------------------
- Runs the RPS-controlled Tic-Tac-Toe game.
- Requires a webcam and the trained RPS model.
"""

import random
from tic_tac_toe_game import TicTacToe, get_best_move
from rps_inference import get_player_rps_move

def rps_winner(human, computer):
    # Returns 'human', 'computer', or 'draw'
    beats = {'rock': 'scissors', 'scissors': 'paper', 'paper': 'rock'}
    if human == computer:
        return 'draw'
    elif beats[human] == computer:
        return 'human'
    else:
        return 'computer'

def play_tic_tac_toe(first_player):
    game = TicTacToe()
    human = 'X' if first_player == 'human' else 'O'
    computer = 'O' if human == 'X' else 'X'
    current = 'X'
    game.display()
    while True:
        if current == human:
            while True:
                try:
                    move = input("Enter your move as 'row col' (e.g., 0 2): ")
                    row, col = map(int, move.strip().split())
                    if game.make_move(row, col, human):
                        break
                    else:
                        print("Invalid move. Try again.")
                except Exception:
                    print("Invalid input. Try again.")
        else:
            print("Computer is making a move...")
            row, col = get_best_move(game.board.copy(), computer)
            game.make_move(row, col, computer)
        game.display()
        if game.check_winner(current):
            winner = "You" if current == human else "Computer"
            print(f"{winner} win!")
            return winner
        if game.is_draw():
            print("It's a draw!")
            return "Draw"
        current = computer if current == human else human

def main():
    print("Welcome to RPS-Controlled Tic-Tac-Toe!")
    while True:
        print("\n--- Rock-Paper-Scissors Round ---")
        human_rps = get_player_rps_move()
        if human_rps is None:
            print("RPS gesture not detected. Exiting game.")
            break
        computer_rps = random.choice(['rock', 'paper', 'scissors'])
        print(f"Your move: {human_rps} | Computer move: {computer_rps}")
        winner = rps_winner(human_rps, computer_rps)
        if winner == 'draw':
            print("RPS round is a draw. Replaying RPS...")
            continue
        print(f"{'You' if winner == 'human' else 'Computer'} win the RPS round and go first in Tic-Tac-Toe!")
        print("\n--- Tic-Tac-Toe Game ---")
        play_tic_tac_toe(winner)
        again = input("Play another round? (y/n): ").strip().lower()
        if again != 'y':
            print("Thanks for playing!")
            break

if __name__ == '__main__':
    main() 