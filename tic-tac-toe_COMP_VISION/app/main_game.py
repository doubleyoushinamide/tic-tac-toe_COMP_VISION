"""
Main Game Orchestration
-----------------------
- Runs the RPS-controlled Tic-Tac-Toe game.
- Requires a webcam and the trained RPS model.
"""

import random
from app.tic_tac_toe_game import TicTacToe, get_best_move
from scripts.rps_inference import get_player_rps_move

def rps_winner(human, computer):
    # Returns 'human', 'computer', or 'draw'
    beats = {'rock': 'scissors', 'scissors': 'paper', 'paper': 'rock'}
    if human == computer:
        return 'draw'
    elif beats[human] == computer:
        return 'human'
    else:
        return 'computer'

def play_tic_tac_toe():
    game = TicTacToe()
    human = 'X'
    computer = 'O'
    game.display()
    while True:
        # RPS round before every move
        while True:
            print("\n--- Rock-Paper-Scissors Round (for this move) ---")
            human_rps = get_player_rps_move()
            if human_rps is None:
                print("RPS gesture not detected. Exiting game.")
                return "Exit"
            computer_rps = random.choice(['rock', 'paper', 'scissors'])
            print(f"Your move: {human_rps} | Computer move: {computer_rps}")
            winner = rps_winner(human_rps, computer_rps)
            if winner == 'draw':
                print("RPS round is a draw. Replaying RPS...")
                continue
            break
        if winner == 'human':
            # Human makes a move
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
            # Computer makes a move
            print("Computer is making a move...")
            row, col = get_best_move(game.board.copy(), computer)
            game.make_move(row, col, computer)
        game.display()
        # Check for winner or draw
        for player in [human, computer]:
            if game.check_winner(player):
                winner_str = "You" if player == human else "Computer"
                print(f"{winner_str} win!")
                return winner_str
        if game.is_draw():
            print("It's a draw!")
            return "Draw"

def main():
    print("Welcome to RPS-Controlled Tic-Tac-Toe!")
    while True:
        print("\n--- New Tic-Tac-Toe Game ---")
        play_tic_tac_toe()
        again = input("Play another round? (y/n): ").strip().lower()
        if again != 'y':
            print("Thanks for playing!")
            break

if __name__ == '__main__':
    main() 