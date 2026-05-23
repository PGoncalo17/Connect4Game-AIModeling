import pygame
import time   # For main
import numpy as np # For main

from Connect4Board import Connect4Board
from Connect4Gui import Connect4Gui
from HumanPlayer import HumanPlayer
from RandomPlayer import RandomAIPlayer
from MinimaxAIPlayer import MinimaxAIPlayer
from MCTSAIPlayer import MCTSAIPlayer



# =========================
# GAME LOOP
# =========================

class Connect4Game:

    def __init__(self):
        pass

    def run_game(self, player1, player2, headless = False, rows = 6, cols = 7, n_connect = 4 ):

        board = Connect4Board(rows, cols, n_connect)

        gui = Connect4Gui(board,rows,cols)
        if (not headless):
            gui.init(board)

        players = [player1, player2]
        turn = 0
        game_over = False

        while not game_over:
            current_player = players[turn]
            if (not headless):
                gui.deal_with_events(board, current_player)

            move = current_player.get_move(board)

            if move is not None and move in board.get_valid_moves():
                row, col = board.drop_piece(move, current_player.piece)

                if board.check_winner(current_player.piece):
                    if(not headless):
                       gui.update_winner(current_player)
                    else:
                        print(f"Player {current_player.piece} wins!")

                    game_over = True

                elif board.is_board_full():
                    if (not headless):
                        gui.draw_game()
                    print("Draw!!!!")
                    game_over = True

                if(not headless):
                    gui.draw_board(board)
                turn = (turn + 1) % 2

            # AI delay (optional for visibility)
            if not headless and not isinstance(current_player, HumanPlayer):
                pygame.time.wait(300)

            if game_over and not headless:
                gui.game_over()


# =========================
# RUN CONFIGURATION
# =========================

if __name__ == "__main__":
    p1 = RandomAIPlayer(piece=1)
    p2 = MinimaxAIPlayer(piece=1)
    p3 = MCTSAIPlayer(piece=2)
    p4 = HumanPlayer(piece=2)
    game = Connect4Game()
    durations = []                                                          # List to store the duration of each game
    for i in range(2):
        start_time = time.time()                                            # Record the start time of the game
        game.run_game(p3, p4, headless=False)

        end_time = time.time()                                              # Record the end time of the game
        duration = end_time - start_time                                    # Calculate how long the game lasted
        durations.append(duration)                                          # Add to list

        print(f"Game {i+1} finished in {duration:.2f} seconds")

    # Calculate metrics
    avg_duration = np.sum(durations) / np.size(durations)
    max_duration = np.max(durations)
    min_duration = np.min(durations)

    print("\n--- Final Statistics ---")
    print(f"Average Duration: {avg_duration:.2f} seconds")
    print(f"Maximum Duration: {max_duration:.2f} seconds")
    print(f"Minimum Duration: {min_duration:.2f} seconds")
        