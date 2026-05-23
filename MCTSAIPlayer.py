import numpy as np
import random
from Player import Player

# Helper class representing a node in the MCTS tree
class MCTSNode:
    def __init__(self, board, parent=None, last_move=None, player_who_moved=None):
        self.board = board                                                              # The board state at this node
        self.parent = parent                                                            # Reference to the parent node
        self.last_move = last_move                                                      # The move (column) that led to this state
        self.player_who_moved = player_who_moved                                        # ID of the player who made the last move
        
        self.children = {}                                                              # Dictionary mapping move -> MCTSNode object
        self.visits = 0                                                                 # Number of times this node has been simulated
        self.wins = 0                                                                   # Number of wins recorded from this node
        self.untried_moves = board.get_valid_moves()                                    # Moves not yet expanded into children

    # Check if all possible moves from this state have been added to the tree
    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    # Check if the game has ended in this node's state
    def is_terminal(self):
        return self.board.check_winner(1) or self.board.check_winner(2) or self.board.is_board_full()

    # Calculate the UCB score for tree navigation
    def ucb1_score(self, total_parent_visits, exploration_constant=1.414):
        if self.visits == 0:                                                              # Prioritize unvisited nodes by returning infinity
            return np.inf
        return (self.wins / self.visits) + exploration_constant * np.sqrt(np.log(total_parent_visits) / self.visits)

    # Select the child node with the highest UCB score
    def best_child(self, exploration_constant=1.414):
        best_score = -np.inf
        best_children = []
        
        for child in self.children.values():
            score = child.ucb1_score(self.visits, exploration_constant)
            if score > best_score:
                best_score = score
                best_children = [child]
            elif score == best_score:
                best_children.append(child)
                
        return random.choice(best_children)                                 # Randomly break ties among best children


class MCTSAIPlayer(Player):
    def __init__(self, piece, max_iterations = 6000):                          # Class constructure
        super().__init__(piece)
        self.max_iterations = max_iterations                                # Number of search cycles to perform

    def get_move(self, board):                                              # Method called by the game engine to get the next move
        valid_moves = board.get_valid_moves()
        if not valid_moves:
            return None
        if len(valid_moves) == 1:
            return valid_moves[0]

        
        opponent = 2 if self.piece == 1 else 1                                  # Define who moved before this state (the opponent)
        root = MCTSNode(board.copy(), player_who_moved=opponent)

        
        for _ in range(self.max_iterations):                                    # Execute the 4 MCTS phases for the specified number of iterations
            node = root
            
            # --- 1. SELECTION ---
            # Traverse the tree using UCB1 until an expandable or terminal node is reached
            while node.is_fully_expanded() and not node.is_terminal():
                node = node.best_child()

            # --- 2. EXPANSION ---
            # If the node is not terminal, create a new child node from an untried move
            if not node.is_terminal() and node.untried_moves:
                move = random.choice(node.untried_moves)                                    # Pick a random untried move
                node.untried_moves.remove(move)
            
                next_player = 2 if node.player_who_moved == 1 else 1                        # Determine whose turn it is in the simulation
                new_board = node.board.copy()
                new_board.drop_piece(move, next_player)
                
                child_node = MCTSNode(new_board, parent=node, last_move=move, player_who_moved=next_player)     # Create and link the child node
                node.children[move] = child_node
                node = child_node                                                           # Move to the new node for the simulation phase

            # --- 3. SIMULATION ---
            # Perform a fast playout with random moves until a game outcome is reached
            sim_board = node.board.copy()
            current_sim_player = 2 if node.player_who_moved == 1 else 1
            winner = 0
            
            while True:
                if sim_board.check_winner(1):
                    winner = 1
                    break
                if sim_board.check_winner(2):
                    winner = 2
                    break
                if sim_board.is_board_full():
                    winner = 0 
                    break
                    
                moves = sim_board.get_valid_moves()
                random_move = random.choice(moves)
                sim_board.drop_piece(random_move, current_sim_player)
                current_sim_player = 2 if current_sim_player == 1 else 1

            # --- 4. BACKPROPAGATION ---
            # Update visits and win statistics from the leaf node back to the root
            while node is not None:
                node.visits += 1
                if winner == 0:
                    node.wins += 0.5                                # Reward draws with half a point
                elif node.player_who_moved == winner:
                    node.wins += 1                                  # Reward the node if the move led to a win
                node = node.parent

        # Final decision: Select the move that led to the most visited child node
        best_move = max(root.children.items(), key=lambda item: item[1].visits)[0]
        
        return best_move