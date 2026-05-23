from Player import Player  
import numpy as np        

class MinimaxAIPlayer(Player): 
    def __init__(self, piece, max_depth = 6):                   # Constructor: receives the piece and search depth
        super().__init__(piece)                                
        self.max_depth = max_depth                              

    
    def get_move(self, board):                                  # Method called by the game engine to get the next move
        valid_moves = board.get_valid_moves()                   # Get list of columns where a piece can be dropped 
        if not valid_moves:                                     # If no moves are possible (board full), return None
            return None
        if len(valid_moves) == 1:                               # For efficiency, if only one move is available, take it
            return valid_moves[0]
        
        best_score = -np.inf                                    # Initialize the values
        best_move = valid_moves[0]                              
        alpha = -np.inf                                         
        beta = np.inf                                            

        for col in valid_moves:                                                             # Iterate through each possible column to find the best one
            newBoard = board.copy()                                                         # Simulates a play
            newBoard.drop_piece(col, self.piece)                                            
            score = self.minimax(newBoard, self.max_depth - 1, alpha, beta, False)          # Call minimax to predict future outcomes (it is now the opponent's turn, is_maximizing=False)

            if score > best_score:            # If this path leads to a better score, updates the vars
                best_score = score            
                best_move = col               

            alpha = max(alpha, best_score)    # Update Alpha with the best score found at this level

        return best_move                      # Return the best column chosen
    

    def minimax(self, board, depth, alpha, beta, is_maximizing):
        opponent = 2 if self.piece == 1 else 1                                      # Identify the opponent's piece

        if board.check_winner(self.piece):                                          # Terminal case: If we win, return a high positive score
            return 1000000 + depth             
        if board.check_winner(opponent):                                            # Terminal case: If opponent wins, return a high negative score
            return -1000000 - depth            
        if board.is_board_full() or depth == 0:                                    
            return self.evaluate_board(board, self.piece) 
        
        valid_moves = board.get_valid_moves()                                        # List all possible moves in this simulated state
        
        if is_maximizing:                                                            # Agent's turn (seeks to maximize score)
            max_eval = -np.inf                
            for col in valid_moves:                                             
                newBoard = board.copy()        
                newBoard.drop_piece(col, self.piece)                                
                ev = self.minimax(newBoard, depth - 1, alpha, beta, False)           # Recursive call for opponent's turn
                max_eval = max(max_eval, ev)  
                alpha = max(alpha, ev)        
                if beta <= alpha:                                                    # Pruning condition: Min already has a better option elsewhere 
                    break                     
            return max_eval
        else:                                                                        # Opponent's turn (seeks to minimize our score)
            min_eval = np.inf                  
            for col in valid_moves:            
                newBoard = board.copy()      
                newBoard.drop_piece(col, opponent) 
                ev = self.minimax(newBoard, depth - 1, alpha, beta, True)           # Recursive call for agent's turn
                min_eval = min(min_eval, ev)  
                beta = min(beta, ev)                                                # Update the Beta boundary
                if beta <= alpha:                                                   # Pruning condition: Max already has a better option
                    break                   
            return min_eval

    
    def evaluate_board(self, board, player):                                        # Heuristic Function: Assigns a score to non-terminal board states
        opponent = 2 if player == 1 else 1     
        score = 0                              
        n = board.n_connect                                                         # Number of pieces needed to win (N) 

        center_col = board.cols // 2                                                # Strategic Priority: The center column is more valuable for connections
        for r in range(board.rows):          
            if board.grid[r][center_col] == player:                                 # Check board state using grid
                score += 4                                                          # Gain points for controlling the center
            elif board.grid[r][center_col] == opponent:
                score -= 4                                                          # Lose points if the opponent controls the center

        
        def evaluate_segment(segment):                                              # Helper function to score sub-segments of size N
            win_score = 0                      
            p_count = segment.count(player)                                         # Count our pieces in the segment
            o_count = segment.count(opponent)                                       # Count opponent pieces in the segment
            empty_count = segment.count(0)                                          # Count empty slots

            if p_count == n:                                                        # segment contains a winning line
                win_score += 100000
            elif p_count == n - 1 and empty_count == 1:                             # Threat: N-1 pieces and 1 empty space 
                win_score += 50 
            elif p_count == n - 2 and empty_count == 2:                             # Potential: N-2 pieces and 2 empty spaces 
                win_score += 10 

            if o_count == n - 1 and empty_count == 1:                               # Block priority: Opponent threat 
                win_score -= 80                                                     # High penalty to force blocking the opponent
            elif o_count == n - 2 and empty_count == 2:                             # Opponent building potential 
                win_score -= 15 

            return win_score

        for r in range(board.rows):                                                 # Horizontal analysis 
            row_array = list(board.grid[r, :])                                      # Extract full row
            for c in range(board.cols - n + 1):
                segment = row_array[c:c+n]                                          # Create sliding segments of size N
                score += evaluate_segment(segment)

        for c in range(board.cols):                                                 # Vertical analysis 
            col_array = list(board.grid[:, c])                                      # Extract full column
            for r in range(board.rows - n + 1):
                segment = col_array[r:r+n]
                score += evaluate_segment(segment)

        
        for r in range(n - 1, board.rows):                                          # Positive Diagonal analysis (/)
            for c in range(board.cols - n + 1):
                segment = [board.grid[r-i][c+i] for i in range(n)]
                score += evaluate_segment(segment)

        
        for r in range(board.rows - n + 1):                                         # Negative Diagonal analysis (\)
            for c in range(board.cols - n + 1):
                segment = [board.grid[r+i][c+i] for i in range(n)]
                score += evaluate_segment(segment)

        return score                                                                # Return total utility value for the state