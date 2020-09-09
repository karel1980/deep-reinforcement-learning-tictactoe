import gym
from gym import spaces
from gym.envs.registration import register
import numpy as np

register(
    id='tictactoe-v0',
    entry_point='tictactoe:TicTacToeEnv',
)

INVALID_MOVE_REWARD = -3
WINNING_MOVE_REWARD = 1
# We already give a negative reward for invalid moves. I don't think rewarding for valid moves is useful.
VALID_MOVE_REWARD = 0
# Note: only the odd player can get draw reward, so we should probably keep it 0
DRAW_REWARD = 0


class TicTacToeEnv(gym.Env):
    reward_range = (-np.inf, np.inf)
    observation_space = spaces.MultiDiscrete([2 for _ in range(0, 3*3*3)])
    action_space = spaces.Discrete(9) 

    def seed(self, seed=None):
        pass
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.current_player = 0
        self.board = [None]*9
        self.moves = 0
        self.done = False
        return self._one_hot_board()
    
    def step(self, action):
        if self.done:
            raise Exception("Game is done. Please reset before playing again")
        info = dict(state="in_progress")
        
        reward = 0

        if self.board[action] != None:
            reward = INVALID_MOVE_REWARD
            info = {"state": "done", "reason": "Illegal move"}
            self.done = True
            return self._one_hot_board(), reward, self.done, info

        self.board[action] = self.current_player
        self.moves += 1
        if self.moves == 9:
            # Stop when board is full
            info = {'state': 'draw'}
            reward = DRAW_REWARD
            self.done = True
        
        winner = self.get_winner()
        if winner is None:
            # keep playing
            reward = VALID_MOVE_REWARD
            self.current_player = 1 - self.current_player
        elif winner == self.current_player:
            # Big reward for winning the game
            reward = WINNING_MOVE_REWARD
            self.done = True
            info = {'state':'winner winner chicken dinner'}
        elif winner != self.current_player:
            # We shouldn't get here because you only get to become a winner in your own round
            self.done = True
            raise Exception("Illegal state: opponent won?")

        return self._one_hot_board(), reward, self.done, info
    
    def get_winner(self):
        for i in range(3):
            # check row number i
            if (self.board[i*3] != None) and (self.board[i*3] == self.board[i*3+1]) and (self.board[i*3] == self.board[i*3+2]):
                return self.board[i*3]
            
            # check column number i
            if (self.board[i] != None) and (self.board[i] == self.board[i+3]) and (self.board[i] == self.board[i+6]):
                return self.board[i]
            
        # check first diagonal
        if (self.board[0] != None) and (self.board[0] == self.board[4]) and (self.board[0] == self.board[8]):
            return self.board[0]
        
        # check second diagonal
        if (self.board[2] != None) and (self.board[2] == self.board[4]) and (self.board[2] == self.board[6]):
                return self.board[2]

        return None
    
    def render(self, mode="human"):
        board = [ ' ' if cell is None else cell for cell in self.board ]
        print ("%d %d %d"%board[:3])
        print ("%d %d %d"%board[3:6])
        print ("%d %d %d"%board[6:9])
        
    def _one_hot_board(self):
        onehot = np.zeros(3*9)
        for i in range(len(self.board)):
            value = 0 if self.board[i] == None else self.board[i] + 1
            onehot[i*3+value] = 1
        return onehot
