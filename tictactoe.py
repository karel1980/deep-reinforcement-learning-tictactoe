import gym
from gym import spaces
from gym.envs.registration import register
import numpy as np

register(
    id='tictactoe-v0',
    entry_point='tictactoe:TicTacToeEnv',
)

class TicTacToeEnv(gym.Env):
    reward_range = (-np.inf, np.inf)
    observation_space = spaces.MultiDiscrete([2 for _ in range(0, 3*3*3)])
    action_space = spaces.Discrete(9) 

    def seed(self, seed=None):
        pass
    
    def reset(self):
        self.current_player = 0
        self.board = np.zeros(9, dtype='int')
        return self._one_hot_board()
    
    def step(self, action):
        exp = dict(state="in_progress")
        
        reward = 0
        done = False

        if self.board[action] != 0:
            reward = -10
            exp = {"state": "done", "reason": "Illegal move"}
            done = True
            return self._one_hot_board(), reward, done, exp

        self.board[action] = self.current_player + 1

        winner = self.get_winner()
        if winner is None:
            reward = 0
        elif winner == self.current_player:
            #win
            reward = 1
            done = True
        elif winner != self.current_player:
            #lose
            reward = -1
            done = True

        self.current_player = 1 - self.current_player

        return self._one_hot_board(), reward, done, exp
    
    def get_winner(self):
        for i in range(3):
            # check row number i
            if self.board[i*3] != 0 and self.board[i*3] == self.board[i*3+1] and self.board[i*3] == self.board[i*3+2]:
                return self.board[i*3]
            
            # check column number i
            if self.board[i] != 0 and self.board[i] == self.board[i+3] and self.board[i] == self.board[i+6]:
                return self.board[i]
            
        # check first diagonal
        if self.board[0] != 0 and self.board[0] == self.board[4] and self.board[0] == self.board[8]:
            return self.board[0]
        
        # check second diagonal
        if self.board[2] != 0 and self.board[2] == self.board[4] and self.board[2] == self.board[6]:
                return self.board[2]

        return None
    
    def render(self, mode="human"):
        print ("%d %d %d"%(self.board[0],self.board[1],self.board[2],))
        print ("%d %d %d"%(self.board[3],self.board[4],self.board[5],))
        print ("%d %d %d"%(self.board[6],self.board[7],self.board[8],))

    def _one_hot_board(self):
        onehot = np.zeros(3*9)
        for i in range(len(self.board)):
            value = self.board[i]
            onehot[i*3+value] = 1
        return onehot
