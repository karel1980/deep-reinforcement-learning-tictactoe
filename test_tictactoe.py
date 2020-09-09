import gym
from tictactoe import TicTacToeEnv
from nose.tools import assert_equal, raises
import numpy as np

def test_env_creation():
    env = gym.make('tictactoe-v0')
    print(env)
    assert_equal(env.board, [None for i in range(9)])
    assert_equal(env.current_player, 0)
    assert_equal(env.done, False)

def test_first_step():
    env = gym.make('tictactoe-v0')
    new_state, reward, done, info = env.step(0)
    assert_equal(reward, 1)
    np.testing.assert_array_equal(new_state, np.array([0,1,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0]))
    assert_equal(env.current_player, 1)
    assert_equal(done, False)
    assert_equal(info, {'state': 'in_progress'})

def test_valid_second_step():
    env = gym.make('tictactoe-v0')
    new_state, reward, done, info = env.step(0)
    new_state, reward, done, info = env.step(1)
    assert_equal(reward, 1)
    np.testing.assert_array_equal(new_state, np.array([0,1,0,0,0,1,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0]))
    assert_equal(env.current_player, 0)
    assert_equal(done, False)
    assert_equal(info, {'state': 'in_progress'})

def test_invalid_step():
    env = gym.make('tictactoe-v0')
    new_state, reward, done, info = env.step(0)
    new_state, reward, done, info = env.step(0)
    assert_equal(reward, -10)
    np.testing.assert_array_equal(new_state, np.array([0,1,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0]))
    assert_equal(env.current_player, 1)
    assert_equal(done, True)
    assert_equal(info, {'state': 'done', 'reason': 'Illegal move'})
    
@raises(Exception)
def test_error_when_step_on_done_env():
    env = gym.make('tictactoe-v0')
    new_state, reward, done, info = env.step(0)
    new_state, reward, done, info = env.step(0)
    new_state, reward, done, info = env.step(1)
    
def test_step_winning_move():
    env = gym.make('tictactoe-v0')
    env.step(0)
    env.step(3)
    env.step(1)
    env.step(4)
    state,  reward, done, info = env.step(2)
    np.testing.assert_array_equal(state,  np.array([0,1,0,0,1,0,0,1,0,0,0,1,0,0,1,1,0,0,1,0,0,1,0,0,1,0,0]))
    assert_equal(reward, 10)
    assert_equal(done,  True)
    assert_equal(info, {'state': 'winner winner chicken dinner'})
