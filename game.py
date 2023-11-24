#!/usr/bin/env python
# coding: utf-8

import numpy as np
from pettingzoo.classic import tictactoe_v3
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# WARNING:absl:At this time, the v2.11+ optimizer `tf.keras.optimizers.Adam` runs slowly on M1/M2 Macs, please use the legacy Keras optimizer instead, located at `tf.keras.optimizers.legacy.Adam`.
# from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.legacy import Adam


def create_env():
    env = tictactoe_v3.env(render_mode=None)
    env.reset()
    return env


def create_model():
    model = Sequential([
        Dense(64, input_shape=(18,), activation='relu'),
        Dense(32, activation='relu'),
        Dense(9, activation='linear')
    ])
    model.compile(optimizer=Adam(lr=0.001), loss='mse')
    return model


def print_env(env, agent):
    print_board(env.last()[0]['observation'])


def print_board(board, agent='player_1'):
    print(format_board(board, agent))


def format_board(board, agent='player_1'):
    result = ""
    for i in range(3):
        for j in range(3):
            sign = "_"
            if board[i][j][1 if agent == 'player_1' else 0] == 1:
                sign = "X"
            if board[i][j][0 if agent == 'player_1' else 1] == 1:
                sign = "O"
            result += sign
        result += "\n"
    return result


class RandomPlayer:
    def __init__(self, name, action_space):
        self.name = name
        self.action_space = action_space

    def get_action(self, env):
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            return None
        else:
            mask = observation['action_mask']
            action = self.action_space.sample(mask)
            return action


def record_game_episodes(env, players, num_games, epsilon):
    episodes = []
    for i in range(num_games):
        observations = []
        rewards = []
        actions = []
        env.reset()
        player_map = dict()
        player_map[env.agents[0]] = players[0]
        player_map[env.agents[1]] = players[1]

        game_moves = 0
        # self play until termination
        # for each move, record (observation, action, reward)
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            observations.append(observation)
            rewards.append(reward)
            # print(agent, "board")
            # print_board(observation['observation'])
            # print(agent, "reward", reward)

            if termination or truncation:
                action = None
            else:
                game_moves += 1
                if np.random.random() < epsilon:
                    # random move
                    mask = observation["action_mask"]
                    action = env.action_space(agent).sample(mask)
                    # print(agent, "EXPLORE")
                else:
                    player = player_map[agent]
                    action = player.get_action(env)
                    # print(agent, "EXPLOIT")

            # print(agent, "action", action)
            actions.append(action)
            env.step(action)

        # print("game_moves", game_moves)
        episode = (observations, rewards, actions)

        episodes.append(episode)

    return episodes


def convert_episodes_to_memory(episodes):
    memory = []
    # first store the events for agent 1
    for episode in episodes:
        observations, rewards, actions = episode
        for i in range(0, len(observations) - 2):
            state = observations[i]['observation']
            action = actions[i]
            reward = rewards[i + 2]
            next_state = observations[i + 2]['observation']
            # print("S,A,N", state, action, next_state)
            memory.append((state, action, reward, next_state))
    return memory


def train_on_memory(model, memory, num_epochs=10, batch_size=10):
    states = []
    targets = []

    for memory_entry in memory:
        state, action, reward, next_state = memory_entry
        states.append(state.flatten())

        # print("board")
        # print_board(state)
        # print("action", action)
        # print("reward", reward)

        predictions = model.predict(np.array([state.flatten(), next_state.flatten()]), verbose=False)

        best_future_action = predictions[1].argmax()
        # print("best future action", best_future_action)
        best_future_reward = predictions[1][best_future_action]
        # print("best future reward", best_future_reward)

        target = predictions[0]
        # print("old predictions", target)
        target[action] = reward + 0.95 * best_future_reward
        # print("updated predictions", target)
        targets.append(target)

    x = np.array(states)
    y = np.array(targets)
    model.fit(x, y, epochs=num_epochs, batch_size=batch_size, verbose=False)


# TODO: epsilon should probably better be a property of ModelAgent. We can even control the epsilon decay from there
def play_and_train(model, players, env, num_games=10, num_epochs=10, batch_size=10, epsilon=0):
    episodes = record_game_episodes(env, players, num_games, epsilon)
    memory = convert_episodes_to_memory(episodes)
    train_on_memory(model, memory, num_epochs, batch_size)


def play_and_train_iterations(model, players, env, train_params):
    for i in range(train_params.num_iterations):
        epsilon = train_params.epsilon_start + (
                1.0 * (train_params.epsilon_end - train_params.epsilon_start) * i / train_params.num_iterations)
        print("iteration %d/%d with epsilon %g" % (i, train_params.num_iterations, epsilon))
        play_and_train(model, players, env, train_params.num_games, train_params.num_epochs, train_params.batch_size,
                       epsilon)

        print("TODO: evaluate how many moves against a random client. This should go up as the model learns to make valid moves")

class ModelPlayer:
    def __init__(self, name, model):
        self.name = name
        self.model = model

    def get_action(self, env):
        observation, reward, termination, truncation, info = env.last()
        if termination or truncation:
            return None
        else:
            x = np.array([observation['observation'].flatten()])
            return self.model.predict(x, verbose=False)[0].argmax()
