#!/usr/bin/env python
# coding: utf-8
import os
import logging

import tensorflow as tf

from evaluate import evaluate_n_games
from game import ModelPlayer, create_env, create_model, print_env, record_game_episodes, convert_episodes_to_memory, \
    train_on_memory, RandomPlayer

AGENT_PARAMS_FILE = "tictactoe-params"


class TrainParams:
    def __init__(self, num_iterations=100, num_games=1000, num_epochs=10, batch_size=100, epsilon_start=0.99,
                 epsilon_end=0.01, verbose=False):
        self.num_iterations = num_iterations
        self.num_games = num_games
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.verbose = verbose


def load_train_save(filename=AGENT_PARAMS_FILE, train_params=TrainParams()):
    model = load_model(filename)
    train(model, train_params)
    model.save(filename)


def train(model, train_params):
    env = create_env()
    player1 = ModelPlayer('ai 1', model)
    player2 = ModelPlayer('ai 2', model)
    players = [player1, player2]

    play_and_train_iterations(model, players, env, train_params)


def load_model(filename, create_if_missing=True):
    if os.path.exists(filename):
        return tf.keras.models.load_model(filename)

    if not create_if_missing:
        raise Exception("model file missing: %s" % filename)

    return create_model()


def save_model(model, filename=AGENT_PARAMS_FILE):
    model.save(AGENT_PARAMS_FILE)


def play_and_train_iterations(model, players, env, train_params):
    for i in range(train_params.num_iterations):
        epsilon = train_params.epsilon_start + (
                1.0 * (train_params.epsilon_end - train_params.epsilon_start) * i / train_params.num_iterations)
        print("iteration %d/%d with epsilon %g" % (i, train_params.num_iterations, epsilon))
        play_and_train(model, players, env, train_params.num_games, train_params.num_epochs, train_params.batch_size,
                       epsilon, verbose=train_params.verbose)

        p1, p2, draw, moves = evaluate_n_games(env, players, 100)
        print("outcomes and moves in 100 games: ", p1, p2, draw, moves)
        env.reset()
        p1, p2, draw, moves = evaluate_n_games(env,
                                               [players[0], RandomPlayer('random2', env.action_space(env.agents[0]))],
                                               100)
        print("outcomes and moves in 100 games against random: ", p1, p2, draw, moves)


def play_and_train(model, players, env, num_games=10, num_epochs=10, batch_size=10, epsilon=0, verbose=False):
    print("playing games")
    episodes = record_game_episodes(env, players, num_games, epsilon)
    print("converting to memory")
    memory = convert_episodes_to_memory(episodes)
    print("training on memory")
    train_on_memory(model, memory, num_epochs, batch_size, verbose)


if __name__ == "__main__":
    logging.getLogger("pettingzoo.utils.env_logger").setLevel(logging.ERROR)

    # slow
    train_params = TrainParams(num_iterations=10, num_games=1000, num_epochs=10, batch_size=100, epsilon_start=0.5,
                               epsilon_end=0.01, verbose=True)

    # super fast
    # train_params = TrainParams(num_iterations=5, num_games=20, num_epochs=1, batch_size=100, epsilon_start=0.3, epsilon_end=0.01)

    load_train_save(train_params=train_params)
