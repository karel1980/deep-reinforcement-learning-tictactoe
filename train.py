#!/usr/bin/env python
# coding: utf-8
import os
import logging

import tensorflow as tf

from game import play_and_train_iterations, ModelPlayer, create_env, create_model, print_env

AGENT_PARAMS_FILE = "tictactoe-params"


class TrainParams:
    def __init__(self, num_iterations=100, num_games=1000, num_epochs=10, batch_size=100, epsilon_start=0.99,
                 epsilon_end=0.01):
        self.num_iterations = num_iterations
        self.num_games = num_games
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end


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


if __name__ == "__main__":
    logging.getLogger("pettingzoo.utils.env_logger").setLevel(logging.ERROR)
    train_params = TrainParams(num_iterations=10, num_games=100, num_epochs=10, batch_size=100, epsilon_start=0.99, epsilon_end=0.01)
    load_train_save(train_params=train_params)
