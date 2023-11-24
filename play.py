#!/usr/bin/env python
# coding: utf-8

from game import ModelPlayer, create_env, print_env
from train import load_model, AGENT_PARAMS_FILE
import numpy as np


def play_against_ai():
    model = load_model(AGENT_PARAMS_FILE)
    player1 = ModelPlayer('player 1', model)

    env = create_env()
    env.reset()

    human = np.random.random() < 0.5
    while True:
        if not human:
            user_input = input("your action? ").strip()
            if user_input == "":
                break
            action = int(user_input)
            env.step(action)
        else:
            action = player1.get_action(env)
            print("AI player plays on square", action)
            env.step(action)
            print_env(env, 0)

        human = not human


if __name__ == "__main__":
    play_against_ai()
