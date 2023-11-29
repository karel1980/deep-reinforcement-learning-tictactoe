#!/usr/bin/env python
# coding: utf-8

from random import shuffle

from game import ModelPlayer, create_env, print_env
from train import load_model, AGENT_PARAMS_FILE


def play_against_ai():
    model = load_model(AGENT_PARAMS_FILE)
    ai_player = ModelPlayer('ai player', model)
    human_player = HumanPlayer('you')
    env = create_env()
    env.reset()

    players = [ai_player, human_player]
    shuffle(players)
    current_player_idx = 0
    while True:
        current_player = players[current_player_idx]

        print_env(env, current_player_idx)
        action = current_player.get_action(env)
        if action is None:
            print("stopped")
            break

        print("%s marked square %d"%(current_player.name, action))
        env.step(action)
        current_player_idx = 1 - current_player_idx


class HumanPlayer:
    def __init__(self, name):
        self.name = name

    def get_action(self, env):
        user_input = input("your action? ").strip()
        if user_input == "":
            return None
        return int(user_input)


if __name__ == "__main__":
    play_against_ai()
