#!/usr/bin/env python
# coding: utf-8

import numpy as np

from game import print_board


def evaluate_single_game(env, players, verbose=False):
    env.reset()

    first_player = np.random.randint(2)

    player_map = dict()
    player_map[env.agents[0]] = players[first_player]
    player_map[env.agents[1]] = players[1 - first_player]

    rewards = dict()
    rewards[players[first_player]] = 0
    rewards[players[1 - first_player]] = 0

    num_moves = 0
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if verbose:
            print_board(observation['observation'], agent)
            print("reward", reward)
        current_player = player_map[agent]
        rewards[current_player] += reward
        if termination or truncation:
            action = None
            if verbose:
                print("termination action", action)
            env.step(action)
        else:
            action = current_player.get_action(env)
            num_moves += 1
            if verbose:
                print("player action", action)
            env.step(action)

    for player in rewards:
        if rewards[player] > 0:
            return player, num_moves
        if rewards[player] < 0:
            # we must check for negative rewards as well
            # because making an invalid move doesn't pass a positive rewards to the other player
            # also: goddamn ugly way to find the other player
            other_player = next(filter(lambda p: p != player, rewards))
            return other_player, num_moves
    return None, num_moves


def evaluate_n_games(env, players, num_games):
    total_moves = 0
    player1_wins = 0
    player2_wins = 0
    draws = 0
    for i in range(num_games):
        winner, num_moves = evaluate_single_game(env, players)
        if winner is None:
            draws += 1
        if winner == players[0]:
            player1_wins += 1
        if winner == players[1]:
            player2_wins += 1
        total_moves += num_moves
    return player1_wins, player2_wins, draws, total_moves
