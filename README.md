# Deep Q learning

This repo is an implementation of a Deep Q learning agent.
Note: I'm just figuring stuff out as I go, there are probably plenty of mistakes.

It features the following features:

## train.py
This script loads the model (or creates a new one), trains the model and saves it.

## play.py
This script lets you to play against the model.  This isn't very refined:
 - you have to specify the square using a 0-based index
 - the end of the game isn't properly detected

## tests

Run tests with `pytest`. This contains a test to check if the model plays valid moves.
Initially this will fail, but as you use `train.py` it will get better and the tests will eventually succeed.




