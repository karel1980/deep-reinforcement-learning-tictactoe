from game import *
from train import TrainParams

MAXIMUM_VALID_STEPS_IN_GAME = 12

MINIMUM_VALID_STEPS_IN_GAME = 5


def test_env_creation():
    env = create_env()

    assert env is not None
    assert env.observation_space(env.agents[0])['observation'].shape == (3, 3, 2)


def test_model_creation():
    model = create_model()

    assert model is not None


def test_model_predictions():
    env = create_env()
    model = create_model()
    obs = env.last()[0]['observation'].flatten()

    prediction = model.predict(np.array([obs]))

    assert prediction.shape == (1, 9)


def test_format_board():
    env = create_env()
    env.step(4)

    board = format_board(env.last()[0]['observation'])

    assert board == "___\n_X_\n___\n"


def test_random_player():
    env = create_env()
    player = RandomPlayer('george', env.action_space(env.agents[0]))

    action = player.get_action(env)

    assert 0 <= action < 9


def test_random_player_action():
    env = create_env()
    player = RandomPlayer('random1', env.action_space(env.agents[0]))

    action = player.get_action(env)
    env.step(action)

    board = format_board(env.last()[0]['observation'])
    assert "X" in board


def test_create_memory():
    env = create_env()
    player1 = RandomPlayer('random1', env.action_space(env.agents[0]))
    player2 = RandomPlayer('random2', env.action_space(env.agents[1]))
    players = [player1, player2]

    episodes = record_game_episodes(env, players, 1, 0)

    assert len(episodes) == 1
    assert MINIMUM_VALID_STEPS_IN_GAME <= len(episodes[0][0]) < MAXIMUM_VALID_STEPS_IN_GAME


def test_create_memory_one_hundred():
    env = create_env()
    player1 = RandomPlayer('random1', env.action_space(env.agents[0]))
    player2 = RandomPlayer('random2', env.action_space(env.agents[0]))
    players = [player1, player2]

    episodes = record_game_episodes(env, players, 100, 0)

    assert len(episodes) == 100


def test_train_on_memory():
    env = create_env()
    model = create_model()
    player1 = RandomPlayer('random1', env.action_space(env.agents[0]))
    player2 = RandomPlayer('random2', env.action_space(env.agents[0]))
    players = [player1, player2]
    episodes = record_game_episodes(env, players, 1, 0)
    memory = convert_episodes_to_memory(episodes)

    train_on_memory(model, memory, 1, 1)


def test_play_and_train():
    env = create_env()
    model = create_model()
    player1 = RandomPlayer('random1', env.action_space(env.agents[0]))
    player2 = RandomPlayer('random2', env.action_space(env.agents[0]))
    players = [player1, player2]

    play_and_train(model, players, env, 1, 1, 1)


def test_model_player():
    env = create_env()
    model = create_model()
    player = ModelPlayer('ai1', model)

    env.step(player.get_action(env))


def test_play_and_train_iterations():
    env = create_env()
    model = create_model()
    player1 = ModelPlayer('ai_player_1', model)
    player2 = ModelPlayer('ai_player_2', model)
    players = [player1, player2]

    play_and_train_iterations(model, players, env, TrainParams(1, 1, 1, 1, 1, 0))
