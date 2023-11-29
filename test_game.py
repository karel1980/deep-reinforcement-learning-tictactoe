from game import *
from test_evaluate import DeterministicPlayer
from train import TrainParams, play_and_train_iterations, play_and_train

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

    assert board == "___\n_O_\n___\n"


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

    board = format_board(env.last()[0]['observation'], 1)
    assert "X" in board


def test_record_game_episodes_win():
    env = create_env()
    player1 = DeterministicPlayer([0, 1, 2])
    player2 = DeterministicPlayer([3, 4])
    players = [player1, player2]

    episodes = record_game_episodes(env, players, 1, 0)

    assert len(episodes) == 1
    assert MINIMUM_VALID_STEPS_IN_GAME <= len(episodes[0][0]) < MAXIMUM_VALID_STEPS_IN_GAME


def test_record_game_episodes_draw():
    env = create_env()
    player1 = DeterministicPlayer([0, 2, 4, 5, 7])
    player2 = DeterministicPlayer([1, 3, 6, 8])
    players = [player1, player2]

    episodes = record_game_episodes(env, players, 1, 0)

    assert len(episodes) == 1
    assert format_board(episodes[0][0][0]['observation'], 0) == "___\n___\n___\n"
    assert format_board(episodes[0][0][1]['observation'], 1) == "X__\n___\n___\n"
    assert format_board(episodes[0][0][2]['observation'], 0) == "XO_\n___\n___\n"
    assert format_board(episodes[0][0][3]['observation'], 1) == "XOX\n___\n___\n"
    assert format_board(episodes[0][0][4]['observation'], 0) == "XOX\nO__\n___\n"
    assert format_board(episodes[0][0][5]['observation'], 1) == "XOX\nOX_\n___\n"
    assert format_board(episodes[0][0][6]['observation'], 0) == "XOX\nOX_\nO__\n"
    assert format_board(episodes[0][0][7]['observation'], 1) == "XOX\nOXX\nO__\n"
    assert format_board(episodes[0][0][8]['observation'], 0) == "XOX\nOXX\nO_O\n"
    assert format_board(episodes[0][0][9]['observation'], 1) == "XOX\nOXX\nOXO\n"
    assert format_board(episodes[0][0][10]['observation'], 0) == "XOX\nOXX\nOXO\n"
    assert len(episodes[0][0]) == 11


def test_convert_episodes_to_memory_win():
    env = create_env()
    player1 = DeterministicPlayer([0, 1, 2])
    player2 = DeterministicPlayer([3, 4, 6])
    players = [player1, player2]

    episodes = record_game_episodes(env, players, 1, 0)
    memory = convert_episodes_to_memory(episodes)

    assert len(memory) == 5
    assert MINIMUM_VALID_STEPS_IN_GAME <= len(episodes[0][0]) < MAXIMUM_VALID_STEPS_IN_GAME
    # agent learns
    validate_memory(memory[0], "___\n___\n___\n", 0, 0, "X__\n___\n___\n", 0)
    validate_memory(memory[1], "X__\n___\n___\n", 3, 0, "X__\nO__\n___\n", 1)
    validate_memory(memory[2], "X__\nO__\n___\n", 1, 0, "XX_\nO__\n___\n", 0)
    validate_memory(memory[3], "XX_\nO__\n___\n", 4, -1, "XX_\nOO_\n___\n", 1)
    validate_memory(memory[4], "XX_\nOO_\n___\n", 2, 1, "XXX\nOO_\n___\n", 0)
    assert len(memory) == 5


def test_convert_episodes_to_memory_draw():
    env = create_env()
    player1 = DeterministicPlayer([0, 2, 4, 5, 7])
    player2 = DeterministicPlayer([1, 3, 6, 8])
    players = [player1, player2]

    episodes = record_game_episodes(env, players, 1, 0)
    memory = convert_episodes_to_memory(episodes)

    assert len(episodes) == 1
    assert MINIMUM_VALID_STEPS_IN_GAME <= len(episodes[0][0]) < MAXIMUM_VALID_STEPS_IN_GAME
    # agent learns
    validate_memory(memory[0], "___\n___\n___\n", 0, 0, "X__\n___\n___\n", 0)
    validate_memory(memory[1], "X__\n___\n___\n", 1, 0, "XO_\n___\n___\n", 1)
    validate_memory(memory[2], "XO_\n___\n___\n", 2, 0, "XOX\n___\n___\n", 0)
    validate_memory(memory[3], "XOX\n___\n___\n", 3, 0, "XOX\nO__\n___\n", 1)
    validate_memory(memory[4], "XOX\nO__\n___\n", 4, 0, "XOX\nOX_\n___\n", 0)
    validate_memory(memory[5], "XOX\nOX_\n___\n", 6, 0, "XOX\nOX_\nO__\n", 1)
    validate_memory(memory[6], "XOX\nOX_\nO__\n", 5, 0, "XOX\nOXX\nO__\n", 0)
    validate_memory(memory[7], "XOX\nOXX\nO__\n", 8, 0, "XOX\nOXX\nO_O\n", 1)
    validate_memory(memory[8], "XOX\nOXX\nO_O\n", 7, 0, "XOX\nOXX\nOXO\n", 0)
    assert len(memory) == 9


def validate_memory(entry, state, action, reward, next_state, x_player=0):
    assert format_board(entry[0], x_player) == state
    assert entry[1] == action
    assert entry[2] == reward
    assert format_board(entry[3], x_player) == next_state


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
