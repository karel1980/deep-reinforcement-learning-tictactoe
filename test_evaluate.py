from evaluate import evaluate_single_game, evaluate_n_games
from game import create_env, ModelPlayer
from train import load_model, AGENT_PARAMS_FILE


def test_evaluate_single_game_player1_wins():
    player1 = DeterministicPlayer([0, 1, 2])
    player2 = DeterministicPlayer([3, 4, 8])
    players = [player1, player2]

    winner, num_moves = evaluate_single_game(create_env(), players)

    assert winner == player1
    assert num_moves == 5 or num_moves == 6


def test_evaluate_invalid_move_makes_other_player_win():
    player1 = DeterministicPlayer([0, 0, 0])
    player2 = DeterministicPlayer([1, 2, 3])
    players = [player1, player2]

    winner, num_moves = evaluate_single_game(create_env(), players)

    assert winner == player2
    assert num_moves == 3 or num_moves == 4


def test_evaluate_single_game_player2_wins():
    player1 = DeterministicPlayer([0, 1, 8])
    player2 = DeterministicPlayer([3, 4, 5])
    players = [player1, player2]

    winner, num_moves = evaluate_single_game(create_env(), players)

    assert winner == player2
    assert num_moves == 5 or num_moves == 6


def test_evaluate_single_game_draw():
    # this blows. setting up a deterministic game is hard if the
    # starting player isn't deterministic
    player1 = DeterministicPlayer([0, 1, 5, 6, 8])
    player2 = DeterministicPlayer([3, 4, 2, 7, 8])
    players = [player1, player2]

    winner, num_moves = evaluate_single_game(create_env(), players)

    assert winner is None
    assert num_moves is 9


def test_model_player():
    model = load_model(AGENT_PARAMS_FILE)
    player1 = ModelPlayer('ai 1', model)
    player2 = ModelPlayer('ai 2', model)
    players = [player1, player2]

    # we don't care who wins
    evaluate_single_game(create_env(), players, verbose=True)


def test_evaluate_n_games():
    env = create_env()
    model = load_model(AGENT_PARAMS_FILE)
    player1 = ModelPlayer('ai 1', model)
    player2 = ModelPlayer('ai 2', model)
    players = [player1, player2]

    result = evaluate_n_games(env, players, 20)
    print(result)

    num_player1_wins, num_player2_wins, num_draws, num_moves = result

    assert num_player1_wins + num_player2_wins + num_draws == 20
    assert 20 * 2 < num_moves <= 20 * 9

    # assert that the model is playing optimally (i.e. all games end in draws, no wins or losses)
    # if these fail you need to run train.py for a bit more
    assert num_draws == 20
    assert num_player1_wins == 0
    assert num_player2_wins == 0
    assert num_moves == 180



class DeterministicPlayer:
    def __init__(self, moves):
        self.moves = moves
        self.next_move = 0

    def get_action(self, env):
        if self.next_move < len(self.moves):
            self.next_move += 1
            return self.moves[self.next_move - 1]
        return None
