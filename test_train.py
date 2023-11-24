from train import *


def test_load_train_save():
    load_train_save(AGENT_PARAMS_FILE, TrainParams(1, 1, 1, 1))


