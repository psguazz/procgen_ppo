import numpy as np

ALPHA = 0.003
GAMMA = 0.99
EPS = np.finfo(np.float32).eps.item()
CLIP = 0.2

EPOCHS = 3
TRAINING_EPISODES = 5

WEIGHTS_PATH = "/Users/psg/master_ai/autonomous/project/ppo/weights/"
