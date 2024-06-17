import numpy as np

EPS = np.finfo(np.float32).eps.item()

ALPHA = 0.003
GAMMA = 0.99
LAMBDA = 0.95
CLIP = 0.2

EPOCHS = 4
BATCHES = 8
BATCH_SIZE = 128

WEIGHTS_PATH = "/Users/psg/master_ai/autonomous/project/ppo/weights/"
