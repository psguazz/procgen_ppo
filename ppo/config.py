import numpy as np

ALPHA = 0.003
GAMMA = 0.99
EPS = np.finfo(np.float32).eps.item()
CLIP = 0.2

EPOCHS = 1
BATCHES = 4
BATCH_SIZE = 128

WEIGHTS_PATH = "/Users/psg/master_ai/autonomous/project/ppo/weights/"
