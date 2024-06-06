import numpy as np

ALPHA = 0.01
GAMMA = 0.99
EPS = np.finfo(np.float32).eps.item()

EPOCHS = 3
BATCHES = 4
BATCH_SIZE = 128

# ALPHA = 0.0003
# GAMMA = 0.99
# CLIP = 0.2
#
# ACTOR_PATH = "/Users/psg/master_ai/autonomous/project/ppo/weights/actor.weights.h5"
# CRITIC_PATH = "/Users/psg/master_ai/autonomous/project/ppo/weights/critic.weights.h5"
