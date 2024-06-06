
import numpy as np
import tensorflow as tf
from ppo.config import BATCHES, BATCH_SIZE


class Batch:
    def __init__(self, states, actions, returns, values, log_probs):
        self.states = tf.expand_dims(states, 1)
        self.actions = tf.expand_dims(actions, 1)
        self.returns = tf.expand_dims(returns, 1)
        self.values = tf.expand_dims(values, 1)
        self.log_probs = tf.expand_dims(log_probs, 1)


class TrainingSet:
    def __init__(self):
        self.episodes = []
        self.total_steps = 0
        self.total_rewards = 0

        self.states = np.array([])
        self.actions = np.array([])
        self.returns = np.array([])
        self.values = np.array([])
        self.log_probs = np.array([])

        self.done = False
        self.full = False

    def add(self, episode):
        if not episode.done or self.done or self.full:
            return

        self.episodes.append(episode)
        self.total_steps += episode.steps
        self.total_rewards += episode.total_reward

        self.states = np.concatenate((self.states, episode.states))
        self.actions = np.concatenate((self.actions, episode.actions))
        self.returns = np.concatenate((self.returns, episode.returns))
        self.values = np.concatenate((self.values, episode.values))
        self.log_probs = np.concatenate((self.log_probs, episode.log_probs))

        self.full = self.total_steps >= BATCH_SIZE * BATCHES

    def finalize(self):
        if self.done:
            return

        self.states = tf.convert_to_tensor(self.states, dtype=tf.float32)
        self.actions = tf.convert_to_tensor(self.actions, dtype=tf.float32)
        self.returns = tf.convert_to_tensor(self.returns, dtype=tf.float32)
        self.values = tf.convert_to_tensor(self.values, dtype=tf.float32)
        self.log_probs = tf.convert_to_tensor(self.log_probs, dtype=tf.float32)

        self.done = True

    def batches(self):
        if not self.done:
            return

        indices = np.arange(self.total_steps)
        np.random.shuffle(indices)

        starts = np.arange(0, self.total_steps, BATCH_SIZE)

        for s in starts:
            batch = indices[s:s+BATCH_SIZE]

            yield Batch(
                states=tf.gather(self.states, batch),
                actions=tf.gather(self.actions, batch),
                returns=tf.gather(self.returns, batch),
                values=tf.gather(self.values, batch),
                log_probs=tf.gather(self.log_probs, batch)
            )
