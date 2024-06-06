
import numpy as np
import tensorflow as tf


class Batch:
    def __init__(self, returns, values, log_probs):
        self.returns = tf.expand_dims(returns, 1)
        self.values = tf.expand_dims(values, 1)
        self.log_probs = tf.expand_dims(log_probs, 1)


class TrainingSet:
    def __init__(self):
        self.episodes = []
        self.total_steps = 0
        self.total_rewards = 0

        self.returns = tf.convert_to_tensor([], dtype=tf.float32)
        self.values = tf.convert_to_tensor([], dtype=tf.float32)
        self.log_probs = tf.convert_to_tensor([], dtype=tf.float32)

    def is_full(self):
        return self.total_steps >= BATCH_SIZE * BATCHES

    def add(self, episode):
        if not episode.done:
            return

        self.episodes.append(episode)
        self.total_steps += episode.steps
        self.total_rewards += episode.total_reward

    def batches(self):
        ep = self.episodes[0]

        for s in [0]:
            yield Batch(
                returns=ep.returns,
                values=ep.values,
                log_probs=ep.log_probs
            )
