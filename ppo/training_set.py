import numpy as np
import tensorflow as tf
from ppo.config import BATCHES, BATCH_SIZE


class Batch:
    def __init__(self, states, actions, values, log_probs, returns, advantages):
        self.states = states
        self.actions = actions

        self.values = tf.expand_dims(values, 1)
        self.log_probs = tf.expand_dims(log_probs, 1)
        self.returns = tf.expand_dims(returns, 1)
        self.advantages = tf.expand_dims(advantages, 1)


class TrainingSet:
    def __init__(self):
        self.episodes = []
        self.total_steps = 0
        self.total_rewards = []

        self.full = False
        self.done = False

        self.states = None
        self.actions = None
        self.values = None
        self.log_probs = None
        self.returns = None
        self.advantages = None

    def add(self, episode):
        if not episode.done:
            return

        self.episodes.append(episode)
        self.total_steps += episode.steps
        self.total_rewards.append(episode.total_reward)

        self.full = self.total_steps > BATCHES * BATCH_SIZE

    def finalize(self):
        if not self.full or self.done:
            return

        self.states = tf.concat([e.states for e in self.episodes], 0)
        self.actions = tf.concat([e.actions for e in self.episodes], 0)
        self.values = tf.concat([e.values for e in self.episodes], 0)
        self.log_probs = tf.concat([e.log_probs for e in self.episodes], 0)
        self.returns = tf.concat([e.returns for e in self.episodes], 0)
        self.advantages = tf.concat([e.advantages for e in self.episodes], 0)

        self.done = True

    def batches(self):
        if not self.done:
            return

        indices = np.arange(self.total_steps)
        np.random.shuffle(indices)

        max_start = self.total_steps - (self.total_steps % BATCH_SIZE)
        starts = np.arange(0, max_start, BATCH_SIZE)

        for s in starts:
            batch = indices[s:s+BATCH_SIZE]

            yield Batch(
                states=tf.gather(self.states, batch),
                actions=tf.gather(self.actions, batch),
                values=tf.gather(self.values, batch),
                log_probs=tf.gather(self.log_probs, batch),
                returns=tf.gather(self.returns, batch),
                advantages=tf.gather(self.advantages, batch),
            )
