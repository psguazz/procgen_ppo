import numpy as np
import tensorflow as tf
from ppo.config import BATCHES, BATCH_SIZE


class Batch:
    def __init__(self, states, actions, values, log_probs, returns, advantage):
        self.states = states
        self.actions = actions

        self.values = tf.expand_dims(values, 1)
        self.log_probs = tf.expand_dims(log_probs, 1)
        self.returns = tf.expand_dims(returns, 1)
        self.advantage = tf.expand_dims(advantage, 1)


class TrainingSet:
    def __init__(self):
        self.episodes = []
        self.total_steps = 0
        self.total_rewards = []

        self.full = False

        self.states = None
        self.actions = None
        self.values = None
        self.log_probs = None
        self.returns = None
        self.advantage = None

    def add(self, episode):
        if not episode.done:
            return

        if len(self.episodes) == 0:
            self.states = episode.states
            self.actions = episode.actions
            self.values = episode.values
            self.log_probs = episode.log_probs
            self.returns = episode.returns
            self.advantage = episode.advantage
        else:
            self.states = tf.concat((self.states, episode.states), 0)
            self.actions = tf.concat((self.actions, episode.actions), 0)
            self.values = tf.concat((self.values, episode.values), 0)
            self.log_probs = tf.concat((self.log_probs, episode.log_probs), 0)
            self.returns = tf.concat((self.returns, episode.returns), 0)
            self.advantage = tf.concat((self.advantage, episode.advantage), 0)

        self.episodes.append(episode)
        self.total_steps += episode.steps
        self.total_rewards.append(episode.total_reward)

        self.full = self.total_steps > BATCHES * BATCH_SIZE

    def batches(self):
        if not self.full:
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
                advantage=tf.gather(self.advantage, batch),
            )
