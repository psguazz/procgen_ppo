import numpy as np
import tensorflow as tf
from tensorflow.math import reduce_mean, reduce_std
from ppo.config import GAMMA, LAMBDA

EPS = np.finfo(np.float32).eps.item()

ta_float = {"dtype": tf.float32, "size": 0, "dynamic_size": True}
ta_int = {"dtype": tf.int64, "size": 0, "dynamic_size": True}


class Episode:
    def __init__(self):
        self.states = tf.TensorArray(**ta_float)
        self.actions = tf.TensorArray(**ta_int)
        self.values = tf.TensorArray(**ta_float)
        self.log_probs = tf.TensorArray(**ta_float)
        self.rewards = tf.TensorArray(**ta_float)

        self.returns = None
        self.advantages = None

        self.total_reward = 0
        self.steps = 0

        self.done = False

    def store(self, state, action, value, log_prob, reward):
        if self.done:
            return

        self.states = self.states.write(self.steps, state)
        self.actions = self.actions.write(self.steps, action)
        self.values = self.values.write(self.steps, value)
        self.log_probs = self.log_probs.write(self.steps, log_prob)
        self.rewards = self.rewards.write(self.steps, reward)

        self.total_reward += reward
        self.steps += 1

    def finalize(self):
        if self.done:
            return

        self.states = self.states.stack()
        self.actions = self.actions.stack()
        self.values = self.values.stack()
        self.log_probs = self.log_probs.stack()
        self.rewards = self.rewards.stack()

        self.returns = self._compute_returns()
        self.advantages = self._compute_advantages()

        self.done = True

    def _compute_returns(self):
        if self.done:
            return

        reverse_rewards = self.rewards[::-1]
        reverse_returns = tf.TensorArray(**ta_float)

        discounted_sum = 0

        for i in range(self.steps):
            discounted_sum = reverse_rewards[i] + GAMMA*discounted_sum
            reverse_returns = reverse_returns.write(i, discounted_sum)

        returns = reverse_returns.stack()[::-1]
        returns = self._normalize(returns)

        return returns

    def _compute_advantages(self):
        if self.done:
            return

        next_values = tf.concat((self.values[1:], [0]), 0)
        deltas = self.rewards + GAMMA*next_values - self.values
        reverse_deltas = deltas[::-1]

        gae = 0
        reverse_advantages = tf.TensorArray(**ta_float)

        for i in range(self.steps):
            gae = reverse_deltas[i] + GAMMA*LAMBDA*gae
            reverse_advantages = reverse_advantages.write(i, gae)

        advantages = reverse_advantages.stack()[::-1]
        advantages = self._normalize(advantages)

        return advantages

    def _normalize(self, tensor):
        return (tensor - reduce_mean(tensor)) / (reduce_std(tensor)+EPS)
