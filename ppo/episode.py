import tensorflow as tf
from tensorflow.math import reduce_mean, reduce_std, reduce_sum
from ppo.config import GAMMA, EPS


class Episode:
    def __init__(self):
        opts = {"dtype": tf.float32, "size": 0, "dynamic_size": True}

        self.log_probs = tf.TensorArray(**opts)
        self.values = tf.TensorArray(**opts)
        self.rewards = tf.TensorArray(**opts)

        self.steps = 0
        self.done = False

    def store(self, log_prob, value, reward):
        self.log_probs = self.log_probs.write(self.steps, log_prob)
        self.values = self.values.write(self.steps, value)
        self.rewards = self.rewards.write(self.steps, reward)

        self.steps += 1

    def end(self):
        if self.done:
            return

        self.log_probs = self.log_probs.stack()
        self.values = self.values.stack()
        self.rewards = self.rewards.stack()

    def expected_returns(self):
        if self.done:
            return

        reverse_rewards = self.rewards[::-1]
        reverse_returns = tf.TensorArray(dtype=tf.float32, size=self.steps)

        discounted_sum = 0

        for i in range(self.steps):
            discounted_sum = reverse_rewards[i] + GAMMA*discounted_sum
            reverse_returns = reverse_returns.write(i, discounted_sum)

        returns = reverse_returns.stack()[::-1]
        returns = (returns - reduce_mean(returns)) / (reduce_std(returns)+EPS)

        return returns

    def total_reward(self):
        return reduce_sum(self.rewards).numpy()
