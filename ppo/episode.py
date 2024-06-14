import tensorflow as tf
from ppo.config import GAMMA


opts = {"dtype": tf.float32, "size": 0, "dynamic_size": True}


class Episode:
    def __init__(self):
        self.states = tf.TensorArray(**opts)
        # self.actions = tf.TensorArray(**opts)
        self.values = tf.TensorArray(**opts)
        self.log_probs = tf.TensorArray(**opts)
        self.rewards = tf.TensorArray(**opts)

        self.returns = None

        self.total_reward = 0
        self.steps = 0

        self.done = False

    def store(self, state, action, value, log_prob, reward):
        if self.done:
            return

        self.states = self.states.write(self.steps, state)
        # self.actions = self.actions.write(self.steps, action)
        self.values = self.values.write(self.steps, value)
        self.log_probs = self.log_probs.write(self.steps, log_prob)
        self.rewards = self.rewards.write(self.steps, reward)

        self.total_reward += reward
        self.steps += 1

    def finalize(self):
        if self.done:
            return

        self.states = self.states.stack()
        # self.actions = self.actions.stack()
        self.values = self.values.stack()
        self.log_probs = self.log_probs.stack()
        self.rewards = self.rewards.stack()

        self.returns = self._compute_returns()

        self.done = True

    def _compute_returns(self):
        if self.done:
            return

        reverse_rewards = self.rewards[::-1]
        reverse_returns = tf.TensorArray(**opts)

        discounted_sum = 0

        for i in range(self.steps):
            discounted_sum = reverse_rewards[i] + GAMMA*discounted_sum
            reverse_returns = reverse_returns.write(i, discounted_sum)

        return reverse_returns.stack()[::-1]
