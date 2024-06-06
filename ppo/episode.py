import numpy as np
from ppo.config import GAMMA, EPS


class Episode:
    def __init__(self):
        self.states = np.array([])
        self.actions = np.array([])
        self.log_probs = np.array([])
        self.values = np.array([])
        self.rewards = np.array([])

        self.returns = None

        self.total_reward = 0
        self.steps = 0

        self.done = False

    def store(self, state, action, log_prob, value, reward):
        if self.done:
            return

        self.states = np.append(self.states, state)
        self.actions = np.append(self.actions, action)
        self.log_probs = np.append(self.log_probs, log_prob)
        self.values = np.append(self.values, value)
        self.rewards = np.append(self.rewards, reward)

        self.total_reward += reward
        self.steps += 1

    def finalize(self):
        if self.done:
            return

        self.returns = self._compute_returns()

        self.done = True

    def _compute_returns(self):
        if self.done:
            return

        reverse_rewards = self.rewards[::-1]
        reverse_returns = np.array([])

        discounted_sum = 0

        for i in range(self.steps):
            discounted_sum = reverse_rewards[i] + GAMMA*discounted_sum
            reverse_returns = np.append(reverse_returns, discounted_sum)

        returns = reverse_returns[::-1]
        returns = (returns - np.mean(returns)) / (np.std(returns)+EPS)

        return returns
