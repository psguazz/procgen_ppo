import numpy as np
from ppo.config import GAMMA, EPS


class Episode:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []

        self.returns = None

        self.total_reward = 0
        self.steps = 0

        self.done = False

    def store(self, state, action, log_prob, value, reward):
        if self.done:
            return

        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)

        self.total_reward += reward
        self.steps += 1

    def finalize(self):
        if self.done:
            return

        self.actions = np.array(self.actions)
        self.log_probs = np.array(self.log_probs)
        self.values = np.array(self.values)
        self.rewards = np.array(self.rewards)

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
