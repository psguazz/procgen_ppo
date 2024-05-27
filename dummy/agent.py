import random


class Agent:
    def __init__(self, n_actions, *args):
        self.n_actions = n_actions

    def choose(self, state):
        action = random.randint(0, self.n_actions)
        prob = 1/self.n_actions
        val = 0

        return action, prob, val

    def remember_and_learn(self, *args):
        pass
