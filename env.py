import gym
import numpy as np


FRAME_STACK = 4


class Env:
    def __init__(self, name, training=False):
        self.name = name
        self.name = "CartPole-v1"

        self.env = gym.make(
            self.name,
            # render_mode="human",
        )

        self.n_actions = self.env.action_space.n

        self.reset()

    def step(self, action):
        action = action.numpy()

        state, reward, term, trunc, _ = self.env.step(action)
        self.done = term or trunc
        self.steps += 1

        self.states = (self.states + [state])[-FRAME_STACK:]

        return state, reward

    def reset(self):
        state, _ = self.env.reset()

        self.states = [np.zeros(state.shape)] * (FRAME_STACK-1) + [state]
        self.done = False
        self.steps = 0

        return state
