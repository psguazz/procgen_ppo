import gym
import numpy as np


FRAME_STACK = 4


class Env:
    def __init__(self, name, training=False):
        self.name = name

        levels = 200 if training else 0

        self.env = gym.make(
            self.name,
            render_mode="human",
            use_backgrounds=False,
            distribution_mode="easy",
            start_level=42,
            num_levels=levels
        )

        self.n_actions = self.env.action_space.n

        self.reset()

    def step(self, action):
        action = action.numpy()

        state, reward, term, _ = self.env.step(action)
        self.done = term
        self.steps += 1

        self.states = (self.states + [state])[-FRAME_STACK:]

        return self.states, reward

    def reset(self):
        state = self.env.reset()

        self.states = [np.zeros(state.shape)] * (FRAME_STACK-1) + [state]
        self.done = False
        self.steps = 0

        return self.states
