import gym
import numpy as np


FRAME_STACK = 4


class Env:
    def __init__(self, name, training=False):
        self.name = name

        self.env = gym.make(
            self.name,
            render_mode="human",
            use_backgrounds=False,
            distribution_mode="easy"
        )

        self.n_actions = self.env.action_space.n

        self.reset()

    def step(self, action):
        action = action.numpy()

        state, reward, term, _ = self.env.step(action)
        self.done = term
        self.steps += 1

        self.states = (self.states + [state])[-FRAME_STACK:]

        if reward != 0:
            print(f"S {self.steps} / A {action} / R {reward}")

        if self.done:
            print("Done!")

        return self.states, reward

    def reset(self):
        state = self.env.reset()

        self.states = [np.zeros(state.shape)] * (FRAME_STACK-1) + [state]
        self.done = False
        self.steps = 0

        print("Resetting environment!")

        return self.states
