
import gym


class Env:
    def __init__(self):
        self.env = gym.make(
            "CartPole-v1",
            render_mode="human",
        )

        self.n_actions = self.env.action_space.n

        self.reset()

    def step(self, action):
        self.state, self.reward, term, trunc, _ = self.env.step(action)
        self.done = term or trunc

        return self.state, self.reward

    def reset(self):
        self.state, _ = self.env.reset()
        self.done = False

        return self.state
