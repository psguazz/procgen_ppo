
import gym


class Env:
    def __init__(self):
        self.env = gym.make(
            "procgen:procgen-coinrun-v0",
            render_mode="human",
            distribution_mode="easy"
        )

        self.n_actions = self.env.action_space.n

        self.reset()

    def step(self, action):
        action = action.numpy()

        self.state, self.reward, term, _ = self.env.step(action)
        self.done = term

        return self.state, self.reward

    def reset(self):
        self.state = self.env.reset()
        self.done = False

        return self.state
