import gym


class Env:
    def __init__(self, name, training=False):
        self.name = name

        self.env = gym.make(
            self.name,
            render_mode="human",
            distribution_mode="easy"
        )

        self.n_actions = self.env.action_space.n

        self.reset()

    def step(self, action):
        action = action.numpy()

        self.state, self.reward, term, _ = self.env.step(action)
        self.done = term
        self.steps += 1

        print(f"Step {self.steps} / Action {action} / Reward {self.reward}")

        if self.done:
            print("Done!")

        return self.state, self.reward

    def reset(self):
        self.state = self.env.reset()
        self.done = False
        self.steps = 0

        print("Resetting environment!")

        return self.state
