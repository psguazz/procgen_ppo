class Episode:
    def __init__(self):
        self.total_reward = 0
        self.steps = 0

    def store(self, reward):
        self.total_reward += reward
        self.steps += 1
