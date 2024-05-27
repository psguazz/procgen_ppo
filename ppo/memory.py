BATCH_SIZE = 20


class Memory:
    def __init__(self):
        self.forget()

    def is_full(self):
        return len(self.states) >= BATCH_SIZE * 5

    def remember(self, state, action, prob, value, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def forget(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.values = []
        self.rewards = []
        self.dones = []
