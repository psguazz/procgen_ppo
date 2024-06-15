from ppo.config import BATCHES, BATCH_SIZE


class TrainingSet:
    def __init__(self):
        self.total_steps = 0
        self.total_rewards = []
        self.episodes = []
        self.full = False

    def add(self, episode):
        if not episode.done:
            return

        self.episodes.append(episode)
        self.total_steps += episode.steps
        self.total_rewards.append(episode.total_reward)

        self.full = self.total_steps > BATCHES * BATCH_SIZE

    def batches(self):
        for episode in self.episodes:
            yield episode
