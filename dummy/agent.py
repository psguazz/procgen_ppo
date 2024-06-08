import tensorflow as tf
from dummy.episode import Episode


class Agent:
    def __init__(self, env, **args):
        self.env = env
        self.n_actions = env.n_actions

    def run_new_episode(self):
        episode = Episode()
        state = self.env.reset()

        while not self.env.done:
            action = tf.random.uniform(
                (),
                minval=0,
                maxval=self.n_actions,
                dtype=tf.int32
            )

            state, reward = self.env.step(action)

            episode.store(reward=reward)

        return episode

    def train(self, *args):
        pass
