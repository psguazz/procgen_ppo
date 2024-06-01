import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from ppo.actor_critic import ActorCritic
from ppo.episode import Episode
from ppo.config import ALPHA


class Agent:
    def __init__(self, env):
        self.env = env
        self.episode = None

        self.model = ActorCritic(self.env.n_actions)
        self.model.compile(optimizer=Adam(learning_rate=ALPHA))

    def run_new_episode(self):
        self.episode = Episode()
        state = self.env.reset()

        while not self.env.done:
            action, value, log_prob = self.model.choose(state)
            new_state, reward = self.env.step(action)

            self.episode.store(
                state=state,
                action=action,
                log_prob=log_prob,
                value=value,
                reward=reward,
                new_state=new_state
            )

            state = new_state

        self.episode.end()
        return self.episode.total_reward()

    def train_new_episode(self):
        with tf.GradientTape() as tape:
            return self.run_new_episode()
