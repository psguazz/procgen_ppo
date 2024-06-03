import tensorflow as tf
from tensorflow.math import reduce_sum
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Reduction
from ppo.actor_critic import ActorCritic
from ppo.episode import Episode
from ppo.config import ALPHA


class Agent:
    def __init__(self, env):
        self.env = env

        self.model = ActorCritic(self.env.n_actions)
        self.model.compile(optimizer=Adam(learning_rate=ALPHA))

        self.huber_loss = tf.keras.losses.Huber(reduction=Reduction.SUM)

    def run_new_episode(self):
        episode = Episode()
        state = self.env.reset()

        while not self.env.done:
            action, value, log_prob = self.model.choose(state)
            state, reward = self.env.step(action)

            episode.store(
                log_prob=log_prob,
                value=value,
                reward=reward,
            )

        episode.finalize()

        return episode

    def train_new_episode(self):
        with tf.GradientTape() as tape:
            episode = self.run_new_episode()

            expected_returns = episode.expected_returns
            values = episode.values
            log_probs = episode.log_probs

            advantages = expected_returns - values

            actor_loss = -reduce_sum(log_probs * advantages)
            critic_loss = self.huber_loss(values, expected_returns)

            loss = actor_loss + critic_loss

        params = self.model.trainable_variables
        grads = tape.gradient(loss, params)
        self.model.optimizer.apply_gradients(zip(grads, params))

        return episode
