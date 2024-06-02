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
        self.episode = None

        self.model = ActorCritic(self.env.n_actions)
        self.model.compile(optimizer=Adam(learning_rate=ALPHA))

        self.huber_loss = tf.keras.losses.Huber(reduction=Reduction.SUM)

    def run_new_episode(self):
        self.episode = Episode()
        state = self.env.reset()

        while not self.env.done:
            action, value, log_prob = self.model.choose(state)
            state, reward = self.env.step(action)

            self.episode.store(
                log_prob=log_prob,
                value=value,
                reward=reward,
            )

        self.episode.end()
        return self.episode.total_reward()

    def train_new_episode(self):
        with tf.GradientTape() as tape:
            reward = self.run_new_episode()

            rewards = tf.expand_dims(self.episode.expected_returns(), 1)
            values = tf.expand_dims(self.episode.values, 1)
            log_probs = tf.expand_dims(self.episode.log_probs, 1)

            advantages = rewards - values

            actor_loss = -reduce_sum(log_probs * advantages)
            critic_loss = self.huber_loss(values, rewards)

            loss = actor_loss + critic_loss

        params = self.model.trainable_variables
        grads = tape.gradient(loss, params)
        self.model.optimizer.apply_gradients(zip(grads, params))

        return reward
