import tensorflow as tf
from tensorflow.math import reduce_sum
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Reduction
from ppo.actor_critic import ActorCritic
from ppo.episode import Episode
from ppo.training_set import TrainingSet
from ppo.config import ALPHA, EPOCHS


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
                state=state,
                action=action,
                log_prob=log_prob,
                value=value,
                reward=reward,
            )

        episode.finalize()

        return episode

    def train_new_episodes(self):
        ts = TrainingSet()

        while not ts.full:
            episode = self.run_new_episode()
            ts.add(episode)

        ts.finalize()

        for _ in range(EPOCHS):
            for b in ts.batches():
                with tf.GradientTape() as tape:
                    advantages = b.returns - b.values

                    actor_loss = -reduce_sum(b.log_probs * advantages)
                    critic_loss = self.huber_loss(b.values, b.returns)

                    __import__('pdb').set_trace()
                    loss = actor_loss + critic_loss

                params = self.model.trainable_variables
                grads = tape.gradient(loss, params)
                self.model.optimizer.apply_gradients(zip(grads, params))

        return ts
