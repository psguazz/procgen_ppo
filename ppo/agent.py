import tensorflow as tf
from tensorflow.math import reduce_mean, minimum
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Reduction
from ppo.actor_critic import ActorCritic
from ppo.episode import Episode
from ppo.training_set import TrainingSet
from ppo.config import ALPHA, EPOCHS, CLIP


class Agent:
    def __init__(self, env, training=False):
        self.env = env

        self.model = ActorCritic(self.env.n_actions)
        self.model.compile(optimizer=Adam(learning_rate=ALPHA))

        self.huber_loss = tf.keras.losses.Huber(reduction=Reduction.SUM)

        if not training:
            self.model.load(self.env.name)

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

    def train(self, steps):
        rewards = []

        while steps > 0:
            ts = self._new_training_set()
            steps -= ts.total_steps
            rewards += ts.total_rewards

            self._training_loop(ts)

        return rewards

    def _new_training_set(self):
        ts = TrainingSet()

        while not ts.full:
            episode = self.run_new_episode()
            ts.add(episode)

        ts.finalize()

        return ts

    def _training_loop(self, ts):
        for _ in range(EPOCHS):
            for b in ts.batches():
                with tf.GradientTape() as tape:
                    log_probs, values = self.model.eval(b.states, b.actions)

                    advantages = b.returns - values

                    ratios = tf.math.exp(log_probs - b.log_probs)
                    c_ratios = tf.clip_by_value(ratios, 1-CLIP, 1+CLIP)
                    w_ratios = ratios * advantages
                    wc_ratios = c_ratios * advantages

                    actor_loss = -reduce_mean(minimum(w_ratios, wc_ratios))
                    critic_loss = self.huber_loss(values, b.returns)

                    loss = actor_loss + critic_loss

                params = self.model.trainable_variables
                grads = tape.gradient(loss, params)

                print(grads)
                self.model.optimizer.apply_gradients(zip(grads, params))

        self.model.save(self.env.name)
