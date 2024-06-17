import tensorflow as tf
from tensorflow.math import exp, reduce_mean, minimum
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber, Reduction
from ppo.actor_critic import ActorCritic
from ppo.episode import Episode
from ppo.training_set import TrainingSet
from ppo.config import ALPHA, EPOCHS, CLIP, BATCH_SIZE


class Agent:
    def __init__(self, env, reset=False):
        self.env = env

        self.model = ActorCritic(self.env.n_actions)
        self.model.compile(optimizer=Adam(learning_rate=ALPHA))

        self.huber_loss = Huber(reduction=Reduction.SUM)

        if not reset:
            self.model.load(self.env.name)

    def run_new_episode(self):
        episode = Episode()
        state = self.env.reset()

        while not self.env.done:
            action, value, log_prob = self.model.choose(state)
            new_state, reward = self.env.step(action)

            episode.store(
                state=state,
                action=action,
                value=value,
                log_prob=log_prob,
                reward=reward,
            )

            state = new_state

        episode.finalize()
        print(f"{episode.total_reward} ({episode.steps})")

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
        print(f"Training on {len(ts.episodes)} episodes...")

        for e in range(EPOCHS):
            print(f"{e}/{EPOCHS}")

            for b in ts.batches():
                with tf.GradientTape() as tape:
                    values, log_probs = self.model.eval(b.states, b.actions)

                    assert b.states.shape == (BATCH_SIZE, 4, 64, 64, 3)
                    assert b.actions.shape == (BATCH_SIZE,)

                    assert b.returns.shape == (BATCH_SIZE, 1)
                    assert b.values.shape == (BATCH_SIZE, 1)
                    assert b.log_probs.shape == (BATCH_SIZE, 1)
                    assert b.advantages.shape == (BATCH_SIZE, 1)
                    assert values.shape == (BATCH_SIZE, 1)
                    assert log_probs.shape == (BATCH_SIZE, 1)

                    ratios = exp(log_probs - b.log_probs)
                    c_ratios = tf.clip_by_value(ratios, 1-CLIP, 1+CLIP)
                    w_ratios = b.advantages * ratios
                    wc_ratios = b.advantages * c_ratios

                    actor_loss = reduce_mean(-minimum(w_ratios, wc_ratios))
                    critic_loss = self.huber_loss(values, b.returns)

                    loss = actor_loss + critic_loss

                params = self.model.trainable_variables
                grads = tape.gradient(loss, params)
                self.model.optimizer.apply_gradients(zip(grads, params))

        self.model.save(self.env.name)
