import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from ppo.actor import Actor
from ppo.critic import Critic
from ppo.memory import Memory

ALPHA = 0.0003
GAMMA = 0.99
CLIP = 0.2
EPOCHS = 2


class Agent:
    def __init__(self, n_actions):
        self.actor = Actor(n_actions)
        self.actor.compile(optimizer=Adam(learning_rate=ALPHA))

        self.critic = Critic()
        self.critic.compile(optimizer=Adam(learning_rate=ALPHA))

        self.memory = Memory()

    def preprocess(self, states):
        return tf.convert_to_tensor(states, dtype=tf.float32) / 255.0

    def choose(self, states):
        states = self.preprocess(states)

        probs = self.actor(states)
        dist = tfp.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return list(zip(action.numpy(), log_prob.numpy()))

    def eval(self, states):
        states = self.preprocess(states)

        value = self.critic(states)

        return value.numpy()

    def remember_and_learn(self, s_t, a_t, p_t, r_t1, s_t1, done):
        v_t, v_t1 = self.eval([s_t, s_t1])
        d_t = r_t1 + (1-int(done))*GAMMA*v_t1 - v_t

        self.memory.remember(
            s_t=s_t,
            a_t=a_t,
            p_t=p_t,
            v_t=v_t,
            r_t1=r_t1,
            s_t1=s_t1,
            v_t1=v_t1,
            d_t=d_t
        )

        if self.memory.is_full():
            self.learn()
            self.memory.forget()

    def learn(self):
        for _ in range(EPOCHS):
            for batch in self.memory.batches():
                with tf.GradientTape(persistent=True) as tape:
                    probs = self.actor(batch.s_ts)
                    dist = tfp.distributions.Categorical(probs)
                    probs = dist.log_prob(batch.a_ts)

                    p_ratio = tf.math.exp(probs - batch.p_ts)
                    p_clip = tf.clip_by_value(p_ratio, 1-CLIP, 1+CLIP)
                    p_weights = p_ratio * batch.a_ts
                    p_clip_weights = p_clip * batch.a_ts

                    actor_loss = -tf.math.minimum(p_weights, p_clip_weights)
                    actor_loss = tf.math.reduce_mean(actor_loss)

                    values = self.critic(batch.s_ts)
                    returns = batch.a_ts + batch.v_ts

                    critic_loss = keras.losses.MSE(values, returns)

                actor_params = self.actor.trainable_variables
                actor_grads = tape.gradient(actor_loss, actor_params)
                actor_args = zip(actor_grads, actor_params)
                self.actor.optimizer.apply_gradients(actor_args)

                critic_params = self.critic.trainable_variables
                critic_grads = tape.gradient(critic_loss, critic_params)
                critic_args = zip(critic_grads, critic_params)
                self.critic.optimizer.apply_gradients(critic_args)
