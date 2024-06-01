import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense


class ActorCritic(keras.Model):
    def __init__(self, num_actions):
        super().__init__()

        self.common = Dense(128, activation="relu")
        self.actor = Dense(num_actions, activation="softmax")
        self.critic = Dense(1)

    def call(self, inputs):
        x = self.common(inputs)
        return self.actor(x), self.critic(x)

    def choose(self, state):
        inputs = tf.convert_to_tensor([state])
        probs, value = self.call(inputs)

        dist = tfp.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.numpy()[0], value.numpy()[0], log_prob.numpy()[0]
