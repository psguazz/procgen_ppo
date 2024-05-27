import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense


class Actor(keras.Model):
    def __init__(self, n_actions):
        super(Actor, self).__init__()

        self.l1 = Dense(256, activation="relu")
        self.l2 = Dense(256, activation="relu")
        self.l3 = Dense(n_actions, activation="softmax")

    def call(self, state):
        state = self.l1(state)
        state = self.l2(state)
        state = self.l3(state)

        return state

    def choose(self, state):
        state = tf.convert_to_tensor([state])

        probs = self(state)
        dist = tfp.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        action = action.numpy()[0]
        log_prob = log_prob.numpy()[0]

        return action, log_prob
