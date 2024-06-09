import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Conv3D, Reshape, LSTM
from tensorflow.keras.layers import Bidirectional as BI, TimeDistributed as TD
from tensorflow.keras.initializers import RandomNormal
from ppo.config import WEIGHTS_PATH


class ActorCritic(keras.Model):
    def __init__(self, num_actions):
        super().__init__()

        init = RandomNormal(mean=0., stddev=0.1)

        self.common_layers = [
            Conv3D(16, (1, 8, 8), strides=(1, 4, 4), kernel_initializer=init),
            Conv3D(32, (1, 4, 4), strides=(1, 2, 2), kernel_initializer=init),
            Conv3D(32, (4, 1, 1), strides=(1, 1, 1), kernel_initializer=init),
            Reshape((4, -1)),
            TD(Dense(256, activation="relu", kernel_initializer=init)),
            BI(LSTM(256, kernel_initializer=init))
        ]

        self.actor = Dense(num_actions, kernel_initializer=init)
        self.critic = Dense(1, kernel_initializer=init)

    def call(self, x):
        for layer in self.common_layers:
            x = layer(x)

        return self.actor(x), self.critic(x)

    def choose(self, state):
        inputs = self._preprocess([state])
        logits, value = self.call(inputs)

        action = tf.random.categorical(logits, 1)[0, 0]
        prob = tf.nn.softmax(logits)[0, action]
        log_prob = tf.math.log(prob)

        return [tf.squeeze(x) for x in [action, value, log_prob]]

    def eval(self, states, actions):
        inputs = self._preprocess(states)
        logits, values = self.call(inputs)

        indices = tf.range(logits.shape[0])
        indices = tf.stack([indices, actions], axis=1)

        probs = tf.nn.softmax(logits)
        probs = tf.gather_nd(probs, indices)
        log_probs = tf.math.log(probs)

        return [tf.squeeze(x) for x in [values, log_probs]]

    def save(self, checkpoint):
        if not os.path.isdir(WEIGHTS_PATH):
            os.makedirs(WEIGHTS_PATH)

        self.save_weights(WEIGHTS_PATH + checkpoint + ".weights.h5")

    def load(self, checkpoint):
        try:
            self.load_weights(WEIGHTS_PATH + checkpoint + ".weights.h5")
        except FileNotFoundError:
            print("Weights not found; starting from scratch")

    def _preprocess(self, states):
        return tf.convert_to_tensor(states, dtype=tf.float32) / 255.0
