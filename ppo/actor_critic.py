import os
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv3D, Reshape, Dense, LSTM
from tensorflow.keras.layers import Bidirectional as BI, TimeDistributed as TD

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights")


class ActorCritic(Model):
    def __init__(self, num_actions):
        super().__init__()

        self.common_layers = [
            Conv3D(16, (1, 8, 8), strides=(1, 4, 4)),
            Conv3D(32, (1, 4, 4), strides=(1, 2, 2)),
            Conv3D(32, (4, 1, 1), strides=(1, 1, 1)),
            Reshape((4, -1)),
            TD(Dense(256, activation="relu")),
            BI(LSTM(256))
        ]

        self.actor = Dense(num_actions)
        self.critic = Dense(1)

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

        return action, tf.squeeze(value), log_prob

    def eval(self, states, actions):
        inputs = self._preprocess(states)
        logits, values = self.call(inputs)

        indices = tf.range(logits.shape[0], dtype=tf.int64)
        indices = tf.stack([indices, actions], axis=1)

        probs = tf.nn.softmax(logits)
        probs = tf.gather_nd(probs, indices)
        log_probs = tf.math.log(probs)

        return values, tf.expand_dims(log_probs, 1)

    def save(self, checkpoint):
        if not os.path.isdir(WEIGHTS_PATH):
            os.makedirs(WEIGHTS_PATH)

        path = self._weights_path(checkpoint)
        self.save_weights(path)

    def load(self, checkpoint):
        try:
            path = self._weights_path(checkpoint)
            self.load_weights(path)
        except FileNotFoundError:
            print("Weights not found; starting from scratch")

    def _preprocess(self, states):
        return tf.convert_to_tensor(states, dtype=tf.float32) / 255.0

    def _weights_path(self, checkpoint):
        filename = checkpoint.replace(":", "_")
        filename = f"{filename}.weights.h5"

        return os.path.join(WEIGHTS_PATH, filename)
