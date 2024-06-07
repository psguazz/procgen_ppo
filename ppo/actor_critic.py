import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.initializers import RandomNormal


class ActorCritic(keras.Model):
    def __init__(self, num_actions):
        super().__init__()

        init = RandomNormal(mean=0., stddev=0.1)

        self.common_layers = [
            Conv2D(16, (3, 3), kernel_initializer=init),
            MaxPooling2D((2, 2)),
            Conv2D(32, (3, 3), kernel_initializer=init),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), kernel_initializer=init),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation="relu", kernel_initializer=init),
            Dense(num_actions, kernel_initializer=init)
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

    def _preprocess(self, states):
        return tf.convert_to_tensor(states, dtype=tf.float32) / 255.0
