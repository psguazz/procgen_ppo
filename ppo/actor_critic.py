import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten


class ActorCritic(keras.Model):
    def __init__(self, num_actions):
        super().__init__()

        self.common_layers = [
            Conv2D(16, (3, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(32, (3, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3)),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation="relu"),
            Dense(num_actions)
        ]

        self.actor = Dense(num_actions)
        self.critic = Dense(1)

    def call(self, x):
        for layer in self.common_layers:
            x = layer(x)

        return self.actor(x), self.critic(x)

    def choose(self, state):
        inputs = self._preprocess(state)
        logits, value = self.call(inputs)

        action = tf.random.categorical(logits, 1)[0, 0]
        prob = tf.nn.softmax(logits)[0, action]
        log_prob = tf.math.log(prob)

        return [tf.squeeze(x) for x in [action, value, log_prob]]

    def _preprocess(self, state):
        return tf.convert_to_tensor([state], dtype=tf.float32) / 255.0
