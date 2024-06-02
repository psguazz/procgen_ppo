import tensorflow as tf
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
        logits, value = self.call(inputs)

        action = tf.random.categorical(logits, 1)[0, 0]
        prob = tf.nn.softmax(logits)[0, action]
        log_prob = tf.math.log(prob)

        return [tf.squeeze(x) for x in [action, value, log_prob]]
