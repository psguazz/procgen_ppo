import os
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.saving import serialize_keras_object as serialize
from tensorflow.keras.saving import deserialize_keras_object as deserialize
from tensorflow.keras.layers import Conv3D, Reshape, Dense, LSTM
from tensorflow.keras.layers import Bidirectional as BI, TimeDistributed as TD

MODEL_PATH = os.path.join(os.path.dirname(__file__), "weights")


def model_path(checkpoint):
    filename = checkpoint.replace(":", "_")

    return os.path.join(MODEL_PATH, filename + ".keras")


class ActorCritic(Model):
    def __init__(self, num_actions, common=None, actor=None, critic=None, **k):
        super().__init__(**k)

        self.common = common or [
            Conv3D(16, (1, 8, 8), strides=(1, 4, 4)),
            Conv3D(32, (1, 4, 4), strides=(1, 2, 2)),
            Conv3D(32, (4, 1, 1), strides=(1, 1, 1)),
            Reshape((4, -1)),
            TD(Dense(256, activation="relu")),
            BI(LSTM(256))
        ]

        self.actor = actor or Dense(num_actions)
        self.critic = critic or Dense(1)

    def call(self, x):
        for layer in self.common:
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

    def get_config(self):
        base_config = super().get_config()
        config = {

            "common": [serialize(lr) for lr in self.common],
            "actor": serialize(self.actor),
            "critic": serialize(self.critic)
        }

        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        common = deserialize(config.pop("common"))
        actor = deserialize(config.pop("actor"))
        critic = deserialize(config.pop("critic"))

        return cls(0, common=common, actor=actor, critic=critic, **config)

    def save_model(self, checkpoint):
        if not os.path.isdir(MODEL_PATH):
            os.makedirs(MODEL_PATH)

        path = model_path(checkpoint)
        self.save(path)

    @classmethod
    def load_model(cls, num_actions, checkpoint):
        try:
            path = model_path(checkpoint)
            return load_model(path, custom_objects={"ActorCritic": cls})
        except ValueError:
            print("Weights not found; starting from scratch")

            return cls(num_actions)

    def _preprocess(self, states):
        return tf.convert_to_tensor(states, dtype=tf.float32) / 255.0
