import os
import tensorflow as tf
from tensorflow.keras import layers, ops, Model
from tensorflow.keras.layers import Dense, Conv3D, Reshape, Embedding, Flatten
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
from tensorflow.keras.initializers import RandomNormal
from ppo.config import WEIGHTS_PATH

HEADS = 8
LAYERS = 8
TUBELET_SHAPE = (4, 8, 8)
EMBED_DIM = 128

init = RandomNormal(mean=0., stddev=0.1)
init = None


class TubeletEmbedding(layers.Layer):
    def __init__(self):
        super().__init__()

        self.layers = [
            Conv3D(
                filters=EMBED_DIM,
                kernel_size=TUBELET_SHAPE,
                strides=TUBELET_SHAPE,
                padding="VALID",
                kernel_initializer=init
            ),

            Reshape(target_shape=(-1, EMBED_DIM))
        ]

    def call(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class PositionalEmbedding(layers.Layer):
    def build(self, input_shape):
        _, num_tokens, _ = input_shape

        self.positions = ops.arange(0, num_tokens, 1)
        self.position_embedding = Embedding(
            input_dim=num_tokens,
            output_dim=EMBED_DIM
        )

    def call(self, x):
        encoded_positions = self.position_embedding(self.positions)

        return x + encoded_positions


class TransformerEncoder(layers.Layer):
    def __init__(self):
        super().__init__()

        self.attention = MultiHeadAttention(num_heads=HEADS, key_dim=EMBED_DIM)

        self.layers = [
            LayerNormalization(epsilon=1e-6),
            Dense(512, activation="relu", kernel_initializer=init),
            Dense(EMBED_DIM, kernel_initializer=init),
            LayerNormalization(epsilon=1e-6),
        ]

    def call(self, x):
        x = x + self.attention(x, x)

        for layer in self.layers:
            x = layer(x)

        return x


class ActorCritic(Model):
    def __init__(self, num_actions):
        super().__init__()

        self.common_layers = [
            TubeletEmbedding(),
            PositionalEmbedding(),
        ] + [
            TransformerEncoder() for _ in range(LAYERS)
        ] + [
            Flatten(),
            Dense(256, activation="relu", kernel_initializer=init)
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

        self.save_weights(WEIGHTS_PATH + checkpoint + ".weights.h5")

    def load(self, checkpoint):
        try:
            self.load_weights(WEIGHTS_PATH + checkpoint + ".weights.h5")
        except FileNotFoundError:
            print("Weights not found; starting from scratch")

    def _preprocess(self, states):
        return tf.convert_to_tensor(states, dtype=tf.float32) / 255.0
