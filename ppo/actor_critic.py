import os
import tensorflow as tf
from tensorflow.keras import layers, ops, Model
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
from tensorflow.keras.layers import Dense, Conv3D, Embedding
from tensorflow.keras.layers import Reshape, Flatten, Dropout


HEADS = 4
LAYERS = 4
TUBELET_SHAPE = (4, 8, 8)
EMBED_DIM = 128
DENSE_DIM = 256
DROPOUT = 0.1

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights")


def weights_path(checkpoint):
    filename = checkpoint.replace(":", "_") + ".weights.h5"

    return os.path.join(WEIGHTS_PATH, filename)


class TubeletEmbedding(layers.Layer):
    def __init__(self):
        super().__init__()

        self.layers = [
            Conv3D(
                filters=EMBED_DIM,
                kernel_size=TUBELET_SHAPE,
                strides=TUBELET_SHAPE,
                padding="VALID",
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

    def call(self, embeddings):
        encoded_positions = self.position_embedding(self.positions)

        return embeddings + encoded_positions


class FeedForward(layers.Layer):
    def __init__(self):
        super().__init__()

        self.layers = [
            Dense(EMBED_DIM*4, activation="relu"),
            Dense(EMBED_DIM),
        ]

    def call(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class TransformerEncoder(layers.Layer):
    def __init__(self):
        super().__init__()

        self.attention = MultiHeadAttention(num_heads=HEADS, key_dim=EMBED_DIM)
        self.feed_forward = FeedForward()

        self.attn_norm = LayerNormalization(epsilon=1e-6)
        self.ff_norm = LayerNormalization(epsilon=1e-6)

        self.attn_drop = Dropout(DROPOUT)
        self.ff_drop = Dropout(DROPOUT)

    def call(self, input, training):
        attn = self.attention(input, input)
        attn = self.attn_drop(attn, training=training)
        attn = self.attn_norm(attn + input)

        ff = self.feed_forward(attn)
        ff = self.ff_drop(ff, training=training)
        ff = self.ff_norm(ff + attn)

        return ff


class ActorCritic(Model):
    def __init__(self, num_actions):
        super().__init__()

        self.training = False

        self.embedding_layers = [
            TubeletEmbedding(),
            PositionalEmbedding(),
        ]

        self.encoding_layers = [
            TransformerEncoder() for _ in range(LAYERS)
        ]

        self.output_layers = [
            Flatten(),
            Dense(DENSE_DIM, activation="relu")
        ]

        self.actor = Dense(num_actions)
        self.critic = Dense(1)

    def call(self, x):
        for layer in self.embedding_layers:
            x = layer(x)

        for layer in self.encoding_layers:
            x = layer(x, training=self.training)

        for layer in self.output_layers:
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

    def set_training(self, training):
        self.training = training

    def save(self, checkpoint):
        if not os.path.isdir(WEIGHTS_PATH):
            os.makedirs(WEIGHTS_PATH)

        path = weights_path(checkpoint)
        self.save_weights(path)

    def load(self, checkpoint):
        try:
            path = weights_path(checkpoint)
            self.load_weights(path)
        except FileNotFoundError:
            print("Weights not found; starting from scratch")

    def _preprocess(self, states):
        return tf.convert_to_tensor(states, dtype=tf.float32) / 255.0
