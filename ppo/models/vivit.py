import os
import tensorflow as tf
from tensorflow.keras import layers, ops, Model
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
from tensorflow.keras.layers import Dense, Conv3D, Embedding, AveragePooling3D
from tensorflow.keras.layers import Reshape, Flatten, Dropout
from ppo.models.base import BaseModel


HEADS = 8
LAYERS = 8
TUBELET_SHAPE = (4, 8, 8)
POOL_SIZE = (1, 2, 2)
EMBED_DIM = 128
DENSE_DIM = 256
DROPOUT = 0.1


class TubeletEmbedding(layers.Layer):
    def __init__(self):
        super().__init__()

        self.layers = [
            AveragePooling3D(pool_size=POOL_SIZE, strides=POOL_SIZE),

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


class ActorCritic(BaseModel):
    def __init__(self, num_actions):
        super().__init__("vivit")

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
