from tensorflow.keras.layers import Reshape, Conv3D, Dense, LSTM
from tensorflow.keras.layers import Bidirectional as BI, TimeDistributed as TD
from ppo.models.base import BaseModel


class ActorCritic(BaseModel):
    def __init__(self, num_actions):
        super().__init__("impala")

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
