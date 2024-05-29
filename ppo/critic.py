import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


class Critic(keras.Model):
    def __init__(self):
        super(Critic, self).__init__()

        self.nn_layers = [
            Conv2D(16, (3, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(32, (3, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3)),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation="relu"),
            Dense(1, activation=None)
        ]

    def call(self, states):
        for layer in self.nn_layers:
            states = layer(states)

        return states
