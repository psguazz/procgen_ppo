import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


class Actor(keras.Model):
    def __init__(self, n_actions):
        super(Actor, self).__init__()

        self.nn_layers = [
            Conv2D(16, (3, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(32, (3, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3)),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation="relu"),
            Dense(n_actions, activation="softmax")
        ]

    def call(self, states):
        for layer in self.nn_layers:
            states = layer(states)

        return states
