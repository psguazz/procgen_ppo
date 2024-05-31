import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


class Actor(keras.Model):
    def __init__(self, n_actions):
        super(Actor, self).__init__()

        self.nn_layers = [
            Dense(64, activation="relu"),
            Dense(64, activation="relu"),
            Dense(n_actions, activation="softmax")
        ]

    def call(self, states):
        for layer in self.nn_layers:
            states = layer(states)

        return states
