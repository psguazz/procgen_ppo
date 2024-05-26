import tensorflow.keras as keras
from tensorflow.keras.layers import Dense


class Actor(keras.Model):
    def __init__(self, n_actions):
        super(Actor, self).__init__()

        self.l1 = Dense(256, activation="relu")
        self.l2 = Dense(256, activation="relu")
        self.l3 = Dense(n_actions, activation="softmax")

    def call(self, state):
        state = self.l1(state)
        state = self.l2(state)
        state = self.l3(state)

        return state
