import tensorflow.keras as keras
from tensorflow.keras.layers import Dense


class Critic(keras.Model):
    def __init__(self):
        super(Critic, self).__init__()

        self.l1 = Dense(256, activation="relu")
        self.l2 = Dense(256, activation="relu")
        self.l3 = Dense(1, activation=None)

    def call(self, state):
        state = self.l1(state)
        state = self.l2(state)
        state = self.l3(state)

        return state
