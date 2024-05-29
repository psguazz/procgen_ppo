import tensorflow.keras as keras
from tensorflow.keras.layers import Dense


class Critic(keras.Model):
    def __init__(self):
        super(Critic, self).__init__()

        self.l1 = Dense(256, activation="relu")
        self.l2 = Dense(256, activation="relu")
        self.l3 = Dense(1, activation=None)

    def call(self, states):
        states = self.l1(states)
        states = self.l2(states)
        states = self.l3(states)

        return states
