from tensorflow.keras.optimizers import Adam
from pg.actor import Actor
from pg.critic import Critic
from pg.memory import Memory

ALPHA = 0.0003


class BaseAgent:
    def __init__(self, n_actions):
        self.actor = Actor(n_actions)
        self.actor.compile(optimizer=Adam(learning_rate=ALPHA))

        self.critic = Critic()
        self.critic.compile(optimizer=Adam(learning_rate=ALPHA))

        self.memory = Memory()

    def choose(self, state):
        action, prob = self.actor.choose(state)
        value = self.critic.eval(state)

        return action, prob, value

    def remember_and_learn(self, *args):
        self.memory.remember(*args)

        if self.memory.is_full():
            self.learn()
            self.memory.forget()

    def learn(self):
        raise "Not implemented!"
