import gym
from dummy.sample import Agent

GAME = "procgen:procgen-coinrun-v0"
STEPS = 2000

if __name__ == '__main__':
    env = gym.make("procgen:procgen-coinrun-v0", render_mode="human")
    state = env.reset()

    agent = Agent(n_actions=env.action_space.n)

    for step in range(STEPS):
        action, prob, val = agent.choose(state)
        new_state, reward, done, info = env.step(action)

        agent.remember_and_learn(state, action, prob, val, reward, done)

        if done:
            state = env.reset()
        else:
            state = new_state

    env.close()
