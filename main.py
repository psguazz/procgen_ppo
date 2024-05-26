import gym
from dummy.sample import Agent

GAME = "procgen:procgen-coinrun-v0"
STEPS = 2000
LEARN_BATCH_SIZE = 20

if __name__ == '__main__':
    env = gym.make("procgen:procgen-coinrun-v0", render_mode="human")
    state = env.reset()

    agent = Agent(n_actions=env.action_space.n)

    for step in range(STEPS):
        action, prob, val = agent.choose(state)
        new_state, reward, done, info = env.step(action)

        agent.remember(state, action, prob, val, reward, done)

        if step % LEARN_BATCH_SIZE == 0:
            agent.learn()

        if done:
            state = env.reset()
        else:
            state = new_state

    env.close()
