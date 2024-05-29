import gym
from ppo.agent import Agent

GAME = "procgen:procgen-coinrun-v0"
STEPS = 2000

if __name__ == '__main__':
    env = gym.make("procgen:procgen-coinrun-v0", render_mode="human")
    s_t = env.reset()

    agent = Agent(n_actions=env.action_space.n)

    for step in range(STEPS):
        a_t, p_t = agent.choose([s_t])[0]
        s_t1, r_t1, done, info = env.step(a_t)

        agent.remember_and_learn(s_t, a_t, p_t, r_t1, s_t1, done)

        if done:
            s_t = env.reset()
        else:
            s_t = s_t1

    env.close()
