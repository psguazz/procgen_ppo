import gym
from ppo.agent import Agent

GAME = "procgen:procgen-coinrun-v0"
STEPS = 2000

if __name__ == '__main__':
    env = gym.make(
        "procgen:procgen-coinrun-v0",
        distribution_mode="easy",
        render_mode="human"
    )

    s_t = env.reset()

    agent = Agent(n_actions=env.action_space.n)

    levels = 1
    all_steps = 1
    level_steps = 1

    for step in range(STEPS):
        print(f"{levels} / {level_steps} / {all_steps}")
        all_steps += 1
        level_steps += 1

        a_t, p_t = agent.choose([s_t])[0]
        s_t1, r_t1, done, info = env.step(a_t)

        agent.remember_and_learn(
            s_t=s_t,
            a_t=a_t,
            p_t=p_t,
            r_t1=r_t1,
            s_t1=s_t1,
            done=done
        )

        if done:
            s_t = env.reset()
            level_steps = 0
            levels += 1
        else:
            s_t = s_t1

    env.close()
