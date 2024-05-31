import gym
from ppo.agent import Agent

GAME = "procgen:procgen-coinrun-v0"
STEPS = 20000

if __name__ == '__main__':
    env = gym.make(
        "CartPole-v1",
        render_mode="human",
    )

    s_t, info = env.reset()

    agent = Agent(n_actions=env.action_space.n)

    levels = 1
    all_steps = 1
    level_steps = 1
    scores = [0]

    for step in range(STEPS):
        all_steps += 1
        level_steps += 1

        a_t, p_t = agent.choose([s_t])[0]
        s_t1, r_t1, term, trunc, info = env.step(a_t)
        done = term or trunc

        scores[-1] += r_t1

        agent.remember_and_learn(
            s_t=s_t,
            a_t=a_t,
            p_t=p_t,
            r_t1=r_t1,
            s_t1=s_t1,
            done=done
        )

        if done:
            print(f"Ep {levels}: {level_steps} steps, {scores[-1]} points")

            s_t, info = env.reset()
            level_steps = 0
            levels += 1
            scores.append(0)
        else:
            s_t = s_t1

    print(scores)
    env.close()
