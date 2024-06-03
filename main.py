import tqdm
from env import Env
from ppo.agent import Agent

EPISODES = 20000

if __name__ == '__main__':
    env = Env()
    agent = Agent(env)

    rewards = []

    t = tqdm.trange(EPISODES)
    for _ in t:
        episode = agent.train_new_episode()
        reward = episode.total_reward

        rewards.append(reward)
        avg_reward = sum(rewards) / len(rewards)

        t.set_postfix(reward=reward, average=avg_reward)
