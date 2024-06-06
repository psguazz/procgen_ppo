from env import Env
from ppo.agent import Agent

STEPS = 20000

if __name__ == '__main__':
    env = Env()
    agent = Agent(env)

    rewards = []
    steps = 0
    while steps < STEPS:
        training_set = agent.train_new_episodes()

        steps += training_set.total_steps
        rewards.append(training_set.total_rewards)
        avg_reward = sum(rewards) / len(rewards)

        print(f"{steps} steps / {rewards[-1]} latest / {avg_reward} avg")
