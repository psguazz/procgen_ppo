from argparse import ArgumentParser, BooleanOptionalAction
from env import Env
from ppo.agent import Agent as PPOAgent
from dummy.agent import Agent as DummyAgent

STEPS = 2000000

AGENTS = {
    "ppo": PPOAgent,
    "dummy": DummyAgent,
}


def parse_args():
    parser = ArgumentParser(description="ProcGen Benchmark")

    parser.add_argument("-a", action="store",
                        choices=["ppo", "dummy"],
                        default="ppo",
                        help="Agent to use.")

    parser.add_argument("-g", action="store",
                        choices=["coinrun", "starpilot", "cart"],
                        default="cart",
                        help="Game to run")

    parser.add_argument("--train",
                        action=BooleanOptionalAction,
                        default=False,
                        help="Whether to start training from scratch")

    return parser.parse_args()


def clean_args(args):
    agent = AGENTS[args.a]
    game = f"procgen:procgen-{args.g}-v0"
    training = args.train

    return agent, game, training


def train(Agent, game):
    env = Env(game, training=True)
    agent = Agent(env, training=True)
    rewards = agent.train(STEPS)

    return rewards


def eval(Agent, game):
    env = Env(game)
    agent = Agent(env)

    cum_rewards = [0]

    for _ in range(500):
        ep = agent.run_new_episode()
        cum_rewards.append(cum_rewards[-1] + ep.total_reward)

    return cum_rewards[1:]


if __name__ == '__main__':
    args = parse_args()
    Agent, game, training = clean_args(args)

    if training:
        train(Agent, game)
    else:
        eval(Agent, game)
