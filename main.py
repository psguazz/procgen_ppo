import tensorflow.keras as keras
from argparse import ArgumentParser, BooleanOptionalAction
from env import Env
from ppo.agent import Agent as PPOAgent
from dummy.agent import Agent as DummyAgent

STEPS = 2000000

AGENTS = {
    "ppo": PPOAgent,
    "dummy": DummyAgent,
}


SEED = 42
keras.utils.set_random_seed(SEED)


def parse_args():
    parser = ArgumentParser(description="ProcGen Benchmark")

    parser.add_argument("-a", action="store",
                        choices=["ppo", "dummy"],
                        default="ppo",
                        help="Agent to use.")

    parser.add_argument("-g", action="store",
                        choices=["coinrun", "starpilot"],
                        default="starpilot",
                        help="Game to run")

    parser.add_argument("--train",
                        action=BooleanOptionalAction,
                        default=False,
                        help="If true, the agent will learn from the episodes")

    parser.add_argument("--reset",
                        action=BooleanOptionalAction,
                        default=False,
                        help="If true and training, the agent will start over")

    return parser.parse_args()


def clean_args(args):
    agent = AGENTS[args.a]
    game = f"procgen:procgen-{args.g}-v0"
    training = args.train
    reset = args.reset

    return agent, game, training, reset


def train(Agent, game, reset):
    env = Env(game, training=True)
    agent = Agent(env, reset=reset)
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
    Agent, game, training, reset = clean_args(args)

    if training:
        train(Agent, game, reset)
    else:
        eval(Agent, game)
