#!/usr/bin/env python3
import argparse

import gym
# noinspection PyUnresolvedReferences
import gym_fpi
from baselines import logger
from baselines.common import tf_util as U


def train(env_id, num_timesteps, seed, debug=False):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1, debug=debug).__enter__()

    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    hid_size=64, num_hid_layers=2)

    env = gym.make(env_id)
    env.seed(seed)

    pposgd_simple.learn(env, policy_fn,
                        max_timesteps=num_timesteps,
                        timesteps_per_actorbatch=512,
                        clip_param=0.2,
                        entcoeff=0.0,
                        optim_epochs=100,
                        optim_stepsize=3e-4,
                        optim_batchsize=64,
                        gamma=0.99,
                        lam=0.95,
                        schedule='linear',)
    env.close()


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--env', type=str, default='FpiCompetitionCondensed-v0')
    argparser.add_argument('--num-timesteps', type=int, default=1e6)
    argparser.add_argument('--seed', type=int, default=42)
    argparser.add_argument('--debug', action='store_true', default=False)
    return argparser.parse_args()


def main():
    args = parse_args()
    logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed, debug=args.debug)


if __name__ == '__main__':
    main()
