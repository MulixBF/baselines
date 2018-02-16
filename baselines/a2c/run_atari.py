#!/usr/bin/env python3

from __future__ import absolute_import
from baselines import logger
from baselines.common.cmd_util import make_atari_env, atari_arg_parser
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.a2c.a2c import learn
from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy

def train(env_id, num_timesteps, seed, policy, lrschedule, num_env):
    if policy == u'cnn':
        policy_fn = CnnPolicy
    elif policy == u'lstm':
        policy_fn = LstmPolicy
    elif policy == u'lnlstm':
        policy_fn = LnLstmPolicy
    env = VecFrameStack(make_atari_env(env_id, num_env, seed), 4)
    learn(policy_fn, env, seed, total_timesteps=int(num_timesteps * 1.1), lrschedule=lrschedule)
    env.close()

def main():
    parser = atari_arg_parser()
    parser.add_argument(u'--policy', help=u'Policy architecture', choices=[u'cnn', u'lstm', u'lnlstm'], default=u'cnn')
    parser.add_argument(u'--lrschedule', help=u'Learning rate schedule', choices=[u'constant', u'linear'], default=u'constant')
    args = parser.parse_args()
    logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
        policy=args.policy, lrschedule=args.lrschedule, num_env=16)

if __name__ == u'__main__':
    main()
