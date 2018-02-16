#!/usr/bin/env python3
from __future__ import absolute_import
from baselines import logger
from baselines.acer.acer_simple import learn
from baselines.acer.policies import AcerCnnPolicy, AcerLstmPolicy
from baselines.common.cmd_util import make_atari_env, atari_arg_parser

def train(env_id, num_timesteps, seed, policy, lrschedule, num_cpu):
    env = make_atari_env(env_id, num_cpu, seed)
    if policy == u'cnn':
        policy_fn = AcerCnnPolicy
    elif policy == u'lstm':
        policy_fn = AcerLstmPolicy
    else:
        print u"Policy {} not implemented".format(policy)
        return
    learn(policy_fn, env, seed, total_timesteps=int(num_timesteps * 1.1), lrschedule=lrschedule)
    env.close()

def main():
    parser = atari_arg_parser()
    parser.add_argument(u'--policy', help=u'Policy architecture', choices=[u'cnn', u'lstm', u'lnlstm'], default=u'cnn')
    parser.add_argument(u'--lrschedule', help=u'Learning rate schedule', choices=[u'constant', u'linear'], default=u'constant')
    parser.add_argument(u'--logdir', help =u'Directory for logging')
    args = parser.parse_args()
    logger.configure(args.logdir)
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
          policy=args.policy, lrschedule=args.lrschedule, num_cpu=16)

if __name__ == u'__main__':
    main()
