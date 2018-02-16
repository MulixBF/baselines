import argparse
from baselines import logger
# noinspection PyUnresolvedReferences
import gym_fpi


def train(env_id, num_timesteps, seed, debug=False):
    from baselines.common import set_global_seeds
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.ppo2 import ppo2
    from baselines.ppo2.policies import MlpPolicy

    import gym
    import tensorflow as tf
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)

    session = tf.Session(config=config)
    if debug:
        from tensorflow.python import debug as tf_debug
        session = tf_debug.LocalCLIDebugWrapperSession(session)
        session.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    session.__enter__()

    def make_env():
        return gym.make(env_id)

    env = DummyVecEnv([make_env])
    env = VecNormalize(env)

    set_global_seeds(seed)
    policy = MlpPolicy
    ppo2.learn(policy=policy,
               env=env,
               nsteps=2048,
               nminibatches=32,
               lam=0.95,
               gamma=0.99,
               noptepochs=10,
               log_interval=1,
               ent_coef=0.0,
               lr=3e-4,
               cliprange=0.2,
               total_timesteps=num_timesteps)


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--env', type=str, default='FpiCompetitionCondensed-v0')
    argparser.add_argument('--num-timesteps', type=int, default=1e6)
    argparser.add_argument('--debug', action='store_true', default=False)
    return argparser.parse_args()


def main():
    args = parse_args()
    logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=42, debug=args.debug)


if __name__ == '__main__':
    main()
