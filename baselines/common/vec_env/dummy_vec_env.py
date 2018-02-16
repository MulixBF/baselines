from __future__ import absolute_import
import numpy as np
from . import VecEnv
from itertools import izip
from itertools import imap

class DummyVecEnv(VecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]        
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        self.ts = np.zeros(len(self.envs), dtype=u'int')        
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a,env) in izip(self.actions, self.envs)]
        obs, rews, dones, infos = imap(np.array, izip(*results))
        self.ts += 1
        for (i, done) in enumerate(dones):
            if done: 
                obs[i] = self.envs[i].reset_state()
                self.ts[i] = 0
        self.actions = None
        return np.array(obs), np.array(rews), np.array(dones), infos

    def reset(self):        
        results = [env.reset_state() for env in self.envs]
        return np.array(results)

    def close(self):
        return