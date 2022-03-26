import multiprocessing as mp

import numpy as np

from .wrappers import MultiProcessing, TimeLimit, CollectDataset, RewardObs
from misc import ConfiguredModule


def make_env(config, callbacks):
  suite, task = config._task.split('_', 1)
  print("Suite: "+suite)
  
  if suite == 'prolog':
    from .ProLog import ProLog
    env=ProLog()
  elif suite == 'dummy':
    from .Dummy import DummyEnv
    env=DummyEnv()
  else:
    raise NotImplementedError(suite)
  env = TimeLimit(env, config._time_limit)
  env = CollectDataset(env, callbacks)
  env = RewardObs(env)
  return env

class ShadowPool:
  def apply(self, func, args=()):
    return func(*args)
  
  def join(self):
    pass

def step(x):
  env, act = x
  return env.step(act)

def reset(env):
  return env.reset()

class Environment(ConfiguredModule):
  @property
  def _param_default(self):
      return dict(time_limit=1000, parallel_execute = False)

  def __init__(self, callbacks, **kwargs):
    '''
    args:
      task: name of the environment
      num_envs: number of envs run parallelly
      time_limit: the allowed time for an environment to responde
    '''
    super().__init__(param_prefix='_', **kwargs)
    self.count_episode = 0
    self.count_steps = 0
    self.done = np.zeros((self._num_envs))
    self.env_steps = np.zeros(self._num_envs)

    if self._parallel_execute: self.pool = mp.Pool(mp.cpu_count())
    #self.pool = ShadowPool()

    self.envs = [self.__create_env(callbacks) for _ in range(self._num_envs)]
    self.obs = self.apply_reset(range(self._num_envs))

  def __create_env(self, callbacks):
    x = self._task.split('_', 1)
    if len(x) == 1: suite = x[0]
    elif len(x) == 2: suite, task = x
    else: raise NotImplementedError(self._task)

    if suite == 'prolog':
      from .ProLog import ProLog
      env = self.configure(ProLog)
    elif suite == 'dummy':
      from .Dummy import DummyEnv
      env = DummyEnv()
    else:
      raise NotImplementedError(suite)

    if self._parallel_execute:
      env = MultiProcessing(env)
    env = TimeLimit(env, self._time_limit)
    env = CollectDataset(env, callbacks)
    env = RewardObs(env)
    return env

  def get_obs(self):
    return self.obs

  def apply_actions(self, actions):
    if self._parallel_execute:
      #[self.pool.apply(e.async_step, args=(a,)) for e, a in zip(self.envs, actions)]
      self.pool.map(step, zip(self.envs, actions))
      self.pool.join()
    return [e.step(a) for e, a in zip(self.envs, actions)]

  def apply_reset(self, indices):
    if self._parallel_execute:
      #[self.pool.apply(self.envs[i].async_reset) for i in indices]
      self.pool.map(reset, [self.envs[i] for i in indices])
      self.pool.join()
    return [self.envs[i].reset() for i in indices]

  def step(self, actions):
    results = self.apply_actions(actions)
    obs, _, done = zip(*[p[:3] for p in results])
    obs = list(obs)
    done = np.stack(done)

    # update meta information
    self.count_episode += int(done.sum())
    self.env_steps += 1
    self.count_steps += (done * self.env_steps).sum()
    self.env_steps *= (1 - done)

    # reset env if it reached end
    if done.any():
      indices = [index for index, d in enumerate(done) if d]
      results = self.apply_reset(indices)
      for index, result in zip(indices, results):
        obs[index] = result

  def __dell__(self):
    self.pool.close()
