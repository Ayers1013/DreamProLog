import numpy as np

from .wrappers import *
from misc import ConfiguredModule


def make_env(config, callbacks):
  suite, task = config.task.split('_', 1)
  print("Suite: "+suite)
  
  if suite == 'prolog':
    from .ProLog import ProLog
    env=ProLog()
  elif suite == 'dummy':
    from .Dummy import DummyEnv
    env=DummyEnv()
  else:
    raise NotImplementedError(suite)
  env = wrappers.TimeLimit(env, config.time_limit)
  env = wrappers.CollectDataset(env, callbacks)
  env = wrappers.RewardObs(env)
  return env

class Environment(ConfiguredModule):
  def __init__(self, callbacks, **kwargs):
    '''
    args:
      task: name of the environment
      num_envs: number of envs run parallelly
      time_limit: the allowed time for an environment to responde
    '''
    self.count_episode = 0
    self.count_steps = 0
    self.env_steps = np.zeros(self.num_envs)

    self.envs = [self.__create_env(callbacks) for _ in range(self.num_envs)]

  def __create_env(self, callbacks):
    suite, task = self.task.split('_', 1)
    if suite == 'prolog':
      from .ProLog import ProLog
      env = self.confiugre(ProLog)
    elif suite == 'dummy':
      from .Dummy import DummyEnv
      env = DummyEnv()
    else:
      raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, self.time_limit)
    env = wrappers.CollectDataset(env, callbacks)
    env = wrappers.RewardObs(env)
    return env

  def step(self, actions):
    results = [e.step(a) for e, a in zip(self.envs, actions)]
    obs, _, done = zip(*[p[:3] for p in results])
    obs = list(obs)
    done = np.stack(done)
    self.episode_count += int(done.sum())
    self.env_steps += 1
    self.count_steps += (done * self.env_steps).sum()
    self.env_steps *= (1 - done)