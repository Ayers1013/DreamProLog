from .wrappers import *
#from .process_data import process_episode
import functools


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
