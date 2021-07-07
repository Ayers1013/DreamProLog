import os
import pathlib
import sys
import warnings
import argparse
import functools

import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as prec

warnings.filterwarnings('ignore', '.*box bound precision lowered.*')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['MUJOCO_GL'] = 'egl'

import tools
from envs import make_env
from tools import make_dataset

sys.path.append(str(pathlib.Path(__file__).parent))

import numpy as np
import ruamel.yaml as yaml

tf.get_logger().setLevel('ERROR')

from dreamer import Dreamer, count_steps

def main(logdir, config):
  logdir = pathlib.Path(logdir).expanduser()
  config.traindir = config.traindir or logdir / 'train_eps'
  config.evaldir = config.evaldir or logdir / 'eval_eps'
  config.steps //= config.action_repeat
  config.eval_every //= config.action_repeat
  config.log_every //= config.action_repeat
  config.time_limit //= config.action_repeat
  config.act = getattr(tf.nn, config.act)
  
  if config.debug:
    tf.config.experimental_run_functions_eagerly(True)
    #tf.debugging.experimental.enable_dump_debug_info(str(logdir), tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)
  if config.gpu_growth:
    message = 'No GPU found. To actually train on CPU remove this assert.'
    #print(message)
    #TODO
    #assert tf.config.experimental.list_physical_devices('GPU'), message
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
      tf.config.experimental.set_memory_growth(gpu, True)
  assert config.precision in (16, 32), config.precision
  if config.precision == 16:
    prec.set_policy(prec.Policy('mixed_float16'))
  print('Logdir', logdir)
  logdir.mkdir(parents=True, exist_ok=True)
  config.traindir.mkdir(parents=True, exist_ok=True)
  config.evaldir.mkdir(parents=True, exist_ok=True)
  step = count_steps(config.traindir)
  logger = tools.Logger(logdir, config.action_repeat * step)

  print('Create envs.')
  if config.offline_traindir:
    directory = config.offline_traindir.format(**vars(config))
  else:
    directory = config.traindir
  train_eps = tools.load_episodes(directory, limit=config.dataset_size)
  if config.offline_evaldir:
    directory = config.offline_evaldir.format(**vars(config))
  else:
    directory = config.evaldir
  eval_eps = tools.load_episodes(directory, limit=1)
  make = lambda mode: make_env(config, logger, mode, train_eps, eval_eps)
  train_envs = [make('train') for _ in range(config.envs)]
  eval_envs = [make('eval') for _ in range(config.envs)]
  #acts = train_envs[0].action_space
  #config.num_actions = acts.n if hasattr(acts, 'n') else acts.shape[0]

  prefill = max(0, config.prefill - count_steps(config.traindir))
  print(f'Prefill dataset ({prefill} steps).')

  #True Random
  def sample():
    act_size=train_envs[0].action_space_size
    arr=np.zeros(act_size)
    arr[np.random.randint(act_size)]=1.0
    # arr[np.random.randint(4)]=1.0
    #BUG#002
    return arr

  sample_lambda=0.07
  def sample_smart(o):
    if(np.random.rand()<sample_lambda): return sample()
    axiom_mask=o['axiom_mask'][0]
    act_size=len(axiom_mask)
    s=int(np.sum(axiom_mask))
    r=np.random.randint(s)+1 if s>0 else 0

    ind=0
    for e in axiom_mask:
      if(e==1): r-=1
      if(r==0): break
      ind+=1

    arr=np.zeros(act_size)
    arr[ind]=1.0
    return arr
  
  random_agent = lambda o, d, s: ([sample_smart(o) for _ in d], s)
  tools.simulate(random_agent, train_envs, prefill)
  tools.simulate(random_agent, eval_envs, episodes=1)
  logger.step = config.action_repeat * count_steps(config.traindir)

  print('Simulate agent.')
  output_sign=train_envs[0].output_sign

  train_dataset = make_dataset(train_eps, config, output_sign)
  eval_dataset = iter(make_dataset(eval_eps, config, output_sign))
  agent = Dreamer(config, logger, train_dataset)
  if (logdir / 'variables.pkl').exists():
    agent.load(logdir / 'variables.pkl')
    agent._should_pretrain._once = False
  
  #debugging
  from methods import Reconstructor
  ReC=Reconstructor(agent._wm, 0)
  for _ in range(5):
    ReC.train(agent._dataset,200)
    agent.save(logdir / 'variables.pkl')
  #ReC.tracker.summary()


  state = None
  while agent._step.numpy().item() < config.steps:
    logger.write()
    print('Start evaluation.')
    #video_pred = agent._wm.video_pred(next(eval_dataset))
    #logger.video('eval_openl', video_pred)
    eval_policy = functools.partial(agent, training=False)
    tools.simulate(eval_policy, eval_envs, episodes=1)
    print('Start training.')
    state = tools.simulate(agent, train_envs, config.eval_every, state=state)
    agent.save(logdir / 'variables.pkl')
  for env in train_envs + eval_envs:
    try:
      env.close()
    except Exception:
      pass


class LolArg:
  def __init__(self):
    self.configs=['defaults','prolog','prolog_easy','debug']
    self.logdir='logdir'

if __name__ == '__main__':
  if(False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', required=True)
    parser.add_argument('--logdir', required=True)
    args, remaining = parser.parse_known_args()
  else:
    #TODO remove this!!
    args, remaining =LolArg(), []
    #TODO ...
  print(args,remaining)

  configs = yaml.safe_load(
      (pathlib.Path(__file__).parent / 'configs.yaml').read_text())
  config_ = {}
  for name in args.configs:
    config_.update(configs[name])
  parser = argparse.ArgumentParser()
  for key, value in config_.items():
    arg_type = tools.args_type(value)
    parser.add_argument(f'--{key}', type=arg_type, default=arg_type(value))
  main(args.logdir, parser.parse_args(remaining))
