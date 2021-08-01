import os
import pathlib
import sys
import warnings
import argparse
import functools

import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as prec
from tensorflow.python.util import tf_decorator

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

def load(logdir, config):
  logdir = pathlib.Path(logdir).expanduser()
  config.traindir = config.traindir or logdir / 'train_eps'
  config.evaldir = config.evaldir or logdir / 'eval_eps'
  config.act = getattr(tf.nn, config.act)

  if config.debug and False:
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
  logger = tools.Logger(logdir, step)

  return config, logger

def main(logdir, config):
  logdir = pathlib.Path(logdir).expanduser()
  config.traindir = config.traindir or logdir / 'train_eps'
  config.evaldir = config.evaldir or logdir / 'eval_eps'
  """config.steps //= config.action_repeat
  config.eval_every //= config.action_repeat
  config.log_every //= config.action_repeat
  config.time_limit //= config.action_repeat"""
  config.act = getattr(tf.nn, config.act)
  
  if config.debug and False:
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

  sample_lambda=0.
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
  #tools.simulate(random_agent, eval_envs, episodes=1)
  logger.step = config.action_repeat * count_steps(config.traindir)

  print('Simulate agent.')
  output_sign=train_envs[0].output_sign

  from dataset import DatasetManager
  from controller import Controller
  ctrl=Controller(output_sign)
  dm=DatasetManager(logger, ctrl.get_signature, config.traindir, config.evaldir)

  #train_dataset = make_dataset(train_eps, config, output_sign)
  #eval_dataset = iter(make_dataset(eval_eps, config, output_sign))
  x=next(iter(dm.sample_episode('train', 8, 2)))
  osign=ctrl.get_signature(8,2)
  def test(inp):
    for k in osign.keys():
      if isinstance(osign[k], dict):
        for gk in osign[k].keys():
          print(k, gk, ':\n\t', inp[k][gk].shape, '\t\t', osign[k][gk].shape,'\t', inp[k][gk].dtype,'\t', osign[k][gk].dtype)
      else: print(k, ':\n\t', inp[k].shape, '\t\t', osign[k].shape,'\t', inp[k].dtype,'\t', osign[k].dtype)

  train_dataset=dm.dataset(batch_length=2, batch_size=8)
  eval_dataset=iter(dm.dataset(batch_length=2, batch_size=8))

  """agent = Dreamer(config, logger, train_dataset)
  x=next(iter(train_dataset))
  test(x)

  from gnn.graph_input import feed_gnn_input
  
  gnn=agent._wm.encoder.encoders['gnn']

  @tf.function(input_signature=[osign])
  def fun(x):
    print('Tracing fun.')
    y=feed_gnn_input(x['gnn'],8 ,2, gnn)
    tf.print(y)
    return y #tf.reduce_sum(y, axis=-1)
  
  for _ in range(10):
    x=next(agent._dataset)
    #fun(x)"""



  agent = Dreamer(config, logger, train_dataset)
  if (logdir / 'variables.pkl').exists():
    agent.load(logdir / 'variables.pkl')
    agent._should_pretrain._once = False
  
  for _ in range(10000):
    mets=agent._wm.train(next(agent._dataset))[4]
    if(_%100==0): print(mets)
  #debugging
  from methods import Reconstructor
  ReC=Reconstructor(agent._wm, 0)
  for _ in range(40):
    ReC.train(agent._dataset,500)
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
    self.logdir='debugEpisodes'#'logdir'#

if __name__ == '__main__':
  try:
    print('Running the DreamProlog algorithm.')
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', required=True)
    parser.add_argument('--logdir', required=True)
    args, remaining = parser.parse_known_args()
  except:
    print("There was a problem with the provided arguments. The program will run in the default setting:")
    class DefaultArg:
      def __init__(self):
        self.configs=['defaults','prolog','prolog_easy','debug']
        self.logdir='logdir'#'debugEpisodes'#
    args, remaining =DefaultArg(), []
  print('--configs', " ".join(args.configs), '--logdir', args.logdir)

  configs = yaml.safe_load(
      (pathlib.Path(__file__).parent / 'configs.yaml').read_text())
  config_ = {}
  for name in args.configs:
    config_.update(configs[name])
  parser = argparse.ArgumentParser()
  for key, value in config_.items():
    arg_type = tools.args_type(value)
    parser.add_argument(f'--{key}', type=arg_type, default=arg_type(value))
  #main(args.logdir, parser.parse_args(remaining))
  config=parser.parse_args(remaining)
  from controller import Controller
  ctrl=Controller(config, args.logdir)
  #ctrl.simulate()
  ctrl.train_only_wordModel(1000)