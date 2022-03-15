import os
import pathlib
import sys
import warnings
import argparse
import functools

import collections

#import tensorflow as tf
#from tensorflow.keras.mixed_precision import experimental as prec
#from tensorflow.python.util import tf_decorator

#warnings.filterwarnings('ignore', '.*box bound precision lowered.*')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['MUJOCO_GL'] = 'egl'

import tools
#from envs import make_env
#from tools import make_dataset

sys.path.append(str(pathlib.Path(__file__).parent))

#import numpy as np
import ruamel.yaml as yaml

#tf.get_logger().setLevel('ERROR')

class LolArg:
  def __init__(self):
    self.configs=['defaults','prolog','prolog_easy','debug']
    self.logdir='debugEpisodes'#'logdir'#

def init_config():
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
        self.configs=['prolog']#['defaults','prolog','prolog_easy','debug']
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
  #print(config)
  return config, args.logdir

if __name__ == '__main__':
  config, logdir= init_config()
  #print(config)
  from controller import Controller
  ctrl=Controller(params=config, logdir=logdir)
  #ctrl.simulate()
  #ctrl.train_only_worldModel(2222)
  #ctrl.train_only(10000)
  """from evalute import Judge
  judge=Judge(ctrl.agent)
  for _ in range(10):
    episode=next(ctrl.datasetManager)
    judge.simulate_trajectory(0, episode)"""
  from models_new import WorldModel

  batch_size = 2
  wm = WorldModel(0, ctrl._config)
  shape = (ctrl._config.batch_size, ctrl._config.state_length, ctrl._config.goal_length)
  ds = iter(ctrl.datasetManager.dataset(*shape))
  
  # remove
  import tensorflow as tf
  #train = wm.train
  train = tf.function(wm.train, input_signature = [ctrl.datasetManager.signature(*shape)])
  _metrics = collections.defaultdict(tf.metrics.Mean)
  _should_log = tools.Every(64)
  _logger = ctrl._logger
  ctrl.datasetManager.logging()
  for i in range(2**15):
      data = next(ds)
      metrics = train(data)
      for name, value in metrics.items():
        _metrics[name].update_state(value)
      if _should_log(i):
        for name, mean in _metrics.items():
          _logger.scalar(name, float(mean.result()))
          mean.reset_states()
        _logger.write()
    
    