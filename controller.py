import pathlib
import functools
from pyswip.core import Sread_function

from tensorflow.keras.mixed_precision import experimental as prec

import tensorflow as tf
import numpy as np

from dataset import DatasetManager
from envs import make_env
import tools
from dreamer import Dreamer, count_steps
from misc import ConfiguredModule

class Controller(ConfiguredModule):
  def __init__(self, **kwargs):
    super().__init__(param_prefix='_', **kwargs)
    self.logdir = pathlib.Path(self._logdir).expanduser()
    self.logger = self.preinit()

    self.datasetManager = DatasetManager(self.logger, self.get_signature, self._traindir, self._evaldir)
    
    # TODO clear
    class Namespace:
      def __init__(self, **kwargs):
          self.__dict__.update(kwargs)
    make = lambda mode: make_env(self, self.datasetManager.get_callbacks(mode, Namespace(traindir=self._traindir, evaldir=self._evaldir)))
    self.train_envs = [make('train') for _ in range(self._envs)]
    self.eval_envs = [make('eval') for _ in range(self._envs)]

    self._signature=self.train_envs[0].output_sign

    self.prefill_dataset()

  @property
  def episodes(self):
    return self.datasetManager._train_eps._episodes

  def get_signature(self, batch_size, batch_length):
    _shape=(batch_size, batch_length) if batch_size!=0 else (batch_length,)
    spec=lambda x, dt: tf.TensorSpec(shape=_shape+x, dtype=dt)

    sign=self._signature(batch_size, batch_length)
    sign.update({
      'action': spec((), tf.int32),
      'reward': spec((), tf.float32),
      'discount': spec((), tf.float32)
    })

    return sign

  def preinit(self):
    logdir = pathlib.Path(self._logdir).expanduser()
    self._traindir = self._traindir or logdir / 'train_eps'
    self._evaldir = self._evaldir or logdir / 'eval_eps'

    if self._debug:
      tf.config.experimental_run_functions_eagerly(True)
      tf.debugging.experimental.enable_dump_debug_info(str(logdir), tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)
    if True: #config.gpu_growth:
      message = 'No GPU found. To actually train on CPU remove this assert.'
      #print(message)
      #TODO
      #assert tf.config.experimental.list_physical_devices('GPU'), message
      for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    assert self._precision in (16, 32), self._precision
    if self._precision == 16:
      prec.set_policy(prec.Policy('mixed_float16'))

    print('Logdir', logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    self._traindir.mkdir(parents=True, exist_ok=True)
    self._evaldir.mkdir(parents=True, exist_ok=True)
    step = count_steps(self._traindir)
    if self._logger == 'wandb':
      logger = tools.LoggerWandb(logdir, step, self) 
    elif self._logger == 'empty':
      logger = tools.LoggerEmpty(logdir, step, self) 

    return logger

  def prefill_dataset(self):
    prefill = max(0, self._prefill - count_steps(self._traindir))
    print(f'Prefill dataset ({prefill} steps).')
    def sample(act_size):
      arr=np.zeros(act_size)
      arr[np.random.randint(act_size)]=1.0
      # arr[np.random.randint(4)]=1.0
      #BUG#002
      return arr

    def sample_smart(o):
      if False:
        mask=o['axiom_mask'][0]
        act_size=len(mask)
        s=int(np.sum(mask))
        r=np.random.randint(s)+1 if s>0 else 0

        ind=0
        for e in axiom_mask:
          if(e==1): r-=1
          if(r==0): break
          ind+=1
      else:
        mask=o['action_mask'][0]
        mask_x = np.where(mask>=0, 1, 0)
        ind = np.random.choice(np.arange(len(mask)), p=mask_x/np.sum(mask_x))
      return ind
    
    random_agent = lambda o, d, s: ([sample_smart(o) for _ in d], s)
    tools.simulate(random_agent, self.train_envs, prefill)
    tools.simulate(random_agent, self.eval_envs, episodes=1)
    self.logger.step = count_steps(self._traindir)

  def simulate(self):
    if (self.logdir / 'variables.pkl').exists():
      self.agent.load(self.logdir / 'variables.pkl')
      self.agent._should_pretrain._once = False
    state = None
    while self.agent._step.numpy().item() < self._config.steps:
      self.datasetManager.logging()
      self.logger.write()
      print('Start evaluation.')
      eval_policy = functools.partial(self.agent, training=False)
      tools.simulate(eval_policy, self.eval_envs, episodes=1)
      print('Start training.')
      state = tools.simulate(self.agent, self.train_envs, self._config.eval_every, state=state)
      self.agent.save(self.logdir / 'variables.pkl')

  def __del__(self):
    for env in self.train_envs + self.eval_envs:
      try:
        env.close()
      except Exception:
        pass

  def train_only_worldModel(self, epochs=1, save=False):
    ds=self.agent._dataset
    self.datasetManager.logging()
    print('Train only run.')
    for step in range(1, epochs+1):
      x=next(ds)
      self.agent._train_only_worldModel(x)
      if step%10==0:
        for name, mean in self.agent._metrics.items():
          self.logger.scalar(name, float(mean.result()))
          mean.reset_states()
        self.logger.step+=1
        self.logger.write()
      if save and step%500==0:
        self.agent.save(self.logdir / 'variables.pkl')

  def train_only(self, epochs=1):
    ds=self.agent._dataset
    print('Train only run.')
    for step in range(epochs):
      x=next(ds)
      self.agent._train(x)
      if step%10==0:
        for name, mean in self.agent._metrics.items():
          self.logger.scalar(name, float(mean.result()))
          mean.reset_states()
        self.logger.step+=1
        self.logger.write()
      if step%500==0:
        self.agent.save(self.logdir / 'variables.pkl')

