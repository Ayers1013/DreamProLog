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

class Controller:
  def __init__(self, config, logdir):
    self._logdir = pathlib.Path(logdir).expanduser()

    self._config=config
    self._logger=self.preinit(self._logdir, config)

    self.datasetManager=DatasetManager(self._logger, self.get_signature, config.traindir, config.evaldir)

    #self.datasetManager._scheduled=('m2n140t8_pre_topc', 'small')
    #generator = next(iter(self.datasetManager.sample_episode(
    #  'train', 8, 2, True)))

    #x=generator()
    
    make = lambda mode: make_env(config, self.datasetManager.get_callbacks(mode, config))
    self.train_envs = [make('train') for _ in range(config.envs)]
    self.eval_envs = [make('eval') for _ in range(config.envs)]

    self._signature=self.train_envs[0].output_sign

    #expected length= 1/p, imidiate fail chance= p
    sample_rate=lambda: [0.0, 0.02, 0.1][np.random.randint(3)]
    self.prefill(sample_rate=sample_rate)
    
    '''try:
      while isinstance(x, dict):
        key=next(iter(x.keys()))
        print(key, type(key))
        x=x[key]
    except:
      print('The train_eps is empty.')'''

    #self.agent=Dreamer(config, self._logger, self.datasetManager.dataset(batch_length=2, batch_size=8))
    self.agent=Dreamer(config, self._logger, self.datasetManager, self.get_signature)
    if (self._logdir / 'variables.pkl').exists():
      self.agent.load(self._logdir / 'variables.pkl')
      self.agent._should_pretrain._once = False

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

  def preinit(self, logdir, config):
    #logdir = pathlib.Path(logdir).expanduser()
    config.traindir = config.traindir or logdir / 'train_eps'
    config.evaldir = config.evaldir or logdir / 'eval_eps'
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
    logger = tools.LoggerWandb(logdir, step, config)

    return logger

  def prefill(self, sample_rate=lambda :0.0):
    prefill = max(0, self._config.prefill - count_steps(self._config.traindir))
    print(f'Prefill dataset ({prefill} steps).')
    def sample(act_size):
      arr=np.zeros(act_size)
      arr[np.random.randint(act_size)]=1.0
      # arr[np.random.randint(4)]=1.0
      #BUG#002
      return arr

    def sample_smart(o):
      axiom_mask=o['axiom_mask'][0]
      act_size=len(axiom_mask)
      if(np.random.rand()<sample_rate()): return np.random.randint(act_size)
      s=int(np.sum(axiom_mask))
      r=np.random.randint(s)+1 if s>0 else 0

      ind=0
      for e in axiom_mask:
        if(e==1): r-=1
        if(r==0): break
        ind+=1

      #NOTE ind instead of arr
      '''arr=np.zeros(act_size)
      arr[ind]=1.0
      return arr'''
      return ind
    
    random_agent = lambda o, d, s: ([sample_smart(o) for _ in d], s)
    tools.simulate(random_agent, self.train_envs, prefill)
    tools.simulate(random_agent, self.eval_envs, episodes=1)
    self._logger.step = count_steps(self._config.traindir)

  def simulate(self):
    if (self._logdir / 'variables.pkl').exists():
      self.agent.load(self._logdir / 'variables.pkl')
      self.agent._should_pretrain._once = False
    state = None
    while self.agent._step.numpy().item() < self._config.steps:
      self.datasetManager.logging()
      self._logger.write()
      print('Start evaluation.')
      eval_policy = functools.partial(self.agent, training=False)
      tools.simulate(eval_policy, self.eval_envs, episodes=1)
      print('Start training.')
      state = tools.simulate(self.agent, self.train_envs, self._config.eval_every, state=state)
      self.agent.save(self._logdir / 'variables.pkl')

  def __del__(self):
    for env in self.train_envs + self.eval_envs:
      try:
        env.close()
      except Exception:
        pass

  def train_only_worldModel(self, epochs=1):
    ds=self.datasetManager.dataset(batch_length=2, batch_size=8)
    ds=iter(ds)
    for step in range(epochs):
      x=next(ds)
      self.agent._train_only_worldModel(x)
      if step%10==0:
        for name, mean in self.agent._metrics.items():
          self._logger.scalar(name, float(mean.result()))
          mean.reset_states()
        self._logger.step+=1
        self._logger.write(fps=False)
      if step%500==0:
        self.agent.save(self._logdir / 'variables.pkl')

  def train_only(self, epochs=1):
    ds=self.datasetManager.dataset(batch_length=2, batch_size=8)
    ds=iter(ds)
    print('Train only run.')
    for step in range(epochs):
      x=next(ds)
      self.agent._train(x)
      if step%10==0:
        for name, mean in self.agent._metrics.items():
          self._logger.scalar(name, float(mean.result()))
          mean.reset_states()
        self._logger.step+=1
        self._logger.write(fps=False)
      if step%500==0:
        self.agent.save(self._logdir / 'variables.pkl')

