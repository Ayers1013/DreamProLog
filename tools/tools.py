import datetime
import io
import json
import pathlib
import pickle
import re
import time
import uuid

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.mixed_precision import experimental as prec
from tensorflow_probability import distributions as tfd

import wandb

class AttrDict(dict):

  __setattr__ = dict.__setitem__
  __getattr__ = dict.__getitem__

class LoggerWandb:

  def __init__(self, logdir, step, config):
    wandb.init(project='DreamProLog', entity='ayers', config=vars(config))
    self._logdir = logdir
    self._last_step = None
    self._last_time = None
    self._scalars = {}
    self._images = {}
    self._videos = {}
    self.step = step

  def __del__(self):
    wandb.finish()

  def scalar(self, name, value):
    self._scalars[name] = float(value)

  def image(self, name, value):
    self._images[name] = np.array(value)

  def video(self, name, value):
    self._videos[name] = np.array(value)

  def write(self, fps=False):
    scalars = list(self._scalars.items())
    if fps:
      scalars.append(('fps', self._compute_fps(self.step)))
    print(f'[{self.step}]', ' / '.join(f'{k} {v:.1f}' for k, v in scalars))
    with (self._logdir / 'metrics.jsonl').open('a') as f:
      f.write(json.dumps({'step': self.step, ** dict(scalars)}) + '\n')
    wandb.log(self._scalars)#, step=self.step)
    self._scalars = {}
    self._images = {}
    self._videos = {}

  

class Logger:

  def __init__(self, logdir, step):
    self._logdir = logdir
    self._writer = tf.summary.create_file_writer(str(logdir), max_queue=1000)
    self._last_step = None
    self._last_time = None
    self._scalars = {}
    self._images = {}
    self._videos = {}
    self.step = step

  def scalar(self, name, value):
    self._scalars[name] = float(value)

  def image(self, name, value):
    self._images[name] = np.array(value)

  def video(self, name, value):
    self._videos[name] = np.array(value)

  def write(self, fps=False):
    scalars = list(self._scalars.items())
    if fps:
      scalars.append(('fps', self._compute_fps(self.step)))
    print(f'[{self.step}]', ' / '.join(f'{k} {v:.1f}' for k, v in scalars))
    with (self._logdir / 'metrics.jsonl').open('a') as f:
      f.write(json.dumps({'step': self.step, ** dict(scalars)}) + '\n')
    with self._writer.as_default():
      for name, value in scalars:
        tf.summary.scalar('scalars/' + name, value, self.step)
      for name, value in self._images.items():
        tf.summary.image(name, value, self.step)
      #for name, value in self._videos.items():
      #  video_summary(name, value, self.step)
    self._writer.flush()
    self._scalars = {}
    self._images = {}
    self._videos = {}

  def _compute_fps(self, step):
    if self._last_step is None:
      self._last_time = time.time()
      self._last_step = step
      return 0
    steps = step - self._last_step
    duration = time.time() - self._last_time
    self._last_time += duration
    self._last_step = step
    return steps / duration

def var_nest_names(nest):
  if isinstance(nest, dict):
    items = ' '.join(f'{k}:{var_nest_names(v)}' for k, v in nest.items())
    return '{' + items + '}'
  if isinstance(nest, (list, tuple)):
    items = ' '.join(var_nest_names(v) for v in nest)
    return '[' + items + ']'
  if hasattr(nest, 'name') and hasattr(nest, 'shape'):
    return nest.name + str(nest.shape).replace(', ', 'x')
  if hasattr(nest, 'shape'):
    return str(nest.shape).replace(', ', 'x')
  return '?'

def graph_summary(writer, step, fn, *args):
  def inner(*args):
    tf.summary.experimental.set_step(step.numpy().item())
    with writer.as_default():
      fn(*args)
  return tf.numpy_function(inner, args, [])

def simulate(agent, envs, steps=0, episodes=0, state=None):
  # Initialize or unpack simulation state.
  if state is None:
    step, episode = 0, 0
    done = np.ones(len(envs), np.bool)
    length = np.zeros(len(envs), np.int32)
    obs = [None] * len(envs)
    agent_state = None
  else:
    step, episode, done, length, obs, agent_state = state
  while (steps and step < steps) or (episodes and episode < episodes):
    # Reset envs if necessary.
    if done.any():
      indices = [index for index, d in enumerate(done) if d]
      results = [envs[i].reset() for i in indices]
      for index, result in zip(indices, results):
        obs[index] = result
    # Step agents.

    #NOTE do sth with it, it's ugly
    _obs=None
    if('gnn' in obs[0]):
      _obs={k: np.stack([o['gnn'][k] for o in obs]) for k in obs[0]['gnn']}
      _obs.update({
        'num_nodes':np.array([[len(e['gnn']['ini_nodes'])] for e in obs]),
        'num_symbols':np.array([[len(e['gnn']['ini_symbols'])] for e in obs]),
        'num_clauses':np.array([[len(e['gnn']['ini_clauses'])] for e in obs] )
      })
    obs = {k: np.stack([o[k] for o in obs]) for k in obs[0] if k!='gnn'}
    if _obs!=None:
      obs['gnn']=_obs
    action, agent_state = agent(obs, done, agent_state)
    if isinstance(action, dict):
      action = [
          {k: np.array(action[k][i]) for k in action}
          for i in range(len(envs))]
    else:
      #NOTE This caused a BUG#001
      action = np.array(action)
      #pass
    assert len(action) == len(envs)
    # Step envs.

    results = [e.step(a) for e, a in zip(envs, action)]
    obs, _, done = zip(*[p[:3] for p in results])
    obs = list(obs)
    done = np.stack(done)
    episode += int(done.sum())
    length += 1
    step += (done * length).sum()
    length *= (1 - done)
  # Return new state to allow resuming the simulation.
  return (step - steps, episode - episodes, done, length, obs, agent_state)

class DtypeDist:

  def __init__(self, dist, dtype=None):
    self._dist = dist
    self._dtype = dtype or prec.global_policy().compute_dtype

  @property
  def name(self):
    return 'DtypeDist'

  def __getattr__(self, name):
    return getattr(self._dist, name)

  def mean(self):
    return tf.cast(self._dist.mean(), self._dtype)

  def mode(self):
    return tf.cast(self._dist.mode(), self._dtype)

  def entropy(self):
    return tf.cast(self._dist.entropy(), self._dtype)

  def sample(self, *args, **kwargs):
    return tf.cast(self._dist.sample(*args, **kwargs), self._dtype)




def lambda_return(
    reward, value, pcont, bootstrap, lambda_, axis):
  # Setting lambda=1 gives a discounted Monte Carlo return.
  # Setting lambda=0 gives a fixed 1-step return.
  assert reward.shape.ndims == value.shape.ndims, (reward.shape, value.shape)
  if isinstance(pcont, (int, float)):
    pcont = pcont * tf.ones_like(reward)
  dims = list(range(reward.shape.ndims))
  dims = [axis] + dims[1:axis] + [0] + dims[axis + 1:]
  if axis != 0:
    reward = tf.transpose(reward, dims)
    value = tf.transpose(value, dims)
    pcont = tf.transpose(pcont, dims)
  if bootstrap is None:
    bootstrap = tf.zeros_like(value[-1])
  next_values = tf.concat([value[1:], bootstrap[None]], 0)
  inputs = reward + pcont * next_values * (1 - lambda_)
  returns = static_scan(
      lambda agg, cur: cur[0] + cur[1] * lambda_ * agg,
      (inputs, pcont), bootstrap, reverse=True)
  if axis != 0:
    returns = tf.transpose(returns, dims)
  return returns

def args_type(default):
  def parse_string(x):
    if default is None:
      return x
    if isinstance(default, bool):
      return bool(['False', 'True'].index(x))
    if isinstance(default, int):
      return float(x) if ('e' in x or '.' in x) else int(x)
    if isinstance(default, (list, tuple)):
      return tuple(args_type(default[0])(y) for y in x.split(','))
    return type(default)(x)
  def parse_object(x):
    if isinstance(default, (list, tuple)):
      return tuple(x)
    return x
  return lambda x: parse_string(x) if isinstance(x, str) else parse_object(x)


def static_scan(fn, inputs, start, reverse=False):
  last = start
  outputs = [[] for _ in tf.nest.flatten(start)]
  indices = range(len(tf.nest.flatten(inputs)[0]))
  if reverse:
    indices = reversed(indices)
  for index in indices:
    inp = tf.nest.map_structure(lambda x: x[index], inputs)
    last = fn(last, inp)
    [o.append(l) for o, l in zip(outputs, tf.nest.flatten(last))]
  if reverse:
    outputs = [list(reversed(x)) for x in outputs]
  outputs = [tf.stack(x, 0) for x in outputs]
  return tf.nest.pack_sequence_as(start, outputs)

def static_scan_rssm(fn, inputs, start, reverse=False):
  last = start
  outputs = [[] for _ in tf.nest.flatten(start)]
  max_ind=25
  indices = range(max_ind)#range(len(tf.nest.flatten(inputs)[0]))
  last_ind=len(tf.nest.flatten(inputs)[0])
  if reverse:
    indices = reversed(indices)
  for index in indices:
    if index<last_ind:
      inp = tf.nest.map_structure(lambda x: x[index], inputs)
      last = fn(last, inp)
      [o.append(l) for o, l in zip(outputs, tf.nest.flatten(last))]
    else:
      [o.append(tf.zeros(o[0].shape)) for o in outputs]
  if reverse:
    outputs = [list(reversed(x)) for x in outputs]
  outputs = [tf.stack(x, 0) for x in outputs]
  outputs = [tf.gather(o, range(last_ind)) for o in outputs]
  return tf.nest.pack_sequence_as(start, outputs)


def uniform_mixture(dist, dtype=None):
  if dist.batch_shape[-1] == 1:
    return tfd.BatchReshape(dist, dist.batch_shape[:-1])
  dtype = dtype or prec.global_policy().compute_dtype
  weights = tfd.Categorical(tf.zeros(dist.batch_shape, dtype))
  return tfd.MixtureSameFamily(weights, dist)


def cat_mixture_entropy(dist):
  if isinstance(dist, tfd.MixtureSameFamily):
    probs = dist.components_distribution.probs_parameter()
  else:
    probs = dist.probs_parameter()
  return -tf.reduce_mean(
      tf.reduce_mean(probs, 2) *
      tf.math.log(tf.reduce_mean(probs, 2) + 1e-8), -1)


@tf.function
def cem_planner(
    state, num_actions, horizon, proposals, topk, iterations, imagine,
    objective):
  dtype = prec.global_policy().compute_dtype
  B, P = list(state.values())[0].shape[0], proposals
  H, A = horizon, num_actions
  flat_state = {k: tf.repeat(v, P, 0) for k, v in state.items()}
  mean = tf.zeros((B, H, A), dtype)
  std = tf.ones((B, H, A), dtype)
  for _ in range(iterations):
    proposals = tf.random.normal((B, P, H, A), dtype=dtype)
    proposals = proposals * std[:, None] + mean[:, None]
    proposals = tf.clip_by_value(proposals, -1, 1)
    flat_proposals = tf.reshape(proposals, (B * P, H, A))
    states = imagine(flat_proposals, flat_state)
    scores = objective(states)
    scores = tf.reshape(tf.reduce_sum(scores, -1), (B, P))
    _, indices = tf.math.top_k(scores, topk, sorted=False)
    best = tf.gather(proposals, indices, axis=1, batch_dims=1)
    mean, var = tf.nn.moments(best, 1)
    std = tf.sqrt(var + 1e-6)
  return mean[:, 0, :]


@tf.function
def grad_planner(
    state, num_actions, horizon, proposals, iterations, imagine, objective,
    kl_scale, step_size):
  dtype = prec.global_policy().compute_dtype
  B, P = list(state.values())[0].shape[0], proposals
  H, A = horizon, num_actions
  flat_state = {k: tf.repeat(v, P, 0) for k, v in state.items()}
  mean = tf.zeros((B, H, A), dtype)
  rawstd = 0.54 * tf.ones((B, H, A), dtype)
  for _ in range(iterations):
    proposals = tf.random.normal((B, P, H, A), dtype=dtype)
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(mean)
      tape.watch(rawstd)
      std = tf.nn.softplus(rawstd)
      proposals = proposals * std[:, None] + mean[:, None]
      proposals = (
          tf.stop_gradient(tf.clip_by_value(proposals, -1, 1)) +
          proposals - tf.stop_gradient(proposals))
      flat_proposals = tf.reshape(proposals, (B * P, H, A))
      states = imagine(flat_proposals, flat_state)
      scores = objective(states)
      scores = tf.reshape(tf.reduce_sum(scores, -1), (B, P))
      div = tfd.kl_divergence(
          tfd.Normal(mean, std),
          tfd.Normal(tf.zeros_like(mean), tf.ones_like(std)))
      elbo = tf.reduce_sum(scores) - kl_scale * div
      elbo /= tf.cast(tf.reduce_prod(tf.shape(scores)), dtype)
    grad_mean, grad_rawstd = tape.gradient(elbo, [mean, rawstd])
    e, v = tf.nn.moments(grad_mean, [1, 2], keepdims=True)
    grad_mean /= tf.sqrt(e * e + v + 1e-4)
    e, v = tf.nn.moments(grad_rawstd, [1, 2], keepdims=True)
    grad_rawstd /= tf.sqrt(e * e + v + 1e-4)
    mean = tf.clip_by_value(mean + step_size * grad_mean, -1, 1)
    rawstd = rawstd + step_size * grad_rawstd
  return mean[:, 0, :]


class Every:

  def __init__(self, every):
    self._every = every
    self._last = None

  def __call__(self, step):
    if not self._every:
      return False
    if self._last is None:
      self._last = step
      return True
    if step >= self._last + self._every:
      self._last += self._every
      return True
    return False


class Once:

  def __init__(self):
    self._once = True

  def __call__(self):
    if self._once:
      self._once = False
      return True
    return False


class Until:

  def __init__(self, until):
    self._until = until

  def __call__(self, step):
    if not self._until:
      return True
    return step < self._until


def schedule(string, step):
  try:
    return float(string)
  except ValueError:
    step = tf.cast(step, tf.float32)
    match = re.match(r'linear\((.+),(.+),(.+)\)', string)
    if match:
      initial, final, duration = [float(group) for group in match.groups()]
      mix = tf.clip_by_value(step / duration, 0, 1)
      return (1 - mix) * initial + mix * final
    match = re.match(r'warmup\((.+),(.+)\)', string)
    if match:
      warmup, value = [float(group) for group in match.groups()]
      scale = tf.clip_by_value(step / warmup, 0, 1)
      return scale * value
    match = re.match(r'exp\((.+),(.+),(.+)\)', string)
    if match:
      initial, final, halflife = [float(group) for group in match.groups()]
      return (initial - final) * 0.5 ** (step / halflife) + final
    raise NotImplementedError(string)
