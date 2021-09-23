import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfkl
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision import experimental as prec

import tools


class RSSM(tools.Module):

  def __init__(
      self, stoch=30, deter=200, hidden=200, layers_input=1, layers_output=1,
      shared=False, discrete=False, act=tf.nn.elu, mean_act='none',
      std_act='softplus', min_std=0.1, cell='keras'):
    super().__init__()
    self._stoch = stoch
    self._deter = deter
    self._hidden = hidden
    self._min_std = min_std
    self._layers_input = layers_input
    self._layers_output = layers_output
    self._shared = shared
    self._discrete = discrete
    self._act = act
    self._mean_act = mean_act
    self._std_act = std_act
    self._embed = None
    if cell == 'gru':
      self._cell = tfkl.GRUCell(self._deter)
    elif cell == 'gru_layer_norm':
      self._cell = GRUCell(self._deter, norm=True)
    else:
      raise NotImplementedError(cell)

    #variable
    self.action_embed=None

  def feed_action_embed(self, x):
    self.action_embed=x

  def action_to_embed(self, action):
    return tf.gather(self.action_embed, action)


  def initial(self, batch_size):
    dtype = prec.global_policy().compute_dtype
    if self._discrete:
      state = dict(
          logit=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
          stoch=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
          deter=self._cell.get_initial_state(None, batch_size, dtype))
    else:
      state = dict(
          mean=tf.zeros([batch_size, self._stoch], dtype),
          std=tf.zeros([batch_size, self._stoch], dtype),
          stoch=tf.zeros([batch_size, self._stoch], dtype),
          deter=self._cell.get_initial_state(None, batch_size, dtype))
    return state

  
  def observe(self, embed, action, state=None):
    action=self.action_to_embed(action)

    print('Tracing RSSM observe function.')
    swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(tf.shape(action)[0])
    embed, action = swap(embed), swap(action)
    #NOTE dummy static_scan
    post, prior = tools.static_scan(
        lambda prev, inputs: self.obs_step(prev[0], *inputs),
        (action, embed), (state, state))
    post = {k: swap(v) for k, v in post.items()}
    prior = {k: swap(v) for k, v in prior.items()}
    #print(post, prior)
    return post, prior

  
  def imagine(self, action, state=None):
    action=self.action_to_embed(action)

    swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(tf.shape(action)[0])
    assert isinstance(state, dict), state
    action = swap(action)
    prior = tools.static_scan(self.img_step, action, state)
    prior = {k: swap(v) for k, v in prior.items()}
    return prior

  def get_feat(self, state):
    stoch = state['stoch']
    if self._discrete:
      shape = stoch.shape[:-2] + [self._stoch * self._discrete]
      stoch = tf.reshape(stoch, shape)
    return tf.concat([stoch, state['deter']], -1)

  def get_dist(self, state, dtype=None):
    if self._discrete:
      logit = state['logit']
      logit = tf.cast(logit, tf.float32)
      dist = tfd.Independent(tools.OneHotDist(logit), 1)
      if dtype != tf.float32:
        dist = tools.DtypeDist(dist, dtype or state['logit'].dtype)
    else:
      mean, std = state['mean'], state['std']
      if dtype:
        mean = tf.cast(mean, dtype)
        std = tf.cast(std, dtype)
      dist = tfd.MultivariateNormalDiag(mean, std)
    return dist

  @tf.function
  def obs_step(self, prev_state, prev_action, embed, sample=True):
    if not self._embed:
      self._embed = embed.shape[-1]
    prior = self.img_step(prev_state, prev_action, None, sample)
    if self._shared:
      post = self.img_step(prev_state, prev_action, embed, sample)
    else:
      #This is an anitbug. (Undo an unlocated bug) 
      #embed=tf.reshape(tf.squeeze(embed),[-1,200])
      x = tf.concat([prior['deter'], embed], -1)
      for i in range(self._layers_output):
        x = self.get(f'obi{i}', tfkl.Dense, self._hidden, self._act)(x)
      stats = self._suff_stats_layer('obs', x)
      if sample:
        stoch = self.get_dist(stats).sample()
      else:
        stoch = self.get_dist(stats).mode()
      post = {'stoch': stoch, 'deter': prior['deter'], **stats}
    return post, prior

  @tf.function
  def img_step(self, prev_state, prev_action, embed=None, sample=True):
    prev_stoch = prev_state['stoch']
    if self._discrete:
      shape = prev_stoch.shape[:-2] + [self._stoch * self._discrete]
      prev_stoch = tf.reshape(prev_stoch, shape)
    if self._shared:
      if embed is None:
        shape = prev_action.shape[:-1] + [self._embed]
        embed = tf.zeros(shape, prev_action.dtype)
      x = tf.concat([prev_stoch, prev_action, embed], -1)
    else:
      x = tf.concat([prev_stoch, prev_action], -1)
    for i in range(self._layers_input):
      x = self.get(f'ini{i}', tfkl.Dense, self._hidden, self._act)(x)
    x, deter = self._cell(x, [prev_state['deter']])
    deter = deter[0]  # Keras wraps the state in a list.
    for i in range(self._layers_output):
      x = self.get(f'imo{i}', tfkl.Dense, self._hidden, self._act)(x)
    stats = self._suff_stats_layer('ims', x)
    if sample:
      stoch = self.get_dist(stats).sample()
    else:
      stoch = self.get_dist(stats).mode()
    prior = {'stoch': stoch, 'deter': deter, **stats}
    return prior

  def _suff_stats_layer(self, name, x):
    if self._discrete:
      x = self.get(name, tfkl.Dense, self._stoch * self._discrete, None)(x)
      logit = tf.reshape(x, x.shape[:-1] + [self._stoch, self._discrete])
      return {'logit': logit}
    else:
      x = self.get(name, tfkl.Dense, 2 * self._stoch, None)(x)
      mean, std = tf.split(x, 2, -1)
      mean = {
          'none': lambda: mean,
          'tanh5': lambda: 5.0 * tf.math.tanh(mean / 5.0),
      }[self._mean_act]()
      std = {
          'softplus': lambda: tf.nn.softplus(std),
          'abs': lambda: tf.math.abs(std + 1),
          'sigmoid': lambda: tf.nn.sigmoid(std),
          'sigmoid2': lambda: 2 * tf.nn.sigmoid(std / 2),
      }[self._std_act]()
      std = std + self._min_std
      return {'mean': mean, 'std': std}

  def kl_loss(self, post, prior, balance, free, scale):
    kld = tfd.kl_divergence
    dist = lambda x: self.get_dist(x, tf.float32)
    if balance == 0.5:
      value = kld(dist(prior), dist(post))
      loss = tf.reduce_mean(tf.maximum(value, free))
    else:
      sg = lambda x: tf.nest.map_structure(tf.stop_gradient, x)
      value = kld(dist(prior), dist(sg(post)))
      pri = tf.reduce_mean(value)
      pos = tf.reduce_mean(kld(dist(sg(prior)), dist(post)))
      pri, pos = tf.maximum(pri, free), tf.maximum(pos, free)
      loss = balance * pri + (1 - balance) * pos
    loss *= scale
    mse_loss= (pri-pos)**2
    mse_loss= tf.reduce_mean(mse_loss)
    return loss, value, mse_loss

class GRUCell(tf.keras.layers.AbstractRNNCell):

  def __init__(self, size, norm=False, act=tf.tanh, update_bias=-1, **kwargs):
    super().__init__()
    self._size = size
    self._act = act
    self._norm = norm
    self._update_bias = update_bias
    self._layer = tfkl.Dense(3 * size, use_bias=norm is not None, **kwargs)
    if norm:
      self._norm = tfkl.LayerNormalization(dtype=tf.float32)

  @property
  def state_size(self):
    return self._size

  def call(self, inputs, state):
    state = state[0]  # Keras wraps the state in a list.
    parts = self._layer(tf.concat([inputs, state], -1))
    if self._norm:
      dtype = parts.dtype
      parts = tf.cast(parts, tf.float32)
      parts = self._norm(parts)
      parts = tf.cast(parts, dtype)
    reset, cand, update = tf.split(parts, 3, -1)
    reset = tf.nn.sigmoid(reset)
    cand = self._act(reset * cand)
    update = tf.nn.sigmoid(update + self._update_bias)
    output = update * cand + (1 - update) * state
    return output, [output]
