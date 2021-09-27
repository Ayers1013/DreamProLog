import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfkl
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision import experimental as prec

import tools

#Note std=1.0 changed to std="learned"
class DenseHead(tools.Module):

  def __init__(
      self, shape, layers, units, act=tf.nn.elu, dist='normal', std='learned'):
    self._shape = (shape,) if isinstance(shape, int) else shape
    self._layers = layers
    self._units = units
    self._act = act
    self._dist = dist
    self._std = std

  def __call__(self, features, dtype=None):
    x = features
    for index in range(self._layers):
      x = self.get(f'h{index}', tfkl.Dense, self._units, self._act)(x)
    mean = self.get(f'hmean', tfkl.Dense, np.prod(self._shape))(x)
    mean = tf.reshape(mean, tf.concat(
        [tf.shape(features)[:-1], self._shape], 0))
    if self._std == 'learned':
      std = self.get(f'hstd', tfkl.Dense, np.prod(self._shape))(x)
      std = tf.nn.softplus(std) + 0.01
      std = tf.reshape(std, tf.concat(
          [tf.shape(features)[:-1], self._shape], 0))
    else:
      std = self._std
    if dtype:
      mean, std = tf.cast(mean, dtype), tf.cast(std, dtype)
    if self._dist == 'normal':
      return tfd.Independent(tfd.Normal(mean, std), len(self._shape))
    if self._dist == 'huber':
      return tfd.Independent(
          tools.UnnormalizedHuber(mean, std, 1.0), len(self._shape))
    if self._dist == 'binary':
      return tfd.Independent(tfd.Bernoulli(mean), len(self._shape))
    raise NotImplementedError(self._dist)

