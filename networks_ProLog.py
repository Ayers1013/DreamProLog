import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfkl
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision import experimental as prec

import tools

class DummyEncoder(tools.Module):

  def __init__(self):
      self.dense=tfkl.Dense(200)

  def __call__(self, obs):
    x = tf.reshape(obs['image'], (-1,) + tuple(obs['image'].shape[-2:]))
    x= self.dense(x)
    return x


class DummyDecoder(tools.Module):

  def __init__(self, shape=(1,16)):
      self.dense=tfkl.Dense(16)
      self._shape=shape

  def __call__(self, obs, dtype=None):
    return tfd.Independent(tfd.Normal(self.dense(obs), 1), len(self._shape))


class EncoderWrapper(tools.Module):

  def __init__(self):
      self.dense=tfkl.Dense(16)

  def __call__(self, obs):
    x = tf.reshape(obs['image'], (-1,) + tuple(obs['image'].shape[-2:]))
    return self.dense(x)


class DecoderWrapper(tools.Module):

  def __init__(self, shape=(1,16)):
      self.dense=tfkl.Dense(16)
      self._shape=shape

  def __call__(self, obs, dtype=None):
    return tfd.Independent(tfd.Normal(self.dense(obs), 1), len(self._shape))