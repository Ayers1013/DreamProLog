import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfkl
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision import experimental as prec

from networks import DenseHead

import tools


class DummyEncoder(tools.Module):

  def __init__(self):
    self.dense1=tfkl.Dense(150)
    self.dense2=tfkl.Dense(200)

  def __call__(self, obs):
    x = tf.reshape(obs, (-1,) + tuple(obs.shape[-2:]))
    x=self.dense1(x)
    x= self.dense2(x)
    return x


class DummyDecoder(tools.Module):

  def __init__(self, shape=(1,25)):
    self.dense1=tfkl.Dense(150)
    #Porlog:14, dummy: 25
    self.dense2=tfkl.Dense(25)
    self.dense3=tfkl.Dense(25)
    self._shape=shape

  def __call__(self, obs, dtype=None):
    obs=self.dense1(obs)
    mean=self.dense2(obs)
    std=self.dense3(obs)
    std=tf.keras.activations.sigmoid(std)
    return tfd.Independent(tfd.Normal(mean,std+0.01), len(self._shape))

class ActionEncoder(tools.Module):

  def __init__(self):
    self.dense=tfkl.Dense(8)

  def __call__(self, obs):
    x=self.dense(obs)
    return x

class Encoder(tools.Module):
  def __init__(self, input_pipes, action_embed):
    self._input_pipes=input_pipes
    self._action_embed=action_embed
    self.encoders={}

    #self.encoders['action_space']=ActionEncoder()
    self.encoders['image']=DummyEncoder()

    if('features' in self._input_pipes):
      #NOTE config
      self.encoders['features']=DummyEncoder()
    
    if(self._action_embed):
      self.action_encoder=ActionEncoder()

  def __call__(self, obs):
    embed={}
    for pipe in self._input_pipes:
      embed[pipe]=self.encoders[pipe](obs[pipe])

    action_embed=None
    if(self._action_embed):
      action_embed=self.action_encoder(obs['action_space'])
    
    
    return embed['image'], action_embed
    
