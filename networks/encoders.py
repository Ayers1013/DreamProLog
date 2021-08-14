import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfkl
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision import experimental as prec

import tools

from gnn import MultiGraphNetwork, feed_gnn_input

class DummyEncoder(tools.Module):

  def __init__(self):
    self.dense1=tfkl.Dense(150)
    self.dense2=tfkl.Dense(200)

  def __call__(self, obs):
    #x = tf.reshape(obs, (-1,) + tuple(obs.shape[-2:]))
    #NOTE batch=1
    #x=tf.squeeze(obs, axis=0)
    x=obs
    x=self.dense1(x)
    x= self.dense2(x)
    return x


class DummyDecoder(tools.Module):

  def __init__(self, shape=(1,14)):
    self.dense1=tfkl.Dense(150)
    #Porlog:14, dummy: 25
    self.dense2=tfkl.Dense(14)
    self.dense3=tfkl.Dense(14)
    self._shape=shape

  def __call__(self, obs, dtype=None):
    obs=self.dense1(obs)
    mean=self.dense2(obs)
    std=self.dense3(obs)
    std=tf.keras.activations.sigmoid(std)
    return tfd.Independent(tfd.Normal(mean,std+0.01), len(self._shape))

class Encoder(tools.Module):
  def __init__(self, input_pipes, action_embed):
    self._input_pipes=input_pipes
    self._action_embed=action_embed
    self.encoders={}

    if 'image' in input_pipes:
      self.encoders['image']=DummyEncoder()    

    if 'gnn' in input_pipes:
      self.gnn=MultiGraphNetwork(out_dim=200)
      self.encoders['gnn']=self.gnn.stateEmbed
      self.encoders['action_space']=self.gnn.actionEmbed

  def __call__(self, obs):

    if obs['gnn']['num_nodes'].shape!=(1,1):
      batch_size, batch_length= obs['image'].shape[:2]
      embed=feed_gnn_input(obs['gnn'], batch_size, batch_length, self.encoders['gnn'])
      
      action_embed=self.encoders['action_space'](obs['action_space'])
    else:
      #NOTE For some reason it is wrapped in a list, so [0]
      try:
        embed=self.encoders['gnn'](obs['gnn'][0])
      except:
        inp=tf.nest.map_structure(lambda x: tf.squeeze(x, axis=0), obs['gnn'])
        inp=tf.nest.map_structure(lambda x: tf.cast(x, dtype=tf.int32), inp)
        embed=self.encoders['gnn'](inp)
      
      inp=obs['action_space'][0]
      inp=tf.nest.map_structure(lambda x: tf.cast(x, dtype=tf.int32), inp)
      action_embed=self.encoders['action_space'](inp)

    action_embed=self.encoders['action_space'](obs['action_space'])

    return embed, action_embed
    
class ActionHead(tools.Module):

  def __init__(
      self, layers, units, act=tf.nn.elu, dist='trunc_normal',
      init_std=0.0, min_std=0.1, action_disc=5, temp=0.1, outscale=0, action_embed=None):
    # assert min_std <= 2
    self._size = 64
    self._layers = layers
    self._units = units
    self._dist = dist
    self._act = act
    self._min_std = min_std
    self._init_std = init_std
    self._action_disc = action_disc
    self._temp = temp() if callable(temp) else temp
    self._outscale = outscale
    self._embed=None

  def __call__(self, features, dtype=None):
    x = features
    for index in range(self._layers):
      kw = {}
      if index == self._layers - 1 and self._outscale:
        kw['kernel_initializer'] = tf.keras.initializers.VarianceScaling(
            self._outscale)
      x = self.get(f'h{index}', tfkl.Dense, self._units, self._act, **kw)(x)
    
    x = self.get(f'hout', tfkl.Dense, self._size)(x)
    x=tf.matmul(x, tf.transpose(self._embed))
    x=tf.nn.softmax(x)
    dist=tfd.Categorical(probs=x)

    '''x = self.get(f'hstd_mean', tfkl.Dense, 2*self._size)(x)
    if dtype:
      x = tf.cast(x, dtype)
    mean, std = tf.split(x, 2, -1)
    std = tf.nn.softplus(std + self._init_std) + self._min_std
    dist = tfd.Normal(mean, std)
    dist = tfd.Independent(dist, 1)'''
    return dist

  def feed(self, embed):
    self._embed=embed