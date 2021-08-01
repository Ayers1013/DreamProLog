import pathlib
import uuid
import datetime
import io
import json

from numpy.lib.function_base import select

#from gnn import GraphData

from dataset.process import process_episode, load_episodes
import functools
import tensorflow as tf
import numpy as np

class DatasetConfig:
  """
  This class is responsible for the configuration of a dataset. It has an implented __hash__ and __eq__ function therefore it can be used as a dictionary key.
  NOTE The pipe arg has no use case (yet)
  """
  def __init__(self, batch_length, batch_size, pipes=[]):
    self.pipes=sorted(pipes)
    self.batch_length=batch_length
    self.batch_size=batch_size

  def __hash__(self):
    return hash('-'.join(self.pipes)+str(self.batch_length)+str(self.batch_size))
   
  def __eq__(self, other):
    if(len(self.pipes)!=len(other.pipes)):
      return False
    for i in range(len(self.pipes)):
      if(self.pipes[i]!=other.pipes[i]): return False
    return (self.batch_size==other.batch_size) and (self.batch_length==other.batch_length)


class DatasetManager:
  def __init__(self, logger, output_sign, train_dir, eval_dir):
    self._logger=logger
    self._train_dir=train_dir
    self._eval_dir=eval_dir

    self._datasets={}
    
    self._train_eps=load_episodes(self._train_dir)
    self._eval_eps=load_episodes(self._eval_dir)

    self._output_sign=output_sign
  
  def get_callbacks(self, mode, config):
    return [functools.partial(
      process_episode, config, self._logger, mode, self._train_eps, self._eval_eps)]

  @staticmethod
  def nested_concat():
    return None

  def get_episode(self, random, selected_eps, length=1):
    #To weigth differently the larger lengths
    wfun=lambda x: x
    
    totalLens={l: wfun(len(eps))  if l>= length else 0 for l, eps in selected_eps.items()}
    total=sum(totalLens.values())
    totalLens={l: v/total for l,v in totalLens.items()}
    lengthIndex=random.choice(list(totalLens.keys()), p=list(totalLens.values()))
    episode=random.choice(list(selected_eps[lengthIndex].values()))
    index=np.random.randint(lengthIndex-length+1)
    episode = {k: v[index: index + length] for k, v in episode.items() if k!='action_space'}
    episode['gnn']={k: [episode['gnn'][i][k] for i in range(length)] for k in episode['gnn'][0].keys()}
    return episode


  def sample_episode(self, mode, batch=None, length=None, balance=False, seed=0):
    random = np.random.RandomState(seed)
    assert isinstance(length, int)
    while True:
      #Sample problem
      selected_eps=dict(train=self._train_eps, eval=self._eval_eps)[mode]
      selected_eps= random.choice(list(selected_eps.values()))
      #I know that this is very ugly :( NOTE REPAIR THIS!!
      sample=next(iter(next(iter(selected_eps.values())).values()))

      #NOTE Probably I should use tf.nest 
      eps=[self.get_episode(random, selected_eps, length) for _ in range(batch)]
      def nested_concat(l):
        return {k: [l(i)[k] for i in range(batch)] if not isinstance(l(0)[k], dict) else nested_concat(lambda i: l(i)[k]) for k in l(0).keys()}
      eps=nested_concat(lambda i: eps[i])
      
      _eps={k: np.stack(v) for k,v in eps.items() if k not in ['gnn', 'action_space']}
      if 'gnn' in sample.keys():
        _eps['gnn']={k: tf.ragged.constant(eps['gnn'][k]) for k in eps['gnn'].keys()}
      if 'action_space' in sample.keys():
        _eps['action_space']=sample['action_space']
      yield _eps

  def dataset(self, batch_size, batch_length):
    
    generator = lambda: self.sample_episode(
      'train', batch_size, batch_length)
    output_sign=self._output_sign(batch_size, batch_length)
    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_sign)
    dataset = dataset.prefetch(10)
    return dataset

