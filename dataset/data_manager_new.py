from operator import le
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


class DatasetManager:
  def __init__(self, logger, output_sign, train_dir, eval_dir):
    columns=['Problem', 'count', 'done', 'reward_sum', 'weighted_reward']
    columns+=[k+'_count' for k in ['small', 'medium', 'large']]
    self._table_columns=columns
    self._logger=logger


    self._train_dir=train_dir
    self._eval_dir=eval_dir

    self._datasets={}
    
    self._scheduled=(None,None)
    self._train_eps=load_episodes(self._train_dir)
    self._eval_eps=load_episodes(self._eval_dir)
    
    self._output_sign=output_sign
  
  def get_callbacks(self, mode, config):
    return [functools.partial(
      process_episode, config, self._logger, mode, self._train_eps, self._eval_eps)]

  def sample_episode(self, mode, batch=None, length=None, balance=False, seed=0):
    random = np.random.RandomState(seed)
    assert isinstance(length, int)
    length=self._train_eps._lengthToTag(length)
    while True:
      #Sample problem
      selected_eps=dict(train=self._train_eps, eval=self._eval_eps)[mode]
      problem, lengthIndex=self._scheduled
      sample_ep=lambda x: selected_eps.sample_episode(problem, length, x)

      if balance:
        pos_batch=(batch+4)//5
        neg_batch=batch-pos_batch
        eps=[sample_ep(False) for _ in range(neg_batch)]
        eps+=[sample_ep(True) for _ in range(pos_batch)]
      else:    
        eps=[sample_ep(False) for _ in range(batch)]

      sample_name=selected_eps._storage[problem][length][0][1]
      sample=selected_eps._episodes[sample_name]
      #NOTE Probably I should use tf.nest 
      def nested_concat(l):
        return {k: [l(i)[k] for i in range(batch)] if not isinstance(l(0)[k], dict) else nested_concat(lambda i: l(i)[k]) for k in l(0).keys()}
      eps=nested_concat(lambda i: eps[i])
      
      try:
        _eps={k: np.stack(v) for k,v in eps.items() if k not in ['gnn', 'action_space']}
      except ValueError as err:
        print(f'This episode from the {problem} problem is corrupted.')
        for k,v in eps.items():
          if k not in ['gnn', 'action_space']:
            print(k, type(v[0]))
            print([e.shape for e in v])
        continue
      if 'gnn' in sample.keys():
        _eps['gnn']={k: tf.ragged.constant(eps['gnn'][k]) for k in eps['gnn'].keys()}
      if 'action_space' in sample.keys():
        _eps['action_space']=sample['action_space']
      yield _eps

  def dataset(self, batch_size, batch_length, balance=False):
    
    generator = lambda: self.sample_episode(
      'train', batch_size, batch_length, balance)
    output_sign=self._output_sign(batch_size, batch_length)
    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_sign)
    #dataset = dataset.prefetch(10)
    return dataset

  def logging(self):
    stats=self._train_eps._stats
    scalars={}
    columns=self._table_columns
    data=[]
    for k,v in stats.items():
      if k[:6]=='stats_':
        scalars[k[6:]]=v
      else:
        stat_dict=self._train_eps.get_statistic(v)
        stat_list=[k]
        for name in columns[1:]:
          stat_list.append(stat_dict.get(name, 0))
        data.append(stat_list)
    self._logger._scalars.update(scalars)
    self._logger.table('Dataset', columns, data)
    



  def __iter__(self):
    names=[
      ('small', (32,2)), 
      ('medium', (16, 6)), 
      ('large', (4, 18)),
    ]
    for name, setting in names:
      self._datasets[name]=iter(self.dataset(*setting, True))

    return self
  
  def __next__(self):
    problem=self._train_eps.sample_problem(10)
    length=self._train_eps.sample_lengthIndex(problem, 4)
    self._scheduled=(problem, length)
    return next(self._datasets[length])


