from operator import le
import pathlib
import uuid
import datetime
import io
import json

from numpy.lib.function_base import select

#from gnn import GraphData

from dataset.process import process_episode, load_episodes, TokenParser
import functools
import tensorflow as tf
import numpy as np


class DatasetManager:
  def __init__(self, logger, output_sign, train_dir, eval_dir):
    columns=['Problem', 'count', 'done', 'reward_sum', 'weighted_reward']
    
    self._table_columns=columns
    self._logger=logger


    self._train_dir=train_dir
    self._eval_dir=eval_dir
    
    self._train_eps=load_episodes(self._train_dir)
    self._eval_eps=load_episodes(self._eval_dir)
    
    self._output_sign=output_sign

    self.tokenParser = TokenParser()

  def _sample(self, mode, batch = None, seed=0):
    random = np.random.RandomState(seed)
    selected_eps=dict(train=self._train_eps, eval=self._eval_eps)[mode]

    def pad(narr):
      size = narr.size
      if size>128:
          narr = narr[:128]
          size = 128
      return np.pad(narr, [1, 128-size], constant_values= [(299, 0)])

    parser = self.tokenParser

    def convert_text(state, apply_pad = True):
      state = np.array([pad(np.array(parser.parse(str(goal)[1:]), dtype = np.int32)) for goal in state])
      if apply_pad: state = np.pad(state[:128], [(0, 128-state.shape[0]), (0,0)])
      return state

    while True:
      #Sample problem
      problem = selected_eps.sample_problem()
      sample_ep=lambda: selected_eps.sample_episode(problem)
      sample = sample_ep()
      ep0 = {k: [] for k in sample if k != 'action'}
      actions = []
      ep1 = {k: [] for k in sample if k != 'action'}
      for i in range(batch):
        sample = sample_ep()
        length = len(sample['action'])
        
        #sample episode snapshot along with the next one 
        r = random.randint(len(sample)-1)
        {k: ep0[k].append(v[r]) if k != 'text' else 
          ep0[k].append(convert_text(v[r])) for k, v in sample.items() if k != 'action'}
        actions.append(sample['action'][r+1])
        {k : ep1[k].append(v[r+1]) if k != 'text' else
          ep1[k].append(convert_text(v[r+1])) for k, v in sample.items() if k != 'action'}
      
      # TODO we use different formats to access ._storage and ._meta
      problem = '/'.join(problem.split('/')[:-1])

      # yields (ep0, action, ep1, meta)
      yield ({k: np.stack(v) for k, v in ep0.items()},
        np.stack(actions), 
        {k: np.stack(v) for k, v in ep1.items()},
        convert_text(selected_eps._meta[problem]['action_space_text'], apply_pad=False)
        )
  
  def get_callbacks(self, mode, config):
    return [functools.partial(
      process_episode, config, self._logger, mode, self._train_eps, self._eval_eps)]

  def dataset(self, batch_size, balance=True):
    signature_ep = {
      'action_mask': tf.TensorSpec((batch_size, None), dtype=tf.int32),
      'text': tf.TensorSpec((batch_size, 128, 129), dtype=tf.int32),
      'reward': tf.TensorSpec((batch_size), dtype=tf.float32),
      'discount': tf.TensorSpec((batch_size,), dtype=tf.float32),
    }
    signature_action = tf.TensorSpec((batch_size,), dtype=tf.int32)
    signature_meta = tf.TensorSpec((None, 129), dtype=tf.int32)
    signature = (signature_ep, signature_action, signature_ep, signature_meta)

    generator = lambda: self._sample(
      'train', batch_size, balance)
    #output_sign=self._output_sign(batch_size, self._train_eps._tagToLength(batch_length))
    dataset = tf.data.Dataset.from_generator(generator, output_signature=signature)
    dataset = dataset.prefetch(10)
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

    def __iter__(self, *args, **kwargs):
      return self.dataset(*args, **kwargs)


