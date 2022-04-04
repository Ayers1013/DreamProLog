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

  def get_callbacks(self, mode, config):
    return [functools.partial(
      process_episode, config, self._logger, mode, self._train_eps, self._eval_eps)]
    
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
    return iter(self.dataset(*args, **kwargs))

  def _sample_state(self, mode, batch, state_length, goal_length, seed=0):
    random = np.random.RandomState(seed)
    selected_eps=dict(train=self._train_eps, eval=self._eval_eps)[mode]

    def pad(narr):
      size = narr.size
      if size>goal_length:
          narr = narr[:goal_length]
          size = goal_length
      return np.pad(narr, [1, goal_length-size], constant_values= [(299, 0)])

    parser = self.tokenParser

    def convert_text(state, apply_pad = True):
      state = np.array([pad(np.array(parser.parse(str(goal)[1:]), dtype = np.int32)) for goal in state[:state_length]])
      if apply_pad: state = np.pad(state, [(0, state_length-state.shape[0]), (0,0)])
      return state

    while True:
      #Sample problem
      problem = selected_eps.sample_problem()
      sample_ep=lambda: selected_eps.sample_episode(problem)
      sample = sample_ep()
      ep = {k: [] for k in sample if k != 'action'}
      for i in range(batch):
        sample = sample_ep()
        length = len(sample['action'])
        
        #sample episode snapshot along with the next one 
        r = random.randint(len(sample['action']))
        {k: ep[k].append(v[r]) if k != 'text' else 
          ep[k].append(convert_text(v[r])) for k, v in sample.items() if k != 'action'}
      
      # TODO we use different formats to access ._storage and ._meta
      problem = '/'.join(problem.split('/')[:-1])

      # yields (ep, meta)
      yield (ep,
        convert_text(selected_eps._meta[problem]['action_space_text'], apply_pad=False)
        ) 

  def signature_state(self, batch_size, state_length, goal_length):
    signature_ep = {
      'action_mask': tf.TensorSpec((batch_size, None), dtype=tf.int32),
      'text': tf.TensorSpec((batch_size, state_length, goal_length+1), dtype=tf.int32),
      'reward': tf.TensorSpec((batch_size), dtype=tf.float32),
      'discount': tf.TensorSpec((batch_size,), dtype=tf.float32),
    }
    signature_meta = tf.TensorSpec((None, goal_length+1), dtype=tf.int32)
    signature = (signature_ep, signature_meta)
    return signature

  def dataset_state(self, batch_size, state_length, goal_length):
    generator = lambda: self._sample_state(
      'train', batch_size, state_length, goal_length)
    dataset = tf.data.Dataset.from_generator(generator, output_signature=self.signature_state(batch_size, state_length, goal_length))
    dataset = dataset.prefetch(16)
    return dataset

  def _sample(self, mode, batch, state_length, goal_length, seed=0):
    random = np.random.RandomState(seed)
    selected_eps=dict(train=self._train_eps, eval=self._eval_eps)[mode]

    def pad(narr):
      size = narr.size
      if size>goal_length:
          narr = narr[:goal_length]
          size = goal_length
      return np.pad(narr, [1, goal_length-size], constant_values= [(299, 0)])

    parser = self.tokenParser

    def convert_text(state, apply_pad = True):
      state = np.array([pad(np.array(parser.parse(str(goal)[1:]), dtype = np.int32)) for goal in state[:state_length]])
      if apply_pad: state = np.pad(state, [(0, state_length-state.shape[0]), (0,0)])
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
        r = random.randint(len(sample['action'])-1)
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

  def signature(self, batch_size, state_length, goal_length):
    
    signature_ep = {
      'action_mask': tf.TensorSpec((batch_size, None), dtype=tf.int32),
      'text': tf.TensorSpec((batch_size, state_length, goal_length+1), dtype=tf.int32),
      'reward': tf.TensorSpec((batch_size), dtype=tf.float32),
      'discount': tf.TensorSpec((batch_size,), dtype=tf.float32),
    }
    signature_action = tf.TensorSpec((batch_size,), dtype=tf.int32)
    signature_meta = tf.TensorSpec((None, goal_length+1), dtype=tf.int32)
    signature = (signature_ep, signature_action, signature_ep, signature_meta)
    return signature

  def dataset(self, batch_size, state_length, goal_length, balance=True):
    generator = lambda: self._sample(
      'train', batch_size, state_length, goal_length, balance)
    #output_sign=self._output_sign(batch_size, self._train_eps._tagToLength(batch_length))
    dataset = tf.data.Dataset.from_generator(generator, output_signature=self.signature(batch_size, state_length, goal_length))
    dataset = dataset.prefetch(16)
    return dataset

  def dataset_goal(self, batch_size, goal_length):
    generator = lambda: self._sample_goal('train', batch_size, goal_length)
    dataset = tf.data.Dataset.from_generator(generator, output_signature=tf.TensorSpec((batch_size, goal_length+1), dtype=tf.int32))
    dataset = dataset.prefetch(16)
    return dataset

  def _sample_goal(self, mode, batch, goal_length, seed=0):
    random = np.random.RandomState(seed)
    selected_eps=dict(train=self._train_eps, eval=self._eval_eps)[mode]

    def pad(narr):
      size = narr.size
      if size>goal_length:
          narr = narr[:goal_length]
          size = goal_length
      return np.pad(narr, [1, goal_length-size], constant_values= [(299, 0)])

    parser = self.tokenParser

    def convert_text(goal, apply_pad = True):
      return pad(np.array(parser.parse(str(goal)[1:]), dtype = np.int32))

    while True:
      sample_ep = lambda: selected_eps.sample_episode(selected_eps.sample_problem())

      goals = []
      for i in range(batch):
        sample = sample_ep()
        length = len(sample['action'])
        r = random.randint(length)
        if len(sample['text'][r])>0:
          goal = random.choice(sample['text'][r])
          goals.append(convert_text(goal))
        else:
          goals.append(np.zeros(goal_length+1, np.int32))
      
      yield np.stack(goals)
