from gnn.graph_data import GraphData
import pathlib

import tensorflow as tf
import numpy as np

def sample_episodes_dep(episodes, length=None, balance=False, seed=0):
  random = np.random.RandomState(seed)
  while True:
    episode = random.choice(list(episodes.values()))
    if length:
      total = len(next(iter(episode.values())))
      available = total - length
      #TODO This caused a problem but I am not sure whether this is a proper solution
      if available < 1:
        print(f'Skipped short episode of length {available}.')
        continue
      if balance:
        index = min(random.randint(0, total), available)
      else:
        index = int(random.randint(0, available + 1))
      episode = {k: v[index: index + length] for k, v in episode.items()}
    yield episode


def sample_episodes(episodes, length=None, balance=False, seed=0):
  random = np.random.RandomState(seed)
  while True:
    episode = random.choice(list(episodes.values()))

    if 'gnn' in episode.keys():
        gnn_input=episode['gnn']
        graphs=[GraphData().load_from_dict(g) for g in gnn_input]

        d={
            'num_nodes':[len(e['ini_nodes']) for e in gnn_input],
            'num_symbols':[len(e['ini_symbols']) for e in gnn_input],
            'num_clauses':[len(e['ini_clauses']) for e in gnn_input]
        }

        data = GraphData.ini_list()
        for g in graphs: data.append(g)
        data.flatten()

        d.update(data.convert_to_dict())

        episode['gnn']=d
    
    yield episode

    
def sample_episodes3(episodes, length=None, balance=False, seed=0):
  random = np.random.RandomState(seed)
  while True:
    episode = random.choice(list(episodes.values()))

    if 'gnn' in episode.keys():
        gnn_input=episode['gnn']
        d={
            'num_nodes':[len(ini_nodes) for ini_nodes in gnn_input['ini_nodes']],
            'num_symbols':[len(ini_symbols) for ini_symbols in gnn_input['ini_symbols']],
            'num_clauses':[len(ini_clauses) for ini_clauses in gnn_input['ini_clauses']]
        }
        #shift the data
        node_shift=0
        symbol_shift=0
        clause_shift=0
        for i in range(len(gnn_input['ini_nodes'])):
            gnn_input['node_inputs_1/symbols'][i] += symbol_shift
            gnn_input['node_inputs_1/nodes'][i][gnn_input['nodes_inputs_1/nodes'][i] >= 0] += node_shift
            gnn_input['node_inputs_2/symbols'][i] += symbol_shift
            gnn_input['node_inputs_2/nodes'][i][gnn_input['nodes_inputs_2/nodes'][i] >= 0] += node_shift
            gnn_input['node_inputs_3/symbols'][i] += symbol_shift
            gnn_input['node_inputs_3/nodes'][i][gnn_input['nodes_inputs_3/nodes'][i] >= 0] += node_shift
            
            gnn_input['symbol_inputs/nodes'][i][gnn_input['symbol_inputs/nodes'][i] >= 0] += node_shift
            gnn_input['node_c_inputs/data'][i] += clause_shift
            gnn_input['clause_inputs/data'][i] += node_shift
            
            node_shift+=d['num_nodes'][i]
            symbol_shift+=d['num_symbols'][i]
            clause_shift+=d['num_clauses'][i]

        for key in gnn_input.keys():
            gnn_input[key]=np.concatenate(gnn_input[key])

        gnn_input.update(d)
    
    yield episode

def sample_episode4(episodes, length=None, balance=False, seed=0):
    random = np.random.RandomState(seed)
    while True:
        episode = random.choice(list(episodes.values()))
        yield episode

def deflatten(episode):
  _episode={k:v for k,v in episode.items() if k.find('/')==-1}
  for k,v in episode.items():
    per=k.find('/')
    if per!=-1:
      key=k[:per]
      if key not in _episode.keys():
        _episode[key]={}
      _episode[key][k[per+1:]]=v
    
  return _episode




#NOTE allow_pickle=True
def load_episodes(directory, limit=None):
  directory = pathlib.Path(directory).expanduser()
  episodes = {}
  total = 0
  for filename in reversed(sorted(directory.glob('*.npz'))):
    try:
      with filename.open('rb') as f:
        episode = np.load(f,allow_pickle=True)
        episode = {k: episode[k] for k in episode.keys()}
        episode = deflatten(episode)
    except Exception as e:
      print(f'Could not load episode: {e}')
      continue 
    episodes[str(filename)] = episode
    total += len(episode['reward']) - 1
    if limit and total >= limit:
      break
  return episodes

def make_dataset(episodes, config, output_sign):
  example = episodes[next(iter(episodes.keys()))]
  #types = {k: v.dtype for k, v in example.items()}
  #shapes = {k: (None,) + v.shape[1:] for k, v in example.items()}
  """
  for k, v in example.items():
    if(k not in output_sign.keys()):
      output_sign[k]=tf.TensorSpec(shape=(None,)+v.shape[1:], dtype=v.dtype)"""

  output_sign.update({
    'action': tf.TensorSpec(shape=(None,None), dtype=tf.float32),
    'reward': tf.TensorSpec(shape=(None,), dtype=tf.float32),
    'discount': tf.TensorSpec(shape=(None,), dtype=tf.float32)
  })

  generator = lambda: sample_episode4(
      episodes, config.batch_length, config.oversample_ends)
  dataset = tf.data.Dataset.from_generator(generator, output_signature=output_sign)
  #NOTE batch>1 not implemented yet (It requires ragged tensors.)
  dataset = dataset.batch(1, drop_remainder=True)
  dataset = dataset.prefetch(10)
  return dataset

