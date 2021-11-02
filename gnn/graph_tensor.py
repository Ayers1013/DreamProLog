import tensorflow as tf

def shifted_cumsum(tensor, shape = None):
    if not shape:
        shape = tensor.shape
    flat_size = 1
    for dim in shape:
        flat_size*=dim
    b = tf.math.cumsum(tf.reshape(tensor, [flat_size]))
    b = tf.concat((tf.zeros((1,), tf.int32), b[:-1]), axis=0)
    return tf.reshape(b, tensor.shape)

def shift_non_neg(x, s):
    shift_mask=tf.cast(x+1, dtype=tf.bool)
    x=x+s
    x=tf.where(shift_mask, x, -1)
    return x
    
def shift_ragged(ragged, shift, non_negative = False):
    shape = shift.shape
    if len(shape)==0: return ragged
    index = [0]*len(shape)
    array  = []
    end = True
    while end:
        if non_negative:
            array.append(shift_non_neg(ragged[index], shift[index]))
        else:
            array.append(ragged[index]+shift[index])
        for i in range(len(shape)-1, -1, -1):
            if index[i]==shape[i]-1:
                index[i]=0
                if i==0:
                    end = False
            else:
                index[i]+=1
                break
    return tf.concat(array, axis=0)

def flatten_ragged(ragged, shape):
    if len(shape)==0: return ragged
    index = [0]*len(shape)
    array  = []
    end = True
    while end:
        array.append(ragged[index])
        for i in range(len(shape)-1, -1, -1):
            if index[i]==shape[i]-1:
                index[i]=0
                if i==0:
                    end = False
            else:
                index[i]+=1
                break
    return tf.concat(array, axis=0)



class GraphTensor:
    def __init__(self, data, batch_shape, flat = False):
        self._data = data
        self._batch_shape = batch_shape
        self._flat = flat

    def __getitem__(self, key):
        return self._data[key]

    def flatten_batch(self):
        if len(self._batch_shape)==0: return
        cumsum = {k: shifted_cumsum(self._data['num_'+k]) for k in ['nodes', 'symbols', 'clauses']}

        op_description = {
            'node_inputs_1': {'lens': 0, 'symbols': 'symbols_0', 'nodes': 'nodes_1', 'sgn': 0},
            'node_inputs_2': {'lens': 0, 'symbols': 'symbols_0', 'nodes': 'nodes_1', 'sgn': 0},
            'node_inputs_3': {'lens': 0, 'symbols': 'symbols_0', 'nodes': 'nodes_1', 'sgn': 0},
            'symbol_inputs': {'lens': 0, 'nodes': 'nodes_1', 'sgn': 0},
            'node_c_inputs': {'lens': 0, 'data': 'clauses_1'}, 
            'clause_inputs': {'lens': 0, 'data': 'nodes_0'}, 
            'ini_nodes': 0, 
            'ini_symbols': 0, 
            'ini_clauses': 0,
            'num_nodes': 0,
            'num_symbols': 0,
            'num_clauses': 0
        }
        print('almost done')
        op = lambda x, desc: flatten_ragged(x, self._batch_shape) if desc == 0 else shift_ragged(x, cumsum[desc[:-2]], int(desc[-1]))
        to_tensor = lambda x: x if not isinstance(x, tf.RaggedTensor) else x.to_tensor()
        self._data=tf.nest.map_structure(lambda x, desc: to_tensor(op(x, desc)), self._data, op_description)
      
    def flatten(self):
        episode = self._data
        _episode={k: v for k,v in episode.items() if not isinstance(v, dict)}
        for key, item in episode.items():
            if key not in _episode.keys():
                _episode.update({key+"/"+nkey: v for nkey,v in flatten_ep(item).items()})

        self._data = _episode

    def deflatten(self):
        episode = self._data
        _episode={k:v for k,v in episode.items() if k.find('/')==-1}
        for k,v in episode.items():
            per=k.find('/')
            if per!=-1:
            key=k[:per]
            if key not in _episode.keys():
                _episode[key]={}
            _episode[key][k[per+1:]]=v
            
        self._data = _episode 

    







def gnn_tensor_shape(Spec, include_nums=False):
    outputs={
        'node_inputs_1': ['lens', 'symbols', 'nodes', 'sgn'],
        'node_inputs_2': ['lens', 'symbols', 'nodes', 'sgn'], 
        'node_inputs_3': ['lens', 'symbols', 'nodes', 'sgn'],
        'symbol_inputs': ['lens', 'nodes', 'sgn'],
        'node_c_inputs': ['lens', 'data'], 
        'clause_inputs': ['lens', 'data'], 
    }
    summary_outputs=[
        'node_inputs_1',
        'node_inputs_2', 
        'node_inputs_3',
        'symbol_inputs',
        'node_c_inputs',
        'clause_inputs',
        'ini_nodes', 
        'ini_symbols', 
        'ini_clauses',
        'num_nodes',
        'num_symbols',
        'num_clauses'
    ]

    gnnSpec = {}
    #node_inputs
    for name in summary_outputs[:3]:
        gnnSpec[name]= {Spec((None,)) if key != 'nodes' else Spec((None, 2)) for key in outputs[name]}
    #symbol_inputs
    name = 'symbol_inputs'
    gnnSpec[name]= {Spec((None,)) if key != 'nodes' else Spec((None, 3)) for key in outputs[name]}
    for name in summary_outputs[4:6]:
        gnnSpec[name]= {Spec((None,)) for key in outputs[name]}
    for name in summary_outputs[6:]:
        gnnSpec[name] = Spec((None,))

    return gnnSpec

sign=gnn_tensor_shape(lambda x: tf.TensorSpec(shape=()+x, dtype=tf.int32), True)