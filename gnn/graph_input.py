import tensorflow as tf
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor
from gnn.graph_data import GraphData
from gnn.segments import SegmentsPH

class GraphHyperEdgesAPH(tf.Module):
    def __init__(self):
        super().__init__()
        self.segments=SegmentsPH()

    """
    def __call__(self, x):
        self.segments(x.lens)
        self.symbols=x.symbols
        self.nodes=x.nodes
        self.sgn=x.sgn
    """

    def __call__(self, x):
        self.segments(x['lens'])
        self.symbols=x['symbols']
        self.nodes=x['nodes']
        self.sgn=x['sgn']

class GraphHyperEdgesBPH(tf.Module):
    def __init__(self):
        super().__init__()
        self.segments=SegmentsPH(nonzero=True)

    """
    def __call__(self, x):
        self.segments(x.lens)
        self.nodes=x.nodes
        self.sgn=x.sgn
        """
    
    def __call__(self, x):
        self.segments(x['lens'])
        self.nodes=x['nodes']
        self.sgn=x['sgn']

class GraphEdgesPH(tf.Module):
    def __init__(self, nonzero=False):
        super().__init__()
        self.segments=SegmentsPH(nonzero=nonzero)

    """
    def __call__(self, x):
        self.segments(x.lens)
        self.data=x.data
        """
    
    def __call__(self, x):
        self.segments(x['lens'])
        self.data=x['data']


class GraphInput(tf.Module):
    def __init__(self):
        super().__init__()
        self.node_inputs=tuple(
            GraphHyperEdgesAPH() for _ in range(3)
        )

        self.symbol_inputs = GraphHyperEdgesBPH()
        self.node_c_inputs = GraphEdgesPH()
        self.clause_inputs = GraphEdgesPH(nonzero = True)

        self.node_nums   = SegmentsPH( nonzero = True)
        self.symbol_nums = SegmentsPH( nonzero = True)
        self.clause_nums = SegmentsPH( nonzero = True)

    def __call__(self, x):

        self.node_nums(x['num_nodes'])
        self.symbol_nums(x['num_symbols'])
        self.clause_nums(x['num_clauses'])

        self.ini_nodes=x['ini_nodes']
        self.ini_symbols=x['ini_symbols']
        self.ini_clauses=x['ini_clauses']
        #self.axiom_mask=x['axiom_mask']

        extract= lambda name: {k.split("/")[1]: v for k,v in x.items() if k.split("/")[0]==name}

        for ind, layer in enumerate(self.node_inputs):
            layer(extract("node_inputs_{}".format(ind+1)))

        self.symbol_inputs(extract("symbol_inputs"))
        self.node_c_inputs(extract("node_c_inputs"))
        self.clause_inputs(extract("clause_inputs"))

def shifted_cumsum(x, length):
    x=tf.squeeze(x, axis=-1)
    result=[]
    shift=0
    for i in range(length):
        result.append(shift)
        shift+=x[i]
    return tf.stack(result, axis=0)

#shift only non -1 entries
def shift_non_neg(x, s):
    shift_mask=tf.cast(x+1, dtype=tf.bool)
    x=x+s
    x=tf.where(shift_mask, x, -1)
    return x

def flatten_gnn_input(inp, length):
    inp=tf.nest.map_structure(lambda x: tf.squeeze(x, axis=1), inp)
    cumsum={k: tf.expand_dims(shifted_cumsum(inp['num_'+k], length), axis=-1) for k in ['nodes', 'symbols', 'clauses']}

    #symbol shifts
    for i in range(3):
        name=f'node_inputs_{i+1}/symbols'
        inp[name]=inp[name]+cumsum['symbols']
    
    #clause shifts
    name=f'node_c_inputs/data'
    inp[name]=inp[name]+cumsum['clauses']

    #node shifts
    name=f'clause_inputs/data'
    inp[name]=inp[name]+cumsum['nodes']

    #[length, 1, 1] shape is required for the latter ops
    cumsum['nodes']=tf.expand_dims(cumsum['nodes'], axis=-1)
    for i in range(3):
        name=f'node_inputs_{i+1}/nodes'
        inp[name]=shift_non_neg(inp[name], cumsum['nodes'])
    
    name=f'symbol_inputs/nodes'
    inp[name]=shift_non_neg(inp[name], cumsum['nodes'])

    #concat the other
    flatten = lambda x: tf.concat([x[i] for i in range(length)], axis=0)
    to_tensor = lambda x: x if not isinstance(x, tf.RaggedTensor) else x.to_tensor()
    inp=tf.nest.map_structure(lambda x: to_tensor(flatten(x)), inp)

    return inp


def feed_gnn_input(x, batch_size, batch_length, fun):
    result=tf.TensorArray(tf.float32, batch_length, element_shape=(batch_size, fun._out_dim))
    pos=tf.zeros((1,), dtype=tf.int32)
    def calc_slice(i):
        y=tf.nest.map_structure(lambda inp: inp[:, i:i+1], x)
        y=flatten_gnn_input(y, batch_size)
        z=fun(y)
        z.set_shape((batch_size, z.shape[1]))
    result, _=tf.while_loop(
        lambda inp, i: i < batch_length,
        lambda inp, i: (inp.write(i, calc_slice(i)), i + 1),
        [result, pos]
    )

    print('Almost Traced')
    result=result.stack()
    result=tf.transpose(result, [1,0,2])
    return result

def feed_gnn_input_dep(x, batch_size, batch_length, fun):
    result=[]
    for i in range(batch_length):
        y=tf.nest.map_structure(lambda inp: inp[:, i:i+1], x)
        y=flatten_gnn_input(y, batch_size)
        z=fun(y)
        z.set_shape((batch_size, z.shape[1]))
        result.append(z)
    print('Almost Traced')
    result=tf.stack(result, axis=1)
    return result