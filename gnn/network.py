import tensorflow as tf
import numpy as np
from tensorflow.python.keras.engine import input_layer
from tensorflow.python.ops.gen_batch_ops import batch

# from gnn.tf_helpers import *

from gnn.graph_input import GraphInput
from gnn.segments import Segments
from gnn.graph_conv import graph_start, graph_conv
from gnn.graph_data import GraphData
import tools

#input_sign

def gnn_output_sign(spec, include_nums=False):
    outputs=[
        'node_inputs_1/lens',
        'node_inputs_1/symbols',
        'node_inputs_1/nodes',
        'node_inputs_1/sgn',
        'node_inputs_2/lens',
        'node_inputs_2/symbols',
        'node_inputs_2/nodes', 
        'node_inputs_2/sgn', 
        'node_inputs_3/lens', 
        'node_inputs_3/symbols', 
        'node_inputs_3/nodes', 
        'node_inputs_3/sgn', 
        'symbol_inputs/lens', 
        'symbol_inputs/nodes', 
        'symbol_inputs/sgn', 
        'node_c_inputs/lens', 
        'node_c_inputs/data', 
        'clause_inputs/lens', 
        'clause_inputs/data', 
        'ini_nodes', 
        'ini_symbols', 
        'ini_clauses',
        'num_nodes',
        'num_symbols',
        'num_clauses'
    ]
    #if not include_nums: outputs=outputs[:-3]

    gnnSpec={}
    for name in outputs[:-3]:
        if name=='symbol_inputs/nodes': gnnSpec[name]=spec((None, 3,))
        elif name.find("/")!=-1 and name.split("/")[1]=='nodes': gnnSpec[name]=spec((None, 2,))
        else: gnnSpec[name]=spec((None,))

    if include_nums:
        for name in outputs[-3:]:
            gnnSpec[name]=spec((None,))

    return gnnSpec

sign=gnn_output_sign(lambda x: tf.TensorSpec(shape=()+x, dtype=tf.int32), True)
'''for k in ['num_nodes', 'num_symbols', 'num_clauses']:
    sign[k]=tf.TensorSpec((8,), dtype=tf.int32)'''

class NetworkConfig:
    def __init__(self):
        self.gnn_start_shape = (4,1,4)
        self.gnn_next_shape = (32,64,32)
        #self.next_shape = (11,12,13)
        self.gnn_layers = 10
        self.gnn_hidden_val = 64
        self.gnn_hidden_act = 64


class GraphNetwork(tools.Module):
    def __init__(self, out_dim=32, config=None):
        super().__init__()
        if(config is None):
            self.config=NetworkConfig()
        else:
            self.config=config

        self.input_layer=GraphInput()
        self.start_layer=graph_start(self.config.start_shape, self.input_layer)
        self.conv_layers=[
            graph_conv(self.input_layer, output_dims=self.config.next_shape) for _ in range(self.config.layers)
        ]

        self.dense1=tf.keras.layers.Dense(self.config.hidden_val)
        self.dense2=tf.keras.layers.Dense(self.config.hidden_val)
        self.dense3=tf.keras.layers.Dense(out_dim, activation=tf.sigmoid, use_bias=True)

    def __call__(self, graph_ph):
        print('Tracing gnn network.')

        self.input_layer(graph_ph)
        x=self.start_layer()

        for layer in self.conv_layers:
            x=layer(x)

        nodes, symbols, clauses=x
        x=self.dense1(clauses)
        x = self.input_layer.clause_nums.collapse(x, [tf.math.segment_max, tf.math.segment_mean])
        #x=tf.reshape(x,shape=(1,-1))
        x=self.dense2(x)
        x=self.dense3(x)
        return x

class MultiGraphNetwork(tools.Module):
    def __init__(self, config=None):
        super().__init__()
        if(config is None):
            self.config=NetworkConfig()
        else:
            self.config=config

        #Main body
        self.input_layer=GraphInput()
        self.start_layer=graph_start(self.config.gnn_start_shape, self.input_layer)
        self.conv_layers=[
            graph_conv(self.input_layer, output_dims=self.config.gnn_next_shape) for _ in range(self.config.gnn_layers)
        ]

        #State body
        self._out_dim=self.config.gnn_hidden_val
        self.dense1=tf.keras.layers.Dense(self.config.gnn_hidden_val)
        self.dense2=tf.keras.layers.Dense(self.config.gnn_hidden_val)
        #self.dense3=tf.keras.layers.Dense(self.config.gnn_hidden_val, activation=tf.sigmoid, use_bias=True)
        self.dense3=tf.keras.layers.Dense(self.config.gnn_hidden_val, activation='', use_bias=True)

        #Action body
        self.ax_segments=Segments(nonzero=True)
        self.dense4=tf.keras.layers.Dense(self.config.gnn_hidden_act, activation=tf.sigmoid, use_bias=True)
        #self.dense5=tf.keras.layers.Dense(1, activation=tf.sigmoid, use_bias=True)

    def __call__(self, graph_ph):
        print('Tracing gnn network.')
        
        self.input_layer(graph_ph)
        x=self.start_layer()

        for layer in self.conv_layers:
            x=layer(x)

        return x

    @tf.function(input_signature=[sign])
    def stateEmbed(self, graph_ph):
        print('Tracing gnn state embed function.')
        
        bsign=sign
        #batch_size=graph_ph['node_inputs_1/lens'].shape[0]
        #for k in ['num_nodes', 'num_symbols', 'num_clauses']:
        #    sign[k]=tf.TensorSpec((batch_size,), dtype=tf.int32)
        if graph_ph['num_nodes'].shape!=(1,1):
            tf.nest.map_structure(lambda x, s: x.set_shape(s.shape), graph_ph, bsign)
            

        nodes, symbols, clauses=self.__call__(graph_ph)
        x=self.dense1(clauses)
        x = self.input_layer.clause_nums.collapse(x, [tf.math.segment_max, tf.math.segment_mean])
        x=self.dense2(x)
        x=self.dense3(x)
        return x

    @tf.function(input_signature=[sign])
    def actionEmbed(self, graph_ph):
        print('Tracing gnn action embed function.')
        nodes, symbols, clauses=self.__call__(graph_ph)
        cur_goals = self.input_layer.clause_nums.gather(clauses, 0)
        ci = self.input_layer.clause_inputs
        self.ax_segments(
            self.input_layer.clause_nums.segment_sum(
                ci.segments.lens
            )
        )

        ax_segments_masked, clauses_i_masked = self.ax_segments, ci.segments.segment_indices

        lit_i_masked = ci.data
        clauses_masked = tf.gather(clauses, clauses_i_masked)
        literals_masked = tf.gather(nodes, lit_i_masked)
        action_inputs = tf.concat(
            [clauses_masked, literals_masked, ax_segments_masked.fill(cur_goals)],
            axis = 1,
        )
        hidden = self.dense4(action_inputs)

        return hidden


    


        