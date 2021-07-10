import tensorflow as tf
import numpy as np
from tensorflow.python.keras.engine import input_layer

# from gnn.tf_helpers import *

from gnn.graph_input import GraphInput
from gnn.segments import Segments
from gnn.graph_conv import graph_start, graph_conv
from gnn.graph_data import GraphData
import tools

class NetworkConfig:
    def __init__(self):
        self.threads = 4
        self.start_shape = (4,1,4)
        self.next_shape = (32,64,32)
        #self.next_shape = (11,12,13)
        self.layers = 10
        self.hidden_val = 64
        self.hidden_act = 64
        self.entropy_regularization = 0.1


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

    @tf.function(experimental_relax_shapes=True)
    def __call__(self, graph_ph):
        print('Tracing gnn network.')
        #From string
        #data=GraphData()
        #data.load_from_str(graph_ph[0])
        #graph_ph=[data]

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
    def __init__(self, out_dim=32, config=None):
        super().__init__()
        if(config is None):
            self.config=NetworkConfig()
        else:
            self.config=config

        #Main body
        self.input_layer=GraphInput()
        self.start_layer=graph_start(self.config.start_shape, self.input_layer)
        self.conv_layers=[
            graph_conv(self.input_layer, output_dims=self.config.next_shape) for _ in range(self.config.layers)
        ]

        #State body
        self.dense1=tf.keras.layers.Dense(self.config.hidden_val)
        self.dense2=tf.keras.layers.Dense(self.config.hidden_val)
        self.dense3=tf.keras.layers.Dense(out_dim, activation=tf.sigmoid, use_bias=True)

        #Action body
        self.ax_segments=Segments(nonzero=True)
        self.dense4=tf.keras.layers.Dense(self.config.hidden_act, activation=tf.sigmoid, use_bias=True)
        #self.dense5=tf.keras.layers.Dense(1, activation=tf.sigmoid, use_bias=True)

    def __call__(self, graph_ph):
        print('Tracing gnn network.')

        self.input_layer(graph_ph)
        x=self.start_layer()

        for layer in self.conv_layers:
            x=layer(x)

        return x

    @tf.function(experimental_relax_shapes=True)
    def stateEmbed(self, graph_ph):
        nodes, symbols, clauses=self.__call__(graph_ph)
        x=self.dense1(clauses)
        x = self.input_layer.clause_nums.collapse(x, [tf.math.segment_max, tf.math.segment_mean])
        x=self.dense2(x)
        x=self.dense3(x)
        return x

    @tf.function(experimental_relax_shapes=True)
    def actionEmbed(self, graph_ph):
        nodes, symbols, clauses=self.__call__(graph_ph)
        cur_goals = self.input_layer.clause_nums.gather(clauses, 0)
        ci = self.input_layer.clause_inputs
        self.ax_segments(
            self.input_layer.clause_nums.segment_sum(
                ci.segments.lens
            )
        )

        
        """mask = tf.ones(shape=(200,))
        ax_segments_masked, clauses_i_masked = self.ax_segments.mask_data(
            ci.segments.segment_indices,
            mask, nonzero = True,
        )"""
        ax_segments_masked, clauses_i_masked = self.ax_segments, ci.segments.segment_indices

        lit_i_masked = ci.data
        clauses_masked = tf.gather(clauses, clauses_i_masked)
        literals_masked = tf.gather(nodes, lit_i_masked)
        action_inputs = tf.concat(
            [clauses_masked, literals_masked, ax_segments_masked.fill(cur_goals)],
            axis = 1,
        )
        hidden = self.dense4(action_inputs)
        
        #logits = tf.squeeze(self.dense5(hidden))
        #
        #log_probs = ax_segments_masked.log_softmax(logits)
        #probs = tf.exp(log_probs)
        #sampled_actions = ax_segments_masked.sample(probs)

        return hidden


    


        