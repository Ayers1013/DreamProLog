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
    def __init__(self, config=None):
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
        self.dense3=tf.keras.layers.Dense(32, activation=tf.sigmoid, use_bias=True)

    #@tf.function
    def __call__(self, graph_ph):
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


        