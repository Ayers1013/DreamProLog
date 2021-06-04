from Env_ProLog import ProLog
from gnn.gnn_util import *


env=ProLog()
import tensorflow as tf
import tools
from gnn.graph_placeholder import GraphPlaceholder, feedGraphPlaceholder
from gnn.network import *
from gnn.stop_watch import StopWatch
tf.compat.v1.disable_eager_execution()


class GraphNetwork:
    def __init__(self):
        
        self.config=NetworkConfig()
        
        self.inputs=GraphPlaceholder()
        self.values=graphs_to_values(self.inputs, self.config)
        
        self.model=tf.keras.Model(inputs=self.inputs.entry, outputs=self.values)
        
    def __call__(self, data):
        
        inputs=feedGraphPlaceholder(data)
        
        values=self.model(inputs)
        
        return values
    

from gnn.graph_data import GraphData
data=GraphData()
data.load_from_str("1 1 1 1 1 1 1 1 0 1 0 1,0 1 2 3 4 3 1 0 3 3,-1 -1 -1 -1 -1 -1 2 -1 -1 -1 4 -1 -1 -1 -1 -1 8 -1 10 -1,-1 1 1 -1 1 1 -1 1 1 -1;0 0 1 0 1 0 0 0 1 0 1 0,3 3 3 3,3 -1 5 -1 9 -1 11 -1,-1 1 1 -1;0 0 0 0 0 0 0 0 0 0 0 0,,,;2 2 1 4 1,0 -1 -1 7 -1 -1 1 -1 -1 6 -1 -1 2 -1 -1 3 2 -1 5 4 -1 9 8 -1 11 10 -1 4 -1 -1,-1 1 1 -1 1 -1 1 1 -1 1;2 2 0 1 0 1 2 1 0 1 0 1,0 1 2 4 2 3 3 5 4 4 5;1 1 2 2 3 2,0 0 1 3 5 6 7 1 9 6 11;0 0 1 0 1 0 0 0 3 0 3 0;1 1 0 1 0;0 1 3 3 3 3;0 0 0 0 0 0 1 0 0 0 0")


graphNet=GraphNetwork()

tf.reduce_sum(graphNet([data]))
