#from Env_ProLog import ProLog
#from gnn.gnn_util import *


#env=ProLog()
import tensorflow as tf
import tools
from gnn.graph_placeholder import GraphPlaceholder
from gnn.network import *
from gnn.stop_watch import StopWatch
tf.compat.v1.disable_eager_execution()

from gnn.graph_data import GraphData
data=GraphData()
data.load_from_str("1 1 1 1 1 1 1 1 0 1 0 1,0 1 2 3 4 3 1 0 3 3,-1 -1 -1 -1 -1 -1 2 -1 -1 -1 4 -1 -1 -1 -1 -1 8 -1 10 -1,-1 1 1 -1 1 1 -1 1 1 -1;0 0 1 0 1 0 0 0 1 0 1 0,3 3 3 3,3 -1 5 -1 9 -1 11 -1,-1 1 1 -1;0 0 0 0 0 0 0 0 0 0 0 0,,,;2 2 1 4 1,0 -1 -1 7 -1 -1 1 -1 -1 6 -1 -1 2 -1 -1 3 2 -1 5 4 -1 9 8 -1 11 10 -1 4 -1 -1,-1 1 1 -1 1 -1 1 1 -1 1;2 2 0 1 0 1 2 1 0 1 0 1,0 1 2 4 2 3 3 5 4 4 5;1 1 2 2 3 2,0 0 1 3 5 6 7 1 9 6 11;0 0 1 0 1 0 0 0 3 0 3 0;1 1 0 1 0;0 1 3 3 3 3;0 0 0 0 0 0 1 0 0 0 0")

class GraphNetwork(tf.Module):
    def __init__(self):
        
        self.config=NetworkConfig()
        
        self.graph=tf.Graph()
        self.graph.seed=42
        
        configProto = tf.compat.v1.ConfigProto(inter_op_parallelism_threads=self.config.threads,
                                          intra_op_parallelism_threads=self.config.threads)
        configProto.gpu_options.allow_growth = True
        self.session = tf.compat.v1.Session(graph = self.graph, config=configProto)
        
        with self.session.graph.as_default():

            self.structure = GraphPlaceholder()
            
            self.value_predictions= graphs_to_values(self.structure, self.config)
            
            self.session.run(tf.compat.v1.global_variables_initializer())
            
        self.session.graph.finalize()
                
        
    def __call__(self, batch,  non_destructive = True):
        with StopWatch("data preparation"):
            #d = self.structure.feed([s.graph_data for s in batch], non_destructive)
            d = self.structure.feed(batch, non_destructive)
        with StopWatch("network"):
            return self.session.run((self.value_predictions), d)
        
    def variables(self):
        return [op for op in self.session.graph.get_operations() if op.op_def and op.op_def.name=='VarHandleOp']


graphNet=GraphNetwork()

print(graphNet([data]))