from .gnn_util import input2actionGraph, input2graph
from .network import GraphNetwork, MultiGraphNetwork, gnn_output_sign
from .graph_data import GraphData
from .graph_input import feed_gnn_input
from .graph_tensor import GraphTensor

exctractImage=lambda prolog, input: input2graph(prolog, input).convert_to_dict()
extractActions=lambda prolog, input: input2actionGraph(prolog, input).convert_to_dict()