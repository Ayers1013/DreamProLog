from .gnn_util import input2actionGraph, input2graph
from .network import GraphNetwork, MultiGraphNetwork
from .graph_data import GraphData

exctractImage=lambda prolog, input: input2graph(prolog, input).convert_to_dict()
extractActions=lambda prolog, input: input2actionGraph(prolog, input).convert_to_dict()