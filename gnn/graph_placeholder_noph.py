import tensorflow as tf
from gnn.graph_data import GraphData
from gnn.segments import SegmentsPH

class GraphHyperEdgesAPH:
    def __init__(self):
        with tf.name_scope("GraphHyperEdgesA"):
            self.segments = SegmentsPH(data_shape = None)
            self.symbols = None
            self.nodes = None
            self.sgn = None

    def feed(self, data):
        self.segments.lens = data.lens
        self.symbols = data.symbols
        self.nodes = data.nodes
        self.sgn = data.sgn

class GraphHyperEdgesBPH:
    def __init__(self):
        with tf.name_scope("GraphHyperEdgesB"):
            self.segments = SegmentsPH(data_shape = None, nonzero = True)
            self.nodes = None
            self.sgn = None

    def feed(self, data):
        self.segments.lens = data.lens
        self.nodes = data.nodes
        self.sgn = data.sgn

class GraphEdgesPH:
    def __init__(self, nonzero = False):
        with tf.name_scope("GraphEdges"):
            self.segments = SegmentsPH(data_shape = None, nonzero = nonzero)
            self.data = None

    def feed(self, data):
        self.segments.lens = data.lens
        self.data = data.data

class GraphPlaceholder():
    def __init__(self):
        with tf.name_scope("GraphPlaceholder"):
            self.node_inputs = tuple(
                GraphHyperEdgesAPH() for _ in range(3)
            )
            self.symbol_inputs = GraphHyperEdgesBPH()
            self.node_c_inputs = GraphEdgesPH()
            self.clause_inputs = GraphEdgesPH(nonzero = True)
            self.ini_nodes = None
            self.ini_symbols = None
            self.ini_clauses = None

            self.node_nums   = SegmentsPH(data_shape = None, nonzero = True)
            self.symbol_nums = SegmentsPH(data_shape = None, nonzero = True)
            self.clause_nums = SegmentsPH(data_shape = None, nonzero = True)
            self.axiom_mask = None

    def feed(self, batch, non_destructive = False):
        if non_destructive: batch = [g.clone() for g in batch]
        node_nums = [g.num_nodes for g in batch]
        symbol_nums = [g.num_symbols for g in batch]
        clause_nums = [g.num_clauses for g in batch]

        data = GraphData.ini_list()
        for g in batch: data.append(g)
        data.flatten()

        self.node_nums.lens   = node_nums
        self.symbol_nums.lens = symbol_nums
        self.clause_nums.lens = clause_nums
        self.ini_nodes        = data.ini_nodes
        self.ini_symbols      = data.ini_symbols
        self.ini_clauses      = data.ini_clauses
        self.axiom_mask       = data.axiom_mask

        for hedges_ph, hedges in zip(self.node_inputs, data.node_inputs):
            hedges_ph.feed(hedges)

        self.symbol_inputs.feed(data.symbol_inputs)
        self.node_c_inputs.feed(data.node_c_inputs)
        self.clause_inputs.feed(data.clause_inputs)
