import tensorflow as tf
from gnn_keras.graph_data import GraphData
from gnn_keras.segments import SegmentsPH

class GraphHyperEdgesAPH:
    def __init__(self):
        with tf.name_scope("GraphHyperEdgesA"):
            self.segments = SegmentsPH(data_shape = None)
            self.symbols = tf.keras.Input(shape=[None], name="symbols", dtype=tf.int32)
            self.nodes = tf.keras.Input(shape=[None,2], name="nodes", dtype=tf.int32)
            self.sgn = tf.keras.Input(shape=[None], name="sgn", dtype=tf.float32)

    def feed(self, data):
        return {
            self.segments.lens : data.lens,
            self.symbols : data.symbols,
            self.nodes : data.nodes,
            self.sgn : data.sgn,
        }

class GraphHyperEdgesBPH:
    def __init__(self):
        with tf.name_scope("GraphHyperEdgesB"):
            self.segments = SegmentsPH(data_shape = None, nonzero = True)
            self.nodes = tf.keras.Input(shape=[None,3], name="nodes", dtype=tf.int32)
            self.sgn = tf.keras.Input(shape=[None], name="sgn", dtype=tf.float32)

    def feed(self, data):
        return {
            self.segments.lens : data.lens,
            self.nodes : data.nodes,
            self.sgn : data.sgn,
        }

class GraphEdgesPH:
    def __init__(self, nonzero = False):
        with tf.name_scope("GraphEdges"):
            self.segments = SegmentsPH(data_shape = None, nonzero = nonzero)
            self.data = tf.keras.Input(shape=[None], name="data", dtype=tf.int32)

    def feed(self, data):
        return {
            self.segments.lens : data.lens,
            self.data : data.data,
        }

class GraphPlaceholder():
    def __init__(self):
        with tf.name_scope("GraphPlaceholder"):
            self.node_inputs = tuple(
                GraphHyperEdgesAPH() for _ in range(3)
            )
            self.symbol_inputs = GraphHyperEdgesBPH()
            self.node_c_inputs = GraphEdgesPH()
            self.clause_inputs = GraphEdgesPH(nonzero = True)
            self.ini_nodes = tf.keras.Input(shape=[None], name="ini_nodes", dtype=tf.int32)
            self.ini_symbols = tf.keras.Input(shape=[None], name="ini_symbols", dtype=tf.int32)
            self.ini_clauses = tf.keras.Input(shape=[None], name="ini_clauses", dtype=tf.int32)

            self.node_nums   = SegmentsPH(data_shape = None, nonzero = True)
            self.symbol_nums = SegmentsPH(data_shape = None, nonzero = True)
            self.clause_nums = SegmentsPH(data_shape = None, nonzero = True)
            self.axiom_mask = tf.keras.Input(shape=[None], name="axiom_mask", dtype=tf.int32)

    def feed(self, batch, non_destructive = False):
        if non_destructive: batch = [g.clone() for g in batch]
        node_nums = [g.num_nodes for g in batch]
        symbol_nums = [g.num_symbols for g in batch]
        clause_nums = [g.num_clauses for g in batch]

        data = GraphData.ini_list()
        for g in batch: data.append(g)
        data.flatten()

        res = {
            self.node_nums.lens   : node_nums,
            self.symbol_nums.lens : symbol_nums,
            self.clause_nums.lens : clause_nums,
            self.ini_nodes        : data.ini_nodes,
            self.ini_symbols      : data.ini_symbols,
            self.ini_clauses      : data.ini_clauses,
            self.axiom_mask       : data.axiom_mask,
        }

        for hedges_ph, hedges in zip(self.node_inputs, data.node_inputs):
            res.update(hedges_ph.feed(hedges))

        res.update(self.symbol_inputs.feed(data.symbol_inputs))
        res.update(self.node_c_inputs.feed(data.node_c_inputs))
        res.update(self.clause_inputs.feed(data.clause_inputs))
        
        return res
