import tensorflow as tf
from gnn.graph_data import GraphData
from gnn.segments import SegmentsPH

class GraphHyperEdgesAPH:
    def __init__(self):
        with tf.name_scope("GraphHyperEdgesA"):
            self.segments = SegmentsPH(data_shape = None)
            self.symbols = tf.keras.Input(shape=(), name="symbols", dtype=tf.int32)
            self.nodes = tf.keras.Input(shape=(2), name="nodes", dtype=tf.int32)
            self.sgn = tf.keras.Input(shape=(), name="sgn", dtype=tf.float32)

    @property
    def entry(self):
        return [
            self.segments.lens,
            self.symbols,
            self.nodes,
            self.sgn,
        ]

def feedGraphHyperEdgesAPH(data):
    return [
        data.lens,
        data.symbols,
        data.nodes,
        data.sqn,
    ]

class GraphHyperEdgesBPH:
    def __init__(self):
        with tf.name_scope("GraphHyperEdgesB"):
            self.segments = SegmentsPH(data_shape = None, nonzero = True)
            self.nodes = tf.keras.Input(shape=(3), name="nodes", dtype=tf.int32)
            self.sgn = tf.keras.Input(shape=(), name="sgn", dtype=tf.float32)

    @property
    def entry(self):
        return [
            self.segments.lens,
            self.nodes,
            self.sgn,
        ]

def feedGraphHyperEdgesBPH(data):
    return [
        data.lens,
        data.nodes,
        data.sgn,
    ]


class GraphEdgesPH:
    def __init__(self, nonzero = False):
        with tf.name_scope("GraphEdges"):
            self.segments = SegmentsPH(data_shape = None, nonzero = nonzero)
            self.data = tf.keras.Input(shape=(), name="data", dtype=tf.int32)

    @property
    def entry(self):
        return [
            self.segments.lens,
            self.data,
        ]

def feedGraphEdgesPH(data):
    return [
        data.lens,
        data.data,
    ]


class GraphPlaceholder():
    def __init__(self):
        with tf.name_scope("GraphPlaceholder"):
            self.node_inputs = tuple(
                GraphHyperEdgesAPH() for _ in range(3)
            )
            self.symbol_inputs = GraphHyperEdgesBPH()
            self.node_c_inputs = GraphEdgesPH()
            self.clause_inputs = GraphEdgesPH(nonzero = True)
            self.ini_nodes = tf.keras.Input(shape=(), name="ini_nodes", dtype=tf.int32)
            self.ini_symbols = tf.keras.Input(shape=(), name="ini_symbols", dtype=tf.int32)
            self.ini_clauses = tf.keras.Input(shape=(), name="ini_clauses", dtype=tf.int32)

            self.node_nums   = SegmentsPH(data_shape = None, nonzero = True)
            self.symbol_nums = SegmentsPH(data_shape = None, nonzero = True)
            self.clause_nums = SegmentsPH(data_shape = None, nonzero = True)
            self.axiom_mask = tf.keras.Input(shape=(), name="axiom_mask", dtype=tf.int32)
        
    @property
    def entry(self):
        res=[
            self.node_nums.lens,   
            self.symbol_nums.lens,
            self.clause_nums.lens,
            self.ini_nodes,
            self.ini_symbols,
            self.ini_clauses,
            self.axiom_mask,
        ]

        for hedges_ph in self.node_inputs:
            res+=hedges_ph.entry

        res+=self.symbol_inputs.entry
        res+=self.node_c_inputs.entry
        res+=self.clause_inputs.entry

        return res



def feedGraphPlaceholder(self, batch, non_destructive = False):
    if non_destructive: batch = [g.clone() for g in batch]
    node_nums = [g.num_nodes for g in batch]
    symbol_nums = [g.num_symbols for g in batch]
    clause_nums = [g.num_clauses for g in batch]

    data = GraphData.ini_list()
    for g in batch: data.append(g)
    data.flatten()

    res = [
        node_nums,
        symbol_nums,
        clause_nums,
        data.ini_nodes,
        data.ini_symbols,
        data.ini_clauses,
        data.axiom_mask,
    ]

    for hedges in data.node_inputs:
        res+=feedGraphHyperEdgesAPH(hedges)
    
    res+=feedGraphHyperEdgesBPH(data.symbol_inputs)
    res+=feedGraphEdgesPH(data.node_c_inputs)
    res+=feedGraphEdgesPH(data.clause_inputs)
    
    return res
