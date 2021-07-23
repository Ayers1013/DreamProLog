import tensorflow as tf
from gnn.graph_data import GraphData
from gnn.segments import SegmentsPH

class GraphHyperEdgesAPH(tf.Module):
    def __init__(self):
        super().__init__()
        self.segments=SegmentsPH()

    """
    def __call__(self, x):
        self.segments(x.lens)
        self.symbols=x.symbols
        self.nodes=x.nodes
        self.sgn=x.sgn
    """

    def __call__(self, x):
        self.segments(x['lens'])
        self.symbols=x['symbols']
        self.nodes=x['nodes']
        self.sgn=x['sgn']

class GraphHyperEdgesBPH(tf.Module):
    def __init__(self):
        super().__init__()
        self.segments=SegmentsPH(nonzero=True)

    """
    def __call__(self, x):
        self.segments(x.lens)
        self.nodes=x.nodes
        self.sgn=x.sgn
        """
    
    def __call__(self, x):
        self.segments(x['lens'])
        self.nodes=x['nodes']
        self.sgn=x['sgn']

class GraphEdgesPH(tf.Module):
    def __init__(self, nonzero=False):
        super().__init__()
        self.segments=SegmentsPH(nonzero=nonzero)

    """
    def __call__(self, x):
        self.segments(x.lens)
        self.data=x.data
        """
    
    def __call__(self, x):
        self.segments(x['lens'])
        self.data=x['data']


class GraphInput(tf.Module):
    def __init__(self):
        super().__init__()
        self.node_inputs=tuple(
            GraphHyperEdgesAPH() for _ in range(3)
        )

        self.symbol_inputs = GraphHyperEdgesBPH()
        self.node_c_inputs = GraphEdgesPH()
        self.clause_inputs = GraphEdgesPH(nonzero = True)

        self.node_nums   = SegmentsPH( nonzero = True)
        self.symbol_nums = SegmentsPH( nonzero = True)
        self.clause_nums = SegmentsPH( nonzero = True)

    def __call__(self, x):

        self.node_nums(x['num_nodes'])
        self.symbol_nums(x['num_symbols'])
        self.clause_nums(x['num_clauses'])

        self.ini_nodes=x['ini_nodes']
        self.ini_symbols=x['ini_symbols']
        self.ini_clauses=x['ini_clauses']
        #self.axiom_mask=x['axiom_mask']

        extract= lambda name: {k.split("/")[1]: v for k,v in x.items() if k.split("/")[0]==name}

        for ind, layer in enumerate(self.node_inputs):
            layer(extract("node_inputs_{}".format(ind+1)))

        self.symbol_inputs(extract("symbol_inputs"))
        self.node_c_inputs(extract("node_c_inputs"))
        self.clause_inputs(extract("clause_inputs"))
