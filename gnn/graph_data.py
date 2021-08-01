import numpy as np
from numpy.lib.function_base import _percentile_dispatcher

def str_to_npa(s):
    if s == "": return np.zeros([0], dtype = int)
    return np.array(list(map(int,s.split(' '))))
def npa_to_str(a):
    return ' '.join(map(str,a))

class GraphHyperEdgesA():
    __slots__ = ['lens', 'symbols', 'nodes', 'sgn']

    def __init__(self, raw_data = None):
        if raw_data is not None:
            lens, (symbols, nodes, sgn) = raw_data
            self.lens = np.array(lens)
            self.symbols = np.array(symbols)
            self.nodes = np.array(nodes).reshape((-1,2))
            self.sgn = np.array(sgn)

    def clone(self): # works for ordinary (non-list) version only
        copy = GraphHyperEdgesA()
        copy.lens = np.array(self.lens)
        copy.symbols = np.array(self.symbols)
        copy.nodes = np.array(self.nodes)
        copy.sgn = np.array(self.sgn)
        return copy

    @staticmethod
    def ini_list():
        res = GraphHyperEdgesA()
        res.lens = []
        res.symbols = []
        res.nodes = []
        res.sgn = []
        return res

    def append(self, other): # self : list, other : ordinary
        self.lens.append(other.lens)
        self.symbols.append(other.symbols)
        self.nodes.append(other.nodes)
        self.sgn.append(other.sgn)

    def flatten(self): # list version -> ordinary version
        self.lens    = np.concatenate(self.lens)
        self.symbols = np.concatenate(self.symbols)
        self.nodes   = np.concatenate(self.nodes)
        self.sgn     = np.concatenate(self.sgn)

    # saving / loading

    def __str__(self):
        return ','.join(map(npa_to_str, (
            self.lens, self.symbols, self.nodes.flatten(), self.sgn
        )))

    def load_from_str(self, s):
        self.lens, self.symbols, self.nodes, self.sgn = \
            map(str_to_npa, s.split(','))
        self.nodes = self.nodes.reshape((-1,2))
        return self

    def load_from_dict(self, d, prefix):
        self.lens=np.array(d[prefix+"lens"])
        self.symbols=np.array(d[prefix+"symbols"])
        self.nodes=np.array(d[prefix+"nodes"])
        self.sgn=np.array(d[prefix+"sgn"])

        self.nodes = self.nodes.reshape((-1,2))
        return self

    def convert_to_dict(self, prefix):
        d={}
        d[prefix+'lens']=self.lens.copy()
        d[prefix+'symbols']=self.symbols.copy()
        d[prefix+'nodes']=self.nodes.copy()
        d[prefix+'sgn']=self.sgn.copy()
        return d


class GraphHyperEdgesB():
    __slots__ = ['lens', 'nodes', 'sgn']

    def __init__(self, data = None):
        if data is not None:
            lens, (nodes, sgn) = data
            self.lens = np.array(lens)
            self.nodes = np.array(nodes).reshape((-1,3))
            self.sgn = np.array(sgn)

    def clone(self): # works for ordinary (non-list) version only
        copy = GraphHyperEdgesB()
        copy.lens = np.array(self.lens)
        copy.nodes = np.array(self.nodes)
        copy.sgn = np.array(self.sgn)
        return copy

    @staticmethod
    def ini_list():
        res = GraphHyperEdgesB()
        res.lens = []
        res.nodes = []
        res.sgn = []
        return res

    def append(self, other): # self : list, other : ordinary
        self.lens.append(other.lens)
        self.nodes.append(other.nodes)
        self.sgn.append(other.sgn)

    def flatten(self): # list version -> ordinary version
        self.lens  = np.concatenate(self.lens)
        self.nodes = np.concatenate(self.nodes)
        self.sgn   = np.concatenate(self.sgn)

    # saving / loading

    def __str__(self):
        return ','.join(map(npa_to_str, (
            self.lens, self.nodes.flatten(), self.sgn
        )))

    def load_from_str(self, s):
        self.lens, self.nodes, self.sgn = \
            map(str_to_npa, s.split(','))
        self.nodes = self.nodes.reshape((-1,3))
        return self
    def load_from_dict(self, d, prefix):
        self.lens=np.array(d[prefix+"lens"])
        self.nodes=np.array(d[prefix+"nodes"])
        self.sgn=np.array(d[prefix+"sgn"])

        self.nodes = self.nodes.reshape((-1,3))
        return self

    def convert_to_dict(self, prefix):
        d={}
        d[prefix+'lens']=self.lens.copy()
        d[prefix+'nodes']=self.nodes.copy()
        d[prefix+'sgn']=self.sgn.copy()
        return d

class GraphEdges():
    __slots__ = ['lens', 'data']

    def __init__(self, raw_data = None):
        if raw_data is not None:
            lens, data = raw_data
            self.lens = np.array(lens)
            self.data = np.array(data)

    def clone(self): # works for ordinary (non-list) version only
        copy = GraphEdges()
        copy.lens = np.array(self.lens)
        copy.data = np.array(self.data)
        return copy

    @staticmethod
    def ini_list():
        res = GraphEdges()
        res.lens = []
        res.data = []
        return res

    def append(self, other): # self : list, other : ordinary
        self.lens.append(other.lens)
        self.data.append(other.data)

    def flatten(self): # list version -> ordinary version
        self.lens = np.concatenate(self.lens)
        self.data = np.concatenate(self.data)

    # saving / loading

    def __str__(self):
        return ','.join(map(npa_to_str, (
            self.lens, self.data
        )))

    def load_from_str(self, s):
        self.lens, self.data = \
            map(str_to_npa, s.split(','))
        return self

    def load_from_dict(self, d, prefix):
        self.lens, self.data = (
            np.array(d[prefix+name])
            for name in ("lens", "data")
        )
        self.lens=np.array(d[prefix+"lens"])
        self.data=np.array(d[prefix+"data"])
        return self

    def convert_to_dict(self, prefix):
        d={}
        d[prefix+'lens']=self.lens.copy()
        d[prefix+'data']=self.data.copy()
        return d

class GraphData():
    __slots__ = ['node_inputs', 'symbol_inputs',
                 'node_c_inputs', 'clause_inputs',
                 'ini_nodes', 'ini_symbols', 'ini_clauses',
                 'num_nodes', 'num_symbols', 'num_clauses',
                 'axiom_mask']

    def __init__(self, raw_data = None):
        if raw_data is None: return
        elif isinstance(raw_data, str): self.load_from_str(raw_data)
        elif isinstance(raw_data, dict): self.load_from_dict(raw_data)
        else:
            symb_edges, (node_c_inputs, clause_inputs),\
                (ini_nodes, ini_symbols, ini_clauses), axiom_mask = raw_data

            self.node_inputs = tuple(map(GraphHyperEdgesA, symb_edges[:-1]))
            self.symbol_inputs = GraphHyperEdgesB(symb_edges[-1])
            self.node_c_inputs = GraphEdges(node_c_inputs)
            self.clause_inputs = GraphEdges(clause_inputs)
            self.ini_nodes = np.array(ini_nodes)
            self.ini_symbols = np.array(ini_symbols)
            self.ini_clauses = np.array(ini_clauses)
            self.num_nodes = len(ini_nodes)
            self.num_symbols = len(ini_symbols)
            self.num_clauses = len(ini_clauses)
            self.axiom_mask = np.array(axiom_mask)

    def clone(self): # works for ordinary (non-list) version only
        copy = GraphData()
        copy.node_inputs  = tuple(hedges.clone() for hedges in self.node_inputs)
        copy.symbol_inputs = self.symbol_inputs.clone()
        copy.node_c_inputs  = self.node_c_inputs.clone()
        copy.clause_inputs  = self.clause_inputs.clone()
        copy.ini_nodes = np.array(self.ini_nodes)
        copy.ini_symbols = np.array(self.ini_symbols)
        copy.ini_clauses = np.array(self.ini_clauses)
        copy.num_nodes = self.num_nodes
        copy.num_symbols = self.num_symbols
        copy.num_clauses = self.num_clauses
        copy.axiom_mask = np.array(self.axiom_mask)
        return copy

    def shift(self, node_shift, symbol_shift, clause_shift):

        for hedges in self.node_inputs:
            hedges.symbols += symbol_shift
            hedges.nodes[hedges.nodes >= 0] += node_shift
        self.symbol_inputs.nodes[self.symbol_inputs.nodes >= 0] += node_shift
        self.node_c_inputs.data += clause_shift
        self.clause_inputs.data += node_shift

    @staticmethod
    def ini_list():
        res = GraphData()
        res.node_inputs = tuple(GraphHyperEdgesA.ini_list() for _ in range(3))
        res.symbol_inputs = GraphHyperEdgesB.ini_list()
        res.node_c_inputs = GraphEdges.ini_list()
        res.clause_inputs = GraphEdges.ini_list()
        res.ini_nodes = []
        res.ini_symbols = []
        res.ini_clauses = []
        res.num_nodes = 0
        res.num_symbols = 0
        res.num_clauses = 0
        res.axiom_mask = []
        return res

    # WARNING: self.append(other) modifies (shifts) "other",
    # for non-invasive version use self.append(other.clone())
    def append(self, other):
        other.shift(self.num_nodes, self.num_symbols, self.num_clauses)
        for hedges, hedges_o in zip(self.node_inputs, other.node_inputs):
            hedges.append(hedges_o)
        self.symbol_inputs.append(other.symbol_inputs)
        self.node_c_inputs.append(other.node_c_inputs)
        self.clause_inputs.append(other.clause_inputs)
        self.ini_nodes.append(other.ini_nodes)
        self.ini_symbols.append(other.ini_symbols)
        self.ini_clauses.append(other.ini_clauses)
        self.num_nodes += other.num_nodes
        self.num_symbols += other.num_symbols
        self.num_clauses += other.num_clauses
        self.axiom_mask.append(other.axiom_mask)

    def flatten(self): # list version -> ordinary version
        for hedges in self.node_inputs: hedges.flatten()
        self.symbol_inputs.flatten()
        self.node_c_inputs.flatten()
        self.clause_inputs.flatten()
        self.ini_nodes   = np.concatenate(self.ini_nodes)
        self.ini_symbols = np.concatenate(self.ini_symbols)
        self.ini_clauses = np.concatenate(self.ini_clauses)
        self.axiom_mask = np.concatenate(self.axiom_mask)

    def __str__(self):
        str_data = [hedges.__str__() for hedges in self.node_inputs]
        str_data += [
            self.symbol_inputs.__str__(),
            self.node_c_inputs.__str__(),
            self.clause_inputs.__str__(),
            npa_to_str(self.ini_nodes),
            npa_to_str(self.ini_symbols),
            npa_to_str(self.ini_clauses),
            npa_to_str(self.axiom_mask),
        ]
        return ';'.join(str_data)

    def load_from_str(self, s):
        str_data = s.split(';')
        self.node_inputs = tuple(
            GraphHyperEdgesA().load_from_str(s)
            for s in str_data[:3]
        )
        self.symbol_inputs = GraphHyperEdgesB().load_from_str(str_data[3])
        self.node_c_inputs = GraphEdges().load_from_str(str_data[4])
        self.clause_inputs = GraphEdges().load_from_str(str_data[5])

        self.ini_nodes, self.ini_symbols, self.ini_clauses, self.axiom_mask = \
            map(str_to_npa, str_data[6:])
        #self.axiom_mask = np.ones(len(self.clause_inputs.data), dtype = int)

        self.num_nodes = len(self.ini_nodes)
        self.num_symbols = len(self.ini_symbols)
        self.num_clauses = len(self.ini_clauses)


        return self

    def load_from_dict(self, d, prefix = ""):

        self.node_inputs = tuple(
            GraphHyperEdgesA().load_from_dict(d, "{}node_inputs_{}/".format(prefix, i+1))
            for i in range(3)
        )
        self.symbol_inputs = GraphHyperEdgesB().load_from_dict(d, prefix+"symbol_inputs/")
        self.node_c_inputs = GraphEdges().load_from_dict(d, prefix+"node_c_inputs/")
        self.clause_inputs = GraphEdges().load_from_dict(d, prefix+"clause_inputs/")
        """
        self.ini_nodes, self.ini_symbols, self.ini_clauses = (
            np.array(d[prefix+name])
            for name in ("ini_nodes", "ini_symbols", "ini_clauses")
        )
        """
        self.ini_nodes=np.array(d[prefix+"ini_nodes"])
        self.ini_symbols=np.array(d[prefix+"ini_symbols"])
        self.ini_clauses=np.array(d[prefix+"ini_clauses"])


        self.axiom_mask = np.ones(len(self.clause_inputs.data), dtype = int)

        self.num_nodes = len(self.ini_nodes)
        self.num_symbols = len(self.ini_symbols)
        self.num_clauses = len(self.ini_clauses)
        return self

    def convert_to_dict(self, prefix=""):
        d={}

        for i in range(3):
            d.update(self.node_inputs[i].convert_to_dict("{}node_inputs_{}/".format(prefix, i+1)))

        for name, inst in [
            (prefix+'symbol_inputs/', self.symbol_inputs),
            (prefix+'node_c_inputs/', self.node_c_inputs),
            (prefix+'clause_inputs/', self.clause_inputs)
        ]:
            d.update(inst.convert_to_dict(name))
        
        for name, inst in [
            (prefix+'ini_nodes', self.ini_nodes),
            (prefix+'ini_symbols', self.ini_symbols),
            (prefix+'ini_clauses', self.ini_clauses)
        ]:
            d[name]=inst.copy()
        
        include_nums=True
        if include_nums:
            for name, inst in [
                (prefix+'num_nodes', self.num_nodes),
                (prefix+'num_symbols', self.num_symbols),
                (prefix+'num_clauses', self.num_clauses)
            ]:
                d[name]=[inst]
        return d

        
        
