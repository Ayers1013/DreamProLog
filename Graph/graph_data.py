import tensorflow as tf
import numpy as np

class DataSkeleton:

    def __init__(self, data):
        for attr in self.__slots__:
            #assert attr in data
            setattr(self, attr, data[attr])

    def copy(self):
        copy=self.__class__()
        set_fn=lambda attr, data: setattr(copy, attr, data)
        for attr in self.__slots__:
            data=getattr(self, attr)
            if isinstance(data, DataSkeleton) or isinstance(data, np.ndarray):
                set_fn(attr, data.copy())
            else: 
                set_fn(attr, data)
    
    def to_dict(self):
        dct = {}
        for attr in self.__slots__:
            x=getattr(self, attr)
            if isinstance(x, DataSkeleton):
                dct[attr] = x.to_dict()
            
            else:
                dct[attr]=x
        return dct

    def apply_copy(self, fn):
        copy=self.__class__()
        set_fn=lambda attr, data: setattr(copy, attr, data)
        for attr in self.__slots__:
            data=getattr(self, attr)
            if isinstance(data, DataSkeleton):
                set_fn(attr, fn(data))
            else: 
                set_fn(attr, data)

class TupleSkeleton(DataSkeleton, Tuple):
    def __init__(self, data):
        super().__init__(data)

class GraphHyperEdgesA(DataSkeleton):
    __slots__ = ['lens', 'symbols', 'nodes', 'sgn']
    def __init__(self, data):
        super().__init__(data)

class GraphHyperEdgesB(DataSkeleton):
    __slots__ = ['lens', 'nodes', 'sgn']
    def __init__(self, data):
        super().__init__(data)

class GraphEdges(DataSkeleton):
    __slots__ = ['lens', 'data']
    def __init__(self, data):
        super().__init__(data)

class GraphData(DataSkeleton):
    __slots__ = ['node_inputs_1', 'node_inputs_2', 'node_inputs_3', 'symbol_inputs',
                 'node_c_inputs', 'clause_inputs',
                 'ini_nodes', 'ini_symbols', 'ini_clauses',
                 'num_nodes', 'num_symbols', 'num_clauses']
    def __init__(self, data):
        self.node_inputs_1=GraphHyperEdgesA(data[f'node_inputs_1'])
        self.node_inputs_2=GraphHyperEdgesA(data[f'node_inputs_2'])
        self.node_inputs_3=GraphHyperEdgesA(data[f'node_inputs_3'])
        self.symbol_inputs=GraphHyperEdgesB(data['symbol_inputs'])
        self.node_c_inputs=GraphEdges(data['node_c_inputs'])
        self.clause_inputs=GraphEdges(data['clause_inputs'])
        for attr in [
            'ini_nodes', 'ini_symbols', 'ini_clauses',
            'num_nodes', 'num_symbols', 'num_clauses']:
            setattr(self, attr, data[attr])


class GraphTensor(GraphData):
    def __init__(self, data):
        super(data).__init__()


    


